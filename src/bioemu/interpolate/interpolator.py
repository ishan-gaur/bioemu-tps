import os
import yaml
import hydra

import torch
import numpy as np
from rmsd import kabsch_rotate

from bioemu.datasets.fastfolders import FastFolderTrajectory
from bioemu.models import DiGConditionalScoreModel
from bioemu.sde_lib import SDE
from bioemu.interpolate import actions
from bioemu.interpolate.om_lib import center_zero, assert_center_zero
from bioemu.sample import maybe_download_checkpoint, SUPPORTED_DENOISERS, DEFAULT_DENOISER_CONFIG_DIR
from bioemu.get_embeds import get_colabfold_embeds
from bioemu.openfold.utils.rigid_utils import Rigid
from bioemu.openfold.np.residue_constants import rigid_group_atom_positions


# class Interpolator(OMInterpolatorWrapper):
class Interpolator(torch.nn.Module):
    """
    Tensor Indices:
        B: Batch
        R: Residue
        X: Spatial coordinates (3)
        Es: Embedding Dimension Single (384)
        Ep: Embedding Dimension Pair (128)
    """

    BIOEMU_VERSION = "bioemu-v1.0"
    
    def __init__(self,
        # the user should set these via om_interpolate
        path_length, # N: number of points along path
        dt, # dt * path_length = T: total time of the path
        
        # om_interpolate should set this
        protein_trajectory: FastFolderTrajectory,

        # These user should set these in their config
        gamma=10, # in the actual script I am stepping through, this is a tensor with value 12.0108, shape (num_paths,)
        # set as samp_args.om_gamma * torch.tensor(masses).to(device)
        D=0.015, # set as samp_args.om_d / (trainset.std if args.scale_data else 1.0) ** 2
        # Specify parameters for the interpolator
        latent_time=0.01, # BioEmu ranges from 0.990 to 0.001 or smthg
        initial_guess_level=0.25, # BioEmu ranges from 0.990 to 0.001 or smthg
        # Specify parameters for the optimizer
        lr=2e-1,
        om_steps=500,
        path_batch_size=-1, # Turns off batch optimization--do the whole path

        # These I can set the right defaults and forget
        action_cls=actions.TruncatedAction,
        optimizer=torch.optim.Adam,
        device="cuda" if torch.cuda.is_available() else "cpu",

        # These are parts of the interface I probably don't need to support, just shove them into kwargs
        # or raise if changed from these defaults
        **kwargs,
        # temperature=1.0, # This is the temperature of what? Not the dynamics right? For sampling?
        # model=None,
        # initial_guess_fn=torch.slerp,
        # encode_and_decode=False, # bit mis-leading, this is whether to do optimization with datapoints at latent_time
        # # and then decode to t=eps instead of decoding them to eps, and optimizing them with the score from t_opt=latent_time
        # mlff=False,
        # cg_prior=False,
        # anneal=False, # What is this for
        # log=False, # see if this is logging during interpolation or log scale or smthg
        # add_noise=False, 
        # truncated_gradient=False, # What is this?
        # subsample_points_percent=None,
        # subsample_dimensions_percent=None, # During optimization?
        # sample_latent_time=False,
        # cosine_scheduler=False
    ):
        if protein_trajectory.c_alpha:
            raise NotImplementedError("C-alpha not implemented yet, BioEmu requires at least N-CA-C-CB-O")
        # the OMBasics codebase calls this gamma, but m * gamma is actually zeta
        super().__init__()
        self.device = device

        self.dt = dt
        self.path_length = path_length
        self.path_batch_size = path_batch_size

        self.t_opt = latent_time
        self.t_lat = initial_guess_level

        self.om_steps = om_steps
        self.optimizer = optimizer
        self.lr = lr

        self.gamma = gamma * torch.tensor(protein_trajectory.masses_R).to(device)
        self.D = D / protein_trajectory.std ** 2
        self.action = action_cls(dt=self.dt, xi=(1 / self.gamma))

        self.score_model, self.sdes, self.denoiser = self.get_bioemu_models()
        self.single_embeds_REs, self.pair_embeds_R2Ep = self.seq_embeds(protein_trajectory)
        self.sequence = protein_trajectory.sequence
        self.topology = protein_trajectory.topology

    @classmethod
    def get_bioemu_models(cls, denoiser_type="dpm", denoiser_config_path=None):
        ckpt_path, model_config_path = maybe_download_checkpoint(
            model_name=Interpolator.BIOEMU_VERSION, ckpt_path=None, model_config_path=None
        )
        assert os.path.isfile(ckpt_path), f"Checkpoint {ckpt_path} not found"
        assert os.path.isfile(model_config_path), f"Model config {model_config_path} not found"

        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)

        model_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        score_model: DiGConditionalScoreModel = hydra.utils.instantiate(model_config["score_model"])
        score_model.load_state_dict(model_state)
        sdes: dict[str, SDE] = hydra.utils.instantiate(model_config["sdes"])

        if denoiser_config_path is None:
            assert (
                denoiser_type in SUPPORTED_DENOISERS
            ), f"denoiser_type must be one of {SUPPORTED_DENOISERS}"
            denoiser_config_path = DEFAULT_DENOISER_CONFIG_DIR / f"{denoiser_type}.yaml"

        with open(denoiser_config_path) as f:
            denoiser_config = yaml.safe_load(f)
        denoiser = hydra.utils.instantiate(denoiser_config)
        return score_model, sdes, denoiser

    @classmethod
    def seq_embeds(cls, protein_trajectory: FastFolderTrajectory):
        """
        Get the single and pair embeddings for the protein trajectory.
        Args:
            protein_trajectory: FastFolderTrajectory object
        Returns:
            single_embeds_REs: torch.Tensor of shape (num_atoms, 384)
            pair_embeds_R2Ep: torch.Tensor of shape (num_atoms, num_atoms, 128)
        """
        n = len(protein_trajectory.sequence)

        single_embeds_file, pair_embeds_file = get_colabfold_embeds(
            seq=protein_trajectory.sequence,
            cache_embeds_dir=None
        )
        single_embeds = np.load(single_embeds_file)
        pair_embeds = np.load(pair_embeds_file)
        assert pair_embeds.shape[0] == pair_embeds.shape[1] == n
        assert single_embeds.shape[0] == n
        assert len(single_embeds.shape) == 2
        _, _, n_pair_feats = pair_embeds.shape  # [seq_len, seq_len, n_pair_feats]

        single_embeds = torch.from_numpy(single_embeds)
        pair_embeds = torch.from_numpy(pair_embeds)
        pair_embeds = pair_embeds.view(n**2, n_pair_feats)
        return single_embeds, pair_embeds

    def euclidian_to_frame(self, x_BAX):
        # TODO, this is probably implemented in the bioemu codebase too
        n = len(self.sequence)

        # edges in a fully connected graph
        # formatted as a list of source edges (00..011...1...) and target edges
        # (012...012...0...)
        edge_set_2R2 = torch.cat( 
            [
                torch.arange(n).repeat_interleave(n).view(1, n**2),
                torch.arange(n).repeat(n).view(1, n**2),
            ],
            dim=0,
        )

        # Rigid.from_3_points implements the gram-schmidt algorithm to get the frame representation in alphafold2
        # See AlphaFold supplement for details: https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf#page=26.15
        # Page 27 in PDF reader, section 1.8.1
        # In the implementation below, we zero index the e-vectors
        # so converting from the paper:
        # e_1 = v_1 / ||v_1|| in the algorithm in the paper
        # where v_1 = x_3 - x_2, which is C - C-alpha
        # v_2 = x_1 - x_2, which is N - C-alpha
        # In Rigid.from_3_points, we have
        # e_0 = origin - p_neg_x_axis
        # but CA should definitely be the origin
        # In the openfold codebase, they use this in their datapipeline, supplying as inputs
        # the first three atoms to these three arguments (see here: https://github.com/aqlaboratory/openfold/blob/e938c184a291bf053af3b14c1e3e8bb29aee57e2/openfold/data/data_transforms.py#L875)
        # As seen in the resdiue_constants.py file (https://github.com/aqlaboratory/openfold/blob/e938c184a291bf053af3b14c1e3e8bb29aee57e2/openfold/np/residue_constants.py#L143)
        # This order is N, CA, C, so I think we can do the same here

        # iterate through the atoms of the protein
        r_list_R, Q_list_R = [], []
        for residue in self.topology.residues:
            # get the CA, N, C, CB, O atoms
            N_idx = residue.atom("N").index
            CA_idx = residue.atom("CA").index
            C_idx = residue.atom("C").index
            # frame_3_pts = Rigid.from_3_points(
            #     p_neg_x_axis=x_BAX[:, N_idx, :],
            #     origin=x_BAX[:, CA_idx, :],
            #     p_xy_plane=x_BAX[:, C_idx, :],
            # )
            # frame = frame_3_pts
            frame_from_ref = Rigid.make_transform_from_reference(
                n_xyz = x_BAX[:, N_idx, :],
                ca_xyz = x_BAX[:, CA_idx, :],
                c_xyz = x_BAX[:, C_idx, :],
            )
            frame = frame_from_ref
            # These two should be the same, but their rotations are different
            # the third column of both rotations are the same, but the first two are different
            # this means the z-axes get transformed the same way (which I think is c-alpha to c)
            # When comparing if the inverse tranformation gets you back to the original, I found
            # frame_3_pts.invert().apply(x_BAX[:, N_idx, :])
            # tensor([[-1.4889e+00,  0.0000e+00, -5.9605e-08],
            #         [-1.5129e+00,  0.0000e+00, -4.7684e-07]])
            # frame_from_ref.invert().apply(x_BAX[:, N_idx, :])
            # tensor([[-5.0134e-01,  1.4020e+00, -5.9605e-08],
            #         [-6.4938e-01,  1.3664e+00, -4.7684e-07]])
            # idealized_frame[0][2] (getting this from rigid_group_atom_positions)
            # (-0.525, 1.362, -0.0)
            # Although the second method consistently seems to get the orientation of the N into the
            # canonical octant (see the idealized frame in the rigid_group_atom_positions for details)
            # the errors to reconstructing the idealized frame can actually be much larger than I expected even
            # for these small peptides--TRP_CAGE in this case deviated by somewherebetween 2.5 and 3.0 angstroms on ALA2
            N_idealized_X = torch.tensor(rigid_group_atom_positions[residue.name][0][2])
            N_reconstructed_BX = frame.invert().apply(x_BAX[:, N_idx, :])
            assert torch.all(torch.norm(N_idealized_X[None, :] - N_reconstructed_BX, dim=1) < 3) # angstroms
            CA_idealized = torch.tensor(rigid_group_atom_positions[residue.name][1][2])
            CA_reconstructed_BX = frame.invert().apply(x_BAX[:, CA_idx, :])
            assert torch.all(torch.norm(CA_idealized[None, :] - CA_reconstructed_BX, dim=1) < 3)
            C_idealized = torch.tensor(rigid_group_atom_positions[residue.name][2][2])
            C_reconstructed_BX = frame.invert().apply(x_BAX[:, C_idx, :])
            assert torch.all(torch.norm(C_idealized[None, :] - C_reconstructed_BX, dim=1) < 3)

            r_list_R.append(frame.get_trans()) # BX
            Q_list_R.append(frame.get_rots().get_rot_mats()) # BXX
        
        r_BRX = torch.stack(r_list_R, dim=1)
        Q_BRXX = torch.stack(Q_list_R, dim=1)
            
        return r_BRX, Q_BRXX


    def forward(self, x1, x2, z=None):
        n_paths, n_atoms = x1.shape[0], x1.shape[1]
        return self.om_interpolate(x1, x2)

    def om_interpolate(self, x1, x2, **kwargs): # kwargs originally had z, the atom identities
        if self.path_batch_size != -1:
            raise NotImplementedError("Batch optimization is not implemented yet.")
        
        start_points_BRX = x1
        end_points_BRX = x2

        n_paths, n_atoms = start_points_BRX.shape[0], start_points_BRX.shape[1]
        force_batch_size = n_paths * n_atoms

        x1 = center_zero(x1)
        x2 = center_zero(x2)
        assert_center_zero(x1) # check that the centering worked up to some eps tol 1e-3 angstroms
        assert_center_zero(x2)

        for i in range(n_paths):
            # Crucial: rotate end_points_BRX to match start_points_BRX (since TIC operates on rotationally invariant features)
            end_points_BRX[i] = torch.tensor(
                kabsch_rotate(end_points_BRX[i].cpu(), start_points_BRX[i].cpu())
            ).to(self.device)

        x1 = center_zero(x1)
        x2 = center_zero(x2)
        assert_center_zero(x1) # check that the centering worked up to some eps tol 1e-3 angstroms
        assert_center_zero(x2)

        # skipped this bit
        # x1 = x1 / self.norm_factor
        # x2 = x2 / self.norm_factor

        original_x1 = x1.clone()
        original_x2 = x2.clone()

        # convert to frame coordinates
        r1_BRX, Q1_BRXX = self.euclidian_to_frame(x1)
        r2_BRX, Q2_BRXX = self.euclidian_to_frame(x2)
        # noise to self.t_lat
        # do linear interpolation for r
        # do spherical interpolation for Q


        return {
            "final_path": None
        }
