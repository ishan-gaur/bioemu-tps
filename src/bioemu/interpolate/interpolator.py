import os
import yaml
import hydra
from pathlib import Path

import torch
import numpy as np
from rmsd import kabsch_rotate
import roma
from torch_geometric.data.batch import Batch

from bioemu.datasets.fastfolders import FastFolderTrajectory, BACKBONE_ATOMS
from bioemu.models import DiGConditionalScoreModel
from bioemu.sde_lib import SDE
from bioemu.interpolate import actions
from bioemu.interpolate.om_lib import center_zero, assert_center_zero
from bioemu.sample import maybe_download_checkpoint, SUPPORTED_DENOISERS, DEFAULT_DENOISER_CONFIG_DIR
from bioemu.get_embeds import get_colabfold_embeds
from bioemu.openfold.utils.rigid_utils import Rigid, Rotation
from bioemu.openfold.np.residue_constants import rigid_group_atom_positions
from bioemu.chemgraph import ChemGraph
from bioemu.sde_lib import SDE, CosineVPSDE
from bioemu.so3_sde import SO3SDE, apply_rotvec_to_rotmat
from bioemu.denoiser import dpm_solver
from typing import cast


# class Interpolator(OMInterpolatorWrapper):
class Interpolator(torch.nn.Module):
    """
    Tensor Indices:
        B: Batch
        Bp: Batch * Path
        Bb: Bioemu Batch
        P: Path
        R: Residue
        X: Spatial coordinates (3)
        Es: Embedding Dimension Single (384)
        Ep: Embedding Dimension Pair (128)
    """

    BIOEMU_VERSION = "bioemu-v1.0"
    BIOEMU_T_EPS = 0.001
    BIOEMU_T_MAX = 0.990
    BIOEMU_N_DEFAULT = 50
    
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
        path_batch_size=-1, # -1 Turns off batch optimization--do the whole path
        # if this is -1, the batch size for other bioemu calls will also be set to the path_length

        # These I can set the right defaults and forget
        action_cls=actions.TruncatedAction,
        optimizer=torch.optim.Adam,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_path="/home/ishan/bioemu/interpolate/output",

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
        if path_batch_size != -1:
            self.bioemu_batch_size = path_batch_size
            raise NotImplementedError("Batch optimization is not implemented yet.")
        else:
            self.bioemu_batch_size = path_length

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

        self.gamma = gamma * torch.tensor(protein_trajectory.masses_R).to(self.device)
        self.D = D / protein_trajectory.std ** 2
        self.action = action_cls(dt=self.dt, xi=(1 / self.gamma))

        self.score_model, self.sdes, self.denoiser = self.get_bioemu_models()

        pos_sde = self.sdes["pos"]
        assert isinstance(pos_sde, CosineVPSDE)
        self.pos_sde = cast(CosineVPSDE, pos_sde)

        so3_sde = self.sdes["node_orientations"]
        assert isinstance(so3_sde, SO3SDE)
        self.so3_sde = cast(SO3SDE, so3_sde)

        self.single_embeds_REs, self.pair_embeds_R2Ep = self.seq_embeds(protein_trajectory)
        self.sequence = protein_trajectory.sequence
        self.topology = protein_trajectory.topology
        self.backbone_mask = protein_trajectory.backbone_mask_A
        self.c_alpha_mask = protein_trajectory.c_alpha_mask_A
        self.backbone_atoms = [atom for atom in self.topology.atoms if atom.name in BACKBONE_ATOMS]
        self.atom_to_backbone_idx = {
            atom.index: i
            for i, atom in enumerate(self.backbone_atoms)
        }

        n = len(self.sequence)
        # edges in a fully connected graph
        # formatted as a list of source edges (00..011...1...) and target edges
        # (012...012...0...)
        self.edge_set_2R2 = torch.cat( 
            [
                torch.arange(n).repeat_interleave(n).view(1, n**2),
                torch.arange(n).repeat(n).view(1, n**2),
            ],
            dim=0,
        )
        self.output_path = Path(output_path) / protein_trajectory.molecule.name
        self.output_path.mkdir(parents=True, exist_ok=True) # check if can edit

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
            N_idx = residue.atom("N").index # these indices are over the atomistic representation
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
            transform = frame_from_ref.to(device=self.device)
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
            N_idealized_X = torch.tensor(rigid_group_atom_positions[residue.name][0][2], device=self.device)
            N_reconstructed_BX = transform.invert().apply(x_BAX[:, N_idx, :]).to(self.device)
            assert torch.all(torch.norm(N_idealized_X[None, :] - N_reconstructed_BX, dim=1) < 3) # angstroms
            CA_idealized = torch.tensor(rigid_group_atom_positions[residue.name][1][2], device=self.device)
            CA_reconstructed_BX = transform.invert().apply(x_BAX[:, CA_idx, :]).to(self.device)
            assert torch.all(torch.norm(CA_idealized[None, :] - CA_reconstructed_BX, dim=1) < 3)
            C_idealized = torch.tensor(rigid_group_atom_positions[residue.name][2][2], device=self.device)
            C_reconstructed_BX = transform.invert().apply(x_BAX[:, C_idx, :]).to(self.device)
            assert torch.all(torch.norm(C_idealized[None, :] - C_reconstructed_BX, dim=1) < 3)

            r_list_R.append(transform.get_trans()) # BX
            Q_list_R.append(transform.get_rots().get_rot_mats()) # BXX
        
        r_BRX = torch.stack(r_list_R, dim=1)
        Q_BRXX = torch.stack(Q_list_R, dim=1)
            
        return r_BRX, Q_BRXX

    def frame_to_euclidian(self, r_BRX, Q_BRXX):
        # iterate through the atoms of the protein
        n_backbone = len(self.atom_to_backbone_idx) # only has entries for backbone atoms
        r_BRX, Q_BRXX = r_BRX.to(self.device), Q_BRXX.to(self.device)
        x_BAbX = torch.zeros((r_BRX.shape[0], n_backbone, 3), device=self.device)
        atoms_set = 0
        for i, residue in enumerate(self.topology.residues):
            frame_transform = Rigid(Rotation(rot_mats=Q_BRXX[:, i]), r_BRX[:, i])
            idealized_frame = rigid_group_atom_positions[residue.name] # these are the frames with atoms in canonical positions, and c-alpha at origin
            # these frames are list of tuples: element letter, atom type (ie 0 is backbone, 1, 2, ... are sidechain atoms), and the coordinates
            residue_backbone_atoms = [atom for atom in idealized_frame if atom[0] in BACKBONE_ATOMS]
            idealized_x_AX = torch.stack([torch.tensor(atom[2]) for atom in residue_backbone_atoms], dim=0).to(self.device)
            idealized_x_BAX = torch.tile(idealized_x_AX[None, :, :], (r_BRX.shape[0], 1, 1))
            for atom_res_idx, atom in enumerate(residue_backbone_atoms):
                atom_idx = residue.atom(atom[0]).index
                assert atom_idx in self.atom_to_backbone_idx, f"Attempted to set atom {atom} at index {atom_idx} which is not a backbone atom"
                backbone_idx = self.atom_to_backbone_idx[atom_idx]
                # make sure this atom has not already been set
                assert torch.all(x_BAbX[:, backbone_idx, :] == torch.zeros_like(x_BAbX[:, backbone_idx, :]))
                x_BAbX[:, backbone_idx, :] = frame_transform.apply(idealized_x_BAX[:, atom_res_idx, :])
                atoms_set += 1
        assert atoms_set == n_backbone, f"Not all atoms were set, only {atoms_set} out of {n_backbone}"
        return x_BAbX
    
    def get_latent_samples(self, r_BRX, Q_BRXX):
        self.so3_sde.to(self.device)
        n_paths = r_BRX.shape[0]

        lat_r_BRX = self.pos_sde.sample_marginal(
            x=r_BRX,
            t=self.t_lat * torch.ones((n_paths,), device=self.device),
        )
        lat_Q_BRXX = self.so3_sde.sample_marginal(
            x=Q_BRXX,
            t=self.t_lat * torch.ones((n_paths,), device=self.device),
        )
        batch = Batch.from_data_list([
            ChemGraph(
                node_orientations=lat_Q_BRXX[i],
                pos=lat_r_BRX[i],
                edge_index=self.edge_set_2R2,
                single_embeds=self.single_embeds_REs,
                pair_embeds=self.pair_embeds_R2Ep,
            )
            for i in range(r_BRX.shape[0])
        ]).to(self.device)
        batch = cast(ChemGraph, batch)
        return batch


    def forward(self, x1, x2, z=None):
        return self.om_interpolate(x1, x2)

    def om_interpolate(self, start_x_BAX, end_x_BAX, **kwargs): # kwargs originally had z, the atom identities
        if self.path_batch_size != -1:
            raise NotImplementedError("Batch optimization is not implemented yet.")

        n_paths = start_x_BAX.shape[0]
        n_residues = len(list(self.topology.residues))
        start_x_BAX, end_x_BAX = start_x_BAX.to(self.device), end_x_BAX.to(self.device)
        
        start_x_BAX = center_zero(start_x_BAX)
        end_x_BAX = center_zero(end_x_BAX)
        assert_center_zero(start_x_BAX) # check that the centering worked up to some eps tol 1e-3 angstroms
        assert_center_zero(end_x_BAX)

        for i in range(n_paths):
            # Crucial: rotate end_points_BRX to match start_points_BRX (since TIC operates on rotationally invariant features)
            end_x_BAX[i] = torch.tensor(
                kabsch_rotate(end_x_BAX[i].cpu(), start_x_BAX[i].cpu())
            ).to(self.device)

        start_x_BAX = center_zero(start_x_BAX)
        end_x_BAX = center_zero(end_x_BAX)
        assert_center_zero(start_x_BAX) # check that the centering worked up to some eps tol 1e-3 angstroms
        assert_center_zero(end_x_BAX)

        # skipped this bit TODO figure out if necessary from Sanjeev
        # start_x_BAX = start_x_BAX / self.norm_factor (originally x1 / self.norm_factor)
        # end_x_BAX = end_x_BAX / self.norm_factor

        og_start_x_BAX = start_x_BAX.clone()
        og_end_x_BAX = end_x_BAX.clone()

        # convert to frame coordinates
        start_r_BRX, start_Q_BRXX = self.euclidian_to_frame(start_x_BAX)
        end_r_BRX, end_Q2_BRXX = self.euclidian_to_frame(end_x_BAX)
        end_x_BAbX_reconstructed = self.frame_to_euclidian(end_r_BRX, end_Q2_BRXX) # check that the reconstruction works
        end_x_BabX = og_end_x_BAX[:, self.backbone_mask, :]
        assert torch.norm(end_x_BAbX_reconstructed - end_x_BabX, dim=2).max() < 3, f"Reconstruction error on endpoint too large: {torch.norm(end_x_BAbX_reconstructed - end_x_BabX, dim=2).max().item()}"

        # noise to self.t_lat
        lat_start_batch = self.get_latent_samples(start_r_BRX, start_Q_BRXX)
        lat_start_r_BRX = lat_start_batch.pos.view(
            n_paths, n_residues, 3
        )
        lat_start_Q_BRXX = lat_start_batch.node_orientations.view(
            n_paths, n_residues, 3, 3
        )
        lat_end_batch = self.get_latent_samples(end_r_BRX, end_Q2_BRXX)
        lat_end_r_BRX = lat_end_batch.pos.view(
            n_paths, n_residues, 3
        )
        lat_end_Q_BRXX = lat_end_batch.node_orientations.view(
            n_paths, n_residues, 3, 3
        )

        # Path interpolation in latent space
        interp_level = torch.linspace(
            0, 1, self.path_length, device=self.device
        ) # so finding self.path_length - 2 new points

        # do linear interpolation for r
        lat_r_PBRX = torch.zeros(
            (self.path_length, start_r_BRX.shape[0], start_r_BRX.shape[1], 3),
        )
        lat_r_PBRX[0] = lat_start_r_BRX
        lat_r_PBRX[-1] = lat_end_r_BRX
        for i in range(1, self.path_length - 1):
            lat_r_PBRX[i] = (1 - interp_level[i]) * lat_start_r_BRX + interp_level[i] * lat_end_r_BRX

        # do spherical interpolation for Q
        lat_Q_PBRXX = torch.zeros(
            (self.path_length, start_Q_BRXX.shape[0], start_Q_BRXX.shape[1], 3, 3),
        )
        lat_Q_PBRXX[0] = lat_start_Q_BRXX
        lat_Q_PBRXX[-1] = lat_end_Q_BRXX
        lat_Q_PBRXX = roma.rotmat_slerp(lat_start_Q_BRXX, lat_end_Q_BRXX, interp_level)

        lat_r_BPRX = lat_r_PBRX.permute(1, 0, 2, 3)
        lat_Q_BPRXX = lat_Q_PBRXX.permute(1, 0, 2, 3, 4)

        denoised_r_Bb = []
        denoised_Q_Bb = []
        lat_r_BpRX = lat_r_BPRX.flatten(0, 1)
        lat_Q_BpRXX = lat_Q_BPRXX.flatten(0, 1)
        for i in range(0, lat_r_BpRX.shape[0], self.bioemu_batch_size):
            lat_r_BbRX = lat_r_BpRX[i:i + self.bioemu_batch_size]
            lat_Q_BbRXX = lat_Q_BpRXX[i:i + self.bioemu_batch_size]

            # decode to un-noised time
            # note that this will be over individual points from each path in the batch
            # we will have to reshape the results later
            lat_batch_BbRX = Batch.from_data_list([
                ChemGraph(
                    node_orientations=lat_Q_BbRXX[j],
                    pos=lat_r_BbRX[j],
                    edge_index=self.edge_set_2R2,
                    single_embeds=self.single_embeds_REs,
                    pair_embeds=self.pair_embeds_R2Ep,
                )
                for j in range(lat_r_BbRX.shape[0])
            ]).to(self.device)

            # we are denoising from t_lat to t_eps whereas N
            # for this method was originally intended for t_max to t_eps
            # rescale N accordingly
            N = int(
                Interpolator.BIOEMU_N_DEFAULT * 
                (self.t_lat - Interpolator.BIOEMU_T_EPS) /
                (Interpolator.BIOEMU_T_MAX - Interpolator.BIOEMU_T_EPS)
            )


            with torch.no_grad():
                denoised_batch = dpm_solver(
                    sdes=self.sdes,
                    batch=lat_batch_BbRX,
                    N=N,
                    score_model=self.score_model,
                    max_t=self.t_lat,
                    eps_t=Interpolator.BIOEMU_T_EPS,
                    device=self.device,
                    max_is_start=True
                )
            denoised_BbRX = denoised_batch.pos.view(
                self.bioemu_batch_size, n_residues, 3
            )
            denoised_BbRXX = denoised_batch.node_orientations.view(
                self.bioemu_batch_size, n_residues, 3, 3
            )
            denoised_r_Bb.append(denoised_BbRX.clone())
            denoised_Q_Bb.append(denoised_BbRXX.clone())

        denoised_r_BPRX = torch.cat(denoised_r_Bb, dim=0).reshape(
            n_paths, self.path_length, n_residues, 3
        )
        denoised_Q_BPRX = torch.cat(denoised_Q_Bb, dim=0).reshape(
            n_paths, self.path_length, n_residues, 3, 3
        )

        # Save PDBs from the 0th batch for a quick check
        self.c_alpha_to_pdb(start_x_BAX[0][self.c_alpha_mask], self.output_path / "start.pdb")
        self.c_alpha_to_pdb(end_x_BAX[0][self.c_alpha_mask], self.output_path / "end.pdb")
        self.c_alpha_to_pdb(start_r_BRX[0], self.output_path / "start_frame.pdb")
        self.c_alpha_to_pdb(end_r_BRX[0], self.output_path / "end_frame.pdb")
        self.c_alpha_to_pdb(lat_r_BPRX[0][0], self.output_path / "lat_start.pdb")
        self.c_alpha_to_pdb(lat_r_BPRX[0][-1], self.output_path / "lat_end.pdb")
        for i in range(20, self.path_length, 20):
            self.c_alpha_to_pdb(lat_r_BPRX[0, i, :, :], self.output_path / f"lat_{i}.pdb")
        self.c_alpha_to_pdb(denoised_r_BPRX[0][0], self.output_path / "denoised_start.pdb")
        self.c_alpha_to_pdb(denoised_r_BPRX[0][-1], self.output_path / "denoised_end.pdb")
        for i in range(20, self.path_length, 20):
            self.c_alpha_to_pdb(denoised_r_BPRX[0, i, :, :], self.output_path / f"denoised_{i}.pdb")
        
        return {
            "final_path": None
        }

    def c_alpha_to_pdb(self, x_RX, output_path):
        """
        Write the C-alpha coordinates to a PDB file.
        Args:
            x_BRX: torch.Tensor of shape (num_paths, num_residues, 3)
                Contains the coordinates of the C-alpha atoms
            topology: Topology object
            output_path: Path to save the PDB file
        """
        with open(output_path, "w") as f:
            for i, residue in enumerate(self.topology.residues):
                # get the CA atom
                CA_idx = residue.atom("CA").index
                # write the CA atom to the PDB file
                f.write(f"ATOM  {i+1:5d}  CA  {residue.name:<3} {residue.index:4d}    {x_RX[i, 0]:8.3f}{x_RX[i, 1]:8.3f}{x_RX[i, 2]:8.3f}\n")
            # write the end of the PDB file
            f.write("END\n")
        print(f"Wrote PDB file to {output_path}")
