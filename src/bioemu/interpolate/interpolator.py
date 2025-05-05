import os
import yaml
import hydra
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
from rmsd import kabsch_rotate
import roma
from torch_geometric.data.batch import Batch

from bioemu.datasets.fastfolders import FastFolderTrajectory, BACKBONE_ATOMS
from bioemu.models import DiGConditionalScoreModel
from bioemu.denoiser import _get_score, dpm_solver
from bioemu.sde_lib import SDE
from bioemu.so3_sde import rotvec_to_rotmat
from bioemu.interpolate import actions
from bioemu.interpolate.om_lib import center_zero, assert_center_zero
from bioemu.sample import maybe_download_checkpoint, SUPPORTED_DENOISERS, DEFAULT_DENOISER_CONFIG_DIR
from bioemu.get_embeds import get_colabfold_embeds
from bioemu.openfold.utils.rigid_utils import Rigid, Rotation
from bioemu.openfold.np.residue_constants import rigid_group_atom_positions
from bioemu.chemgraph import ChemGraph
from bioemu.sde_lib import SDE, CosineVPSDE
from bioemu.so3_sde import SO3SDE, apply_rotvec_to_rotmat
from typing import cast, Type


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
        O: OM optimization step
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

        # these should be initialized by the instatiation of the config by hydra
        optimizer: Type[torch.optim.Optimizer],

        # These user should set these in their config
        gamma=10, # in the actual script I am stepping through, this is a tensor with value 12.0108, shape (n_paths,)
        # set as samp_args.om_gamma * torch.tensor(masses).to(device)
        D=0.015, # set as samp_args.om_d / (trainset.std if args.scale_data else 1.0) ** 2
        force_scale=1.0, # fudge factor to reweight forces
        # Specify parameters for the interpolator
        latent_time=0.01, # BioEmu ranges from 0.990 to 0.001 or smthg
        initial_guess_level=0.25, # BioEmu ranges from 0.990 to 0.001 or smthg
        # Specify parameters for the optimizer
        lr=2e-1,
        om_steps=500,
        path_opt_log_step=50, # log paths every 50 steps
        path_batch_size=-1, # -1 Turns off batch optimization--do the whole path
        # if this is -1, the batch size for other bioemu calls will also be set to the path_length

        # These I can set the right defaults and forget
        # action_cls=actions.TruncatedAction,
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
        self.path_opt_log_step = path_opt_log_step
        self.optimizer_cls = optimizer
        self.lr = lr

        self.zeta_Ab = gamma * torch.tensor(protein_trajectory.masses_Ab).to(self.device)
        self.D = D / protein_trajectory.std ** 2
        self.force_scale = force_scale
        # self.action = action_cls(dt=self.dt, xi=(1 / self.gamma))

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
        self.backbone_atoms_RAf = []
        for residue in self.topology.residues:
            residue_frame = rigid_group_atom_positions[residue.name]
            frame_atom_names = [atom[0] for atom in residue_frame]
            residue_backbone_atoms = [atom for atom in residue.atoms if atom.name in BACKBONE_ATOMS]
            residue_backbone_openfold_order = sorted(residue_backbone_atoms, key=lambda atom: frame_atom_names.index(atom.name))
            self.backbone_atoms_RAf.append(residue_backbone_openfold_order)
        self.atom_to_backbone_idx = {
            atom.index: i
            for i, atom in enumerate(
                [a for residue_atoms in self.backbone_atoms_RAf for a in residue_atoms]
            )
        }

        self.F_RAfX, self.frame_mask_RAf = self.get_topology_frames()

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

    def get_topology_frames(self):
        """
        Get the frames for the topology.
        Args:
            topology: Topology object
        Returns:
            frames_AfX: torch.Tensor of shape (n_residues * n_frame_atoms, 3)
                Contains the coordinates of the C-alpha atoms
            frame_mask_Af: torch.Tensor of shape (num_atoms,)
                Contains the mask for the C-alpha atoms
        """
        frames_R = []
        frame_mask_R = []
        for residue in self.topology.residues:
            residue_frame = rigid_group_atom_positions[residue.name]
            topology_atom_indices = [residue.atom(atom[0]).index for atom in residue_frame]
            backbone_atom_indices = list(filter(lambda x: self.backbone_mask[x], topology_atom_indices))
            # if these aren't continuous we won't just be able to concatenate the output euclidian coordinates when we use these
            # to invert the frame transformations
            assert list(range(min(backbone_atom_indices), max(backbone_atom_indices) + 1)) == sorted(backbone_atom_indices), f"Topology atom indices {backbone_atom_indices} are not continuous"
            frame_index_sorted_by_topology_pos = sorted(range(len(backbone_atom_indices)), key=lambda k: backbone_atom_indices[k])
            # although we can deal with arbitrary orders for the other atoms, the A matrix in the score conversion from frame to euclidian
            # requires that we know the 2nd atom is CA
            assert residue_frame[frame_index_sorted_by_topology_pos[1]][0] == "CA", f"Residue frame {residue_frame} does not have CA as the second atom"

            residue_frame_AfX = torch.stack([torch.tensor(residue_frame[i][2]) for i in frame_index_sorted_by_topology_pos])
            frames_R.append(residue_frame_AfX)

        frame_lens = [len(frame) for frame in frames_R]
        for i, frame in enumerate(frames_R):
            frame_padded_AfX = torch.zeros((max(frame_lens), 3))
            frame_padded_AfX[:len(frame), :] = frame
            frames_R[i] = frame_padded_AfX
            frame_mask_R.append(torch.tensor([True] * len(frame) + [False] * (max(frame_lens) - len(frame))))

        frames_RAfX = torch.stack(frames_R)
        frame_mask_RAf = torch.stack(frame_mask_R)
        assert frames_RAfX.shape[:-1] == frame_mask_RAf.shape, f"frames_RAfX shape {frames_RAfX.shape} does not match frame_mask_RAf shape {frame_mask_RAf.shape}"

        frames_RAfX = frames_RAfX.to(self.device)
        frame_mask_RAf = frame_mask_RAf.to(self.device)

        return frames_RAfX, frame_mask_RAf

    def all_atom_to_backbone(self, x_BAX):
        """
        Convert all atom coordinates to backbone coordinates.
        Args:
            x_BAX: torch.Tensor of shape (n_paths, n_atoms, 3)
                Contains the coordinates of the atoms
        Returns:
            x_BAbX: torch.Tensor of shape (n_paths, n_backbone_atoms, 3)
                Contains the coordinates of the backbone atoms
        """
        assert x_BAX.shape[1] == len(list(self.topology.atoms)), f"Number of atoms in x_BAX ({x_BAX.shape[1]}) does not match number of atoms in topology ({len(self.topology.atoms)})"
        assert x_BAX.shape[2] == 3, f"x_BAX should have shape (n_paths, n_atoms, 3), but has shape {x_BAX.shape}"

        # iterate through the atoms of the protein
        n_backbone = len(self.atom_to_backbone_idx)
        x_BAbX = torch.zeros((x_BAX.shape[0], n_backbone, 3), device=self.device)
        for i, idx in enumerate(self.atom_to_backbone_idx):
            x_BAbX[:, i, :] = x_BAX[:, idx, :]
        return x_BAbX

    def all_atom_euclidian_to_frame(self, x_BAX):
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

    def backbone_euclidian_to_frame(self, x_BAbX):
        if x_BAbX.ndim == 2:
            x_BAbX = x_BAbX[None, :, :]
        r_list_R, Q_list_R = [], []
        residue_ptr = 0
        for residue in self.topology.residues:
            N_idx = residue_ptr
            CA_idx = residue_ptr + 1
            C_idx = residue_ptr + 2
            frame_from_ref = Rigid.make_transform_from_reference(
                n_xyz = x_BAbX[:, N_idx, :],
                ca_xyz = x_BAbX[:, CA_idx, :],
                c_xyz = x_BAbX[:, C_idx, :],
            )
            transform = frame_from_ref.to(device=self.device)

            residue_frame = rigid_group_atom_positions[residue.name]

            def check_atom_threshold(idealized_X, reconstructed_BX, atom_name, threshold=4):
                distances = torch.norm(idealized_X[None, :] - reconstructed_BX, dim=1)
                over_threshold = distances >= threshold
                if over_threshold.any():
                    percent_over = over_threshold.float().mean().item() * 100
                    median_over = distances[over_threshold].median().item()
                    print(f"WARNING: {percent_over:.2f}% of {atom_name} atoms exceeded {threshold}Å threshold with median distance {median_over:.2f}Å")
                return distances

            # Check N atoms
            N_idealized_X = torch.tensor(residue_frame[0][2], device=self.device)
            N_frame_reconstructed_BX = transform.invert().apply(x_BAbX[:, N_idx, :]).to(self.device)
            check_atom_threshold(N_idealized_X, N_frame_reconstructed_BX, "N")

            # Check CA atoms
            CA_idealized = torch.tensor(residue_frame[1][2], device=self.device)
            CA_frame_reconstructed_BX = transform.invert().apply(x_BAbX[:, CA_idx, :]).to(self.device)
            check_atom_threshold(CA_idealized, CA_frame_reconstructed_BX, "CA")

            # Check C atoms
            C_idealized = torch.tensor(residue_frame[2][2], device=self.device)
            C_frame_reconstructed_BX = transform.invert().apply(x_BAbX[:, C_idx, :]).to(self.device)
            check_atom_threshold(C_idealized, C_frame_reconstructed_BX, "C")

            backbone_frame_AbX = torch.tensor([
                residue_frame[0][2],
                residue_frame[1][2],
                residue_frame[2][2],
            ], device=self.device)
            backbone_frame_BAbX = torch.tile(backbone_frame_AbX[None, :, :], (x_BAbX.shape[0], 1, 1))
            reconstructed_backbone_BRX = torch.stack([
                transform.apply(backbone_frame_BAbX[:, atom, :])
                for atom in range(backbone_frame_BAbX.shape[1])
            ], dim=1)
            distances = torch.norm(reconstructed_backbone_BRX - x_BAbX[:, residue_ptr:residue_ptr + 3, :], dim=2)
            over_threshold = distances >= 4
            if over_threshold.any():
                percent_over = over_threshold.float().mean().item() * 100
                median_over = distances[over_threshold].median().item()
                median_under = distances[~over_threshold].median().item() if (~over_threshold).any() else 0
                print(f"WARNING: {percent_over:.2f}% of backbone atoms exceeded 4Å threshold with median distance {median_over:.2f}Å. Median distance under threshold: {median_under:.2f}Å")

            r_list_R.append(transform.get_trans()) # BX
            Q_list_R.append(transform.get_rots().get_rot_mats()) # BXX

            residue_ptr += sum([atom[1] == 0 or atom[1] == 3 for atom in residue_frame]) # 0 are N, CA, C, CB, and 3 is O

        flat_backbone_atoms = [atom for residue_atoms in self.backbone_atoms_RAf for atom in residue_atoms]
        assert residue_ptr == len(flat_backbone_atoms), f"Residue pointer {residue_ptr} does not match number of backbone atoms {len(self.backbone_atoms_RAf)}"

        r_BRX = torch.stack(r_list_R, dim=1)
        Q_BRXX = torch.stack(Q_list_R, dim=1)
            
        return r_BRX, Q_BRXX

    def frame_to_euclidian_backbone(self, r_BRX, Q_BRXX):
        # iterate through the atoms of the protein
        n_backbone = len(self.atom_to_backbone_idx) # only has entries for backbone atoms
        r_BRX, Q_BRXX = r_BRX.to(self.device), Q_BRXX.to(self.device)
        x_BAbX = torch.zeros((r_BRX.shape[0], n_backbone, 3), device=self.device)
        atoms_set = 0
        backbone_atoms = []
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
                backbone_atoms.append(atom[0])
            # the euclidian to frame conversion assumes the first three atoms of every residue are N, CA, C
            backbone_indices = sorted([
                self.atom_to_backbone_idx[residue.atom(atom[0]).index]
                for atom in residue_backbone_atoms
            ])
            assert backbone_indices[0] == self.atom_to_backbone_idx[residue.atom("N").index], f"Backbone indices {backbone_indices} do not start with N"
            assert backbone_indices[1] == self.atom_to_backbone_idx[residue.atom("CA").index], f"Backbone indices {backbone_indices} do not have CA as the second atom"
            assert backbone_indices[2] == self.atom_to_backbone_idx[residue.atom("C").index], f"Backbone indices {backbone_indices} do not have C as the third atom"
            assert len(backbone_indices) > 4 or backbone_indices[3] == self.atom_to_backbone_idx[residue.atom("O").index], f"Backbone indices {backbone_indices} do not have O as the fourth atom, when only 4 atoms"
            assert len(backbone_indices) != 5 or backbone_indices[3] == self.atom_to_backbone_idx[residue.atom("CB").index], f"Backbone indices {backbone_indices} do not have CB as the fourth atom, when there are 5 atoms"
            assert len(backbone_indices) != 5 or backbone_indices[4] == self.atom_to_backbone_idx[residue.atom("O").index], f"Backbone indices {backbone_indices} do not have O as the fifth atom, when there are 5 atoms"
            assert len(backbone_indices) > 3 and len(backbone_indices) < 6, f"Backbone indices {backbone_indices} do not have 4 or 5 atoms"

        # check that all the atoms were set in the right order
        backbone_atom_elements = [a.name for residue_atoms in self.backbone_atoms_RAf for a in residue_atoms]
        for i, atom in enumerate(backbone_atoms):
            assert atom == backbone_atom_elements[i], f"Atom {atom} at index {i} does not match expected atom {self.backbone_atoms_RAf[i].name}"

        assert atoms_set == n_backbone, f"Not all atoms were set, only {atoms_set} out of {n_backbone}"
        assert torch.all(torch.norm(self.backbone_euclidian_to_frame(x_BAbX)[0] - r_BRX, dim=2) < 4), f"Reconstruction error too large: {torch.norm(self.backbone_euclidian_to_frame(x_BAbX)[0] - r_BRX, dim=2).max().item()}"
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

    def get_forces(self, x_BAbX, t):
        r_BRX, Q_BRXX = self.backbone_euclidian_to_frame(x_BAbX)
        assert x_BAbX.shape[1] == torch.sum(self.frame_mask_RAf).item(), f"Number of atoms in x_BAbX ({x_BAbX.shape[1]}) does not match number of atoms in frames_RAfX ({torch.sum(self.frame_mask_RAf).item()})"

        batch = Batch.from_data_list([
            ChemGraph(
                node_orientations=Q_BRXX[i],
                pos=r_BRX[i],
                edge_index=self.edge_set_2R2,
                single_embeds=self.single_embeds_REs,
                pair_embeds=self.pair_embeds_R2Ep,
            )
            for i in range(x_BAbX.shape[0])
        ]).to(self.device)
        if not isinstance(t, torch.Tensor):
            t = torch.full((x_BAbX.shape[0],), t, device=self.device)
        score = _get_score(batch=batch, t=t, score_model=self.score_model, sdes=self.sdes)
        score_r_BrX = score["pos"] # Br is batch * residue, which in this case is n_points * n_residues
        score_r_BRX = score_r_BrX.view(*r_BRX.shape)
        score_Q_BrX = score["node_orientations"] # if you look at the Euler-Maruyama integrator for this, it looks like this is a rotation vector
        score_Q_BrXX = rotvec_to_rotmat(score_Q_BrX)
        score_Q_BRXX = score_Q_BrXX.view(*Q_BRXX.shape)

        n_frame_atoms = self.F_RAfX.shape[1]
        A_1Af = torch.zeros((1, n_frame_atoms), device=self.device)
        A_1Af[0, 1] = 1.0 # this is the entry corresponding to the CA atom in X, or the translation r in the frame representation
        B_AfAf = torch.eye(n_frame_atoms, device=self.device) - torch.ones_like(A_1Af).T @ A_1Af
        # F_RXAf = self.F_RAfX.permute(0, 2, 1)
        # F_psinv_RXAf = torch.inverse(F_RXAf @ self.F_RAfX) @ F_RXAf
        F_pinv_RXAf = torch.linalg.pinv(self.F_RAfX)

        score_x_r_comp_BRAfX = torch.einsum("brx, a -> brax", score_r_BRX, A_1Af.squeeze())
        # score_x_Q_comp_BRAfX, residuals, rank, singular_values = torch.linalg.lstsq(F_RXAf, score_Q_BRXX)
        # B_BAfAf = torch.tile(B_AfAf[None, ...], (score_x_Q_comp_BRAfX.shape[0], 1, 1))
        B_RAfAf = torch.tile(B_AfAf[None, ...], (F_pinv_RXAf.shape[0], 1, 1))
        DG_Q_RAfX = (F_pinv_RXAf @ B_RAfAf).permute(0, 2, 1)
        score_x_Q_comp_BRAfX = torch.einsum("brij, rai -> braj", score_Q_BRXX, DG_Q_RAfX)
        score_x_BRAfX = score_x_r_comp_BRAfX + score_x_Q_comp_BRAfX
        # the mask below just gets rid of the padding for the glycine atoms
        score_x_BAbX = score_x_BRAfX.flatten(1, 2)[:, self.frame_mask_RAf.flatten()].reshape(
            score_x_BRAfX.shape[0], -1, 3
        )
        return score_x_BAbX

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
        start_r_BRX, start_Q_BRXX = self.all_atom_euclidian_to_frame(start_x_BAX)
        start_x_BAbX_reconstructed = self.frame_to_euclidian_backbone(start_r_BRX, start_Q_BRXX) # check that the reconstruction works
        start_x_BAbX = self.all_atom_to_backbone(start_x_BAX)
        assert torch.norm(start_x_BAbX_reconstructed - start_x_BAbX, dim=2).max() < 4, f"Reconstruction error on startpoint too large: {torch.norm(start_x_BAbX - start_x_BAbX, dim=2).max().item()}"
        end_r_BRX, end_Q2_BRXX = self.all_atom_euclidian_to_frame(end_x_BAX)
        end_x_BAbX_reconstructed = self.frame_to_euclidian_backbone(end_r_BRX, end_Q2_BRXX) # check that the reconstruction works
        end_x_BAbX = self.all_atom_to_backbone(end_x_BAX)
        assert torch.norm(end_x_BAbX_reconstructed - end_x_BAbX, dim=2).max() < 4, f"Reconstruction error on endpoint too large: {torch.norm(end_x_BAbX_reconstructed - end_x_BAbX, dim=2).max().item()}"

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

        # reset the first and last frames to the original start and end points
        denoised_r_BPRX[:, 0, :, :] = start_r_BRX
        denoised_r_BPRX[:, -1, :, :] = end_r_BRX

        # in the two-for-one-diffusion codebase, they anneal the t_opt from 200 to t_opt over the
        # first 1/4th of the optimizaiton steps (linear schedule)
        # Optimization of path using OM action
        with torch.enable_grad():
            denoised_r_BPRX.requires_grad = False
            denoised_Q_BPRX.requires_grad = False

            denoised_x_BpAbX = self.frame_to_euclidian_backbone(denoised_r_BPRX.flatten(0, 1), denoised_Q_BPRX.flatten(0, 1))

            denoised_st_x_BAbX = denoised_x_BpAbX[0].unsqueeze(0)
            reconstructed_st_x_BAbX = self.frame_to_euclidian_backbone(*self.backbone_euclidian_to_frame(denoised_st_x_BAbX))
            assert torch.all(torch.norm(denoised_st_x_BAbX - reconstructed_st_x_BAbX, dim=2) < 4), f"Reconstruction error on denoised start point too large: {torch.norm(denoised_st_x_BAbX - reconstructed_st_x_BAbX, dim=2).max().item()}"

            denoised_x_BpAbX_reconstructed = self.frame_to_euclidian_backbone(*self.backbone_euclidian_to_frame(denoised_x_BpAbX))
            assert torch.all(torch.norm(denoised_x_BpAbX - denoised_x_BpAbX_reconstructed, dim=2) < 4), f"Reconstruction error on denoised path too large: {torch.norm(denoised_x_BpAbX - denoised_x_BpAbX_reconstructed.flatten(0, 1), dim=2).max().item()}"

            denoised_x_BPAbX = denoised_x_BpAbX.view(
                n_paths, self.path_length, -1, 3 # -1 should be number of backbone atoms--97 for trpcage
            )
            denoised_x_BPAbX[:, 0, :, :] = start_x_BAbX
            denoised_x_BPAbX[:, -1, :, :] = end_x_BAbX


            actions = [[] for _ in range(n_paths)]
            all_path_xs_BO = [[] for _ in range(n_paths)]
            path_terms = [[] for _ in range(n_paths)]
            force_terms = [[] for _ in range(n_paths)]
            laplace_terms = [[] for _ in range(n_paths)]
            for b in range(n_paths):
                pbar = tqdm(range(self.om_steps))
                denoised_x_PAbX = denoised_x_BPAbX[b].clone()
                denoised_x_PAbX.requires_grad = True
                # denoised_x_PAbX.requires_grad = True
                optimizer = self.optimizer_cls(params=[denoised_x_PAbX])
                for i in pbar:
                    # Initialize gradient accumulator
                    optimizer.zero_grad()
                    # grads_accumulator = torch.zeros_like(denoised_x_PAbX)
                    

                    path_displacements_PAbX = denoised_x_PAbX[1:] - denoised_x_PAbX[:-1]
                    path_distances_PAb = torch.linalg.vector_norm(path_displacements_PAbX, dim=2)
                    # Below is the OM term for difference in euclidian position over the time interval
                    # Note mean is over particles and path positions
                    # equations in the paper are for single particles
                    # we sum over particles here just for convenience with autograd
                    # we also rescale by dt because this terms comes from brownian motion
                    distance_term = (path_distances_PAb / (2 * self.dt)).mean()

                    path_forces_PAbX = self.get_forces(denoised_x_PAbX, self.t_opt)
                    path_force_mags_PAb = torch.linalg.vector_norm(path_forces_PAbX, dim=2)
                    # Below is the OM term for the size of the forces at each configuration of the system
                    # Note force = grad potential = score up to constant scaling factors
                    force_term = (path_force_mags_PAb * self.dt * (self.force_scale ** 2) / (2 * self.zeta_Ab.unsqueeze(0) ** 2)).mean()
                    # force_term = torch.zeros_like(distance_term)

                    # TODO, for now we're using the truncated action in the regime of low diffusion coefficient
                    instability_term = (0.0 * self.D * self.dt / self.zeta_Ab).mean() # laplacian of the potential / divergence of the score

                    action = distance_term + force_term + instability_term
                    # grads = torch.autograd.grad(action, denoised_x_PAbX, retain_graph=True)[0]
                    grads = torch.autograd.grad(action, denoised_x_PAbX)[0]
                    action, distance_term, force_term, instability_term = (
                        action.item(),
                        distance_term.item(),
                        force_term.item(),
                        instability_term.item(),
                    )
                    actions[b].append(action)
                    path_terms[b].append(distance_term)
                    force_terms[b].append(force_term)
                    laplace_terms[b].append(instability_term)

                    with torch.no_grad():
                        # Zero out gradients for endpoints (they should be fixed)
                        grads[0], grads[-1] = 0, 0
                        denoised_x_PAbX.grad = grads
                        optimizer.step()
                        optimizer.zero_grad()

                        # not using this now, but could see it being useful before the optimizer step
                        # if add_noise:
                        #     # Add noise to gradients
                        #     _t = (
                        #         torch.tensor([max(1000 - i - 1, diff_time)])
                        #     …   

                    all_path_xs_BO[b].append(denoised_x_PAbX.clone().detach())
                    # want this as a denominator to tell us how much the magnitude of this term
                    # contributes to the total action, hence the abs for the instability,
                    # which can be negative (stable)
                    total_abs = distance_term + force_term + abs(instability_term)
                    path_contribution = (
                        distance_term / total_abs if total_abs != 0 else 0
                    )
                    force_contribution = (
                        force_term / total_abs if total_abs != 0 else 0
                    )
                    laplace_contribution = (
                        abs(instability_term) / total_abs if total_abs != 0 else 0
                    )
                    pbar.set_description(
                        f"OM Action: {action}, Path Contribution: {round(path_contribution*100, 3)}%, Force Contribution: {round(force_contribution * 100, 3)}%, Laplace Contribution: {round(laplace_contribution * 100, 3)}%"
                    )

                    del distance_term, force_term, instability_term, action, grads
                    torch.cuda.empty_cache()

                del denoised_x_PAbX 
                torch.cuda.empty_cache()

        # Save PDBs from the 0th batch for a quick check
        self.c_alpha_to_pdb(start_x_BAX[0][self.c_alpha_mask], self.output_path / "start.pdb")
        self.c_alpha_to_pdb(end_x_BAX[0][self.c_alpha_mask], self.output_path / "end.pdb")
        self.c_alpha_to_pdb(start_r_BRX[0], self.output_path / "start_frame.pdb")
        self.c_alpha_to_pdb(end_r_BRX[0], self.output_path / "end_frame.pdb")
        # self.c_alpha_to_pdb(lat_r_BPRX[0][0], self.output_path / "lat_start.pdb")
        # self.c_alpha_to_pdb(lat_r_BPRX[0][-1], self.output_path / "lat_end.pdb")
        for i in list(range(0, self.path_length, 20)) + [self.path_length - 1]:
        # for i in range(20, self.path_length, 20):
            self.c_alpha_to_pdb(lat_r_BPRX[0, i, :, :], self.output_path / f"lat_{i}.pdb")

        all_path_xs_B = [torch.stack(path_opt_steps) for path_opt_steps in all_path_xs_BO]
        all_path_xs_OBPAbX = torch.stack(all_path_xs_B, dim=0).permute(1, 0, 2, 3, 4)
        # decode the optimized paths (keeping every 50 for future visualization)
        all_path_xs_OBPAbX = torch.concat([all_path_xs_OBPAbX[::self.path_opt_log_step], all_path_xs_OBPAbX[-1:]], dim=0)
        # n_opt_log_steps = all_path_xs_OBPAbX.shape[0]
        # Reset the start/end points--shouldn't need this since we set the grads to 0
        # all_path_xs_OBPAbX[:, :, 0] = torch.tile(start_x_BAbX[None, ...], (n_opt_log_steps, 1, 1, 1))
        # all_path_xs_OBPAbX[:, :, -1] = torch.tile(end_x_BAbX[None, ...], (n_opt_log_steps, 1, 1, 1))
        assert torch.all(
            torch.norm(all_path_xs_OBPAbX[:, :, 0] - start_x_BAbX, dim=2) < 4
        ), f"Reconstruction error on start point too large: {torch.norm(all_path_xs_OBPAbX[:, 0] - start_x_BAbX, dim=2).max().item()}\nSomehow changed during optimization of the paths"
        assert torch.all(
            torch.norm(all_path_xs_OBPAbX[:, :, -1] - end_x_BAbX, dim=2) < 4
        ), f"Reconstruction error on endpoint too large: {torch.norm(all_path_xs_OBPAbX[:, -1] - end_x_BAbX, dim=2).max().item()}\nSomehow changed during optimization of the paths"
        all_path_xs_OBPAbX[0] = denoised_x_BPAbX # reset the first point to the initial denoised point

        for i in range(len(all_path_xs_OBPAbX)):
            for j in list(range(0, self.path_length, 20)) + [self.path_length - 1]:
                sample_RX = self.backbone_euclidian_to_frame(all_path_xs_OBPAbX[i, 0, j])[0][0] # first 0 is to get the r tensor, second is to select the batch index
                self.c_alpha_to_pdb(sample_RX, self.output_path / f"denoised_{j}_step_{i * self.path_opt_log_step}.pdb")

        n_atoms = start_x_BAbX.shape[1]
        all_path_xs_OBPRX = self.backbone_euclidian_to_frame(all_path_xs_OBPAbX.flatten(0, 2))[0].reshape(
            all_path_xs_OBPAbX.shape[0], all_path_xs_OBPAbX.shape[1], all_path_xs_OBPAbX.shape[2], -1, 3
        )
        final_path = all_path_xs_OBPRX[-1].flatten(0, 1)
        n_backbone = all_path_xs_OBPRX.shape[-2]
        all_paths = all_path_xs_OBPRX.reshape(all_path_xs_OBPRX.shape[0], -1, n_backbone, 3)
        actions = torch.tensor(actions).sum(dim=0)
        path_terms = torch.tensor(path_terms).sum(dim=0)
        force_terms = torch.tensor(force_terms).sum(dim=0)
        return {
            "final_path": final_path,
            "all_paths": all_paths,
            "actions": actions,
            "force_terms": force_terms,
            "path_terms": path_terms,
        }

    def c_alpha_to_pdb(self, x_RX, output_path):
        """
        Write the C-alpha coordinates to a PDB file.
        Args:
            x_BRX: torch.Tensor of shape (n_paths, num_residues, 3)
                Contains the coordinates of the C-alpha atoms
            topology: Topology object
            output_path: Path to save the PDB file
        """
        # x_RX /= 10 # angstroms to nanometers
        with open(output_path, "w") as f:
            for i, residue in enumerate(self.topology.residues):
                # get the CA atom
                CA_idx = residue.atom("CA").index
                # write the CA atom to the PDB file
                f.write(f"ATOM  {i+1:5d}  CA  {residue.name:<3} {residue.index:4d}    {x_RX[i, 0]:8.3f}{x_RX[i, 1]:8.3f}{x_RX[i, 2]:8.3f}\n")
            # write the end of the PDB file
            f.write("END\n")
        print(f"Wrote PDB file to {output_path}")
