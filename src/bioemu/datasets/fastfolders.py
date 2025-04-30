import os
from enum import Enum
from pathlib import Path

import torch
import mdtraj as md
import numpy as np

class Molecule(Enum):
    CHIGNOLIN = "CLN025"
    TRP_CAGE = "2JOF"
    BBA = "1FME"
    VILLIN = "2F4K"
    WW_DOMAIN = "GTT"
    NTL9 = "NTL9"
    BBL = "2WAV"
    PROTEIN_B = "PRB"
    HOMEODOMAIN = "UVF"
    PROTEIN_G = "NuG2"
    ALPHA3D = "A3D"
    LAMBDA_REPRESSOR = "lambda"

TEMP_DICT = {
    Molecule.CHIGNOLIN: 340,
    Molecule.TRP_CAGE: 290,
    Molecule.BBA: 325,
    Molecule.VILLIN: 360,
    Molecule.WW_DOMAIN: 360,
    Molecule.NTL9: 355,
    Molecule.BBL: 298,
    Molecule.PROTEIN_B: 340,
    Molecule.HOMEODOMAIN: 360,
    Molecule.PROTEIN_G: 350,
    Molecule.ALPHA3D: 370,
    Molecule.LAMBDA_REPRESSOR: 350,
}

NORM_STDS = {
    Molecule.CHIGNOLIN: 3.113133430480957,
    Molecule.TRP_CAGE: 5.08211088180542,
    Molecule.BBA: 6.294918537139893,
    Molecule.VILLIN: 6.082900047302246,
    Molecule.PROTEIN_G: 6.354289531707764,
    "alanine_fold1": 0.9449278712272644,
    "alanine_fold2": 0.944965124130249,
    "alanine_fold3": 0.9452606439590454,
    "alanine_fold4": 0.9454087018966675,
}

AA_CODE_TO_LETTER = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
}

# default cluster endpoints for testing interpolation
# (obtained by visual inspection of what are hard transition paths to capture)
CLUSTER_ENDPOINTS = {
    Molecule.CHIGNOLIN: [11, 13],
    Molecule.TRP_CAGE: [2, 13],
    Molecule.BBA: [9, 17],
    Molecule.VILLIN: [0, 17],
    Molecule.PROTEIN_G: [11, 14],
}

def verify_path(path: str | os.PathLike, var_name: str) -> Path:
    """Check if the path exists."""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise ValueError(f"Path {path} for variable {var_name} does not exist.")
    return path

class FastFolderTrajectory:
    def __init__(self,
        protein_name: str, # TRP_CAGE
        om_home: str | os.PathLike,
        ref_data_home: str | os.PathLike,
    ):
        """
        Args:
            protein_name (str): Name of the protein.
            om_home (str | os.PathLike): Path to the OpenMM home directory.
            ref_data_home (str | os.PathLike): Path to the reference data home directory.
        Tensor Indices:
            F: Frame (depends on the trajectory, TRP_CAGE has 1044000 total, 1625 starting frames, and 354 ending frames)
            R: Residue (depends on protein, TRP_CAGE has 20)
            X: Spatial coordinates (3)
        """
        # self.mean0 = True # from original class, think not used
        # self.atom_selection = None # from original class, think not used
        atom_selection = "c-alpha"
        om_home = verify_path(om_home, "om_home")
        ref_data_home = verify_path(ref_data_home, "ref_data_home")

        # Mostly just set these up in case, and to reduce feeling of sunk cost for reading through
        # the OM dataset code T_T
        self.molecule = Molecule[protein_name.upper()]
        self.topology = md.load_topology(
            om_home / "datasets" / "folded_pdbs" / 
            f"{self.molecule.value}-0-{atom_selection}.pdb"
        )
        self.std = NORM_STDS[self.molecule]
        self.num_beads = self.topology.n_residues
        self.bead_onehot = torch.eye(self.num_beads)

        ground_truth_traj = torch.load(
            ref_data_home / self.molecule.value / "gt_traj.pt",
            weights_only=True
        ) # shape is (1044000, 20, 3) for 1044000 frames, 20 residues, and 3 spatial coordinates
        ground_truth_traj -= ground_truth_traj.mean(dim=1, keepdims=True) # center
        ground_truth_traj *= 10 # convert to angstroms
        self.ground_truth_traj_FRX = ground_truth_traj

        # To get these run /home/ishan/OMBasics/two-for-one-diffusion/sample.py and set a breakpoint at line 758 for commit 6a8fbfa6
        # where the call sample_interpolations_from_model as seen below
        # endpoint_1 = gt_traj[::100][start_points]
        # endpoint_2 = gt_traj[::100][end_points]
        # Then run something of the form below in the debugger. This is for the example where the protein is TRP_CAGE (2JOF)
        # torch.save(endpoint_1, '/data/ishan/reference_md_sims/2JOF/start_points.pt')
        # torch.save(endpoint_2, '/data/ishan/reference_md_sims/2JOF/end_points.pt')
        # If desired later on, we'd need to move the TICA plot stuff from sample into this repo to generate these ourselves
        self.start_points_FRX = torch.load(
            ref_data_home / self.molecule.value / "start_points.pt",
            weights_only=True
        ) # these are start points sampled from every 100 frames of the ground truth trajectory
        self.end_points_FRX = torch.load(
            ref_data_home / self.molecule.value / "end_points.pt",
            weights_only=True
        ) # these are end points sampled from every 100 frames of the ground truth trajectory

        # It's a bit annoying to find some of these sequences to make sure, but for example, it looks like TRP-CAGE
        # can be found at https://www.rcsb.org/sequence/2M7D
        self.sequence = "".join(
            [AA_CODE_TO_LETTER[residue.name] for residue in self.topology.residues]
        )
