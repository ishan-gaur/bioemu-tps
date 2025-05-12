import os
import re
import networkx as nx
from enum import Enum
from pathlib import Path
from bioemu.openfold.np.residue_constants import rigid_group_atom_positions

import torch
import mdtraj as md
import numpy as np

class AtomSelection(Enum):
    PROTEIN = "protein"
    A_CARBON = "c-alpha"
    BACKBONE_5 = "backbone_5"
    ALL = "all"

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

BACKBONE_ATOMS = [
    "N", "CA", "C", "O", "CB"
]

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
            A: Atom (depends on the trajectory, TRP_CAGE has 20 residues and 272 atoms)
            Ab: Backbone Atoms (depends on the trajectory, TRP_CAGE has 20 residues and 272 atoms)
            R: Residue (C-alpha; number depends on the trajectory, TRP_CAGE has 20 residues)
            X: Spatial coordinates (3)
        """
        # self.mean0 = True # from original class, think not used
        # self.atom_selection = None # from original class, think not used
        atom_selection = AtomSelection.BACKBONE_5
        self.c_alpha = (atom_selection == AtomSelection.A_CARBON)
        om_home = verify_path(om_home, "om_home")
        ref_data_home = verify_path(ref_data_home, "ref_data_home")

        # Mostly just set these up in case, and to reduce feeling of sunk cost for reading through
        # the OM dataset code T_T
        self.molecule = Molecule[protein_name.upper()]
        # Conver the MAE to PDB ising ChimeraX's Save-as functionality
        self.topology = md.load_topology(
            om_home / "datasets" / "folded_pdbs" / 
            f"{self.molecule.value}-from-mae.pdb"
        )

        # Remove all hydrogens    
        table, bonds = self.topology.to_dataframe()
        hydrogen_indices = [atom.index for atom in self.topology.atoms if atom.element.symbol == "H"]
        for i in hydrogen_indices[::-1]:
            self.topology.delete_atom_by_index(i)

        # Remove extra atoms not in the frames
        table, bonds = self.topology.to_dataframe()
        st_top_len = len(list(self.topology.atoms))
        for residue in self.topology.residues:
            residue_frame = rigid_group_atom_positions[residue.name]
            # residue frame elements are tuples of (atom symbol, residue index, (x, y, z))
            frame_atoms = [atom[0] for atom in residue_frame]
            top_residue_atoms = [atom.name for atom in residue.atoms]
            assert set(frame_atoms).issubset(set(top_residue_atoms))
            # had to comment this out because sometimes there is an extra oxygen on the residues like on tyrosine or serine
            # compared to what's in the residue frame
            # assert len(list(residue.atoms)) == len(residue_frame)
            extra_atoms = [a for a in top_residue_atoms if a not in frame_atoms]
            if len(extra_atoms) == 0: continue
            print(f"Extra atoms in {residue.name}: {extra_atoms}")
            for a in extra_atoms:
                self.topology.delete_atom_by_index(residue.atom(a).index)
        end_top_len = len(list(self.topology.atoms))
        print(f"Removed {st_top_len - end_top_len} extra atoms from the topology")
        print(f"Topology has {len(list(self.topology.atoms))} atoms and {len(list(self.topology.bonds))} bonds")

        # Reorder the atoms for each residue according to the frame order
        # See format of dataframe here https://mdtraj.org/1.9.4/api/generated/mdtraj.Topology.html
        table, bonds = self.topology.to_dataframe()
        assert np.all(bonds[:, 3:] == 0)
        residue_ptr = 0
        n_atoms = len(list(self.topology.atoms))
        topology_to_frame = [None for _ in range(len(list(self.topology.atoms)))]
        for residue in self.topology.residues:
            residue_frame = rigid_group_atom_positions[residue.name]
            frame_to_topology = [residue.atom(atom[0]).index for atom in residue_frame]
            for frame_pos, top_idx in enumerate(frame_to_topology):
                topology_to_frame[residue_ptr + frame_pos] = top_idx
            # make sure the residue's atoms are consecutively laid out in the topology
            # otherwise using this residue pointer doesn't make sense
            assert residue_ptr == min(frame_to_topology)
            # residue_ptr + len(residue_frame) is the start of the next residue
            assert residue_ptr + len(residue_frame) - 1 == max(frame_to_topology)
            residue_ptr += len(residue_frame)
        assert None not in topology_to_frame
        assert len(set(topology_to_frame)) == len(topology_to_frame)
        assert len(topology_to_frame) == n_atoms
        topology_to_frame = np.array(topology_to_frame)

        # Reorder the atoms in the topology according to the frame order
        table.index = topology_to_frame
        table = table.sort_index()

        # Bonds use the "serial" number
        serial = table["serial"].values
        bonds = np.array([b for b in bonds if b[0] in serial and b[1] in serial])
        serial_to_index = {s: i for i, s in zip(table.index, serial)}
        assert len(serial_to_index.values()) == len(table.index) # make sure all bonds have atoms still in the table
        # for some reason this is not true, even after deleting the hydrogens
        # assert len(np.unique(bonds)) == len(table.index) # make sure all atoms have bonds still
        bonds = np.array([[serial_to_index[b[0]], serial_to_index[b[1]], 0, 0] for b in bonds]) # not sure what the last two columns are
        table["serial"] = table.index

        self.topology = md.Topology.from_dataframe(table, bonds)

        # Get masks to help get atoms of interest from the all atom topologies
        # WARNING: Although there are 20 residues, the number of atoms is not 20*5 = 100
        # because some residues have less than 5 atoms (e.g. GLY)
        self.backbone_mask_A = torch.tensor([(atom.name in BACKBONE_ATOMS) for atom in self.topology.atoms])
        self.c_alpha_mask_A = torch.tensor([(atom.name == "CA") for atom in self.topology.atoms])

        # It's a bit annoying to find some of these sequences to make sure, but for example, it looks like TRP-CAGE
        # can be found at https://www.rcsb.org/sequence/2M7D
        self.sequence = "".join(
            [AA_CODE_TO_LETTER[residue.name] for residue in self.topology.residues]
        )

        self.ground_truth_traj_FAX = torch.load(
            ref_data_home / self.molecule.value / "gt_traj_all_atom.pt",
            weights_only=False
        ) # shape is (1044000, 272, 3) 
        # why does this not work with weights_only=True?
        # because it is a numpy array for some of these...
        if isinstance(self.ground_truth_traj_FAX, np.ndarray):
            self.ground_truth_traj_FAX = torch.tensor(self.ground_truth_traj_FAX, dtype=torch.float32)
        # no longer needed after converting to pdb
        # mae_to_pdb_map = mae_to_pdb_atom_mapping(self.molecule, om_home, ref_data_home, forward=True)
        # ground_truth_traj = ground_truth_traj[:, mae_to_pdb_map, :]

        # # make sure the permutation and masks worked by comparing the masked [X] this works
        # # traj to the coarse-grained one
        # self.ground_truth_traj_FRX = ground_truth_traj[:, self.c_alpha_mask_A]
        # pre_saved_gt_traj = torch.load(
        #     ref_data_home / self.molecule.value / "gt_traj.pt",
        #     weights_only=True
        # )
        # assert torch.allclose(
        #     self.ground_truth_traj_FRX,
        #     pre_saved_gt_traj,
        #     atol=1e-5
        # ), "Ground truth trajectory does not match the pre-saved one. Check the mae_to_pdb mapping or backbone mask."
        self.ground_truth_traj_FAX -= self.ground_truth_traj_FAX.mean(dim=1, keepdims=True) # center
        # Structure files, including this trajectory seem to be saved in nm by default
        # however, in bioemu/src/bioemu/convert_chemgraph.py, the C-O bond length is in angstroms
        # so convert everything to angstroms
        self.ground_truth_traj_FAX = to_angstrom(self.ground_truth_traj_FAX) # convert to angstroms

        # These are the target start and end points for the interpolation
        # To get these run /home/ishan/OMBasics/two-for-one-diffusion/sample.py and set a breakpoint at line 758 for commit 6a8fbfa6
        # Right after the calls to
        # start_points = cluster_assignments == clusters[0]
        # end_points = cluster_assignments == clusters[1]
        # Then run something of the form below in the debugger. This is for the example where the protein is TRP_CAGE (2JOF)
        # torch.save(start_points, '/data/ishan/reference_md_sims/2JOF/start_points.pt')
        # torch.save(end_points, '/data/ishan/reference_md_sims/2JOF/end_points.pt')
        # If desired later on, we'd need to move the TICA plot stuff from sample into this repo to generate these ourselves
        self.start_points_F = torch.load(
            ref_data_home / self.molecule.value / "start_points.pt",
            weights_only=False # TODO why did this fail when set to True?
        )
        self.end_points_F = torch.load(
            ref_data_home / self.molecule.value / "end_points.pt",
            weights_only=False # TODO why did this fail when set to True?
        ) 
        # these are end points sampled from every 100 frames of the ground truth trajectory
        self.start_points_FAX = self.ground_truth_traj_FAX[::100][self.start_points_F]
        self.end_points_FAX = self.ground_truth_traj_FAX[::100][self.end_points_F]

        # Misc properties
        self.std = NORM_STDS[self.molecule]
        self.num_beads = self.topology.n_residues
        self.bead_onehot_RR = torch.eye(self.num_beads)
        self.masses_R = [sum([a.element.mass for a in r.atoms]) for r in self.topology.residues] # masses of the residues
        self.masses_Ab = [a.element.mass for i, a in enumerate(self.topology.atoms) if self.backbone_mask_A[i]] # masses of the backbone atoms


def to_angstrom(x):
    """
    Convert from nanometer to angstrom.
    """
    return x * 10.0

def mae_to_pdb_atom_mapping(molecule, om_home, ref_data_home, forward=True):
    """
    In the case of all-atom proteins, we need to correct for the fact that the pdb and mae/dcd files have different atom orderings.
    """

    pdb_topology = md.load_topology(
        om_home / "datasets" / "folded_pdbs" / 
        f"{molecule.value}.pdb"
    ) # 1 chain, 20 residues, 284 atoms, 290 bonds
    pdb_bonds = torch.tensor(
        [(bond[0].index, bond[1].index) for bond in pdb_topology.bonds]
    ) # (290, 2)
    mae_bonds = extract_bonds_from_mae(
        om_home / "datasets" / "folded_maes" /
        f"{molecule.value}-0-protein.mae"
    ) # (278, 2) TODO why are there 12 less bonds?
    if forward:
        return recover_permutation(mae_bonds, pdb_bonds)
    return recover_permutation(pdb_bonds, mae_bonds)


def extract_bonds_from_mae(file_path):
    """
    Extract bond indices from a .mae file
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    bond_section = False
    bonds = []

    for line in lines:
        line = line.strip()

        # Detect the start of the m_bond block
        if line.startswith("m_bond"):
            bond_section = True
            continue

        # Detect the end of the block
        if bond_section and line.startswith("}"):
            break

        # Skip the header inside the block (first few lines)
        if bond_section and ":::" in line:
            continue

        # Extract bond data
        if bond_section:
            parts = re.split(r"\s+", line)  # Split by whitespace
            if len(parts) >= 4:  # Ensure valid data row
                i_m_from, i_m_to = int(parts[1]), int(parts[2])
                bonds.append([i_m_from, i_m_to])

    # Convert to torch.Tensor
    bond_tensor = torch.tensor(bonds, dtype=torch.int64) - 1

    return bond_tensor


def recover_permutation(bonds1, bonds2):
    """
    Recovers the permutation mapping node indices in the permuted graph (bonds1)
    to those in the original one (bonds2) using vf2pp_isomorphism from networkx.

    Args:
        bonds1 (torch.Tensor): Permuted graph edges of shape [N, 2]
        bonds2 (torch.Tensor): Original graph edges of shape [N, 2]

    Returns:
        dict or None: A dictionary mapping original node indices to permuted ones if an isomorphism exists, None otherwise.
    """
    # Create NetworkX graphs
    G1 = nx.Graph()
    G2 = nx.Graph()

    G1.add_edges_from(bonds1.tolist())
    G2.add_edges_from(bonds2.tolist())

    # Compute isomorphism
    iso_mapping = nx.vf2pp_isomorphism(G1, G2)

    if iso_mapping is None:
        return None

    # Convert mapping to tensor
    max_node = max(max(G1.nodes), max(G2.nodes)) + 1
    perm_tensor = torch.full((max_node,), -1, dtype=torch.long)

    for perm, orig in iso_mapping.items():
        perm_tensor[orig] = perm

    return perm_tensor