import numpy as np

def coords_to_pdb(coords, atom_name="CA", res_name="ALA"):
    """
    Convert an (N, 3) numpy array of coordinates to a PDB string.
    """
    pdb = ""
    for i, (x, y, z) in enumerate(coords):
        pdb += (
            f"ATOM  {i+1:5d} {atom_name:>2s}  {res_name} A{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
    pdb += "END\n"
    return pdb
