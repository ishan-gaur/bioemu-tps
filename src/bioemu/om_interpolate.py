import hydra
import yaml
from pathlib import Path


def main(protein_name: str = "TRP_CAGE", num_trajectories: int = 2):
    """
    Main function to get sampled transition paths for fast-folding proteins from the DE Shaw dataset
    by minimizing the OM Action using the BioEmu score function.
    Args:
        protein_name (str): Name of the protein. E.g. 'TRP_CAGE', 'CHIGNOLIN', 'BBA', 'VILLIN', 'PROTEIN_G'.
        num_trajectories (int): Number of trajectories to sample.
    Tensor Indices:
        B: Batch
        R: Residue
        X: Spatial coordinates (3)
    """
    fastfolders_config_path = Path(__file__).parent / "config" / "datasets" / "fastfolders.yaml"
    with open(fastfolders_config_path) as f:
        fastfolders_config = yaml.safe_load(f)
    Trajectory = hydra.utils.instantiate(fastfolders_config)
    protein_trajectory = Trajectory(protein_name=protein_name)
    
    start_points_BRX = protein_trajectory.start_points_FRX.repeat(
                num_trajectories // len(protein_trajectory.start_points_FRX) + 1, 1, 1
            )[:num_trajectories]
    end_points_BRX = protein_trajectory.end_points_FRX.repeat(
                num_trajectories // len(protein_trajectory.end_points_FRX) + 1, 1, 1
            )[:num_trajectories]
    print(start_points_BRX.shape, end_points_BRX.shape)


if __name__ == "__main__":
    import logging
    import fire

    logging.basicConfig(level=logging.DEBUG)
    fire.Fire(main)