import os
import yaml
import hydra
from pathlib import Path

import torch

from bioemu.datasets.fastfolders import FastFolderTrajectory, Molecule
from bioemu.interpolate import sample_interpolations_from_model, OMInterpolatorWrapper
import bioemu.interpolate as actions
from bioemu.interpolate import Interpolator

from typing import cast


def main(
    protein_name: str = "TRP_CAGE",
    num_trajectories: int = 2,
    path_length: int = 100,
    dt: float = 0.001,
    sample_batch_size: int = 2,
    output_dir: None | str | os.PathLike = None,
    seed: int = 0,
):
    """
    Main function to get sampled transition paths for fast-folding proteins from the DE Shaw dataset
    by minimizing the OM Action using the BioEmu score function.
    Args:
        protein_name (str): Name of the protein. E.g. 'TRP_CAGE', 'CHIGNOLIN', 'BBA', 'VILLIN', 'PROTEIN_G'.
        num_trajectories (int): Number of trajectories to sample.
    Tensor Indices:
        B: Batch
        R: Residue
        Ab: Backbone atom
        X: Spatial coordinates (3)
    """
    assert protein_name in Molecule.__members__, f"Protein {protein_name} not found in the dataset. Please check the name."
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "output" / protein_name
        output_dir.mkdir(parents=True, exist_ok=True) # make sure the directory is editable
    fastfolders_config_path = Path(__file__).parent / "config" / "datasets" / "fastfolders.yaml"
    with open(fastfolders_config_path) as f:
        fastfolders_config = yaml.safe_load(f)
    Trajectory = hydra.utils.instantiate(fastfolders_config)
    protein_trajectory = Trajectory(protein_name=protein_name)
    protein_trajectory = cast(FastFolderTrajectory, protein_trajectory)
    
    start_points_BAX = protein_trajectory.start_points_FAX.repeat(
                num_trajectories // len(protein_trajectory.start_points_FAX) + 1, 1, 1
            )[:num_trajectories]
    end_points_BAX = protein_trajectory.end_points_FAX.repeat(
                num_trajectories // len(protein_trajectory.end_points_FAX) + 1, 1, 1
            )[:num_trajectories]

    interpolator_config_path = Path(__file__).parent / "config" / "interpolator" / "interpolator.yaml"
    with open(interpolator_config_path) as f:
        interpolator_config = yaml.safe_load(f)
    FastFolderInterpolator = hydra.utils.instantiate(interpolator_config)
    
    
    torch.manual_seed(seed)
    bioemu_interpolator = FastFolderInterpolator(
        dt=dt,
        path_length=path_length,
        protein_trajectory=protein_trajectory,
    )

    paths = sample_interpolations_from_model(
        interpolator=bioemu_interpolator,
        endpoint_1_samples=start_points_BAX,
        endpoint_2_samples=end_points_BAX,
        batch_size=sample_batch_size,
        verbose=False,
        z=None,
    )

    print(paths)





if __name__ == "__main__":
    import logging
    import fire

    logging.basicConfig(level=logging.DEBUG)
    fire.Fire(main)