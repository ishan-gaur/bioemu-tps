import os
import yaml
import hydra
from omegaconf import DictConfig
from pathlib import Path

import torch

from bioemu.datasets.fastfolders import FastFolderTrajectory, Molecule
from bioemu.interpolate import Interpolator, sample_interpolations_from_model
from matplotlib import pyplot as plt
from bioemu.datasets.fastfolders import CLUSTER_ENDPOINTS

from typing import cast

# To use this script, you need the all atom MAE structure of a protein that has been converted to PDB using ChimeraX
# You also need the all-atom and c-alpha coarse-grained all atom trajectories
# You also need to set a breakpoint in the OM interpolate sample.py script and save the start and end points

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
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
    exp_tag = None
    if exp_tag is not None:
        input(f"WARNING: You are running the experiment with the tag {exp_tag}. This may not be the intended experiment. Press Enter to continue or Ctrl+C to exit.")
    protein_name = cfg.protein_name
    num_trajectories = cfg.num_trajectories
    path_length = cfg.path_length
    dt = cfg.dt
    # sample_batch_size = cfg.sample_batch_size
    output_dir = cfg.output_dir
    seed = cfg.seed

    assert protein_name in Molecule.__members__, f"Protein {protein_name} not found in the dataset. Please check the name."
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    exp_append_name = f"t_lat={cfg.interpolator.initial_guess_level}_t_opt={cfg.interpolator.latent_time}_physical_params_dt={dt}_path_len={path_length}_num_traj={num_trajectories}_{exp_tag}"
    eval_folder = output_dir / protein_name.lower() / f"main_eval_output_bioemu_interpolate_{exp_append_name}"
    eval_folder.mkdir(parents=True, exist_ok=True)

    Trajectory = hydra.utils.instantiate(cfg.dataset)
    protein_trajectory = Trajectory(protein_name=protein_name)
    protein_trajectory = cast(FastFolderTrajectory, protein_trajectory)
    
    start_points_BAX = protein_trajectory.start_points_FAX.repeat(
                num_trajectories // len(protein_trajectory.start_points_FAX) + 1, 1, 1
            )[:num_trajectories]
    end_points_BAX = protein_trajectory.end_points_FAX.repeat(
                num_trajectories // len(protein_trajectory.end_points_FAX) + 1, 1, 1
            )[:num_trajectories]

    FastFolderInterpolator = hydra.utils.instantiate(cfg.interpolator)
    torch.manual_seed(seed)
    bioemu_interpolator = FastFolderInterpolator(
        dt=dt,
        path_length=path_length,
        protein_trajectory=protein_trajectory,
    )
    bioemu_interpolator = cast(Interpolator, bioemu_interpolator)

    paths = sample_interpolations_from_model(
        interpolator=bioemu_interpolator,
        endpoint_1_samples=start_points_BAX,
        endpoint_2_samples=end_points_BAX,
        batch_size=num_trajectories,
        verbose=False,
        z=None,
    )

    actions = paths["actions"]
    path_terms = paths["path_terms"]
    force_terms = paths["force_terms"]

    # make a line plot where actions, path terms, and force terms are plotted using matplotlib and save as png
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(actions, label="Action")
    ax.plot(path_terms, label="Path Norm Loss")
    ax.plot(force_terms, label="Force Norm Loss")
    ax.legend()
    ax.set_title("Actions, Path Norms, and Force Norms")
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Value")
    ax.set_yscale("log")
    plt.savefig(str(eval_folder) + "/actions_path_force_terms.png")
    print(f"Saved actions, path terms, and force terms plot to {eval_folder}/actions_path_force_terms.png")

    # History of paths along the optimization
    all_paths = paths["all_paths"]
    # Save paths
    torch.save(
        all_paths,
        str(str(eval_folder) + f"/path_history-bioemu_interpolate.pt"),
    )
    torch.save(
        paths["sampled_mol"],
        str(str(eval_folder) + f"/sample-bioemu_interpolate.pt"),
    )
    # save the atom selection mb
    # save the model...
    print(f"""Now run the following command in the OMBasics repo/environment to evaluate the paths:
from datasets.dataset_utils_empty import AtomSelection
from evaluate.evaluate_fastfolders import evaluate_fastfolders
evaluate_fastfolders(
    "{protein_name.lower()}",
    "bioemu_interpolate",
    "{exp_append_name}",
    checkpoint_folder="{str(output_dir)}",
    reference_folder="{str(output_dir.parent)}/evaluate/saved_references",
    pdb_folder="{str(output_dir.parent)}/datasets",
    atom_selection=AtomSelection.A_CARBON,
    model=None,
    num_paths={num_trajectories},
    endpoints={CLUSTER_ENDPOINTS[protein_trajectory.molecule]},
    compute_rates=False,
    log=False,
    gif=True,
)
""")
    print("Evaluation complete.")





if __name__ == "__main__":
    import logging
    # import fire

    logging.basicConfig(level=logging.DEBUG)
    main()
    # fire.Fire(main)