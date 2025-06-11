import hydra
import torch
import numpy as np
from pathlib import Path
from bioemu.datasets.fastfolders import FastFolderTrajectory, Molecule
from bioemu.interpolate import Interpolator
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import cast
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
@torch.no_grad()
def main(cfg: DictConfig) -> None:
    save_dir = "/home/ishan/bioemu/output"
    num_frames = 1000

    # initialize the fastfolder and interpolator classes from config
    protein_name = cfg.protein_name
    path_length = cfg.path_length
    dt = cfg.dt
    # sample_batch_size = cfg.sample_batch_size
    seed = cfg.seed

    assert protein_name in Molecule.__members__, f"Protein {protein_name} not found in the dataset. Please check the name."
    eval_folder = Path(save_dir) / protein_name.upper() / "force_predictions"
    eval_folder.mkdir(parents=True, exist_ok=True)

    Trajectory = hydra.utils.instantiate(cfg.dataset)
    protein_trajectory = Trajectory(protein_name=protein_name, center_scale_traj=False)
    protein_trajectory = cast(FastFolderTrajectory, protein_trajectory)
    
    FastFolderInterpolator = hydra.utils.instantiate(cfg.interpolator)
    torch.manual_seed(seed)
    bioemu_interpolator = FastFolderInterpolator(
        dt=dt,
        path_length=path_length,
        protein_trajectory=protein_trajectory,
    )
    bioemu_interpolator = cast(Interpolator, bioemu_interpolator)
    bioemu_interpolator = bioemu_interpolator.cuda()

    output_dir = Path(save_dir).expanduser().resolve()
    output_dir = eval_folder / protein_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # set a seed so the same random ones are always picked
    torch.manual_seed(seed)
    np.random.seed(seed)

    # pick a random indices and calculate forces
    if not (output_dir / "force_unit_vec_FAbX.pt").exists() or not (output_dir / "start_frames_FAbX.pt").exists():
        indices = np.random.choice(
            len(protein_trajectory.ground_truth_traj_FAX) - 1, num_frames, replace=False
        )
        next_frame = indices + 1
        start_frames_FAX = protein_trajectory.ground_truth_traj_FAX[indices]
        next_frames_FAX = protein_trajectory.ground_truth_traj_FAX[next_frame]
        start_frames_FAbX = bioemu_interpolator.all_atom_to_backbone(start_frames_FAX)
        next_frames_FAbX = bioemu_interpolator.all_atom_to_backbone(next_frames_FAX)
        force_unit_vec_FAbX = start_frames_FAbX - next_frames_FAbX
        force_unit_vec_FAbX /= torch.linalg.vector_norm(force_unit_vec_FAbX, dim=-1, keepdim=True)
        start_frames_FAbX = start_frames_FAbX.cpu()
        force_unit_vec_FAbX = force_unit_vec_FAbX.cpu()
        torch.save(
            start_frames_FAbX,
            output_dir / f"start_frames_FAbX.pt",
        )
        torch.save(
            force_unit_vec_FAbX,
            output_dir / f"force_unit_vec_FAbX.pt",
        )

    start_frames_FAbX = torch.load(
        output_dir / "start_frames_FAbX.pt"
    )
    start_frames_FRX, start_frames_FQXX = bioemu_interpolator.backbone_euclidian_to_frame(start_frames_FAbX.cuda())
    start_frames_FAbX = start_frames_FAbX.cpu()
    torch.save(
        start_frames_FRX,
        output_dir / f"start_frames_FRX.pt",
    )
    # force_unit_vec_FAbX = torch.load(
    #     output_dir / "force_unit_vec_FAbX.pt"
    # )
    # since this was their best
    force_unit_vec_FRX = torch.load(
        output_dir / f"two-for-one-normalized_scores_FRX_t_0.01.pt",
    )
    # pick a discretization of the noise levels
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 0.990]
    # then use the interpolator to get forces at those levels
    for t in noise_levels:
        if (output_dir / f"normalized_scores_FAbX_t_{t}.pt").exists():
            continue
        normalized_scores_B = []
        batch_size = 200
        # make sure to batch
        for i in tqdm(range(0, len(start_frames_FAbX), batch_size), desc=f"Calculating forces for {t}"):
            start_frames_BAbX = start_frames_FAbX[i : i + batch_size]
            score_BAbX = bioemu_interpolator.get_forces(
                start_frames_BAbX.cuda(), t=t
            )
            score_BAbX /= torch.linalg.vector_norm(score_BAbX, dim=-1, keepdim=True)
            normalized_scores_B.append(score_BAbX.clone().detach().cpu())
        normalized_scores_FAbX = torch.cat(normalized_scores_B, dim=0)
        # save each one to the output directory under the protein being tested
        torch.save(
            normalized_scores_FAbX,
            output_dir / f"normalized_scores_FAbX_t_{t}.pt",
        )

    # for overdamped langevin, average of the displacements points in the same direction as the forces
    # so we are just going to make the predicted score and displacements
    # unit vectors and look at the cosine similarity
    cosine_similarity_means = []
    cosine_similarity_stds = []
    backbone_to_CA = []
    # want to reconstruct the backbone to C-alpha mask
    # for this we have to go through all atoms and track if they were
    # the C-alpha atom if they were in the backbone
    for i in range(len(protein_trajectory.c_alpha_mask_A)):
        if protein_trajectory.backbone_mask_A[i]:
            backbone_to_CA.append(
                protein_trajectory.c_alpha_mask_A[i]
            )
    backbone_to_CA_Ab = torch.tensor(backbone_to_CA)
    for t in noise_levels:
        normalized_scores_FAbX = torch.load(
            output_dir / f"normalized_scores_FAbX_t_{t}.pt"
        )
        normalized_scores_FRX = normalized_scores_FAbX[:, backbone_to_CA_Ab]
        # residue_averaged_score = 
        # cosine similarity
        # cosine_similarity_FAb = torch.einsum(
        #     "ijk,ijk->ij", force_unit_vec_FAbX, normalized_scores_FAbX
        # )
        cosine_similarity_FR = torch.einsum(
            "ijk,ijk->ij", force_unit_vec_FRX, normalized_scores_FRX
        )
        cosine_similarity_means.append(cosine_similarity_FR.mean())
        cosine_similarity_stds.append(cosine_similarity_FR.std())

    # plot the cosine similarity
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        noise_levels,
        cosine_similarity_means,
        yerr=cosine_similarity_stds,
        fmt="o-",
        capsize=5,
    )
    plt.xlabel("Noise Level")
    plt.ylabel("Cosine Similarity")
    plt.title(f"Cosine Similarity of BioEmu Forces at Different Noise Levels with Two-for-One Model")
    plt.savefig(
        output_dir / f"cosine_similarity_two_for_one.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"Cosine similarity plot saved to {output_dir / 'cosine_similarity_two_for_one.png'}"
    )


if __name__ == "__main__":
    main()