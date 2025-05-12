# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script for sampling from a trained model."""

import logging
import os
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from bioemu.chemgraph import ChemGraph
from bioemu.convert_chemgraph import save_pdb_and_xtc
from bioemu.get_embeds import get_colabfold_embeds
from bioemu.models import DiGConditionalScoreModel
from bioemu.sde_lib import SDE
from bioemu.seq_io import parse_sequence, write_fasta
from bioemu.utils import (
    count_samples_in_output_dir,
    format_npz_samples_filename,
    print_traceback_on_exception,
)

logger = logging.getLogger(__name__)

DEFAULT_DENOISER_CONFIG_DIR = Path(__file__).parent / "config/denoiser/"
SupportedDenoisersLiteral = Literal["dpm", "heun"]
SUPPORTED_DENOISERS = list(typing.get_args(SupportedDenoisersLiteral))

from bioemu.sample import maybe_download_checkpoint, generate_batch
from bioemu.interpolate import Interpolator
from bioemu.datasets.fastfolders import FastFolderTrajectory, Molecule
from bioemu.datasets.fastfolders import CLUSTER_ENDPOINTS
from typing import cast

@hydra.main(version_base=None, config_path="config", config_name="config")
@torch.no_grad()
def main(
    cfg: DictConfig,
    num_samples: int = 1000,
    save_dir: str | Path = "/home/ishan/bioemu/output",
    batch_size_100: int = 10,
    model_name: str | None = "bioemu-v1.0",
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    denoiser_type: SupportedDenoisersLiteral | None = "dpm",
    denoiser_config_path: str | Path | None = None,
    cache_embeds_dir: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
    msa_host_url: str | None = None,
    filter_samples: bool = True,
) -> None:
    """
    Generate samples for a specified sequence, using a trained model.

    Args:
        sequence: Amino acid sequence for which to generate samples, or a path to a .fasta file, or a path to an .a3m file with MSAs.
            If it is not an a3m file, then colabfold will be used to generate an MSA and embedding.
        num_samples: Number of samples to generate. If `output_dir` already contains samples, this function will only generate additional samples necessary to reach the specified `num_samples`.
        output_dir: Directory to save the samples. Each batch of samples will initially be dumped as .npz files. Once all batches are sampled, they will be converted to .xtc and .pdb.
        batch_size_100: Batch size you'd use for a sequence of length 100. The batch size will be calculated from this, assuming
           that the memory requirement to compute each sample scales quadratically with the sequence length.
        model_name: Name of pretrained model to use. The model will be retrieved from huggingface. If not set,
           this defaults to `bioemu-v1.0`. If this is set, you do not need to provide `ckpt_path` or `model_config_path`.
        ckpt_path: Path to the model checkpoint. If this is set, `model_name` will be ignored.
        model_config_path: Path to the model config, defining score model architecture and the corruption process the model was trained with.
           Only required if `ckpt_path` is set.
        denoiser_type: Denoiser to use for sampling, if `denoiser_config_path` not specified. Comes in with default parameter configuration. Must be one of ['dpm', 'heun']
        denoiser_config_path: Path to the denoiser config, defining the denoising process.
        cache_embeds_dir: Directory to store MSA embeddings. If not set, this defaults to `COLABFOLD_DIR/embeds_cache`.
        cache_so3_dir: Directory to store SO3 precomputations. If not set, this defaults to `~/sampling_so3_cache`.
        msa_host_url: MSA server URL. If not set, this defaults to colabfold's remote server. If sequence is an a3m file, this is ignored.
        filter_samples: Filter out unphysical samples with e.g. long bond distances or steric clashes.
    """
    protein_name = cfg.protein_name
    num_trajectories = cfg.num_trajectories
    path_length = cfg.path_length
    dt = cfg.dt
    # sample_batch_size = cfg.sample_batch_size
    seed = cfg.seed

    assert protein_name in Molecule.__members__, f"Protein {protein_name} not found in the dataset. Please check the name."
    eval_folder = Path(save_dir) / protein_name.upper() / "iid_samples"
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

    output_dir = Path(save_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)  # Fail fast if output_dir is non-writeable

    ckpt_path, model_config_path = maybe_download_checkpoint(
        model_name=model_name, ckpt_path=ckpt_path, model_config_path=model_config_path
    )

    assert os.path.isfile(ckpt_path), f"Checkpoint {ckpt_path} not found"
    assert os.path.isfile(model_config_path), f"Model config {model_config_path} not found"

    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    if cache_so3_dir is not None:
        model_config["sdes"]["node_orientations"]["cache_dir"] = cache_so3_dir

    # User may have provided an MSA file instead of a sequence. This will be used for embeddings.
    sequence = protein_trajectory.sequence
    msa_file = sequence if str(sequence).endswith(".a3m") else None

    if msa_file is not None and msa_host_url is not None:
        logger.warning(f"msa_host_url is ignored because MSA file {msa_file} is provided.")

    # Parse FASTA or A3M file if sequence is a file path. Extract the actual sequence.
    sequence = parse_sequence(sequence)

    fasta_path = output_dir / "sequence.fasta"
    if fasta_path.is_file():
        if parse_sequence(fasta_path) != sequence:
            raise ValueError(
                f"{fasta_path} already exists, but contains a sequence different from {sequence}!"
            )
    else:
        # Save FASTA file in output_dir
        write_fasta([sequence], fasta_path)

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

    logger.info(
        f"Sampling {num_samples} structures for sequence of length {len(sequence)} residues..."
    )
    batch_size = int(batch_size_100 * (100 / len(sequence)) ** 2)
    if batch_size == 0:
        logger.warning(f"Sequence {sequence} may be too long. Attempting with batch_size = 1.")
        batch_size = 1
    logger.info(f"Using batch size {min(batch_size, num_samples)}")

    # existing_num_samples = count_samples_in_output_dir(output_dir)
    existing_num_samples = 0
    # logger.info(f"Found {existing_num_samples} previous samples in {output_dir}.")
    for seed in tqdm(
        range(existing_num_samples, num_samples, batch_size), desc="Sampling batches..."
    ):
        n = min(batch_size, num_samples - seed)
        npz_path = output_dir / format_npz_samples_filename(seed, n)
        # if npz_path.exists():
        #     raise ValueError(
        #         f"Not sure why {npz_path} already exists when so far only {existing_num_samples} samples have been generated."
        #     )
        logger.info(f"Sampling {seed=}")
        torch.manual_seed(seed)
        n = len(sequence)

        single_embeds_file, pair_embeds_file = get_colabfold_embeds(
            seq=sequence,
            cache_embeds_dir=cache_embeds_dir,
            msa_file=msa_file,
            msa_host_url=msa_host_url,
        )
        single_embeds = np.load(single_embeds_file)
        pair_embeds = np.load(pair_embeds_file)
        assert pair_embeds.shape[0] == pair_embeds.shape[1] == n
        assert single_embeds.shape[0] == n
        assert len(single_embeds.shape) == 2
        _, _, n_pair_feats = pair_embeds.shape  # [seq_len, seq_len, n_pair_feats]

        single_embeds, pair_embeds = torch.from_numpy(single_embeds), torch.from_numpy(pair_embeds)
        pair_embeds = pair_embeds.view(n**2, n_pair_feats)

        edge_index = torch.cat(
            [
                torch.arange(n).repeat_interleave(n).view(1, n**2),
                torch.arange(n).repeat(n).view(1, n**2),
            ],
            dim=0,
        )
        pos = torch.full((n, 3), float("nan"))
        node_orientations = torch.full((n, 3, 3), float("nan"))

        chemgraph = ChemGraph(
            edge_index=edge_index,
            pos=pos,
            node_orientations=node_orientations,
            single_embeds=single_embeds,
            pair_embeds=pair_embeds,
        )
        context_batch = Batch.from_data_list([chemgraph for _ in range(batch_size)])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sampled_chemgraph_batch = denoiser(
            sdes=sdes,
            device=device,
            batch=context_batch,
            score_model=score_model,
        )
        assert isinstance(sampled_chemgraph_batch, Batch)
        sampled_chemgraphs = sampled_chemgraph_batch.to_data_list()
        pos = torch.stack([x.pos for x in sampled_chemgraphs]).to("cpu")
        node_orientations = torch.stack([x.node_orientations for x in sampled_chemgraphs]).to("cpu")

        batch = {"pos": pos, "node_orientations": node_orientations}
        batch = {k: v.cpu().numpy() for k, v in batch.items()}
        np.savez(npz_path, **batch, sequence=sequence)

    logger.info("Converting samples to .pdb and .xtc...")
    samples_files = sorted(list(output_dir.glob("batch_*.npz")))
    sequences = [np.load(f)["sequence"].item() for f in samples_files]
    if set(sequences) != {sequence}:
        raise ValueError(f"Expected all sequences to be {sequence}, but got {set(sequences)}")
    positions = torch.tensor(np.concatenate([np.load(f)["pos"] for f in samples_files]))
    # node_orientations = torch.tensor(
    #     np.concatenate([np.load(f)["node_orientations"] for f in samples_files])
    # )
    positions = torch.as_tensor(positions, dtype=torch.float32)
    # node_orientations = torch.as_tensor(node_orientations, dtype=torch.float32)
    # backbone_atoms = bioemu_interpolator.frame_to_euclidian_backbone(positions, node_orientations)
    torch.save(
        positions,
        str(str(eval_folder) + f"/backbone-samples-bioemu_iid.pt"),
    )
    # torch.save(
    #     positions,
    #     str(str(eval_folder) + f"/samples_t=0.25-bioemu_iid.pt"),
    # )

if __name__ == "__main__":
    main()