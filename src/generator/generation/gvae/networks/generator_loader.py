"""Checkpoint loader shared by the main model and ablation baselines."""

from __future__ import annotations

from pathlib import Path

import torch

from gvae.networks.mlp_baseline import build_mlp_baseline_from_checkpoint
from gvae.networks.structure_generator import build_structure_generator_from_checkpoint


def load_generator(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    family = checkpoint.get("model_config", {}).get(
        "model_family",
        "conditional_structure_vae",
    )
    if family == "direct_mlp":
        model = build_mlp_baseline_from_checkpoint(checkpoint, device)
    elif family == "conditional_structure_vae":
        model = build_structure_generator_from_checkpoint(checkpoint, device)
    else:
        raise ValueError(f"Unsupported generator model_family: {family}")
    return model, checkpoint
