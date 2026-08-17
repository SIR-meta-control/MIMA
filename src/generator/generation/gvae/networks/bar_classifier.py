"""Bar type classifier shared by training and sampling scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gvae.core.constants import BAR_ORDER, VREQ_FORMAT


class BarClassifier(nn.Module):
    """Small MLP for p(v | vreq)."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        output_dim: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_bar_classifier_from_checkpoint(
    checkpoint: dict,
    dropout_override: float | None = None,
) -> BarClassifier:
    model_config = checkpoint["model_config"]
    model = BarClassifier(
        input_dim=int(model_config.get("input_dim", len(VREQ_FORMAT))),
        hidden_dim=int(model_config["hidden_dim"]),
        output_dim=int(model_config.get("output_dim", len(BAR_ORDER))),
        dropout=(
            float(model_config.get("dropout", 0.0))
            if dropout_override is None
            else dropout_override
        ),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_bar_classifier(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[BarClassifier, np.ndarray, np.ndarray, dict]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    model = build_bar_classifier_from_checkpoint(checkpoint, dropout_override=0.0)
    mean = np.asarray(checkpoint["input_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["input_std"], dtype=np.float32)
    return model, mean, std, checkpoint


def predict_bar_probabilities(
    model: nn.Module,
    vreq: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    x = (vreq[None, :].astype(np.float32) - mean[None, :]) / std[None, :]
    with torch.no_grad():
        logits = model(torch.from_numpy(x)) / temperature
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    return probs / probs.sum()


def parse_vreq(text: str) -> np.ndarray:
    text = text.strip()
    if text.startswith("["):
        values = json.loads(text)
    else:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]

    if len(values) != len(VREQ_FORMAT):
        raise ValueError(f"vreq must have {len(VREQ_FORMAT)} values, got {len(values)}")
    return np.asarray(values, dtype=np.float32)

