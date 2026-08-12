"""Deterministic Vreq-only MLP baseline for structure-generation ablations."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from gvae.core.constants import BAR_ORDER, VREQ_FORMAT
from gvae.robot.geometry import normalize_quaternion
from gvae.networks.structure_generator import ScaleGraphEstimator


class DirectMLPStructureGenerator(nn.Module):
    """Map Vreq directly to bar probabilities and type-specific structures.

    Bar type is not an input feature. The network predicts all three structure
    heads from Vreq, while ``bar_v`` only selects which output head is read.
    """

    model_family = "direct_mlp"
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (320, 320, 320, 320),
        num_bar_types: int = len(BAR_ORDER),
        dropout: float = 0.0,
        input_mean: tuple[float, ...] | list[float] | None = None,
        input_std: tuple[float, ...] | list[float] | None = None,
        scale_mode: str = "direct_mlp",
        bar_embedding_dim: int = 16,
        scale_gnn_hidden_dims: tuple[int, int, int] = (128, 256, 256),
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")
        if scale_mode not in {"direct_mlp", "gnn"}:
            raise ValueError(f"Unknown scale mode: {scale_mode}")

        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.num_bar_types = int(num_bar_types)
        self.dropout = float(dropout)
        self.scale_mode = scale_mode
        self.bar_embedding_dim = int(bar_embedding_dim)
        self.scale_gnn_hidden_dims = tuple(int(value) for value in scale_gnn_hidden_dims)
        self.structure_target_dim = 8 * 7 + 3
        self.target_dim = self.structure_target_dim + (
            3 if scale_mode == "direct_mlp" else 0
        )

        dimensions = (len(VREQ_FORMAT),) + self.hidden_dims
        layers = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.extend(
                [
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.trunk = nn.Sequential(*layers)
        self.bar_head = nn.Linear(self.hidden_dims[-1], self.num_bar_types)
        self.structure_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.hidden_dims[-1], self.target_dim),
                )
                for _ in range(self.num_bar_types)
            ]
        )
        if scale_mode == "gnn":
            self.bar_embedding = nn.Embedding(
                self.num_bar_types,
                self.bar_embedding_dim,
            )
            self.scale_estimator = ScaleGraphEstimator(
                bar_embedding_dim=self.bar_embedding_dim,
                hidden_dims=self.scale_gnn_hidden_dims,
                dropout=dropout,
            )
        else:
            self.bar_embedding = None
            self.scale_estimator = None

        mean = input_mean if input_mean is not None else [0.0] * len(VREQ_FORMAT)
        std = input_std if input_std is not None else [1.0] * len(VREQ_FORMAT)
        self.register_buffer("input_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer(
            "input_std",
            torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6),
        )

    def encode_vreq(self, vreq: torch.Tensor) -> torch.Tensor:
        normalized = (vreq - self.input_mean) / self.input_std
        return self.trunk(normalized)

    def predict_bar_logits(self, vreq: torch.Tensor) -> torch.Tensor:
        return self.bar_head(self.encode_vreq(vreq))

    def predict_bar_probabilities(
        self,
        vreq: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        return F.softmax(self.predict_bar_logits(vreq) / temperature, dim=-1)

    def _all_raw_outputs(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack([head(hidden) for head in self.structure_heads], dim=1)

    def _select_raw(
        self,
        all_raw: torch.Tensor,
        bar_v: torch.Tensor,
    ) -> torch.Tensor:
        gather_index = bar_v.long().view(-1, 1, 1).expand(-1, 1, self.target_dim)
        return all_raw.gather(dim=1, index=gather_index).squeeze(1)

    def estimate_scale(
        self,
        nodes: torch.Tensor,
        leg_angle: torch.Tensor,
        bar_v: torch.Tensor,
    ) -> torch.Tensor:
        if self.scale_estimator is None or self.bar_embedding is None:
            raise RuntimeError("Scale GNN is not initialized")
        normalized_nodes = torch.cat(
            [nodes[..., :3], normalize_quaternion(nodes[..., 3:])],
            dim=-1,
        )
        quaternion = normalized_nodes[..., 3:]
        quaternion_sign = torch.where(
            quaternion[..., :1] < 0.0,
            -torch.ones_like(quaternion[..., :1]),
            torch.ones_like(quaternion[..., :1]),
        )
        normalized_nodes = torch.cat(
            [normalized_nodes[..., :3], quaternion * quaternion_sign],
            dim=-1,
        )
        return self.scale_estimator(
            normalized_nodes,
            leg_angle,
            bar_v,
            self.bar_embedding(bar_v.long()),
        )

    def _unpack_raw(
        self,
        raw: torch.Tensor,
        bar_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        nodes_raw = raw[:, : 8 * 7].reshape(-1, 8, 7)
        leg_start = 8 * 7
        nodes = torch.cat(
            [
                nodes_raw[..., :3],
                normalize_quaternion(nodes_raw[..., 3:]),
            ],
            dim=-1,
        )
        leg_angle = raw[:, leg_start : leg_start + 3]
        if self.scale_mode == "gnn":
            scale = self.estimate_scale(nodes, leg_angle, bar_v)
        else:
            scale = F.softplus(raw[:, leg_start + 3 : leg_start + 6])
        return {
            "nodes": nodes,
            "leg_angle": leg_angle,
            "scale": scale,
        }

    def decode(
        self,
        vreq: torch.Tensor,
        bar_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode_vreq(vreq)
        all_raw = self._all_raw_outputs(hidden)
        decoded = self._unpack_raw(self._select_raw(all_raw, bar_v), bar_v)
        decoded["bar_logits"] = self.bar_head(hidden)
        return decoded

    def forward(
        self,
        vreq: torch.Tensor,
        bar_v: torch.Tensor,
        nodes: torch.Tensor | None = None,
        leg_angle: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        del scale, sample
        decoded = self.decode(vreq, bar_v)
        if self.scale_mode == "gnn" and nodes is not None and leg_angle is not None:
            decoded["scale_teacher"] = self.estimate_scale(
                nodes,
                leg_angle,
                bar_v,
            )
        return decoded

    def sample(
        self,
        vreq: torch.Tensor,
        bar_v: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        repeated_vreq = vreq.repeat_interleave(num_samples, dim=0)
        repeated_bar = bar_v.repeat_interleave(num_samples, dim=0)
        return self.decode(repeated_vreq, repeated_bar)


def build_mlp_baseline_from_checkpoint(
    checkpoint: dict,
    device: torch.device | str = "cpu",
) -> DirectMLPStructureGenerator:
    config = checkpoint["model_config"]
    model = DirectMLPStructureGenerator(
        hidden_dims=tuple(int(value) for value in config["hidden_dims"]),
        num_bar_types=int(config.get("num_bar_types", len(BAR_ORDER))),
        dropout=0.0,
        input_mean=config.get("input_mean"),
        input_std=config.get("input_std"),
        scale_mode=str(config.get("scale_mode", "direct_mlp")),
        bar_embedding_dim=int(config.get("bar_embedding_dim", 16)),
        scale_gnn_hidden_dims=tuple(
            int(value)
            for value in config.get("scale_gnn_hidden_dims", (128, 256, 256))
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_mlp_baseline(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[DirectMLPStructureGenerator, dict]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    return build_mlp_baseline_from_checkpoint(checkpoint, device), checkpoint
