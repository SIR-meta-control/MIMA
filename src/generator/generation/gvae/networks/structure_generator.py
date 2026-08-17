"""Conditional VAE for third-stage robot structure generation.

The generator keeps condition semantics separated before fusion:

  xyz limits -> xyz encoder
  task bits  -> task encoder
  bar type   -> learned embedding

It also learns a conditional prior p(z | condition), and uses type-specific
decoder heads so each bar subspace can model its own structure distribution.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from gvae.core.constants import (
    BAR_ORDER,
    LEG_MOUNT_NODE_INDICES,
    TYPE_GRAPH_EDGES,
)
from gvae.robot.geometry import normalize_quaternion


class GraphMessagePassingLayer(nn.Module):
    """Dense message-passing layer for the fixed eight-node robot graphs."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_projection = nn.Linear(input_dim, output_dim)
        self.neighbor_projection = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbor_mean = torch.bmm(adjacency, node_features) / degree
        output = self.self_projection(node_features)
        output = output + self.neighbor_projection(neighbor_mean)
        return self.dropout(F.relu(self.norm(output)))


class ScaleGraphEstimator(nn.Module):
    """Estimate robot dimensions from generated structure geometry."""

    def __init__(
        self,
        bar_embedding_dim: int,
        hidden_dims: tuple[int, int, int] = (128, 256, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != 3:
            raise ValueError("ScaleGraphEstimator requires exactly three hidden dimensions")

        node_feature_dim = 3 + 4 + 3 + 1 + bar_embedding_dim
        dimensions = (node_feature_dim,) + tuple(hidden_dims)
        self.layers = nn.ModuleList(
            [
                GraphMessagePassingLayer(
                    dimensions[index],
                    dimensions[index + 1],
                    dropout=dropout,
                )
                for index in range(3)
            ]
        )
        pooled_dim = hidden_dims[-1] * 2
        self.output_head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

        adjacency = torch.eye(8, dtype=torch.float32).repeat(len(BAR_ORDER), 1, 1)
        for bar_index, bar_type in enumerate(BAR_ORDER):
            for source, target in TYPE_GRAPH_EDGES[bar_type]:
                adjacency[bar_index, source, target] = 1.0
                adjacency[bar_index, target, source] = 1.0
        self.register_buffer("adjacency", adjacency)

        leg_mask = torch.zeros((8, 1), dtype=torch.float32)
        leg_mask[list(LEG_MOUNT_NODE_INDICES)] = 1.0
        self.register_buffer("leg_mount_mask", leg_mask)

    def forward(
        self,
        nodes: torch.Tensor,
        leg_angle: torch.Tensor,
        bar_v: torch.Tensor,
        bar_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = nodes.shape[0]
        leg_mask = self.leg_mount_mask.to(dtype=nodes.dtype, device=nodes.device)
        leg_mask = leg_mask.unsqueeze(0).expand(batch_size, -1, -1)
        leg_features = leg_angle.unsqueeze(1).expand(-1, 8, -1) * leg_mask
        bar_features = bar_embedding.unsqueeze(1).expand(-1, 8, -1)

        node_features = torch.cat(
            [nodes[..., :3], nodes[..., 3:], leg_features, leg_mask, bar_features],
            dim=-1,
        )
        adjacency = self.adjacency[bar_v.long()].to(
            dtype=nodes.dtype,
            device=nodes.device,
        )
        for layer in self.layers:
            node_features = layer(node_features, adjacency)

        mean_pool = node_features.mean(dim=1)
        max_pool = node_features.max(dim=1).values
        return F.softplus(self.output_head(torch.cat([mean_pool, max_pool], dim=-1)))


class ScaleMLPEstimator(nn.Module):
    """Topology-free scale ablation using the same per-node input features."""

    def __init__(
        self,
        bar_embedding_dim: int,
        hidden_dims: tuple[int, int, int] = (320, 352, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != 3:
            raise ValueError("ScaleMLPEstimator requires exactly three hidden dimensions")

        node_feature_dim = 3 + 4 + 3 + 1 + bar_embedding_dim
        dimensions = (8 * node_feature_dim,) + tuple(hidden_dims)
        layers = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.extend(
                [
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.layers = nn.Sequential(*layers)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

        leg_mask = torch.zeros((8, 1), dtype=torch.float32)
        leg_mask[list(LEG_MOUNT_NODE_INDICES)] = 1.0
        self.register_buffer("leg_mount_mask", leg_mask)

    def forward(
        self,
        nodes: torch.Tensor,
        leg_angle: torch.Tensor,
        bar_v: torch.Tensor,
        bar_embedding: torch.Tensor,
    ) -> torch.Tensor:
        del bar_v
        batch_size = nodes.shape[0]
        leg_mask = self.leg_mount_mask.to(dtype=nodes.dtype, device=nodes.device)
        leg_mask = leg_mask.unsqueeze(0).expand(batch_size, -1, -1)
        leg_features = leg_angle.unsqueeze(1).expand(-1, 8, -1) * leg_mask
        bar_features = bar_embedding.unsqueeze(1).expand(-1, 8, -1)
        node_features = torch.cat(
            [nodes[..., :3], nodes[..., 3:], leg_features, leg_mask, bar_features],
            dim=-1,
        )
        hidden = self.layers(node_features.flatten(start_dim=1))
        return F.softplus(self.output_head(hidden))


class ConditionalStructureVAE(nn.Module):
    """Generate nodes, leg angles, and scale conditioned on vreq and bar type."""

    model_family = "conditional_structure_vae"

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        condition_dim: int = 128,
        xyz_hidden_dim: int = 32,
        task_hidden_dim: int = 32,
        bar_embedding_dim: int = 16,
        num_bar_types: int = 3,
        dropout: float = 0.0,
        scale_mode: str = "gnn",
        scale_gnn_hidden_dims: tuple[int, int, int] = (128, 256, 256),
        scale_mlp_hidden_dims: tuple[int, int, int] = (320, 352, 256),
    ) -> None:
        super().__init__()
        if scale_mode not in {"gnn", "mlp", "decoder"}:
            raise ValueError(f"Unknown scale mode: {scale_mode}")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.xyz_hidden_dim = xyz_hidden_dim
        self.task_hidden_dim = task_hidden_dim
        self.bar_embedding_dim = bar_embedding_dim
        self.num_bar_types = num_bar_types
        self.dropout = dropout
        self.scale_mode = scale_mode
        self.scale_gnn_hidden_dims = tuple(scale_gnn_hidden_dims)
        self.scale_mlp_hidden_dims = tuple(scale_mlp_hidden_dims)
        self.structure_target_dim = 8 * 7 + 3
        self.target_dim = self.structure_target_dim + (3 if scale_mode == "decoder" else 0)

        self.xyz_encoder = nn.Sequential(
            nn.Linear(3, xyz_hidden_dim),
            nn.ReLU(),
            nn.Linear(xyz_hidden_dim, xyz_hidden_dim),
            nn.ReLU(),
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(3, task_hidden_dim),
            nn.ReLU(),
            nn.Linear(task_hidden_dim, task_hidden_dim),
            nn.ReLU(),
        )
        self.bar_embedding = nn.Embedding(num_bar_types, bar_embedding_dim)

        fusion_dim = xyz_hidden_dim + task_hidden_dim + bar_embedding_dim
        self.condition_encoder = nn.Sequential(
            nn.Linear(fusion_dim, condition_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU(),
        )

        self.posterior_encoder = nn.Sequential(
            nn.Linear(condition_dim + self.target_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.posterior_mean = nn.Linear(hidden_dim, latent_dim)
        self.posterior_logvar = nn.Linear(hidden_dim, latent_dim)

        self.prior_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_trunk = nn.Sequential(
            nn.Linear(condition_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, self.target_dim) for _ in range(num_bar_types)]
        )
        if scale_mode == "gnn":
            self.scale_estimator = ScaleGraphEstimator(
                bar_embedding_dim=bar_embedding_dim,
                hidden_dims=self.scale_gnn_hidden_dims,
                dropout=dropout,
            )
        elif scale_mode == "mlp":
            self.scale_estimator = ScaleMLPEstimator(
                bar_embedding_dim=bar_embedding_dim,
                hidden_dims=self.scale_mlp_hidden_dims,
                dropout=dropout,
            )
        else:
            self.scale_estimator = None

    def encode_condition(self, vreq: torch.Tensor, bar_v: torch.Tensor) -> torch.Tensor:
        xyz = self.xyz_encoder(vreq[:, :3])
        task = self.task_encoder(vreq[:, 3:6])
        bar = self.bar_embedding(bar_v.long())
        return self.condition_encoder(torch.cat([xyz, task, bar], dim=-1))

    def flatten_target(
        self,
        nodes: torch.Tensor,
        leg_angle: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        parts = [nodes.reshape(nodes.shape[0], -1), leg_angle]
        if self.scale_mode == "decoder":
            parts.append(scale)
        return torch.cat(parts, dim=-1)

    def encode_posterior(
        self,
        condition: torch.Tensor,
        nodes: torch.Tensor,
        leg_angle: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.flatten_target(nodes, leg_angle, scale)
        hidden = self.posterior_encoder(torch.cat([condition, target], dim=-1))
        return self.posterior_mean(hidden), self.posterior_logvar(hidden)

    def encode_prior(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.prior_encoder(condition)
        return self.prior_mean(hidden), self.prior_logvar(hidden)

    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if not sample:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std * temperature

    def _decode_raw(
        self,
        condition: torch.Tensor,
        bar_v: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.decoder_trunk(torch.cat([condition, z], dim=-1))
        all_heads = torch.stack([head(hidden) for head in self.decoder_heads], dim=1)
        gather_index = bar_v.long().view(-1, 1, 1).expand(-1, 1, self.target_dim)
        return all_heads.gather(dim=1, index=gather_index).squeeze(1)

    def _unpack_raw(
        self,
        raw: torch.Tensor,
        bar_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        nodes_raw = raw[:, : 8 * 7].reshape(-1, 8, 7)
        leg_start = 8 * 7
        leg_angle = raw[:, leg_start : leg_start + 3]

        nodes = torch.cat(
            [
                nodes_raw[:, :, :3],
                normalize_quaternion(nodes_raw[:, :, 3:]),
            ],
            dim=-1,
        )
        if self.scale_mode in {"gnn", "mlp"}:
            scale = self.estimate_scale(nodes, leg_angle, bar_v)
        else:
            scale = F.softplus(raw[:, leg_start + 3 : leg_start + 6])
        return {
            "nodes": nodes,
            "leg_angle": leg_angle,
            "scale": scale,
        }

    def estimate_scale(
        self,
        nodes: torch.Tensor,
        leg_angle: torch.Tensor,
        bar_v: torch.Tensor,
    ) -> torch.Tensor:
        if self.scale_estimator is None:
            raise RuntimeError("Structure-based scale estimator is not initialized")
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

    def decode_from_condition(
        self,
        condition: torch.Tensor,
        bar_v: torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._unpack_raw(self._decode_raw(condition, bar_v, z), bar_v)

    def decode(
        self,
        vreq: torch.Tensor,
        bar_v: torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        condition = self.encode_condition(vreq, bar_v)
        return self.decode_from_condition(condition, bar_v, z)

    def forward(
        self,
        vreq: torch.Tensor,
        bar_v: torch.Tensor,
        nodes: torch.Tensor,
        leg_angle: torch.Tensor,
        scale: torch.Tensor,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        condition = self.encode_condition(vreq, bar_v)
        posterior_mean, posterior_logvar = self.encode_posterior(
            condition,
            nodes,
            leg_angle,
            scale,
        )
        prior_mean, prior_logvar = self.encode_prior(condition)
        z = self.reparameterize(posterior_mean, posterior_logvar, sample=sample)
        decoded = self.decode_from_condition(condition, bar_v, z)
        if self.scale_mode in {"gnn", "mlp"}:
            decoded["scale_teacher"] = self.estimate_scale(
                nodes,
                leg_angle,
                bar_v,
            )
        decoded.update(
            {
                "posterior_mean": posterior_mean,
                "posterior_logvar": posterior_logvar,
                "prior_mean": prior_mean,
                "prior_logvar": prior_logvar,
                # Kept as aliases for simple logging/backward compatibility.
                "mean": posterior_mean,
                "logvar": posterior_logvar,
            }
        )
        return decoded

    def sample(
        self,
        vreq: torch.Tensor,
        bar_v: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        vreq_rep = vreq.repeat_interleave(num_samples, dim=0)
        bar_rep = bar_v.repeat_interleave(num_samples, dim=0)
        condition = self.encode_condition(vreq_rep, bar_rep)
        prior_mean, prior_logvar = self.encode_prior(condition)
        z = self.reparameterize(
            prior_mean,
            prior_logvar,
            sample=True,
            temperature=temperature,
        )
        decoded = self.decode_from_condition(condition, bar_rep, z)
        decoded["prior_mean"] = prior_mean
        decoded["prior_logvar"] = prior_logvar
        return decoded


def build_structure_generator_from_checkpoint(
    checkpoint: dict,
    device: torch.device | str = "cpu",
) -> ConditionalStructureVAE:
    """Construct an inference model from a saved training checkpoint."""
    config = checkpoint["model_config"]
    scale_gnn_hidden_dims = tuple(
        int(value)
        for value in config.get("scale_gnn_hidden_dims", (128, 256, 256))
    )
    scale_mlp_hidden_dims = tuple(
        int(value)
        for value in config.get("scale_mlp_hidden_dims", (320, 352, 256))
    )
    model = ConditionalStructureVAE(
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        condition_dim=int(config.get("condition_dim", 128)),
        xyz_hidden_dim=int(config.get("xyz_hidden_dim", 32)),
        task_hidden_dim=int(config.get("task_hidden_dim", 32)),
        bar_embedding_dim=int(config.get("bar_embedding_dim", 16)),
        num_bar_types=int(config.get("num_bar_types", 3)),
        dropout=0.0,
        # Checkpoints saved before the Scale GNN used decoder-predicted scale.
        scale_mode=str(config.get("scale_mode", "decoder")),
        scale_gnn_hidden_dims=scale_gnn_hidden_dims,
        scale_mlp_hidden_dims=scale_mlp_hidden_dims,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_structure_generator(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[ConditionalStructureVAE, dict]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    return build_structure_generator_from_checkpoint(checkpoint, device), checkpoint
