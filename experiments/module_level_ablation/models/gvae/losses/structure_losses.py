"""Loss assembly for third-stage structure VAE training."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from gvae.robot.constraints import (
    edge_angle_consistency_loss,
    edge_spacing_loss,
    size_constraint_loss,
    task_constraint_loss,
    type_geometry_loss,
)
from gvae.robot.geometry import (
    pose_parameter_loss,
    pose_to_transform,
    transform_position_rotation_loss,
)


def conditional_kl_loss(
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    """KL(q(z|condition,target) || p(z|condition)) for diagonal Gaussians."""
    posterior_var = posterior_logvar.exp()
    prior_var = prior_logvar.exp()
    kl = (
        prior_logvar
        - posterior_logvar
        + (posterior_var + (posterior_mean - prior_mean).pow(2)) / prior_var.clamp_min(1e-8)
        - 1.0
    )
    return 0.5 * kl.sum(dim=-1).mean()


def compute_structure_losses(
    batch: dict,
    pred: dict,
    graph_outputs: dict,
    edge_angles_pred: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    nodes_pred = pred["nodes"]
    edge_t_pred = graph_outputs["edge_t"]
    leg_base_t_pred = graph_outputs["leg_base_t"]

    edge_t_gt = pose_to_transform(batch["edge_pose"])
    leg_base_t_gt = pose_to_transform(batch["leg_base"])

    losses = {
        "nodes": pose_parameter_loss(nodes_pred, batch["nodes"]),
        "edge_pose": transform_position_rotation_loss(edge_t_pred, edge_t_gt),
        "leg_base": transform_position_rotation_loss(leg_base_t_pred, leg_base_t_gt),
        "leg_angle": F.mse_loss(pred["leg_angle"], batch["leg_angle"]),
        "scale": F.mse_loss(pred["scale"], batch["scale"]),
        "spacing": edge_spacing_loss(edge_t_pred, graph_outputs["s_edge_spacing"]),
        "geometry": type_geometry_loss(edge_t_pred, list(batch["bar_type"])),
        "angle_equal": edge_angle_consistency_loss(edge_angles_pred),
        "size": size_constraint_loss(batch["vreq"], pred["scale"]),
        "task": task_constraint_loss(
            batch["vreq"],
            batch["bar_v"],
            pred["scale"],
            edge_angles_pred,
        ),
        "kl": conditional_kl_loss(
            pred["posterior_mean"],
            pred["posterior_logvar"],
            pred["prior_mean"],
            pred["prior_logvar"],
        ),
    }
    if "scale_teacher" in pred:
        losses["scale_teacher"] = F.mse_loss(
            pred["scale_teacher"],
            batch["scale"],
        )

    total = edge_t_pred.new_tensor(0.0)
    metrics = {}
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
        metrics[name] = float(value.detach().cpu().item())
    metrics["total"] = float(total.detach().cpu().item())
    return total, metrics


def compute_direct_mlp_losses(
    batch: dict,
    pred: dict,
    graph_outputs: dict,
    edge_angles_pred: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Losses for the deterministic Vreq-only MLP ablation."""
    edge_t_pred = graph_outputs["edge_t"]
    leg_base_t_pred = graph_outputs["leg_base_t"]
    losses = {
        "nodes": pose_parameter_loss(pred["nodes"], batch["nodes"]),
        "edge_pose": transform_position_rotation_loss(
            edge_t_pred,
            pose_to_transform(batch["edge_pose"]),
        ),
        "leg_base": transform_position_rotation_loss(
            leg_base_t_pred,
            pose_to_transform(batch["leg_base"]),
        ),
        "leg_angle": F.mse_loss(pred["leg_angle"], batch["leg_angle"]),
        "scale": F.mse_loss(pred["scale"], batch["scale"]),
        "bar": F.cross_entropy(pred["bar_logits"], batch["bar_v"]),
        "spacing": edge_spacing_loss(
            edge_t_pred,
            graph_outputs["s_edge_spacing"],
        ),
        "geometry": type_geometry_loss(edge_t_pred, list(batch["bar_type"])),
        "angle_equal": edge_angle_consistency_loss(edge_angles_pred),
        "size": size_constraint_loss(batch["vreq"], pred["scale"]),
        "task": task_constraint_loss(
            batch["vreq"],
            batch["bar_v"],
            pred["scale"],
            edge_angles_pred,
        ),
    }
    if "scale_teacher" in pred:
        losses["scale_teacher"] = F.mse_loss(
            pred["scale_teacher"],
            batch["scale"],
        )

    total = edge_t_pred.new_tensor(0.0)
    metrics = {}
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
        metrics[name] = float(value.detach().cpu().item())
    metrics["bar_accuracy"] = float(
        (pred["bar_logits"].argmax(dim=-1) == batch["bar_v"])
        .float()
        .mean()
        .detach()
        .cpu()
        .item()
    )
    metrics["total"] = float(total.detach().cpu().item())
    return total, metrics
