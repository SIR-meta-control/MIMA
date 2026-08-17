"""Constraint losses for third-stage structure generation."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from gvae.core.constants import (
    EDGE_ANGLE_EQUALITY_INDICES,
    LOAD_Q2_RANGE,
    PACK_LIMIT_SCALE,
    TYPE_COLLINEAR_EDGE_PAIRS,
    TYPE_PARALLEL_EDGE_GROUPS,
)
from gvae.robot.geometry import wrap_angle


def _edge_z(edge_t: torch.Tensor, index: int) -> torch.Tensor:
    return edge_t[:, index, :3, 2]


def _edge_pos(edge_t: torch.Tensor, index: int) -> torch.Tensor:
    return edge_t[:, index, :3, 3]


def parallel_loss(edge_t: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
    ref = F.normalize(_edge_z(edge_t, indices[0]), dim=-1)
    losses = []
    for idx in indices[1:]:
        cur = F.normalize(_edge_z(edge_t, idx), dim=-1)
        losses.append(torch.linalg.cross(ref, cur, dim=-1).pow(2).sum(dim=-1))
    if not losses:
        return edge_t.new_tensor(0.0)
    return torch.stack(losses, dim=0).mean()


def collinear_loss(edge_t: torch.Tensor, pair: tuple[int, int]) -> torch.Tensor:
    i, j = pair
    zi = F.normalize(_edge_z(edge_t, i), dim=-1)
    zj = F.normalize(_edge_z(edge_t, j), dim=-1)
    pi = _edge_pos(edge_t, i)
    pj = _edge_pos(edge_t, j)

    axis_loss = torch.linalg.cross(zi, zj, dim=-1).pow(2).sum(dim=-1)
    offset_loss = torch.linalg.cross(zi, pj - pi, dim=-1).pow(2).sum(dim=-1)
    return (axis_loss + offset_loss).mean()


def type_geometry_loss(edge_t: torch.Tensor, bar_types: list[str]) -> torch.Tensor:
    losses = []
    for row, bar_type in enumerate(bar_types):
        current = edge_t[row : row + 1]
        for pair in TYPE_COLLINEAR_EDGE_PAIRS.get(bar_type, ()):
            losses.append(collinear_loss(current, pair))
        for group in TYPE_PARALLEL_EDGE_GROUPS.get(bar_type, ()):
            losses.append(parallel_loss(current, group))
    if not losses:
        return edge_t.new_tensor(0.0)
    return torch.stack(losses).mean()


def edge_spacing_loss(edge_t: torch.Tensor, target_spacing: torch.Tensor) -> torch.Tensor:
    positions = edge_t[:, :, :3, 3]
    shifted = torch.roll(positions, shifts=-1, dims=1)
    spacing = torch.linalg.norm(positions - shifted, dim=-1)
    return F.mse_loss(spacing, target_spacing.unsqueeze(0).expand_as(spacing))


def edge_angle_consistency_loss(
    edge_angles: torch.Tensor,
    indices: tuple[int, ...] = EDGE_ANGLE_EQUALITY_INDICES,
) -> torch.Tensor:
    """Enforce the learned moving-link angles to match each other."""
    selected = edge_angles[:, indices]
    sin_mean = torch.sin(selected).mean(dim=1, keepdim=True)
    cos_mean = torch.cos(selected).mean(dim=1, keepdim=True)
    circular_mean = torch.atan2(sin_mean, cos_mean)
    return wrap_angle(selected - circular_mean).pow(2).mean()


def size_constraint_loss(vreq: torch.Tensor, scale_pred: torch.Tensor) -> torch.Tensor:
    excess = torch.clamp(scale_pred - vreq[:, :3], min=0.0)
    return excess.pow(2).sum(dim=-1).mean()


def task_constraint_loss(
    vreq: torch.Tensor,
    bar_v: torch.Tensor,
    scale_pred: torch.Tensor,
    edge_angles: torch.Tensor,
) -> torch.Tensor:
    load = vreq[:, 3] > 0.5
    inspect = vreq[:, 4] > 0.5
    pack = vreq[:, 5] > 0.5
    loss = scale_pred.new_tensor(0.0)

    if load.any():
        q2 = edge_angles[load, 2]
        low, high = LOAD_Q2_RANGE
        load_q2_loss = torch.clamp(low - q2, min=0.0).pow(2) + torch.clamp(
            q2 - high,
            min=0.0,
        ).pow(2)
        load_type_loss = (bar_v[load].float() - 2.0).pow(2)
        loss = loss + load_q2_loss.mean() + load_type_loss.mean()

    if inspect.any():
        loss = loss + (bar_v[inspect].float() - 0.0).pow(2).mean()

    if pack.any():
        pack_limit = torch.as_tensor(
            PACK_LIMIT_SCALE,
            dtype=scale_pred.dtype,
            device=scale_pred.device,
        )
        pack_scale_loss = torch.clamp(scale_pred[pack] - pack_limit, min=0.0).pow(2)
        pack_type_loss = (bar_v[pack].float() - 0.0).pow(2)
        loss = loss + pack_scale_loss.sum(dim=-1).mean() + pack_type_loss.mean()

    return loss
