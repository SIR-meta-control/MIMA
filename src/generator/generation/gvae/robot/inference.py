"""Hard decoding, constraint evaluation, scoring, and diversity selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F

from gvae.core.constants import (
    BAR_ORDER,
    EDGE_ANGLE_EQUALITY_INDICES,
    LOAD_Q2_RANGE,
    PACK_LIMIT_SCALE,
    TYPE_COLLINEAR_EDGE_PAIRS,
    TYPE_PARALLEL_EDGE_GROUPS,
)
from gvae.robot.geometry import (
    EdgeAngleRecoverer,
    GraphImputationLayer,
    normalize_quaternion,
    transform_to_pose,
    wrap_angle,
)


@dataclass(frozen=True)
class ConstraintThresholds:
    """Filtering thresholds calibrated against source and generated structures."""

    quaternion_norm_max: float = 1e-4
    spacing_rmse_max: float = 0.015
    spacing_error_max: float = 0.03
    geometry_error_max: float = 0.03
    angle_rmse_max: float = 0.03
    size_excess_max: float = 1e-4
    task_excess_max: float = 1e-4

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def eligible_bar_indices(vreq: np.ndarray | torch.Tensor) -> list[int]:
    """Return bar subspaces compatible with the hard task rules."""
    values = torch.as_tensor(vreq).detach().cpu().flatten()
    if values.numel() != 6:
        raise ValueError(f"vreq must contain 6 values, got {values.numel()}")

    load = bool(values[3] > 0.5)
    inspect = bool(values[4] > 0.5)
    pack = bool(values[5] > 0.5)

    if load and (inspect or pack):
        raise ValueError(
            "The current hard rules are infeasible: load requires 6-bar, "
            "while inspect/pack require 4-bar."
        )
    if load:
        return [2]
    if inspect or pack:
        return [0]
    return list(range(len(BAR_ORDER)))


def _geometry_error_per_candidate(
    edge_t: torch.Tensor,
    bar_v: torch.Tensor,
) -> torch.Tensor:
    errors = edge_t.new_zeros(edge_t.shape[0])
    for v, bar_type in enumerate(BAR_ORDER):
        mask = bar_v == v
        if not mask.any():
            continue

        current = edge_t[mask]
        residuals = []
        for i, j in TYPE_COLLINEAR_EDGE_PAIRS.get(bar_type, ()):
            zi = F.normalize(current[:, i, :3, 2], dim=-1)
            zj = F.normalize(current[:, j, :3, 2], dim=-1)
            pi = current[:, i, :3, 3]
            pj = current[:, j, :3, 3]
            residuals.append(torch.linalg.cross(zi, zj, dim=-1).norm(dim=-1))
            residuals.append(
                torch.linalg.cross(zi, pj - pi, dim=-1).norm(dim=-1)
            )

        for group in TYPE_PARALLEL_EDGE_GROUPS.get(bar_type, ()):
            ref = F.normalize(current[:, group[0], :3, 2], dim=-1)
            for edge_index in group[1:]:
                direction = F.normalize(current[:, edge_index, :3, 2], dim=-1)
                residuals.append(
                    torch.linalg.cross(ref, direction, dim=-1).norm(dim=-1)
                )

        if residuals:
            errors[mask] = torch.stack(residuals, dim=-1).max(dim=-1).values
    return errors


def _angle_rmse_per_candidate(edge_angles: torch.Tensor) -> torch.Tensor:
    selected = edge_angles[:, EDGE_ANGLE_EQUALITY_INDICES]
    circular_mean = torch.atan2(
        torch.sin(selected).mean(dim=-1, keepdim=True),
        torch.cos(selected).mean(dim=-1, keepdim=True),
    )
    residual = wrap_angle(selected - circular_mean)
    return torch.sqrt(residual.pow(2).mean(dim=-1))


def _task_metrics(
    vreq: torch.Tensor,
    bar_v: torch.Tensor,
    scale: torch.Tensor,
    edge_angles: torch.Tensor,
) -> dict[str, torch.Tensor]:
    load = vreq[:, 3] > 0.5
    inspect = vreq[:, 4] > 0.5
    pack = vreq[:, 5] > 0.5

    bar_valid = torch.ones_like(load)
    bar_valid = torch.where(load, bar_v == 2, bar_valid)
    bar_valid = torch.where(inspect | pack, bar_v == 0, bar_valid)

    q2 = edge_angles[:, 2]
    q2_low, q2_high = LOAD_Q2_RANGE
    q2_excess = torch.maximum(
        torch.clamp(q2_low - q2, min=0.0),
        torch.clamp(q2 - q2_high, min=0.0),
    )
    q2_excess = torch.where(load, q2_excess, torch.zeros_like(q2_excess))

    pack_limit = torch.as_tensor(
        PACK_LIMIT_SCALE,
        dtype=scale.dtype,
        device=scale.device,
    )
    pack_excess = torch.clamp(scale - pack_limit, min=0.0).max(dim=-1).values
    pack_excess = torch.where(pack, pack_excess, torch.zeros_like(pack_excess))

    return {
        "bar_type_valid": bar_valid,
        "load_q2_excess": q2_excess,
        "pack_scale_excess": pack_excess,
    }


class HardConstraintDecoder:
    """Normalize nodes, derive graph fields, evaluate constraints, and score."""

    def __init__(
        self,
        graph_layer: GraphImputationLayer,
        angle_recoverer: EdgeAngleRecoverer,
        thresholds: ConstraintThresholds | None = None,
    ) -> None:
        self.graph_layer = graph_layer
        self.angle_recoverer = angle_recoverer
        self.thresholds = thresholds or ConstraintThresholds()

    def decode_and_evaluate(
        self,
        nodes: torch.Tensor,
        leg_angle: torch.Tensor,
        scale: torch.Tensor,
        vreq: torch.Tensor,
        bar_v: torch.Tensor,
        bar_probability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        repaired_nodes = torch.cat(
            [nodes[..., :3], normalize_quaternion(nodes[..., 3:])],
            dim=-1,
        )
        graph = self.graph_layer(repaired_nodes)
        edge_angles = self.angle_recoverer(graph["edge_t"])
        edge_pose = transform_to_pose(graph["edge_t"])
        leg_base = transform_to_pose(graph["leg_base_t"])

        quaternion_error = (
            torch.linalg.norm(repaired_nodes[..., 3:], dim=-1) - 1.0
        ).abs().max(dim=-1).values

        edge_positions = graph["edge_t"][:, :, :3, 3]
        spacing = torch.linalg.norm(
            edge_positions - torch.roll(edge_positions, shifts=-1, dims=1),
            dim=-1,
        )
        spacing_error = (
            spacing - graph["s_edge_spacing"].unsqueeze(0)
        ).abs()
        spacing_rmse = torch.sqrt(spacing_error.pow(2).mean(dim=-1))
        spacing_error_max = spacing_error.max(dim=-1).values

        geometry_error = _geometry_error_per_candidate(graph["edge_t"], bar_v)
        angle_rmse = _angle_rmse_per_candidate(edge_angles)
        size_excess = torch.clamp(scale - vreq[:, :3], min=0.0).max(dim=-1).values
        size_margin = ((vreq[:, :3] - scale) / vreq[:, :3].clamp_min(1e-6)).min(
            dim=-1
        ).values
        task = _task_metrics(vreq, bar_v, scale, edge_angles)

        finite = (
            torch.isfinite(repaired_nodes).flatten(1).all(dim=-1)
            & torch.isfinite(edge_pose).flatten(1).all(dim=-1)
            & torch.isfinite(leg_base).flatten(1).all(dim=-1)
            & torch.isfinite(leg_angle).all(dim=-1)
            & torch.isfinite(scale).all(dim=-1)
        )

        threshold = self.thresholds
        checks = {
            "finite": finite,
            "quaternion": quaternion_error <= threshold.quaternion_norm_max,
            "spacing": (spacing_rmse <= threshold.spacing_rmse_max)
            & (spacing_error_max <= threshold.spacing_error_max),
            "geometry": geometry_error <= threshold.geometry_error_max,
            "angle": angle_rmse <= threshold.angle_rmse_max,
            "size": size_excess <= threshold.size_excess_max,
            "bar_type": task["bar_type_valid"],
            "task": (task["load_q2_excess"] <= threshold.task_excess_max)
            & (task["pack_scale_excess"] <= threshold.task_excess_max),
        }
        valid = torch.stack(list(checks.values()), dim=-1).all(dim=-1)

        quaternion_score = torch.exp(
            -torch.square(quaternion_error / threshold.quaternion_norm_max)
        )
        spacing_score = torch.exp(
            -torch.square(spacing_rmse / threshold.spacing_rmse_max)
        )
        geometry_score = torch.exp(
            -torch.square(geometry_error / threshold.geometry_error_max)
        )
        angle_score = torch.exp(
            -torch.square(angle_rmse / threshold.angle_rmse_max)
        )
        constraint_score = torch.stack(
            [quaternion_score, spacing_score, geometry_score, angle_score],
            dim=-1,
        ).mean(dim=-1)

        size_score = torch.where(
            checks["size"],
            torch.clamp(0.5 + size_margin / 0.5, min=0.0, max=1.0),
            torch.exp(
                -torch.square(
                    size_excess / max(threshold.size_excess_max, 1e-8)
                )
            ),
        )
        task_violation = torch.maximum(
            task["load_q2_excess"],
            task["pack_scale_excess"],
        )
        task_score = torch.where(
            checks["bar_type"] & checks["task"],
            torch.ones_like(task_violation),
            torch.exp(
                -torch.square(
                    task_violation / max(threshold.task_excess_max, 1e-8)
                )
            )
            * checks["bar_type"].to(task_violation.dtype),
        )
        bar_score = bar_probability.clamp(min=0.0, max=1.0)
        overall_score = (
            0.45 * constraint_score
            + 0.20 * size_score
            + 0.25 * task_score
            + 0.10 * bar_score
        )
        confidence = torch.pow(constraint_score.clamp_min(1e-8), 0.45)
        confidence = confidence * torch.pow(size_score.clamp_min(1e-8), 0.20)
        confidence = confidence * torch.pow(task_score.clamp_min(1e-8), 0.25)
        confidence = confidence * torch.pow(bar_score.clamp_min(1e-8), 0.10)

        return {
            "nodes": repaired_nodes,
            "edge_angles": edge_angles,
            "edge_pose": edge_pose,
            "leg_base": leg_base,
            "leg_angle": leg_angle,
            "scale": scale,
            "valid": valid,
            "checks": checks,
            "metrics": {
                "quaternion_norm_error": quaternion_error,
                "spacing_rmse": spacing_rmse,
                "spacing_error_max": spacing_error_max,
                "geometry_error_max": geometry_error,
                "angle_rmse": angle_rmse,
                "size_excess_max": size_excess,
                "size_margin_min": size_margin,
                "load_q2_excess": task["load_q2_excess"],
                "pack_scale_excess": task["pack_scale_excess"],
            },
            "scores": {
                "constraint": constraint_score,
                "size": size_score,
                "task": task_score,
                "bar_type": bar_score,
                "overall": overall_score,
                "confidence": confidence,
            },
        }


def candidate_feature(candidate: dict) -> np.ndarray:
    """Build a sign-invariant feature vector for diversity selection."""
    nodes = np.asarray(candidate["structure"]["nodes"], dtype=np.float64)
    positions = nodes[:, :3] / 0.5

    quaternions = nodes[:, 3:]
    signs = np.where(quaternions[:, :1] < 0.0, -1.0, 1.0)
    quaternions = quaternions * signs

    scale = np.asarray(candidate["structure"]["global"]["scale"], dtype=np.float64) / 0.5
    leg_angle = (
        np.asarray(candidate["structure"]["global"]["leg_angle"], dtype=np.float64)
        / np.pi
    )
    return np.concatenate(
        [positions.reshape(-1), quaternions.reshape(-1), scale, leg_angle]
    )


def select_diverse_top_k(
    candidates: list[dict],
    top_k: int,
    min_distance: float = 0.02,
    min_per_bar: int = 0,
) -> list[dict]:
    """Greedily keep high-scoring candidates that are not near duplicates."""
    if top_k <= 0:
        return []
    ordered = sorted(candidates, key=lambda item: item["score"], reverse=True)
    selected = []
    selected_features = []
    selected_ids = set()
    deferred = []

    coverage_candidates = []
    if min_per_bar > 0:
        for bar_type in BAR_ORDER:
            matches = [
                candidate
                for candidate in ordered
                if candidate["bar_type"] == bar_type
            ]
            coverage_candidates.extend(matches[:min_per_bar])
        coverage_candidates.sort(key=lambda item: item["score"], reverse=True)

    for candidate in coverage_candidates[:top_k]:
        feature = candidate_feature(candidate)
        if not selected_features:
            distance = float("inf")
        else:
            distance = min(
                float(np.sqrt(np.mean(np.square(feature - other))))
                for other in selected_features
            )
        candidate["diversity_distance"] = (
            None if not np.isfinite(distance) else distance
        )
        candidate["diversity_passed"] = distance >= min_distance
        candidate["selection_reason"] = "bar_coverage"
        selected.append(candidate)
        selected_features.append(feature)
        selected_ids.add(id(candidate))

    for candidate in ordered:
        if id(candidate) in selected_ids:
            continue
        feature = candidate_feature(candidate)
        if not selected_features:
            distance = float("inf")
        else:
            distance = min(
                float(np.sqrt(np.mean(np.square(feature - other))))
                for other in selected_features
            )

        candidate["diversity_distance"] = (
            None if not np.isfinite(distance) else distance
        )
        if distance >= min_distance and len(selected) < top_k:
            candidate["diversity_passed"] = True
            candidate["selection_reason"] = "score_and_diversity"
            selected.append(candidate)
            selected_features.append(feature)
            selected_ids.add(id(candidate))
        else:
            candidate["diversity_passed"] = False
            deferred.append((candidate, feature, distance))

    for candidate, feature, distance in deferred:
        if len(selected) >= top_k:
            break
        candidate["diversity_distance"] = (
            None if not np.isfinite(distance) else distance
        )
        candidate["selection_reason"] = "score_fill"
        selected.append(candidate)
        selected_features.append(feature)

    selected.sort(key=lambda item: item["score"], reverse=True)
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank
    return selected
