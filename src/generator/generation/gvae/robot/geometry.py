"""Differentiable geometry utilities and graph-imputation layer."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gvae.core.constants import LEG_MOUNT_NODE_INDICES


EDGE_ANGLE_RULES = {
    0: {"prev": 7, "sign": 1.0},
    1: {"prev": 0, "sign": 1.0},
    2: {"prev": 1, "sign": 1.0},
    3: {"prev": 2, "sign": 1.0},
    4: {"prev": 3, "sign": -1.0},
    5: {"prev": 4, "sign": 1.0},
    6: {"prev": 5, "sign": 1.0},
}

GRAPH_IMPUTATION_KEYS = ("T_node_leg", "T_node_edge", "S_edge_spacing")


def normalize_quaternion(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=eps)


def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = normalize_quaternion(quat)
    w, x, y, z = quat.unbind(dim=-1)
    two = 2.0

    row0 = torch.stack(
        [
            1.0 - two * (y * y + z * z),
            two * (x * y - w * z),
            two * (x * z + w * y),
        ],
        dim=-1,
    )
    row1 = torch.stack(
        [
            two * (x * y + w * z),
            1.0 - two * (x * x + z * z),
            two * (y * z - w * x),
        ],
        dim=-1,
    )
    row2 = torch.stack(
        [
            two * (x * z - w * y),
            two * (y * z + w * x),
            1.0 - two * (x * x + y * y),
        ],
        dim=-1,
    )
    return torch.stack([row0, row1, row2], dim=-2)


def pose_to_transform(pose: torch.Tensor) -> torch.Tensor:
    """Convert [..., 7] [x, y, z, qw, qx, qy, qz] to [..., 4, 4]."""
    trans = pose[..., :3]
    quat = pose[..., 3:]
    rot = quaternion_to_matrix(quat)

    out_shape = pose.shape[:-1] + (4, 4)
    transform = torch.zeros(out_shape, dtype=pose.dtype, device=pose.device)
    transform[..., :3, :3] = rot
    transform[..., :3, 3] = trans
    transform[..., 3, 3] = 1.0
    return transform


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert [..., 3, 3] rotation matrices to [qw, qx, qy, qz]."""
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0))
    qx = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0))
    qy = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0))
    qz = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0))

    qx = torch.copysign(qx, m21 - m12)
    qy = torch.copysign(qy, m02 - m20)
    qz = torch.copysign(qz, m10 - m01)
    return normalize_quaternion(torch.stack([qw, qx, qy, qz], dim=-1))


def transform_to_pose(transform: torch.Tensor) -> torch.Tensor:
    """Convert [..., 4, 4] transforms to [..., 7] pose parameters."""
    pos = transform[..., :3, 3]
    quat = matrix_to_quaternion(transform[..., :3, :3])
    return torch.cat([pos, quat], dim=-1)


def transform_position_rotation_loss(
    pred_t: torch.Tensor,
    target_t: torch.Tensor,
) -> torch.Tensor:
    pos_loss = F.mse_loss(pred_t[..., :3, 3], target_t[..., :3, 3])
    rot_loss = F.mse_loss(pred_t[..., :3, :3], target_t[..., :3, :3])
    return pos_loss + rot_loss


def pose_parameter_loss(pred_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    pred_pos = pred_pose[..., :3]
    target_pos = target_pose[..., :3]
    pred_quat = normalize_quaternion(pred_pose[..., 3:])
    target_quat = normalize_quaternion(target_pose[..., 3:])

    pos_loss = F.mse_loss(pred_pos, target_pos)
    quat_direct = (pred_quat - target_quat).pow(2).mean(dim=-1)
    quat_flipped = (pred_quat + target_quat).pow(2).mean(dim=-1)
    quat_loss = torch.minimum(quat_direct, quat_flipped).mean()
    return pos_loss + quat_loss


def rotation_z_np(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return transform


def quaternion_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    w, x, y, z = quat
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
            ],
            [
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
            ],
            [
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def edge_row_to_transform_np(edge_row: list[float] | np.ndarray) -> np.ndarray:
    row = np.asarray(edge_row, dtype=np.float64)
    pose = row[1:] if row.shape[0] == 8 else row
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_matrix_np(pose[3:])
    transform[:3, 3] = pose[:3]
    return transform


def derive_edge_angle_static_transforms(reference_edges: list[list[float]]) -> np.ndarray:
    """Derive fixed A_k transforms used to recover edge angles from edge poses."""
    static_transforms = np.zeros((7, 4, 4), dtype=np.float64)
    for edge_idx, rule in EDGE_ANGLE_RULES.items():
        prev_idx = int(rule["prev"])
        sign = float(rule["sign"])
        angle = float(reference_edges[edge_idx][0])
        t_prev = edge_row_to_transform_np(reference_edges[prev_idx])
        t_curr = edge_row_to_transform_np(reference_edges[edge_idx])
        t_prev_to_curr = np.linalg.inv(t_prev) @ t_curr
        static_transforms[edge_idx] = t_prev_to_curr @ rotation_z_np(-sign * angle)
    return static_transforms


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class EdgeAngleRecoverer(nn.Module):
    """Recover edges[:, 0] from edge transform matrices."""

    def __init__(self, static_transforms: np.ndarray) -> None:
        super().__init__()
        if static_transforms.shape != (7, 4, 4):
            raise ValueError(
                f"static_transforms must have shape (7, 4, 4), got {static_transforms.shape}"
            )
        self.register_buffer(
            "static_transforms",
            torch.as_tensor(static_transforms, dtype=torch.float32),
        )
        prev_indices = [EDGE_ANGLE_RULES[i]["prev"] for i in range(7)]
        signs = [EDGE_ANGLE_RULES[i]["sign"] for i in range(7)]
        self.register_buffer("prev_indices", torch.as_tensor(prev_indices, dtype=torch.long))
        self.register_buffer("signs", torch.as_tensor(signs, dtype=torch.float32))

    def forward(self, edge_t: torch.Tensor) -> torch.Tensor:
        batch_size = edge_t.shape[0]
        angles = torch.zeros((batch_size, 8), dtype=edge_t.dtype, device=edge_t.device)
        static = self.static_transforms.to(dtype=edge_t.dtype, device=edge_t.device)
        signs = self.signs.to(dtype=edge_t.dtype, device=edge_t.device)

        for edge_idx in range(7):
            prev_idx = int(self.prev_indices[edge_idx].item())
            t_prev = edge_t[:, prev_idx]
            t_curr = edge_t[:, edge_idx]
            t_zero = torch.matmul(t_prev, static[edge_idx].expand(batch_size, -1, -1))
            r_diff = torch.matmul(
                t_zero[:, :3, :3].transpose(-1, -2),
                t_curr[:, :3, :3],
            )
            angles[:, edge_idx] = signs[edge_idx] * torch.atan2(
                r_diff[:, 1, 0],
                r_diff[:, 0, 0],
            )
        return wrap_angle(angles)


class GraphImputationLayer(nn.Module):
    """Hard graph-imputation layer: nodes -> edges and leg_base."""

    edge_node_indices = (1, 2, 3, 4, 5, 6, 7, 0)
    leg_node_indices = LEG_MOUNT_NODE_INDICES

    def __init__(self, graph_imputation_path: str | Path) -> None:
        super().__init__()
        data = load_graph_imputation_yaml(graph_imputation_path)
        self.register_buffer(
            "t_node_leg",
            torch.as_tensor(data["T_node_leg"], dtype=torch.float32),
        )
        self.register_buffer(
            "t_node_edge",
            torch.as_tensor(data["T_node_edge"], dtype=torch.float32),
        )
        self.register_buffer(
            "s_edge_spacing",
            torch.as_tensor(data["S_edge_spacing"], dtype=torch.float32),
        )

    def forward(self, nodes: torch.Tensor) -> dict[str, torch.Tensor]:
        nodes_t = pose_to_transform(nodes)
        edge_source = nodes_t[:, self.edge_node_indices]
        leg_source = nodes_t[:, self.leg_node_indices]

        t_node_edge = self.t_node_edge.to(dtype=nodes.dtype, device=nodes.device)
        t_node_leg = self.t_node_leg.to(dtype=nodes.dtype, device=nodes.device)

        edge_t = torch.matmul(edge_source, t_node_edge.unsqueeze(0))
        leg_base_t = torch.matmul(leg_source, t_node_leg.unsqueeze(0))
        return {
            "nodes_t": nodes_t,
            "edge_t": edge_t,
            "leg_base_t": leg_base_t,
            "s_edge_spacing": self.s_edge_spacing.to(
                dtype=nodes.dtype,
                device=nodes.device,
            ),
        }


def load_graph_imputation_yaml(path: str | Path) -> dict[str, list]:
    """Load graph-imputation constants from a small YAML file.

    The project stores these constants as top-level YAML keys whose values are
    flow-style lists. This keeps the file valid YAML and human-readable while
    avoiding a hard runtime dependency on PyYAML.
    """
    data = {}
    current_key = None
    current_lines = []

    def flush_current() -> None:
        nonlocal current_key, current_lines
        if current_key is None:
            return
        value_text = "\n".join(current_lines).strip()
        if not value_text:
            raise ValueError(f"Missing value for graph-imputation key {current_key!r}")
        data[current_key] = ast.literal_eval(value_text)
        current_key = None
        current_lines = []

    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        is_top_level_key = raw_line == raw_line.lstrip() and ":" in raw_line
        if is_top_level_key:
            flush_current()
            key, value = raw_line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key not in GRAPH_IMPUTATION_KEYS:
                raise ValueError(f"Unknown graph-imputation key {key!r} in {path}")
            current_key = key
            if value:
                current_lines.append(value)
            continue

        if current_key is None:
            raise ValueError(f"Invalid graph-imputation YAML line {line_number}: {raw_line}")
        current_lines.append(raw_line)

    flush_current()

    missing = [key for key in GRAPH_IMPUTATION_KEYS if key not in data]
    if missing:
        raise ValueError(f"Missing graph-imputation keys in {path}: {missing}")
    return data
