from __future__ import annotations

import torch


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Pure PyTorch fallback for pointnet2_ops.furthest_point_sample.

    The legacy InternVL point-cloud encoder only samples a very small number of
    centers, so this compatibility implementation is fast enough for the
    refactored training/evaluation workflow.
    """

    if xyz.ndim != 3 or xyz.size(-1) != 3:
        raise ValueError(f"xyz must have shape [B, N, 3], got {tuple(xyz.shape)}")
    if npoint <= 0:
        raise ValueError("npoint must be positive")

    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    sample_count = min(npoint, num_points)
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch_size, num_points), float("inf"), device=device)
    farthest = torch.zeros(batch_size, dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    xyz_float = xyz.float()

    for i in range(sample_count):
        centroids[:, i] = farthest
        centroid = xyz_float[batch_indices, farthest].view(batch_size, 1, 3)
        dist = torch.sum((xyz_float - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1).indices

    if sample_count < npoint:
        centroids[:, sample_count:] = centroids[:, sample_count - 1 : sample_count]
    return centroids


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather [B, C, N] features at [B, S] indices, matching pointnet2_ops."""

    if features.ndim != 3:
        raise ValueError(f"features must have shape [B, C, N], got {tuple(features.shape)}")
    if idx.ndim != 2:
        raise ValueError(f"idx must have shape [B, S], got {tuple(idx.shape)}")
    if features.size(0) != idx.size(0):
        raise ValueError("features and idx batch dimensions must match")

    idx = idx.long()
    expanded_idx = idx.unsqueeze(1).expand(-1, features.size(1), -1)
    return torch.gather(features, 2, expanded_idx)
