from __future__ import annotations

import torch
from torch import nn


class KNN(nn.Module):
    """Small torch.cdist-based fallback for knn_cuda.KNN.

    It supports the call pattern used by the vendored InternVL point-cloud
    encoder: KNN(k=group_size, transpose_mode=True)(xyz, center), where xyz is
    [B, N, 3] and center is [B, G, 3].
    """

    def __init__(self, k: int, transpose_mode: bool = False) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.transpose_mode = transpose_mode

    def forward(self, xyz: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transpose_mode:
            points = xyz
            centers = query
        else:
            points = xyz.transpose(1, 2) if xyz.ndim == 3 and xyz.size(1) == 3 else xyz
            centers = query.transpose(1, 2) if query.ndim == 3 and query.size(1) == 3 else query

        if points.ndim != 3 or centers.ndim != 3:
            raise ValueError("xyz and query must both be rank-3 tensors")
        if points.size(0) != centers.size(0) or points.size(-1) != centers.size(-1):
            raise ValueError("xyz and query must share batch and coordinate dimensions")

        k = min(self.k, points.size(1))
        distances = torch.cdist(centers.float(), points.float(), p=2)
        values, indices = torch.topk(distances, k=k, dim=-1, largest=False, sorted=True)
        if k < self.k:
            pad_count = self.k - k
            values = torch.cat([values, values[..., -1:].expand(-1, -1, pad_count)], dim=-1)
            indices = torch.cat([indices, indices[..., -1:].expand(-1, -1, pad_count)], dim=-1)
        return values.to(dtype=xyz.dtype), indices.long()
