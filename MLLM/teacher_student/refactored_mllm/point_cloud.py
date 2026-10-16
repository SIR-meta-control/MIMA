from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SAMPLE_POINTS = 2048


def coerce_point_cloud(points: Any) -> np.ndarray:
    if isinstance(points, str):
        stripped = points.strip()
        if stripped.startswith("["):
            points = json.loads(stripped)

    if isinstance(points, list) and points and isinstance(points[0], dict):
        points = [
            (point["x"], point["y"], point["z"])
            for point in points
            if {"x", "y", "z"}.issubset(point)
        ]

    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("Point cloud must be an Nx3 array using x/y/z columns")

    array = array[:, :3]
    finite_rows = np.isfinite(array).all(axis=1)
    array = array[finite_rows]
    if array.size == 0:
        raise ValueError("Point cloud has no finite x/y/z points")
    return array


def load_point_cloud(path: str | Path) -> np.ndarray:
    point_cloud_path = Path(path)
    if not point_cloud_path.exists():
        raise FileNotFoundError(point_cloud_path)

    suffix = point_cloud_path.suffix.lower()
    if suffix == ".npy":
        return coerce_point_cloud(np.load(point_cloud_path))
    if suffix in {".csv", ".txt", ".xyz"}:
        return _load_text_point_cloud(point_cloud_path)
    if suffix == ".ply":
        return _load_ascii_ply(point_cloud_path)
    return _load_open3d_point_cloud(point_cloud_path)


def sample_point_cloud(
    points: np.ndarray,
    sample_points: int = DEFAULT_SAMPLE_POINTS,
    seed: int = 0,
) -> np.ndarray:
    array = coerce_point_cloud(points)
    if len(array) == sample_points:
        return array.copy()

    rng = np.random.default_rng(seed)
    replace = len(array) < sample_points
    indices = rng.choice(len(array), size=sample_points, replace=replace)
    return array[indices]


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    array = coerce_point_cloud(points).astype(float, copy=True)
    centroid = np.mean(array, axis=0)
    array -= centroid
    radius = np.max(np.sqrt(np.sum(array**2, axis=1)))
    if radius > 0.0:
        array /= radius
    return array


def load_raw_point_cloud(
    points: Any,
    sample_points: int = DEFAULT_SAMPLE_POINTS,
    normalize: bool = True,
    seed: int = 0,
) -> np.ndarray:
    sampled = sample_point_cloud(
        coerce_point_cloud(points),
        sample_points=sample_points,
        seed=seed,
    )
    return normalize_point_cloud(sampled) if normalize else sampled


def _load_text_point_cloud(path: Path) -> np.ndarray:
    try:
        return coerce_point_cloud(np.loadtxt(path, delimiter=","))
    except Exception:
        try:
            return coerce_point_cloud(np.loadtxt(path))
        except Exception:
            return coerce_point_cloud(pd.read_csv(path).to_numpy())


def _load_ascii_ply(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header: list[str] = []
        for line in handle:
            header.append(line.rstrip("\n"))
            if line.strip() == "end_header":
                break
        else:
            raise ValueError(f"PLY file is missing end_header: {path}")

        if not any(line.strip() == "format ascii 1.0" for line in header):
            return _load_open3d_point_cloud(path)

        vertex_count = None
        for line in header:
            parts = line.split()
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                break
        if vertex_count is None:
            raise ValueError(f"PLY file is missing vertex count: {path}")

        rows = []
        for _ in range(vertex_count):
            parts = handle.readline().split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return coerce_point_cloud(rows)


def _load_open3d_point_cloud(path: Path) -> np.ndarray:
    try:
        import open3d as o3d
    except Exception as exc:
        raise ValueError(
            "Reading this point-cloud format requires Open3D, or provide "
            ".npy/.csv/.txt/.xyz/ascii .ply point clouds."
        ) from exc

    cloud = o3d.io.read_point_cloud(str(path))
    return coerce_point_cloud(np.asarray(cloud.points, dtype=float))
