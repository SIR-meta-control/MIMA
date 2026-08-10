"""Deterministic 16-feature adapter used by the RF, DT, and GBT ablations."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .schema import FEATURE_COLUMNS


def extract_depth_features(depth: np.ndarray) -> dict[str, float]:
    array = np.asarray(depth, dtype=float)
    if array.ndim != 2:
        raise ValueError("depth array must be 2D")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("depth array must have at least 2 rows and 2 columns")

    valid = np.isfinite(array) & (array > 0.0)
    finite = array[valid]
    if finite.size == 0:
        raise ValueError("depth array has no positive finite values")

    fill_value = float(np.nanmedian(finite))
    cleaned = np.where(valid, array, fill_value)
    gradient_magnitude = np.zeros_like(array, dtype=float)
    for gradient in np.gradient(cleaned):
        gradient_magnitude += gradient**2
    finite_gradients = np.sqrt(gradient_magnitude)
    finite_gradients = finite_gradients[np.isfinite(finite_gradients)]
    discontinuity = (
        float(np.percentile(finite_gradients, 95)) if finite_gradients.size else 0.0
    )
    discontinuity = max(discontinuity, 0.0)
    return {
        "depth_min_clearance_m": float(np.percentile(finite, 10)),
        "depth_obstacle_height_m": discontinuity,
        "depth_step_height_m": discontinuity,
        "depth_slope_deg": float(math.degrees(math.atan(discontinuity))),
    }


def extract_point_cloud_features(points: np.ndarray) -> dict[str, float]:
    xyz = _coerce_points(points)
    width = _percentile_range(xyz[:, 0])
    forward_depth = _percentile_range(xyz[:, 1])
    ground_threshold = np.percentile(xyz[:, 2], 25)
    ground_band = xyz[xyz[:, 2] <= ground_threshold, 2]
    roughness = float(np.std(ground_band)) if ground_band.size else 0.0
    return {
        "pc_corridor_width_m": width,
        "pc_ground_roughness": max(roughness, 0.0),
        "pc_turn_angle_deg": _estimate_turn_angle(xyz),
        "pc_free_space_area_m2": max(width * forward_depth, 0.0),
    }


def extract_rgb_features(rgb: np.ndarray) -> dict[str, float]:
    array = np.asarray(rgb, dtype=float)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("RGB image must be an HxWx3 array")
    rgb_array = np.nan_to_num(array[:, :, :3], nan=0.0, posinf=255.0, neginf=0.0)
    if rgb_array.max(initial=0.0) > 1.0:
        rgb_array = np.clip(rgb_array, 0.0, 255.0) / 255.0
    red, green, blue = (rgb_array[:, :, index] for index in range(3))
    brightness = float(np.mean(rgb_array))
    redness = float(np.mean(np.maximum(red - np.maximum(green, blue), 0.0)))
    contrast = float(np.std(rgb_array))
    return {
        "rgb_tunnel_score": _clip01(1.0 - brightness),
        "rgb_step_score": _clip01(contrast),
        "rgb_obstacle_score": _clip01(max(redness, contrast)),
        "rgb_person_score": 0.0,
        "rgb_open_ground_score": _clip01(brightness * (1.0 - redness)),
    }


def build_feature_frame(
    *,
    rgb_path: str | Path,
    depth_path: str | Path,
    point_cloud_path: str | Path,
    sample_id: str = "sensor_sample",
    scenario: str = "open_ground",
) -> pd.DataFrame:
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    row: dict[str, object] = {
        "sample_id": sample_id,
        "scenario": scenario,
        "split": "inference",
    }
    row.update(extract_rgb_features(rgb))
    row.update(extract_depth_features(np.load(depth_path)))
    row.update(extract_point_cloud_features(np.load(point_cloud_path)))
    # The reported MuJoCo baselines were sensor-only; command features were neutral.
    row.update({"cmd_load": 0.0, "cmd_inspect": 0.0, "cmd_pack": 0.0})
    return pd.DataFrame([row])[["sample_id", "scenario", "split", *FEATURE_COLUMNS]]


def _coerce_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("point cloud must be an Nx3 array using x/y/z columns")
    array = array[:, :3]
    array = array[np.isfinite(array).all(axis=1)]
    if array.size == 0:
        raise ValueError("point cloud has no finite x/y/z points")
    return array


def _percentile_range(values: np.ndarray) -> float:
    return float(max(np.percentile(values, 95) - np.percentile(values, 5), 0.0))


def _estimate_turn_angle(points: np.ndarray) -> float:
    xy = points[:, :2]
    if len(xy) < 3:
        return 0.0
    centered = xy - xy.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    return float(abs(math.degrees(math.atan2(vh[0, 1], vh[0, 0]))))


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0)) if math.isfinite(value) else 0.0
