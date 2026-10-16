from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from refactored_mllm.point_cloud import coerce_point_cloud, load_point_cloud


def load_rgb_array(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def decode_rgb_payload(payload: str | bytes) -> np.ndarray:
    raw = _decode_payload_bytes(payload)
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"))


def load_depth_array(path: str | Path) -> np.ndarray:
    depth_path = Path(path)
    if not depth_path.exists():
        raise FileNotFoundError(depth_path)

    suffix = depth_path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(depth_path), dtype=float)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        with Image.open(depth_path) as image:
            return np.asarray(image, dtype=float)
    try:
        return np.asarray(np.loadtxt(depth_path, delimiter=","), dtype=float)
    except Exception:
        return np.asarray(np.loadtxt(depth_path), dtype=float)


def extract_rgb_features(
    rgb: np.ndarray, scenario: str = "open_ground"
) -> dict[str, float]:
    del scenario
    array = np.asarray(rgb, dtype=float)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("RGB image must be an HxWx3 array")

    rgb_array = np.nan_to_num(array[:, :, :3], nan=0.0, posinf=255.0, neginf=0.0)
    if rgb_array.max(initial=0.0) > 1.0:
        rgb_array = np.clip(rgb_array, 0.0, 255.0) / 255.0

    red = rgb_array[:, :, 0]
    green = rgb_array[:, :, 1]
    blue = rgb_array[:, :, 2]
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


def extract_depth_features(depth: np.ndarray) -> dict[str, float]:
    array = np.asarray(depth, dtype=float)
    if array.ndim != 2:
        raise ValueError("Depth array must be 2D")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("Depth array must have at least 2 rows and 2 columns")

    valid = np.isfinite(array) & (array > 0.0)
    finite = array[valid]
    if finite.size == 0:
        raise ValueError("Depth array has no positive finite values")

    fill_value = float(np.nanmedian(finite))
    cleaned = np.where(valid, array, fill_value)
    gradients = np.gradient(cleaned)
    gradient_magnitude = np.zeros_like(array, dtype=float)
    for gradient in gradients:
        gradient_magnitude += gradient**2
    gradient_magnitude = np.sqrt(gradient_magnitude)
    finite_gradients = gradient_magnitude[np.isfinite(gradient_magnitude)]
    discontinuity = (
        float(np.percentile(finite_gradients, 95)) if finite_gradients.size else 0.0
    )
    discontinuity = max(discontinuity, 0.0)

    return {
        "depth_min_clearance_m": float(np.percentile(finite, 10)),
        "depth_obstacle_height_m": float(discontinuity),
        "depth_step_height_m": float(discontinuity),
        "depth_slope_deg": float(math.degrees(math.atan(discontinuity))),
    }


def extract_depth_features_from_point_cloud(points: np.ndarray) -> dict[str, float]:
    array = coerce_point_cloud(points)
    y_forward = array[:, 1]
    z_height = array[:, 2]
    positive_forward = y_forward[np.isfinite(y_forward) & (y_forward > 0.0)]
    if positive_forward.size:
        min_clearance = float(np.percentile(positive_forward, 10))
    else:
        min_clearance = float(max(_percentile_range(y_forward), 0.0))

    ground = float(np.percentile(z_height, 10))
    obstacle_height = float(max(np.percentile(z_height, 90) - ground, 0.0))
    ground_band = z_height[z_height <= np.percentile(z_height, 25)]
    roughness = float(np.std(ground_band)) if ground_band.size else 0.0

    return {
        "depth_min_clearance_m": min_clearance,
        "depth_obstacle_height_m": obstacle_height,
        "depth_step_height_m": float(max(roughness, 0.0)),
        "depth_slope_deg": 0.0,
    }


def extract_point_cloud_features(points: np.ndarray) -> dict[str, float]:
    array = coerce_point_cloud(points)
    x_lateral = array[:, 0]
    y_forward = array[:, 1]
    z_height = array[:, 2]

    width = _percentile_range(x_lateral)
    forward_depth = _percentile_range(y_forward)
    ground_threshold = np.percentile(z_height, 25)
    ground_band = z_height[z_height <= ground_threshold]
    roughness = float(np.std(ground_band)) if ground_band.size else 0.0

    return {
        "pc_corridor_width_m": width,
        "pc_ground_roughness": float(max(roughness, 0.0)),
        "pc_turn_angle_deg": _estimate_xy_turn_angle(array),
        "pc_free_space_area_m2": float(max(width * forward_depth, 0.0)),
    }


def extract_task_features(text: str = "") -> dict[str, float]:
    lowered = text.lower()
    return {
        "cmd_load": float(any(word in lowered for word in ("load", "carry", "cargo"))),
        "cmd_inspect": float(
            any(
                word in lowered for word in ("inspect", "inspection", "check", "survey")
            )
        ),
        "cmd_pack": float(
            any(word in lowered for word in ("pack", "packing", "parcel"))
        ),
    }


def load_or_decode_point_cloud(
    *,
    path: str | Path | None = None,
    data: Any | None = None,
) -> np.ndarray:
    if data is not None:
        return coerce_point_cloud(data)
    if path is not None:
        return load_point_cloud(path)
    raise ValueError("point cloud requires either path or data")


def _decode_payload_bytes(payload: str | bytes) -> bytes:
    if isinstance(payload, bytes):
        return payload

    stripped = payload.strip()
    if "," in stripped and stripped.lower().startswith("data:"):
        stripped = stripped.split(",", 1)[1]

    try:
        return base64.b64decode(stripped, validate=True)
    except Exception:
        return payload.encode("latin1")


def _percentile_range(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(max(np.percentile(finite, 95) - np.percentile(finite, 5), 0.0))


def _estimate_xy_turn_angle(points: np.ndarray) -> float:
    xy = coerce_point_cloud(points)[:, [0, 1]]
    if len(xy) < 3:
        return 0.0

    centered = xy - xy.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0

    direction = vh[0]
    return float(abs(math.degrees(math.atan2(direction[1], direction[0]))))


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))
