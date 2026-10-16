from __future__ import annotations

from pathlib import Path

import numpy as np

from refactored_mllm.features import load_depth_array, load_rgb_array
from refactored_mllm.point_cloud import load_point_cloud


SENSOR_SCALE_PREFIX = "Raw sensor statistics (metric values in meters):"

SENSOR_PROMPT_FEATURES = (
    ("pc_y75", "pc_y_p75_m"),
    ("pc_x75", "pc_x_p75_m"),
    ("depth_left", "depth_left_mean_m"),
    ("depth_right", "depth_right_mean_m"),
    ("pc_z25", "pc_z_p25_m"),
    ("rgb_std_r", "rgb_std_r"),
    ("rgb_gray50", "rgb_gray_p50"),
    ("pc_y50", "pc_y_p50_m"),
    ("pc_z50", "pc_z_p50_m"),
    ("depth_bottom", "depth_bottom_mean_m"),
    ("depth_p90", "depth_p90_m"),
    ("rgb_mean_g", "rgb_mean_g"),
    ("rgb_std_g", "rgb_std_g"),
    ("pc_z_mean", "pc_center_z_m"),
    ("pc_count_k", "pc_point_count_k"),
    ("rgb_mean_b", "rgb_mean_b"),
    ("depth_valid", "depth_valid_ratio"),
    ("pc_z99", "pc_z_p99_m"),
    ("pc_z95", "pc_z_p95_m"),
    ("pc_z90", "pc_z_p90_m"),
)


def sensor_scale_metadata(
    depth_path: str | Path,
    point_cloud_path: str | Path,
    *,
    rgb_path: str | Path | None = None,
) -> dict[str, float]:
    depth = np.asarray(load_depth_array(depth_path), dtype=float)
    if depth.ndim != 2:
        raise ValueError(f"Depth array must be 2D: {depth_path}")
    valid = np.isfinite(depth) & (depth > 0.0)
    valid_depth = depth[valid]
    points = np.asarray(load_point_cloud(point_cloud_path), dtype=float)
    points = points[np.isfinite(points).all(axis=1), :3]
    if valid_depth.size == 0 or points.size == 0:
        raise ValueError(f"Cannot extract sensor scale metadata for {depth_path}")

    fill_value = float(np.median(valid_depth))
    cleaned_depth = np.where(valid, depth, fill_value)
    height, width = cleaned_depth.shape
    h0, h1 = height // 4, max(height // 4 + 1, (3 * height) // 4)
    w0, w1 = width // 4, max(width // 4 + 1, (3 * width) // 4)
    gradient_magnitude = np.zeros_like(cleaned_depth, dtype=float)
    for gradient in np.gradient(cleaned_depth):
        gradient_magnitude += gradient**2
    gradient_magnitude = np.sqrt(gradient_magnitude)

    point_center = points.mean(axis=0)
    radius = np.sqrt(((points - point_center) ** 2).sum(axis=1)).max()
    metadata = {
        "depth_valid_ratio": float(valid.mean()),
        "depth_mean_m": float(valid_depth.mean()),
        "depth_std_m": float(valid_depth.std()),
        **{
            f"depth_p{percentile:02d}_m": float(np.percentile(valid_depth, percentile))
            for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        },
        "depth_center_mean_m": float(cleaned_depth[h0:h1, w0:w1].mean()),
        "depth_top_mean_m": float(cleaned_depth[: max(1, height // 3), :].mean()),
        "depth_bottom_mean_m": float(cleaned_depth[(2 * height) // 3 :, :].mean()),
        "depth_left_mean_m": float(cleaned_depth[:, : max(1, width // 3)].mean()),
        "depth_right_mean_m": float(cleaned_depth[:, (2 * width) // 3 :].mean()),
        "depth_gradient_p95_m": float(np.percentile(gradient_magnitude, 95)),
        "pc_point_count_k": float(len(points) / 1000.0),
        "pc_center_x_m": float(point_center[0]),
        "pc_center_y_m": float(point_center[1]),
        "pc_center_z_m": float(point_center[2]),
        "pc_radius_m": float(radius),
    }
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        values = points[:, axis_index]
        for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            metadata[f"pc_{axis_name}_p{percentile:02d}_m"] = float(
                np.percentile(values, percentile)
            )
        metadata[f"pc_{axis_name}_range_05_95_m"] = float(
            np.percentile(values, 95) - np.percentile(values, 5)
        )
        metadata[f"pc_{axis_name}_std_m"] = float(values.std())

    if rgb_path is not None:
        rgb = np.asarray(load_rgb_array(rgb_path), dtype=float)[:, :, :3]
        if rgb.max(initial=0.0) > 1.0:
            rgb = np.clip(rgb, 0.0, 255.0) / 255.0
        gray = rgb.mean(axis=2)
        for channel_index, channel_name in enumerate(("r", "g", "b")):
            metadata[f"rgb_mean_{channel_name}"] = float(
                rgb[:, :, channel_index].mean()
            )
            metadata[f"rgb_std_{channel_name}"] = float(rgb[:, :, channel_index].std())
        metadata.update(
            {
                "rgb_gray_p10": float(np.percentile(gray, 10)),
                "rgb_gray_p50": float(np.percentile(gray, 50)),
                "rgb_gray_p90": float(np.percentile(gray, 90)),
                "rgb_dark_ratio": float((gray < 0.2).mean()),
            }
        )
    return metadata


def format_sensor_scale(metadata: dict[str, object], decimals: int = 3) -> str:
    values = [
        f"{short_name}={float(metadata[key]):.{decimals}f}"
        for short_name, key in SENSOR_PROMPT_FEATURES
    ]
    return f"{SENSOR_SCALE_PREFIX} " + ",".join(values) + "."
