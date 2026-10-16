from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from refactored_mllm.features import (
    decode_rgb_payload,
    extract_depth_features,
    extract_depth_features_from_point_cloud,
    extract_point_cloud_features,
    extract_rgb_features,
    extract_task_features,
    load_depth_array,
    load_or_decode_point_cloud,
    load_rgb_array,
)
from refactored_mllm.models import (
    format_vr_prediction,
    postprocess_requirement_predictions,
)
from refactored_mllm.point_cloud import coerce_point_cloud
from refactored_mllm.schema import (
    FEATURE_COLUMNS,
    META_COLUMNS,
    PREDICTION_COLUMNS,
    REQUIREMENT_COLUMNS,
)


DEFAULT_STUDENT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "student"
STUDENT_MODEL_FILENAME = "student.joblib"
STUDENT_METADATA_FILENAME = "metadata.json"
DISTILL_FEATURE_VERSION = "distill_sensor_summary_v2_sampled"

# Regular-grid sampling bounds summary work without changing the API or feature schema.
DEFAULT_SUMMARY_MAX_SAMPLES = 65536

RGB_SUMMARY_COLUMNS = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "rgb_gray_p10",
    "rgb_gray_p50",
    "rgb_gray_p90",
    "rgb_dark_ratio",
    "rgb_bright_ratio",
]

DEPTH_SUMMARY_COLUMNS = [
    "depth_valid_ratio",
    "depth_mean_m",
    "depth_std_m",
    "depth_p01_m",
    "depth_p05_m",
    "depth_p10_m",
    "depth_p25_m",
    "depth_p50_m",
    "depth_p75_m",
    "depth_p90_m",
    "depth_p95_m",
    "depth_p99_m",
    "depth_center_mean_m",
    "depth_top_mean_m",
    "depth_bottom_mean_m",
    "depth_left_mean_m",
    "depth_right_mean_m",
    "depth_gradient_p95",
]

POINT_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
POINT_SUMMARY_COLUMNS = [
    "pc_point_count_k",
    *[
        f"pc_{axis}_p{percentile:02d}_m"
        for axis in ("x", "y", "z")
        for percentile in POINT_PERCENTILES
    ],
    "pc_x_range_05_95_m",
    "pc_y_range_05_95_m",
    "pc_z_range_05_95_m",
    "pc_x_mean_m",
    "pc_y_mean_m",
    "pc_z_mean_m",
    "pc_x_std_m",
    "pc_y_std_m",
    "pc_z_std_m",
    "pc_xy_area_05_95_m2",
    "pc_xz_area_05_95_m2",
    "pc_yz_area_05_95_m2",
]

DISTILL_FEATURE_COLUMNS = (
    list(FEATURE_COLUMNS)
    + RGB_SUMMARY_COLUMNS
    + DEPTH_SUMMARY_COLUMNS
    + POINT_SUMMARY_COLUMNS
)


def build_distill_feature_frame(
    *,
    rgb_path: str | Path | None = None,
    rgb_payload: str | bytes | None = None,
    depth_path: str | Path | None = None,
    depth_array: Any | None = None,
    point_cloud_path: str | Path | None = None,
    point_cloud_data: Any | None = None,
    sample_id: str = "sensor_sample",
    scenario: str = "open_ground",
    split: str = "inference",
    task_command: str = "",
    summary_max_samples: int | None = DEFAULT_SUMMARY_MAX_SAMPLES,
) -> pd.DataFrame:
    if rgb_payload is not None:
        rgb = decode_rgb_payload(rgb_payload)
    elif rgb_path is not None:
        rgb = load_rgb_array(rgb_path)
    else:
        raise ValueError("RGB input requires rgb_path or rgb_payload")

    point_cloud = load_or_decode_point_cloud(
        path=point_cloud_path,
        data=point_cloud_data,
    )

    depth = None
    if depth_array is not None:
        depth = np.asarray(depth_array, dtype=float)
    elif depth_path is not None:
        depth = load_depth_array(depth_path)

    sampled_rgb = _regular_sample_array(rgb, summary_max_samples)
    if depth is not None:
        sampled_depth = _regular_sample_array(depth, summary_max_samples)
        depth_features = extract_depth_features(sampled_depth)
        depth_summary = _extract_depth_summary(sampled_depth)
    else:
        depth_features = extract_depth_features_from_point_cloud(point_cloud)
        depth_summary = _zero_features(DEPTH_SUMMARY_COLUMNS)

    row: dict[str, object] = {
        "sample_id": sample_id,
        "scenario": scenario,
        "split": split,
    }
    row.update(extract_rgb_features(sampled_rgb, scenario=scenario))
    row.update(depth_features)
    row.update(extract_point_cloud_features(point_cloud))
    row.update(extract_task_features(task_command))
    row.update(_extract_rgb_summary(sampled_rgb, max_samples=None))
    row.update(depth_summary)
    row.update(
        _extract_point_cloud_summary(point_cloud, max_samples=summary_max_samples)
    )

    frame = pd.DataFrame([row])
    return validate_distill_feature_frame(frame)


def validate_distill_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [
        column
        for column in META_COLUMNS + DISTILL_FEATURE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing distill feature columns: {missing}")

    validated = frame.copy()
    numeric = validated[DISTILL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad_columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(
            f"Distill feature columns contain non-numeric values: {bad_columns}"
        )
    validated[DISTILL_FEATURE_COLUMNS] = numeric
    return validated[META_COLUMNS + DISTILL_FEATURE_COLUMNS]


def load_manifest_records(manifests: Sequence[str | Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for manifest_value in manifests:
        manifest = Path(manifest_value)
        base_dir = manifest.parent
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = str(record.get("sample_id") or "")
                if not sample_id:
                    raise ValueError(f"Missing sample_id in {manifest}:{line_number}")
                if sample_id in seen:
                    raise ValueError(
                        f"Duplicate sample_id across manifests: {sample_id}"
                    )
                seen.add(sample_id)
                record = _absolutize_path_fields(record, base_dir)
                record["source_manifest"] = str(manifest.resolve())
                records.append(record)
    return records


def build_distill_dataset_from_records(
    records: Sequence[dict[str, object]],
    *,
    summary_max_samples: int | None = DEFAULT_SUMMARY_MAX_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_frames: list[pd.DataFrame] = []
    truth_rows: list[dict[str, object]] = []
    for record in records:
        feature_frames.append(
            build_distill_feature_frame(
                rgb_path=_optional_path(record, "rgb_path"),
                depth_path=_optional_path(record, "depth_path"),
                point_cloud_path=_optional_path(record, "point_cloud_path"),
                sample_id=str(record["sample_id"]),
                scenario=str(record.get("scenario", "open_ground")),
                split=str(record.get("split", "inference")),
                task_command=str(record.get("task_command", "")),
                summary_max_samples=summary_max_samples,
            )
        )
        truth_rows.append(
            {
                "sample_id": str(record["sample_id"]),
                **{column: float(record[column]) for column in REQUIREMENT_COLUMNS},
            }
        )
    features = pd.concat(feature_frames, ignore_index=True)
    truth = pd.DataFrame(truth_rows)
    return features, truth


def load_student_bundle(model_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    directory = Path(model_dir)
    metadata_path = directory / STUDENT_METADATA_FILENAME
    model_path = directory / STUDENT_MODEL_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return joblib.load(model_path), metadata


def predict_student_frame(
    model: Any,
    features: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    feature_columns = list(metadata.get("feature_columns", DISTILL_FEATURE_COLUMNS))
    missing = [column for column in feature_columns if column not in features.columns]
    if missing:
        raise ValueError(f"Missing student feature columns: {missing}")
    x = features[feature_columns].apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any():
        bad_columns = x.columns[x.isna().any()].tolist()
        raise ValueError(f"Student features contain non-numeric values: {bad_columns}")
    raw_prediction = model.predict(x)
    prediction_frame = pd.DataFrame(raw_prediction, columns=PREDICTION_COLUMNS)
    return postprocess_requirement_predictions(prediction_frame)


def predict_student_from_payload(
    payload: dict[str, Any],
    *,
    model_dir: str | Path = DEFAULT_STUDENT_MODEL_DIR,
) -> dict[str, object]:
    model, metadata = load_student_bundle(model_dir)
    frame = build_distill_feature_frame_from_payload(
        payload,
        summary_max_samples=student_summary_max_samples(metadata),
    )
    return format_student_prediction(
        frame,
        predict_student_frame(model, frame, metadata),
        model_dir=model_dir,
        metadata=metadata,
    )


def build_distill_feature_frame_from_payload(
    payload: dict[str, Any],
    *,
    summary_max_samples: int | None = DEFAULT_SUMMARY_MAX_SAMPLES,
) -> pd.DataFrame:
    task_command = str(
        payload.get("task_command")
        or payload.get("instruction")
        or payload.get("text")
        or ""
    )
    point_cloud_data = (
        payload.get("pointcloud_data")
        or payload.get("point_cloud_data")
        or payload.get("point_cloud")
        or payload.get("points")
    )
    point_cloud_path = (
        payload.get("point_cloud_path")
        or payload.get("pointcloud_path")
        or payload.get("pc_path")
    )
    image_payload = payload.get("image")
    rgb_path = payload.get("rgb_path") or payload.get("image_path")
    depth_path = payload.get("depth_path")
    depth_array = (
        payload.get("depth") if "depth" in payload else payload.get("depth_data")
    )

    return build_distill_feature_frame(
        rgb_payload=image_payload,
        rgb_path=rgb_path,
        depth_path=depth_path,
        depth_array=depth_array,
        point_cloud_path=point_cloud_path,
        point_cloud_data=point_cloud_data,
        sample_id=str(payload.get("sample_id", "sensor_sample")),
        scenario=str(payload.get("scenario", "open_ground")),
        task_command=task_command,
        summary_max_samples=summary_max_samples,
    )


def student_summary_max_samples(metadata: dict[str, Any]) -> int | None:
    feature_config = metadata.get("feature_config")
    if isinstance(feature_config, dict) and "summary_max_samples" in feature_config:
        value = feature_config["summary_max_samples"]
        return None if value is None else int(value)
    # v1 bundles predate sampling and must retain full-resolution feature extraction.
    return None


def format_student_prediction(
    frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    *,
    model_dir: str | Path,
    metadata: dict[str, Any],
) -> dict[str, object]:
    prediction_row = prediction_frame.iloc[0]
    return {
        "sample_id": str(frame.loc[0, "sample_id"]),
        "scenario": str(frame.loc[0, "scenario"]),
        "model": str(metadata.get("model_name", "distilled_student")),
        "student_model_dir": str(model_dir),
        "feature_version": str(
            metadata.get("feature_version", DISTILL_FEATURE_VERSION)
        ),
        "v_r": format_vr_prediction(prediction_row),
        "prediction_columns": {
            column: float(prediction_row[column]) for column in prediction_frame.columns
        },
    }


def _extract_rgb_summary(
    rgb: np.ndarray,
    *,
    max_samples: int | None = DEFAULT_SUMMARY_MAX_SAMPLES,
) -> dict[str, float]:
    array = np.asarray(_regular_sample_array(rgb, max_samples), dtype=float)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("RGB image must be an HxWx3 array")
    rgb_array = np.nan_to_num(array[:, :, :3], nan=0.0, posinf=255.0, neginf=0.0)
    if rgb_array.max(initial=0.0) > 1.0:
        rgb_array = np.clip(rgb_array, 0.0, 255.0) / 255.0
    gray = rgb_array.mean(axis=2)
    return {
        "rgb_mean_r": _safe_float(rgb_array[:, :, 0].mean()),
        "rgb_mean_g": _safe_float(rgb_array[:, :, 1].mean()),
        "rgb_mean_b": _safe_float(rgb_array[:, :, 2].mean()),
        "rgb_std_r": _safe_float(rgb_array[:, :, 0].std()),
        "rgb_std_g": _safe_float(rgb_array[:, :, 1].std()),
        "rgb_std_b": _safe_float(rgb_array[:, :, 2].std()),
        "rgb_gray_p10": _safe_percentile(gray, 10),
        "rgb_gray_p50": _safe_percentile(gray, 50),
        "rgb_gray_p90": _safe_percentile(gray, 90),
        "rgb_dark_ratio": _safe_float((gray < 0.2).mean()),
        "rgb_bright_ratio": _safe_float((gray > 0.8).mean()),
    }


def _extract_depth_summary(depth: np.ndarray) -> dict[str, float]:
    array = np.asarray(depth, dtype=float)
    valid = np.isfinite(array) & (array > 0.0)
    finite = array[valid]
    if finite.size == 0:
        return _zero_features(DEPTH_SUMMARY_COLUMNS)
    fill_value = float(np.nanmedian(finite))
    cleaned = np.where(valid, array, fill_value)
    h, w = cleaned.shape
    h0, h1 = h // 4, max(h // 4 + 1, (3 * h) // 4)
    w0, w1 = w // 4, max(w // 4 + 1, (3 * w) // 4)
    gradients = np.gradient(cleaned)
    gradient_magnitude = np.zeros_like(cleaned, dtype=float)
    for gradient in gradients:
        gradient_magnitude += gradient**2
    gradient_magnitude = np.sqrt(gradient_magnitude)
    return {
        "depth_valid_ratio": _safe_float(valid.mean()),
        "depth_mean_m": _safe_float(finite.mean()),
        "depth_std_m": _safe_float(finite.std()),
        "depth_p01_m": _safe_percentile(finite, 1),
        "depth_p05_m": _safe_percentile(finite, 5),
        "depth_p10_m": _safe_percentile(finite, 10),
        "depth_p25_m": _safe_percentile(finite, 25),
        "depth_p50_m": _safe_percentile(finite, 50),
        "depth_p75_m": _safe_percentile(finite, 75),
        "depth_p90_m": _safe_percentile(finite, 90),
        "depth_p95_m": _safe_percentile(finite, 95),
        "depth_p99_m": _safe_percentile(finite, 99),
        "depth_center_mean_m": _safe_float(cleaned[h0:h1, w0:w1].mean()),
        "depth_top_mean_m": _safe_float(cleaned[: max(1, h // 3), :].mean()),
        "depth_bottom_mean_m": _safe_float(cleaned[(2 * h) // 3 :, :].mean()),
        "depth_left_mean_m": _safe_float(cleaned[:, : max(1, w // 3)].mean()),
        "depth_right_mean_m": _safe_float(cleaned[:, (2 * w) // 3 :].mean()),
        "depth_gradient_p95": _safe_percentile(
            gradient_magnitude[np.isfinite(gradient_magnitude)], 95
        ),
    }


def _extract_point_cloud_summary(
    points: np.ndarray,
    *,
    max_samples: int | None = DEFAULT_SUMMARY_MAX_SAMPLES,
) -> dict[str, float]:
    full_array = coerce_point_cloud(points)
    array = _regular_sample_array(full_array, max_samples, point_cloud=True)
    row: dict[str, float] = {"pc_point_count_k": _safe_float(len(full_array) / 1000.0)}
    axes = {"x": array[:, 0], "y": array[:, 1], "z": array[:, 2]}
    for axis, values in axes.items():
        finite = values[np.isfinite(values)]
        for percentile in POINT_PERCENTILES:
            row[f"pc_{axis}_p{percentile:02d}_m"] = _safe_percentile(finite, percentile)
        row[f"pc_{axis}_mean_m"] = _safe_float(finite.mean()) if finite.size else 0.0
        row[f"pc_{axis}_std_m"] = _safe_float(finite.std()) if finite.size else 0.0
    row["pc_x_range_05_95_m"] = row["pc_x_p95_m"] - row["pc_x_p05_m"]
    row["pc_y_range_05_95_m"] = row["pc_y_p95_m"] - row["pc_y_p05_m"]
    row["pc_z_range_05_95_m"] = row["pc_z_p95_m"] - row["pc_z_p05_m"]
    row["pc_xy_area_05_95_m2"] = row["pc_x_range_05_95_m"] * row["pc_y_range_05_95_m"]
    row["pc_xz_area_05_95_m2"] = row["pc_x_range_05_95_m"] * row["pc_z_range_05_95_m"]
    row["pc_yz_area_05_95_m2"] = row["pc_y_range_05_95_m"] * row["pc_z_range_05_95_m"]
    return row


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return _safe_float(np.percentile(finite, percentile))


def _regular_sample_array(
    array: np.ndarray,
    max_samples: int | None,
    *,
    point_cloud: bool = False,
) -> np.ndarray:
    sampled = np.asarray(array)
    if max_samples is None or max_samples <= 0:
        return sampled
    sample_count = len(sampled) if point_cloud else sampled.size
    if sampled.ndim == 3:
        sample_count = int(np.prod(sampled.shape[:-1]))
    if sample_count <= max_samples:
        return sampled
    stride = max(int(np.ceil(np.sqrt(sample_count / max_samples))), 1)
    if sampled.ndim == 3:
        return sampled[::stride, ::stride, :]
    if point_cloud:
        return sampled[:: max(int(np.ceil(sample_count / max_samples)), 1)]
    if sampled.ndim == 2:
        return sampled[::stride, ::stride]
    return sampled[:: max(int(np.ceil(sample_count / max_samples)), 1)]


def _safe_float(value: Any) -> float:
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return value


def _zero_features(columns: Sequence[str]) -> dict[str, float]:
    return {column: 0.0 for column in columns}


def _optional_path(record: dict[str, object], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    return str(value)


def _absolutize_path_fields(
    record: dict[str, object],
    base_dir: Path,
) -> dict[str, object]:
    converted = dict(record)
    for key, value in list(converted.items()):
        if not key.endswith("_path") or not isinstance(value, str):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = base_dir / path
        converted[key] = str(path.resolve())
    return converted
