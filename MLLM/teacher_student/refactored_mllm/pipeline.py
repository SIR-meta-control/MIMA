from __future__ import annotations

from pathlib import Path
from typing import Any

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
    DEFAULT_MODEL_DIR,
    format_vr_prediction,
    load_model_bundle,
    predict_requirement_frame,
)
from refactored_mllm.schema import FEATURE_COLUMNS, META_COLUMNS, validate_feature_frame


def build_sensor_feature_frame(
    *,
    rgb_path: str | Path | None = None,
    rgb_payload: str | bytes | None = None,
    depth_path: str | Path | None = None,
    depth_array: Any | None = None,
    point_cloud_path: str | Path | None = None,
    point_cloud_data: Any | None = None,
    sample_id: str = "sensor_sample",
    scenario: str = "open_ground",
    task_command: str = "",
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

    if depth_array is not None:
        depth_features = extract_depth_features(depth_array)
    elif depth_path is not None:
        depth_features = extract_depth_features(load_depth_array(depth_path))
    else:
        depth_features = extract_depth_features_from_point_cloud(point_cloud)

    row: dict[str, object] = {
        "sample_id": sample_id,
        "scenario": scenario,
        "split": "inference",
    }
    row.update(extract_rgb_features(rgb, scenario=scenario))
    row.update(depth_features)
    row.update(extract_point_cloud_features(point_cloud))
    row.update(extract_task_features(task_command))

    frame = pd.DataFrame([row])[META_COLUMNS + FEATURE_COLUMNS]
    return validate_feature_frame(frame)


def predict_vr_from_sensor_paths(
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    model_key: str = "dt",
    rgb_path: str | Path,
    depth_path: str | Path | None = None,
    point_cloud_path: str | Path,
    sample_id: str = "sensor_sample",
    scenario: str = "open_ground",
    task_command: str = "",
) -> dict[str, object]:
    frame = build_sensor_feature_frame(
        rgb_path=rgb_path,
        depth_path=depth_path,
        point_cloud_path=point_cloud_path,
        sample_id=sample_id,
        scenario=scenario,
        task_command=task_command,
    )
    return predict_vr_from_feature_frame(
        frame,
        model_dir=model_dir,
        model_key=model_key,
    )


def predict_vr_from_payload(
    payload: dict[str, Any],
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    default_model_key: str = "dt",
) -> dict[str, object]:
    model_key = str(
        payload.get("model") or payload.get("model_key") or default_model_key
    )
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

    frame = build_sensor_feature_frame(
        rgb_payload=image_payload,
        rgb_path=rgb_path,
        depth_path=depth_path,
        point_cloud_path=point_cloud_path,
        point_cloud_data=point_cloud_data,
        sample_id=str(payload.get("sample_id", "sensor_sample")),
        scenario=str(payload.get("scenario", "open_ground")),
        task_command=task_command,
    )
    return predict_vr_from_feature_frame(
        frame,
        model_dir=model_dir,
        model_key=model_key,
    )


def predict_vr_from_feature_frame(
    frame: pd.DataFrame,
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    model_key: str = "dt",
) -> dict[str, object]:
    validated = validate_feature_frame(frame)
    model, metadata = load_model_bundle(model_dir, model_key)
    feature_columns = metadata.get("feature_columns", FEATURE_COLUMNS)
    if list(feature_columns) != list(FEATURE_COLUMNS):
        raise ValueError("Exported model feature columns do not match current schema")

    prediction_frame = predict_requirement_frame(model, validated)
    prediction_row = prediction_frame.iloc[0]
    sample_id = str(validated.loc[0, "sample_id"])
    scenario = str(validated.loc[0, "scenario"])
    return {
        "sample_id": sample_id,
        "scenario": scenario,
        "model": model_key.lower(),
        "v_r": format_vr_prediction(prediction_row),
        "prediction_columns": {
            column: float(prediction_row[column]) for column in prediction_frame.columns
        },
        "features": {
            column: float(validated.loc[0, column]) for column in FEATURE_COLUMNS
        },
    }
