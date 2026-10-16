from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from refactored_mllm.schema import (
    ENV_REQUIREMENT_COLUMNS,
    FEATURE_COLUMNS,
    PREDICTION_COLUMNS,
    REQUIREMENT_COLUMNS,
    TASK_REQUIREMENT_COLUMNS,
)


MODEL_FILENAME_BY_KEY = {
    "dt": "dt.joblib",
    "rf": "rf.joblib",
    "gbt": "gbt.joblib",
}
METADATA_FILENAME = "metadata.json"
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "mima_requirement_vector"
    / "models"
    / "baselines"
)

LEGACY_SKLEARN_LOSS_TYPES = (
    "CyHalfSquaredError",
    "CyHalfPoissonLoss",
    "CyHalfGammaLoss",
    "CyHalfTweedieLoss",
    "CyHalfTweedieLossIdentity",
    "CyHalfBinomialLoss",
)


class BehaviorTreeVRModel:
    def predict(self, features: pd.DataFrame) -> list[tuple[float, ...]]:
        return [self._predict_row(row) for _, row in features.iterrows()]

    def _predict_row(self, row: pd.Series) -> tuple[float, ...]:
        wp_m = float(row["pc_corridor_width_m"])
        hp_m = float(row["depth_min_clearance_m"])
        dp_m = float(row["pc_free_space_area_m2"]) / max(wp_m, 1e-6)
        hs_m = float(row["depth_step_height_m"])
        fl = float(row["cmd_load"])
        fi = float(row["cmd_inspect"])
        fp = float(row["cmd_pack"])
        return (wp_m, hp_m, dp_m, hs_m, fl, fi, fp)


def allowed_model_keys() -> list[str]:
    return sorted([*MODEL_FILENAME_BY_KEY, "bt"])


def load_model_bundle(
    model_dir: str | Path, model_key: str
) -> tuple[Any, dict[str, Any]]:
    normalized_key = model_key.lower()
    if normalized_key == "bt":
        return BehaviorTreeVRModel(), {
            "model_name": "behavior_tree",
            "feature_columns": list(FEATURE_COLUMNS),
            "requirement_columns": list(REQUIREMENT_COLUMNS),
        }

    if normalized_key not in MODEL_FILENAME_BY_KEY:
        allowed = ", ".join(allowed_model_keys())
        raise ValueError(f"model must be one of: {allowed}")

    directory = Path(model_dir)
    metadata_path = directory / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_files = metadata.get("model_files", MODEL_FILENAME_BY_KEY)
    model_filename = model_files.get(normalized_key)
    if not model_filename:
        raise KeyError(f"metadata is missing model file for {normalized_key}")

    model_path = directory / model_filename
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    ensure_sklearn_legacy_loss_unpickle_shims()
    return joblib.load(model_path), metadata


def predict_requirement_frame(model: Any, features: pd.DataFrame) -> pd.DataFrame:
    raw_prediction = model.predict(features[FEATURE_COLUMNS])
    prediction_frame = pd.DataFrame(raw_prediction, columns=PREDICTION_COLUMNS)
    return postprocess_requirement_predictions(prediction_frame)


def postprocess_requirement_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    processed = predictions.copy()
    for column in ENV_REQUIREMENT_COLUMNS:
        prediction_column = f"{column}_pred"
        processed[prediction_column] = pd.to_numeric(
            processed[prediction_column],
            errors="coerce",
        ).clip(lower=0.0)
    for column in TASK_REQUIREMENT_COLUMNS:
        prediction_column = f"{column}_pred"
        processed[prediction_column] = pd.to_numeric(
            processed[prediction_column],
            errors="coerce",
        ).clip(lower=0.0, upper=1.0)
    return processed


def format_vr_prediction(prediction_row: pd.Series) -> dict[str, float]:
    return {
        requirement_column: float(prediction_row[prediction_column])
        for requirement_column, prediction_column in zip(
            REQUIREMENT_COLUMNS,
            PREDICTION_COLUMNS,
            strict=True,
        )
    }


def ensure_sklearn_legacy_loss_unpickle_shims(loss_module: Any | None = None) -> None:
    if loss_module is None:
        try:
            import sklearn._loss._loss as loss_module
        except Exception:
            return

    for type_name in LEGACY_SKLEARN_LOSS_TYPES:
        helper_name = f"__pyx_unpickle_{type_name}"
        if hasattr(loss_module, helper_name) or not hasattr(loss_module, type_name):
            continue
        setattr(loss_module, helper_name, _legacy_cython_unpickle)


def _legacy_cython_unpickle(
    __pyx_type: Any, __pyx_checksum: Any, __pyx_state: Any
) -> Any:
    del __pyx_checksum
    instance = __pyx_type.__new__(__pyx_type)
    if __pyx_state:
        state = __pyx_state
        if isinstance(state, tuple) and len(state) >= 3:
            state = state[2]
        if isinstance(state, dict):
            for key, value in state.items():
                try:
                    setattr(instance, key, value)
                except AttributeError:
                    pass
    return instance
