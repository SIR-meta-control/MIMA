"""Load, train, and run the paper's conventional requirement-vector baselines."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

from .baseline_features import build_feature_frame
from .schema import (
    ENVIRONMENT_REQUIREMENT_KEYS,
    FEATURE_COLUMNS,
    PREDICTION_COLUMNS,
    REQUIREMENT_KEYS,
    TASK_REQUIREMENT_KEYS,
)


MODEL_FILENAMES = {"dt": "dt.joblib", "rf": "rf.joblib", "gbt": "gbt.joblib"}
LEGACY_SKLEARN_LOSS_TYPES = (
    "CyHalfSquaredError",
    "CyHalfPoissonLoss",
    "CyHalfGammaLoss",
    "CyHalfTweedieLoss",
    "CyHalfTweedieLossIdentity",
    "CyHalfBinomialLoss",
)


def build_unfitted_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "dt": DecisionTreeRegressor(
            max_depth=6, min_samples_leaf=1, random_state=random_state
        ),
        "rf": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=1,
            random_state=random_state,
        ),
        "gbt": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=random_state,
            )
        ),
    }


def train_and_export(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    random_state: int = 42,
    source_manifests: Iterable[str | Path] = (),
) -> dict[str, Path]:
    required = ["split", *FEATURE_COLUMNS, *REQUIREMENT_KEYS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"training table is missing columns: {missing}")
    train = frame.loc[frame["split"] == "train"].copy()
    if train.empty:
        raise ValueError("training table must contain at least one split=train row")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for key, model in build_unfitted_models(random_state).items():
        model.fit(train[list(FEATURE_COLUMNS)], train[list(REQUIREMENT_KEYS)])
        path = directory / MODEL_FILENAMES[key]
        joblib.dump(model, path)
        written[key] = path

    metadata = {
        "model_files": MODEL_FILENAMES,
        "model_names": {
            "dt": "w/o MLLM -> DT",
            "rf": "w/o MLLM -> RF",
            "gbt": "w/o MLLM -> GBT",
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "requirement_columns": list(REQUIREMENT_KEYS),
        "prediction_columns": list(PREDICTION_COLUMNS),
        "random_state": random_state,
        "train_samples": int(len(train)),
        "source_manifests": [Path(path).as_posix() for path in source_manifests],
    }
    metadata_path = directory / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    written["metadata"] = metadata_path
    return written


def predict_from_sensor_paths(
    *,
    model_dir: str | Path,
    model_key: str,
    rgb_path: str | Path,
    depth_path: str | Path,
    point_cloud_path: str | Path,
    sample_id: str = "sensor_sample",
    scenario: str = "open_ground",
) -> dict[str, Any]:
    model, metadata = load_model(model_dir, model_key)
    if metadata.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("model feature columns do not match the 16-feature schema")
    frame = build_feature_frame(
        rgb_path=rgb_path,
        depth_path=depth_path,
        point_cloud_path=point_cloud_path,
        sample_id=sample_id,
        scenario=scenario,
    )
    raw = np.asarray(model.predict(frame[list(FEATURE_COLUMNS)]), dtype=float)
    if raw.shape != (1, len(REQUIREMENT_KEYS)):
        raise ValueError(f"unexpected model output shape: {raw.shape}")
    values = _postprocess(dict(zip(REQUIREMENT_KEYS, raw[0])))
    return {
        "sample_id": sample_id,
        "scenario": scenario,
        "model": model_key.lower(),
        "v_r": values,
        "features": {
            column: float(frame.loc[0, column]) for column in FEATURE_COLUMNS
        },
    }


def load_model(model_dir: str | Path, model_key: str) -> tuple[Any, dict[str, Any]]:
    key = model_key.lower()
    if key not in MODEL_FILENAMES:
        raise ValueError(f"model must be one of: {', '.join(sorted(MODEL_FILENAMES))}")
    directory = Path(model_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    model_path = directory / metadata.get("model_files", MODEL_FILENAMES)[key]
    _install_legacy_sklearn_shims()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(model_path)
    return model, metadata


def _postprocess(values: dict[str, float]) -> dict[str, float]:
    result = {key: float(value) for key, value in values.items()}
    for key in ENVIRONMENT_REQUIREMENT_KEYS:
        result[key] = max(result[key], 0.0)
    for key in TASK_REQUIREMENT_KEYS:
        result[key] = min(max(result[key], 0.0), 1.0)
    return result


def _legacy_cython_unpickle(cython_type: Any, checksum: Any, state: Any) -> Any:
    del checksum
    instance = cython_type.__new__(cython_type)
    if isinstance(state, tuple) and len(state) >= 3:
        state = state[2]
    if isinstance(state, dict):
        for key, value in state.items():
            try:
                setattr(instance, key, value)
            except AttributeError:
                pass
    return instance


def _install_legacy_sklearn_shims() -> None:
    try:
        import sklearn._loss._loss as loss_module
    except Exception:
        return
    for type_name in LEGACY_SKLEARN_LOSS_TYPES:
        helper_name = f"__pyx_unpickle_{type_name}"
        if not hasattr(loss_module, helper_name) and hasattr(loss_module, type_name):
            setattr(loss_module, helper_name, _legacy_cython_unpickle)
