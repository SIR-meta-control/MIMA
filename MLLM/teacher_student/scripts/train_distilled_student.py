from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refactored_mllm.distillation import (  # noqa: E402
    DEFAULT_STUDENT_MODEL_DIR,
    DEFAULT_SUMMARY_MAX_SAMPLES,
    DISTILL_FEATURE_COLUMNS,
    DISTILL_FEATURE_VERSION,
    STUDENT_METADATA_FILENAME,
    STUDENT_MODEL_FILENAME,
    build_distill_dataset_from_records,
    load_manifest_records,
    predict_student_frame,
)
from refactored_mllm.metrics import (  # noqa: E402
    DEFAULT_TOLERANCES_M,
    evaluate_vr_tolerance,
    per_column_error_summary,
)
from refactored_mllm.schema import PREDICTION_COLUMNS, REQUIREMENT_COLUMNS  # noqa: E402


DEFAULT_MANIFEST_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "distill"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a compact sensor-to-v_r student model for Jetson-style "
            "onboard inference. Teacher prediction CSV is optional; when it is "
            "omitted this is a GT-supervised student baseline."
        )
    )
    parser.add_argument("--train-manifest", action="append", type=Path, default=None)
    parser.add_argument("--test-manifest", action="append", type=Path, default=None)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", default="distilled_student_split_v1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / DEFAULT_STUDENT_MODEL_DIR
    )
    parser.add_argument(
        "--model-type",
        choices=("extratrees", "rf", "hgbt", "mlp"),
        default="extratrees",
    )
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional explicit model name stored in metadata and service responses.",
    )
    parser.add_argument(
        "--summary-max-samples",
        type=int,
        default=DEFAULT_SUMMARY_MAX_SAMPLES,
        help="Maximum regularly sampled RGB/depth/point-cloud values used by summary features.",
    )
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs used by tree models. Use 1 for low-overhead service inference.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--teacher-predictions",
        type=Path,
        default=None,
        help="Optional CSV with sample_id and *_pred columns from InternVL teacher.",
    )
    parser.add_argument(
        "--teacher-weight",
        type=float,
        default=0.0,
        help="Blend target = (1-w)*GT + w*teacher. Use 0 when no teacher CSV.",
    )
    parser.add_argument(
        "--supervision-mode",
        choices=("auto", "gt-only", "teacher-only", "gt-teacher"),
        default="auto",
        help="Explicit ablation label and validation for the training target source.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        action="append",
        default=None,
        help="Tolerance in meters. Can be passed multiple times.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.summary_max_samples <= 0:
        raise ValueError("--summary-max-samples must be positive")
    supervision_mode = _resolve_supervision_mode(args)
    train_records, test_records, split_metadata = _load_train_test_records(args)
    tolerances = args.tolerance or list(DEFAULT_TOLERANCES_M)

    train_features, train_truth = build_distill_dataset_from_records(
        train_records,
        summary_max_samples=args.summary_max_samples,
    )
    test_features, test_truth = build_distill_dataset_from_records(
        test_records,
        summary_max_samples=args.summary_max_samples,
    )
    teacher_used = args.teacher_predictions is not None and args.teacher_weight > 0.0
    train_target = _build_training_target(
        train_truth,
        teacher_predictions=args.teacher_predictions,
        teacher_weight=args.teacher_weight,
    )

    model = _make_model(args)
    model.fit(
        train_features[DISTILL_FEATURE_COLUMNS], train_target[REQUIREMENT_COLUMNS]
    )

    train_predictions = _predict_with_sample_ids(model, train_features)
    test_predictions = _predict_with_sample_ids(model, test_features)
    train_metrics, _ = evaluate_vr_tolerance(
        train_truth,
        train_predictions,
        method=_method_name(args, teacher_used),
        tolerances_m=tolerances,
    )
    test_metrics, test_errors = evaluate_vr_tolerance(
        test_truth,
        test_predictions,
        method=_method_name(args, teacher_used),
        tolerances_m=tolerances,
    )
    test_summary = per_column_error_summary(
        test_truth,
        test_predictions,
        tolerance_m=0.01,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    train_features.to_csv(args.output_dir / "train_features.csv", index=False)
    test_features.to_csv(args.output_dir / "test_features.csv", index=False)
    train_truth.to_csv(args.output_dir / "train_truth.csv", index=False)
    test_truth.to_csv(args.output_dir / "test_truth.csv", index=False)
    train_predictions.to_csv(args.output_dir / "train_predictions.csv", index=False)
    test_predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    test_errors.to_csv(args.output_dir / "test_errors.csv", index=False)
    test_summary.to_csv(args.output_dir / "test_per_column_errors.csv", index=False)
    pd.DataFrame([train_metrics]).to_csv(
        args.output_dir / "train_metrics.csv", index=False
    )
    pd.DataFrame([test_metrics]).to_csv(
        args.output_dir / "test_metrics.csv", index=False
    )

    metadata = {
        "model_name": _method_name(args, teacher_used),
        "model_type": args.model_type,
        "model_params": {
            "n_estimators": int(args.n_estimators),
            "max_depth": args.max_depth,
            "min_samples_leaf": int(args.min_samples_leaf),
            "n_jobs": int(args.n_jobs),
            "random_state": int(args.random_state),
        },
        "feature_version": DISTILL_FEATURE_VERSION,
        "feature_config": {
            "summary_max_samples": int(args.summary_max_samples),
        },
        "feature_columns": list(DISTILL_FEATURE_COLUMNS),
        "requirement_columns": list(REQUIREMENT_COLUMNS),
        "prediction_columns": list(PREDICTION_COLUMNS),
        **split_metadata,
        "n_train": int(len(train_features)),
        "n_test": int(len(test_features)),
        "teacher_predictions": str(args.teacher_predictions)
        if args.teacher_predictions
        else None,
        "teacher_weight": float(args.teacher_weight),
        "teacher_used": bool(teacher_used),
        "supervision_mode": supervision_mode,
        "metric_primary": "vr_success_rate@0.01m",
        "tolerances_m": [float(value) for value in tolerances],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    joblib.dump(model, args.model_dir / STUDENT_MODEL_FILENAME)
    (args.model_dir / STUDENT_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


def _load_train_test_records(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if args.train_manifest and args.test_manifest:
        return (
            load_manifest_records(args.train_manifest),
            load_manifest_records(args.test_manifest),
            {
                "split_strategy": "explicit_manifest",
                "train_manifests": [str(path) for path in args.train_manifest],
                "test_manifests": [str(path) for path in args.test_manifest],
            },
        )
    if args.train_manifest or args.test_manifest:
        raise ValueError("Pass both --train-manifest and --test-manifest, or neither.")

    manifests = sorted(args.manifest_root.rglob("manifest.jsonl"))
    if len(manifests) == 1:
        records = load_manifest_records(manifests)
        train_records, test_records = _hash_split_records(
            records,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
        )
        return (
            train_records,
            test_records,
            {
                "split_strategy": "hash",
                "split_seed": args.split_seed,
                "test_ratio": float(args.test_ratio),
                "source_manifests": [str(path) for path in manifests],
            },
        )
    if not manifests:
        raise FileNotFoundError(f"No manifest.jsonl found under {args.manifest_root}")
    raise ValueError(
        "Multiple manifests found. Pass --train-manifest and --test-manifest explicitly."
    )


def _hash_split_records(
    records: Sequence[dict[str, object]],
    *,
    test_ratio: float,
    seed: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test-ratio must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("Need at least two records to split")
    test_count = min(max(round(len(records) * test_ratio), 1), len(records) - 1)
    ranked = sorted(
        records,
        key=lambda record: (
            hashlib.sha256(f"{seed}:{record['sample_id']}".encode("utf-8")).hexdigest(),
            str(record["sample_id"]),
        ),
    )
    test_ids = {str(record["sample_id"]) for record in ranked[:test_count]}
    train_records: list[dict[str, object]] = []
    test_records: list[dict[str, object]] = []
    for record in records:
        split_record = dict(record)
        if str(record["sample_id"]) in test_ids:
            split_record["split"] = "test"
            test_records.append(split_record)
        else:
            split_record["split"] = "train"
            train_records.append(split_record)
    return train_records, test_records


def _make_model(args: argparse.Namespace) -> object:
    if args.model_type == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=args.n_jobs,
        )
    if args.model_type == "rf":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=args.n_jobs,
        )
    if args.model_type == "hgbt":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                random_state=args.random_state,
                max_iter=500,
                learning_rate=0.04,
                l2_regularization=0.001,
            )
        )
    if args.model_type == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 128, 64),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=0.001,
                        alpha=0.0001,
                        max_iter=2000,
                        early_stopping=True,
                        random_state=args.random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model type: {args.model_type}")


def _build_training_target(
    truth: pd.DataFrame,
    *,
    teacher_predictions: Path | None,
    teacher_weight: float,
) -> pd.DataFrame:
    if teacher_predictions is None or teacher_weight <= 0.0:
        return truth[["sample_id", *REQUIREMENT_COLUMNS]].copy()
    if not 0.0 <= teacher_weight <= 1.0:
        raise ValueError("--teacher-weight must be between 0 and 1")

    teacher = pd.read_csv(teacher_predictions)
    teacher_columns = ["sample_id", *PREDICTION_COLUMNS]
    missing = [column for column in teacher_columns if column not in teacher.columns]
    if missing:
        raise ValueError(f"Teacher predictions are missing columns: {missing}")
    merged = truth[["sample_id", *REQUIREMENT_COLUMNS]].merge(
        teacher[teacher_columns],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if merged[PREDICTION_COLUMNS].isna().any().any():
        missing_ids = merged.loc[
            merged[PREDICTION_COLUMNS].isna().any(axis=1), "sample_id"
        ]
        raise ValueError(
            "Teacher predictions do not cover all training samples: "
            f"{missing_ids.head(10).tolist()}"
        )
    target = merged[["sample_id", *REQUIREMENT_COLUMNS]].copy()
    for requirement_column, prediction_column in zip(
        REQUIREMENT_COLUMNS,
        PREDICTION_COLUMNS,
        strict=True,
    ):
        target[requirement_column] = (1.0 - teacher_weight) * merged[
            requirement_column
        ].astype(float) + teacher_weight * merged[prediction_column].astype(float)
    return target


def _resolve_supervision_mode(args: argparse.Namespace) -> str:
    has_teacher = args.teacher_predictions is not None
    weight = float(args.teacher_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("--teacher-weight must be between 0 and 1")
    inferred = (
        "gt-only"
        if not has_teacher or weight == 0.0
        else "teacher-only"
        if weight == 1.0
        else "gt-teacher"
    )
    if args.supervision_mode == "auto":
        return inferred
    if args.supervision_mode != inferred:
        raise ValueError(
            f"--supervision-mode {args.supervision_mode!r} conflicts with "
            f"teacher_predictions={has_teacher} and teacher_weight={weight:g}"
        )
    return str(args.supervision_mode)


def _predict_with_sample_ids(model: object, features: pd.DataFrame) -> pd.DataFrame:
    predictions = predict_student_frame(
        model,
        features,
        {"feature_columns": list(DISTILL_FEATURE_COLUMNS)},
    )
    predictions.insert(0, "sample_id", features["sample_id"].astype(str).to_numpy())
    return predictions


def _method_name(args: argparse.Namespace, teacher_used: bool) -> str:
    if args.model_name:
        return str(args.model_name)
    prefix = "distilled" if teacher_used else "gt_supervised"
    suffix = f"_{args.n_estimators}" if args.model_type in {"extratrees", "rf"} else ""
    return f"{prefix}_student_{args.model_type}{suffix}"


if __name__ == "__main__":
    main()
