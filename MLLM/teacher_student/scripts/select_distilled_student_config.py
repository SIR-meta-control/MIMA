from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refactored_mllm.distillation import (  # noqa: E402
    DEFAULT_SUMMARY_MAX_SAMPLES,
    DISTILL_FEATURE_COLUMNS,
    build_distill_dataset_from_records,
    load_manifest_records,
    predict_student_frame,
)
from refactored_mllm.metrics import evaluate_vr_tolerance  # noqa: E402
from refactored_mllm.schema import PREDICTION_COLUMNS  # noqa: E402


DEFAULT_ESTIMATORS = (8, 16, 32, 64, 128, 256, 512, 800)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a compact ExtraTrees student using only the inner split. "
            "The fastest model within a configurable success-rate drop is selected."
        )
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--n-estimators",
        type=int,
        action="append",
        default=None,
        help="Candidate tree count; repeat for multiple candidates.",
    )
    parser.add_argument(
        "--summary-max-samples",
        type=int,
        action="append",
        default=None,
        help="Candidate sensor-summary sample cap; repeat for multiple candidates.",
    )
    parser.add_argument("--max-success-drop-pp", type=float, default=5.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary_candidates = tuple(
        sorted(set(args.summary_max_samples or [DEFAULT_SUMMARY_MAX_SAMPLES]))
    )
    if not summary_candidates or summary_candidates[0] <= 0:
        raise ValueError("--summary-max-samples values must be positive")
    if args.max_success_drop_pp < 0.0:
        raise ValueError("--max-success-drop-pp must be non-negative")
    candidates = tuple(sorted(set(args.n_estimators or DEFAULT_ESTIMATORS)))
    if not candidates or candidates[0] <= 0:
        raise ValueError("--n-estimators values must be positive")

    train_records = load_manifest_records([args.train_manifest])
    validation_records = load_manifest_records([args.validation_manifest])
    rows: list[dict[str, float | int]] = []
    train_count = 0
    validation_count = 0
    for summary_max_samples in summary_candidates:
        train_features, train_truth = build_distill_dataset_from_records(
            train_records,
            summary_max_samples=summary_max_samples,
        )
        feature_start = time.perf_counter()
        validation_features, validation_truth = build_distill_dataset_from_records(
            validation_records,
            summary_max_samples=summary_max_samples,
        )
        feature_ms_per_sample = (
            (time.perf_counter() - feature_start) * 1000.0 / len(validation_features)
        )
        train_count = len(train_features)
        validation_count = len(validation_features)
        x_train = train_features[DISTILL_FEATURE_COLUMNS]
        y_train = train_truth.drop(columns="sample_id")
        for n_estimators in candidates:
            model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=1,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            )
            model.fit(x_train, y_train)
            metadata = {"feature_columns": list(DISTILL_FEATURE_COLUMNS)}
            predict_student_frame(model, validation_features, metadata)
            prediction_start = time.perf_counter()
            repeat_count = 10
            predictions = None
            for _ in range(repeat_count):
                predictions = predict_student_frame(
                    model, validation_features, metadata
                )
            prediction_ms_per_sample = (
                (time.perf_counter() - prediction_start)
                * 1000.0
                / repeat_count
                / len(validation_features)
            )
            assert predictions is not None
            predictions.insert(
                0,
                "sample_id",
                validation_features["sample_id"].to_numpy(),
            )
            metrics, _ = evaluate_vr_tolerance(
                validation_truth,
                predictions[["sample_id", *PREDICTION_COLUMNS]],
                method=(
                    f"inner_gt_only_extratrees_{n_estimators}_s{summary_max_samples}"
                ),
                tolerances_m=(0.01,),
            )
            geometry = geometry_metrics(validation_truth, predictions)
            rows.append(
                {
                    "summary_max_samples": summary_max_samples,
                    "n_estimators": n_estimators,
                    "feature_ms_per_sample": round(feature_ms_per_sample, 6),
                    "prediction_ms_per_sample": round(prediction_ms_per_sample, 6),
                    "estimated_inference_ms_per_sample": round(
                        feature_ms_per_sample + prediction_ms_per_sample,
                        6,
                    ),
                    "all_vr_success_at_1cm": float(metrics["vr_success_rate@0.01m"]),
                    "mae_vr": float(metrics["mae_vr"]),
                    **geometry,
                }
            )

    ranking = pd.DataFrame(rows)
    selected = select_candidate(ranking, max_success_drop_pp=args.max_success_drop_pp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking.sort_values("n_estimators").to_csv(
        args.output_dir / "candidates.csv", index=False
    )
    payload = {
        "selection_data": "strict_teacher_inner_val_v1 only",
        "selection_rule": (
            "lowest measured inner inference time within max_success_drop_pp "
            "of the best geometry_success_at_1cm"
        ),
        "max_success_drop_pp": float(args.max_success_drop_pp),
        "summary_max_samples_candidates": list(summary_candidates),
        "n_train": int(train_count),
        "n_validation": int(validation_count),
        "selected": selected,
        "candidates_csv": str(args.output_dir / "candidates.csv"),
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


def geometry_metrics(
    truth: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, float]:
    merged = truth[["sample_id", "wp_m", "hp_m"]].merge(
        predictions[["sample_id", "wp_m_pred", "hp_m_pred"]],
        on="sample_id",
        validate="one_to_one",
    )
    width_error = pd.to_numeric(merged["wp_m_pred"]) - pd.to_numeric(merged["wp_m"])
    height_error = pd.to_numeric(merged["hp_m_pred"]) - pd.to_numeric(merged["hp_m"])
    tolerance_success = width_error.abs().le(0.01 + 1e-12) & height_error.abs().le(
        0.01 + 1e-12
    )
    exact_success = width_error.abs().le(1e-9) & height_error.abs().le(1e-9)
    return {
        "geometry_success_at_1cm": round(float(tolerance_success.mean()) * 100.0, 6),
        "geometry_exact_success": round(float(exact_success.mean()) * 100.0, 6),
        "geometry_overestimate_rate": round(
            float((width_error.gt(1e-9) | height_error.gt(1e-9)).mean()) * 100.0,
            6,
        ),
        "width_overestimate_rate": round(float(width_error.gt(1e-9).mean()) * 100.0, 6),
        "height_overestimate_rate": round(
            float(height_error.gt(1e-9).mean()) * 100.0, 6
        ),
    }


def select_candidate(
    candidates: pd.DataFrame,
    *,
    max_success_drop_pp: float,
) -> dict[str, float | int]:
    required = {
        "n_estimators",
        "summary_max_samples",
        "estimated_inference_ms_per_sample",
        "geometry_success_at_1cm",
        "geometry_exact_success",
        "geometry_overestimate_rate",
        "width_overestimate_rate",
        "height_overestimate_rate",
        "mae_vr",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate table is missing columns: {missing}")
    best_success = float(candidates["geometry_success_at_1cm"].max())
    threshold = best_success - float(max_success_drop_pp)
    eligible = candidates[candidates["geometry_success_at_1cm"] >= threshold - 1e-12]
    selected = eligible.sort_values(
        [
            "estimated_inference_ms_per_sample",
            "summary_max_samples",
            "n_estimators",
            "geometry_success_at_1cm",
            "geometry_exact_success",
            "geometry_overestimate_rate",
            "width_overestimate_rate",
            "height_overestimate_rate",
            "mae_vr",
        ],
        ascending=[True, True, True, False, False, True, True, True, True],
        kind="stable",
    ).iloc[0]
    result = selected.to_dict()
    result["best_geometry_success_at_1cm"] = best_success
    result["eligible_success_threshold"] = threshold
    return result


if __name__ == "__main__":
    main()
