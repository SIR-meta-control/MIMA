from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refactored_mllm.schema import REQUIREMENT_COLUMNS


SUCCESS_PREFIX = "vr_success_rate@"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select an InternVL checkpoint using only inner-validation predictions."
    )
    parser.add_argument("--evaluations-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument(
        "--primary-metric",
        choices=("geometry-exact", "geometry-tolerance"),
        default="geometry-exact",
        help="Primary inner-validation metric used to rank checkpoints.",
    )
    parser.add_argument(
        "--max-tolerance-drop-pp",
        type=float,
        default=3.0,
        help=(
            "When exact geometry is primary, only checkpoints within this many "
            "percentage points of the best 1 cm geometry success are eligible."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = []
    for metrics_path in sorted(args.evaluations_root.glob("*/metrics.csv")):
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        checkpoint = metrics_path.parent.name
        success_column = _success_column(metrics, args.tolerance)
        geometry_metrics = _geometry_metrics(metrics_path.parent, args.tolerance)
        rows.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_step": _checkpoint_step(checkpoint),
                "success_at_tolerance": float(metrics[success_column]),
                **geometry_metrics,
                "mae_vr": float(metrics["mae_vr"]),
                "metrics_path": str(metrics_path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No */metrics.csv found under {args.evaluations_root}")
    if args.max_tolerance_drop_pp < 0.0:
        raise ValueError("--max-tolerance-drop-pp must be non-negative")
    candidates = pd.DataFrame(rows)
    best_tolerance_success = float(candidates["geometry_success_at_tolerance"].max())
    tolerance_threshold = best_tolerance_success - float(args.max_tolerance_drop_pp)
    candidates["eligible_for_selection"] = (
        candidates["geometry_success_at_tolerance"] >= tolerance_threshold - 1e-12
    )
    primary = (
        "geometry_exact_success"
        if args.primary_metric == "geometry-exact"
        else "geometry_success_at_tolerance"
    )
    secondary = (
        "geometry_success_at_tolerance"
        if args.primary_metric == "geometry-exact"
        else "geometry_exact_success"
    )
    ranking = candidates.sort_values(
        [
            "eligible_for_selection",
            primary,
            secondary,
            "geometry_overestimate_rate",
            "width_overestimate_rate",
            "height_overestimate_rate",
            "success_at_tolerance",
            "mae_vr",
            "checkpoint_step",
        ],
        ascending=[False, False, False, True, True, True, False, True, True],
        kind="stable",
    )
    best = ranking.loc[ranking["eligible_for_selection"]].iloc[0].to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output.with_suffix(".csv"), index=False)
    args.output.write_text(
        json.dumps(
            {
                "selection_data": "strict_teacher_inner_val_v1 only",
                "selection_metric": primary,
                "eligibility_rule": (
                    f"geometry_success_at_tolerance within {args.max_tolerance_drop_pp:g} pp "
                    "of the best inner-validation checkpoint"
                ),
                "best_geometry_success_at_tolerance": best_tolerance_success,
                "eligible_geometry_success_threshold": tolerance_threshold,
                "tie_breakers": [
                    secondary,
                    "geometry_overestimate_rate",
                    "width_overestimate_rate",
                    "height_overestimate_rate",
                    f"all_vr_success_rate@{args.tolerance:g}m",
                    "mae_vr",
                    "earlier_checkpoint_step",
                ],
                "best": best,
                "ranking_csv": str(args.output.with_suffix(".csv")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(best, indent=2))


def _checkpoint_step(name: str) -> int:
    try:
        return int(name.rsplit("checkpoint-", 1)[1])
    except (IndexError, ValueError):
        return 2**31 - 1


def _success_column(metrics: dict[str, object], tolerance: float) -> str:
    target = float(tolerance)
    for column in metrics:
        if not str(column).startswith(SUCCESS_PREFIX):
            continue
        suffix = str(column)[len(SUCCESS_PREFIX) :]
        if suffix.endswith("m"):
            suffix = suffix[:-1]
        try:
            if abs(float(suffix) - target) <= 1e-12:
                return str(column)
        except ValueError:
            continue
    raise KeyError(f"No success-rate column found for tolerance {tolerance:g} m")


def _geometry_metrics(evaluation_dir: Path, tolerance: float) -> dict[str, float]:
    truth = pd.read_csv(evaluation_dir / "truth.csv")
    predictions = pd.read_csv(evaluation_dir / "predictions.csv")
    geometry_columns = [
        column for column in ("wp_m", "hp_m") if column in REQUIREMENT_COLUMNS
    ]
    merged = truth[["sample_id", *geometry_columns]].merge(
        predictions[["sample_id", *[f"{column}_pred" for column in geometry_columns]]],
        on="sample_id",
        validate="one_to_one",
    )
    success = pd.Series(True, index=merged.index)
    exact = pd.Series(True, index=merged.index)
    for column in geometry_columns:
        error = pd.to_numeric(merged[f"{column}_pred"]) - pd.to_numeric(merged[column])
        success &= error.abs().le(float(tolerance) + 1e-12)
        exact &= error.abs().le(1e-9)
    width_error = pd.to_numeric(merged["wp_m_pred"]) - pd.to_numeric(merged["wp_m"])
    height_error = pd.to_numeric(merged["hp_m_pred"]) - pd.to_numeric(merged["hp_m"])
    return {
        "geometry_success_at_tolerance": round(float(success.mean()) * 100.0, 6),
        "geometry_exact_success": round(float(exact.mean()) * 100.0, 6),
        "geometry_overestimate_rate": round(
            float((width_error.gt(1e-9) | height_error.gt(1e-9)).mean()) * 100.0,
            6,
        ),
        "width_overestimate_rate": round(
            float(width_error.gt(1e-9).mean()) * 100.0,
            6,
        ),
        "height_overestimate_rate": round(
            float(height_error.gt(1e-9).mean()) * 100.0,
            6,
        ),
    }


if __name__ == "__main__":
    main()
