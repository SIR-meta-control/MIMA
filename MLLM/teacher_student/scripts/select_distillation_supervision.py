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

from scripts.select_distilled_student_config import geometry_metrics


DEFAULT_MODES = ("gt_only", "teacher_only", "gt_teacher")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the student supervision mode using only inner-validation outputs."
    )
    parser.add_argument("--ablations-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = []
    for mode in args.mode or DEFAULT_MODES:
        mode_dir = args.ablations_root / mode
        truth = pd.read_csv(mode_dir / "test_truth.csv")
        predictions = pd.read_csv(mode_dir / "test_predictions.csv")
        metrics = pd.read_csv(mode_dir / "test_metrics.csv").iloc[0]
        rows.append(
            {
                "supervision_mode": mode,
                "all_vr_success_at_1cm": float(metrics["vr_success_rate@0.01m"]),
                "mae_vr": float(metrics["mae_vr"]),
                **geometry_metrics(truth, predictions),
            }
        )
    ranking = rank_supervision_modes(pd.DataFrame(rows))
    best = ranking.iloc[0].to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output.with_suffix(".csv"), index=False)
    payload = {
        "selection_data": "strict_teacher_inner_val_v1 only",
        "selection_metric": "geometry_success_at_1cm",
        "tie_breakers": [
            "all_vr_success_at_1cm",
            "geometry_overestimate_rate",
            "width_overestimate_rate",
            "height_overestimate_rate",
            "mae_vr",
            "fixed_mode_order",
        ],
        "best": best,
        "ranking_csv": str(args.output.with_suffix(".csv")),
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def rank_supervision_modes(rows: pd.DataFrame) -> pd.DataFrame:
    order = {mode: index for index, mode in enumerate(DEFAULT_MODES)}
    ranked = rows.copy()
    ranked["fixed_mode_order"] = (
        ranked["supervision_mode"].map(order).fillna(len(order))
    )
    return ranked.sort_values(
        [
            "geometry_success_at_1cm",
            "all_vr_success_at_1cm",
            "geometry_overestimate_rate",
            "width_overestimate_rate",
            "height_overestimate_rate",
            "mae_vr",
            "fixed_mode_order",
        ],
        ascending=[False, False, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


if __name__ == "__main__":
    main()
