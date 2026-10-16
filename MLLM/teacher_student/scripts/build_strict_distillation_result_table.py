from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd


METHODS = (
    ("teacher", "Full-MIMA"),
    ("gt_only", "MIMA-Distilled-GT-only"),
    ("teacher_only", "MIMA-Distilled-Teacher-only"),
    ("gt_teacher", "MIMA-Distilled-GT+Teacher"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final strict teacher/distillation table."
    )
    parser.add_argument("--share-dir", type=Path, required=True)
    parser.add_argument("--active-metadata", type=Path, required=True)
    parser.add_argument("--service-benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    teacher_selection = _load_json(args.share_dir / "teacher_checkpoint_selection.json")
    supervision_selection = _load_json(
        args.share_dir / "student_supervision_selection.json"
    )
    config_selection = _load_json(args.share_dir / "student_config_selection.json")
    active_metadata = _load_json(args.active_metadata)
    benchmark = _load_json(args.service_benchmark)

    rows = []
    for slug, label in METHODS:
        outer = (
            pd.read_csv(args.share_dir / f"{slug}_outer_metrics.csv").iloc[0].to_dict()
        )
        strict = _load_json(args.share_dir / f"{slug}_strict_summary.json")["aggregate"]
        rows.append(
            {
                "method": label,
                "outer_n": int(outer["n_test"]),
                "outer_success_01m": float(outer["vr_success_rate@0.01m"]),
                "outer_success_02m": float(outer["vr_success_rate@0.02m"]),
                "outer_mae_vr": float(outer["mae_vr"]),
                "odd_execution_rows": int(strict["execution_rows"]),
                "odd_strict_success_01m": float(strict["strict_success_rate_01cm"]),
                "odd_strict_success_02m": float(strict["strict_success_rate_02cm"]),
                "odd_chain_success": float(strict["chain_success_rate"]),
                "odd_physical_success": float(strict["physical_success_rate"]),
                "odd_failed_count": int(strict["failed_count"]),
            }
        )

    payload = {
        "teacher_selection": teacher_selection,
        "student_config_selection": config_selection,
        "student_supervision_selection": supervision_selection,
        "active_student_metadata": active_metadata,
        "service_benchmark": benchmark,
        "methods": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        args.output_dir / "strict_distillation_results.csv", index=False
    )
    (args.output_dir / "strict_distillation_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
