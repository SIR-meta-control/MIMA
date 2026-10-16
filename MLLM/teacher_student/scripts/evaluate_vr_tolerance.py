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

from refactored_mllm.metrics import (  # noqa: E402
    DEFAULT_TOLERANCES_M,
    evaluate_vr_tolerance,
    per_column_error_summary,
)
from refactored_mllm.schema import REQUIREMENT_COLUMNS  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v_r predictions with bidirectional absolute-error tolerances."
    )
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--method", default="v_r-predictions")
    parser.add_argument(
        "--tolerance",
        type=float,
        action="append",
        default=None,
        help="Tolerance in meters. Can be passed multiple times.",
    )
    parser.add_argument(
        "--columns",
        default=",".join(REQUIREMENT_COLUMNS),
        help="Comma-separated v_r columns to require for success.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    requirement_columns = [
        column.strip() for column in args.columns.split(",") if column.strip()
    ]
    tolerances = args.tolerance or list(DEFAULT_TOLERANCES_M)
    truth = pd.read_csv(args.truth)
    predictions = pd.read_csv(args.predictions)

    metrics, errors = evaluate_vr_tolerance(
        truth,
        predictions[
            ["sample_id", *[f"{column}_pred" for column in requirement_columns]]
        ],
        method=args.method,
        requirement_columns=requirement_columns,
        tolerances_m=tolerances,
    )
    summary = per_column_error_summary(
        truth,
        predictions,
        requirement_columns=requirement_columns,
        tolerance_m=0.03,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(summary.to_string(index=False), flush=True)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(args.output_dir / "metrics.csv", index=False)
        errors.to_csv(args.output_dir / "errors.csv", index=False)
        summary.to_csv(args.output_dir / "per_column_errors.csv", index=False)
        print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
