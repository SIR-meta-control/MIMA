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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert InternVL evaluation CSV output to AblationMLLM cache JSONL."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    predictions = pd.read_csv(args.predictions)
    required = ["sample_id", *[f"{column}_pred" for column in REQUIREMENT_COLUMNS]]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")
    if predictions["sample_id"].duplicated().any():
        raise ValueError("Prediction CSV contains duplicate sample_id values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in predictions.to_dict("records"):
            cache_row = {
                "sample_id": str(row["sample_id"]),
                "success": True,
                "model": args.model,
            }
            cache_row.update(
                {column: float(row[f"{column}_pred"]) for column in REQUIREMENT_COLUMNS}
            )
            handle.write(json.dumps(cache_row, ensure_ascii=False) + "\n")
    print(f"wrote {len(predictions)} rows to {args.output}")


if __name__ == "__main__":
    main()
