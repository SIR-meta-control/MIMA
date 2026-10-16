from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refactored_mllm.distillation import (  # noqa: E402
    build_distill_dataset_from_records,
    load_manifest_records,
    load_student_bundle,
    predict_student_frame,
    student_summary_max_samples,
)
from refactored_mllm.schema import REQUIREMENT_COLUMNS  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a distilled-student prediction cache for canonical full-chain evaluation."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model, metadata = load_student_bundle(args.model_dir)
    records = load_manifest_records([args.manifest])
    features, _ = build_distill_dataset_from_records(
        records,
        summary_max_samples=student_summary_max_samples(metadata),
    )
    predictions = predict_student_frame(model, features, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sample_id, prediction in zip(
            features["sample_id"].astype(str),
            predictions.to_dict("records"),
            strict=True,
        ):
            row = {
                "sample_id": sample_id,
                "success": True,
                "model": str(metadata.get("model_name", "distilled_student")),
            }
            row.update(
                {
                    requirement: float(prediction[f"{requirement}_pred"])
                    for requirement in REQUIREMENT_COLUMNS
                }
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(predictions)} rows to {args.output}")


if __name__ == "__main__":
    main()
