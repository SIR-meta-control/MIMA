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

from refactored_mllm.distillation import (  # noqa: E402
    DEFAULT_STUDENT_MODEL_DIR,
    build_distill_dataset_from_records,
    load_manifest_records,
    load_student_bundle,
    predict_student_frame,
    student_summary_max_samples,
)
from refactored_mllm.metrics import (  # noqa: E402
    DEFAULT_TOLERANCES_M,
    evaluate_vr_tolerance,
    per_column_error_summary,
)
from refactored_mllm.schema import PREDICTION_COLUMNS  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifest.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "student_eval"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a distilled student model on MuJoCo sensor manifests."
    )
    parser.add_argument("--manifest", action="append", type=Path, default=None)
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / DEFAULT_STUDENT_MODEL_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method", default=None)
    parser.add_argument("--tolerance", type=float, action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifests = args.manifest or [DEFAULT_MANIFEST]
    tolerances = args.tolerance or list(DEFAULT_TOLERANCES_M)
    model, metadata = load_student_bundle(args.model_dir)
    features, truth = build_distill_dataset_from_records(
        load_manifest_records(manifests),
        summary_max_samples=student_summary_max_samples(metadata),
    )
    predictions = predict_student_frame(model, features, metadata)
    predictions.insert(0, "sample_id", features["sample_id"].astype(str).to_numpy())
    method = args.method or str(metadata.get("model_name", "distilled_student"))
    metrics, errors = evaluate_vr_tolerance(
        truth,
        predictions[["sample_id", *PREDICTION_COLUMNS]],
        method=method,
        tolerances_m=tolerances,
    )
    summary = per_column_error_summary(truth, predictions, tolerance_m=0.01)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_dir / "features.csv", index=False)
    truth.to_csv(args.output_dir / "truth.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    errors.to_csv(args.output_dir / "errors.csv", index=False)
    summary.to_csv(args.output_dir / "per_column_errors.csv", index=False)
    pd.DataFrame([metrics]).to_csv(args.output_dir / "metrics.csv", index=False)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "manifests": [str(path) for path in manifests],
                "model_dir": str(args.model_dir),
                "model_metadata": metadata,
                "metrics": metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
