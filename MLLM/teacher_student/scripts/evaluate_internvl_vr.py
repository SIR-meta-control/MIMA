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

from refactored_mllm.internvl.inference import InternVLVRPredictor
from refactored_mllm.metrics import evaluate_vr_tolerance, per_column_error_summary
from refactored_mllm.schema import REQUIREMENT_COLUMNS


PREDICTION_COLUMNS = [f"{column}_pred" for column in REQUIREMENT_COLUMNS]
TRUTH_COLUMNS = ["sample_id", "scenario", *REQUIREMENT_COLUMNS]
FAILURE_COLUMNS = ["sample_id", "error_type", "error"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned InternVL v_r predictions."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-num", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method", default="InternVL-vr-finetuned")
    parser.add_argument(
        "--tolerance",
        type=float,
        action="append",
        default=None,
        help="Tolerance in meters. Can be passed multiple times.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_manifest(args.manifest, limit=args.limit)
    predictor = InternVLVRPredictor(
        checkpoint=args.checkpoint,
        device=args.device,
        max_num=args.max_num,
    )

    truth_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for index, record in enumerate(records, start=1):
        sample_id = str(record["sample_id"])
        truth_rows.append(_truth_row(record))
        try:
            paths = _resolve_paths(args.manifest.parent, record)
            prediction = predictor.predict(
                rgb_path=paths["rgb_path"],
                depth_path=paths["depth_path"],
                point_cloud_path=paths["point_cloud_path"],
            )
            vr = prediction["v_r"]
            row = {"sample_id": sample_id, "raw_response": prediction["raw_response"]}
            row.update({f"{column}_pred": vr[column] for column in REQUIREMENT_COLUMNS})
            prediction_rows.append(row)
        except Exception as exc:
            failure_rows.append(
                {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if not args.continue_on_error:
                raise
        if index % 25 == 0 or index == len(records):
            print(f"evaluated {index}/{len(records)} samples", flush=True)

    truth = pd.DataFrame(truth_rows, columns=TRUTH_COLUMNS)
    predictions = pd.DataFrame(
        prediction_rows,
        columns=["sample_id", "raw_response", *PREDICTION_COLUMNS],
    )
    failures = pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS)

    truth_path = args.output_dir / "truth.csv"
    predictions_path = args.output_dir / "predictions.csv"
    failures_path = args.output_dir / "failures.csv"
    metrics_path = args.output_dir / "metrics.csv"
    per_column_path = args.output_dir / "per_column_errors.csv"

    truth.to_csv(truth_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    failures.to_csv(failures_path, index=False)

    if len(predictions) != len(truth):
        raise RuntimeError(
            f"Only {len(predictions)}/{len(truth)} predictions succeeded; "
            f"see {failures_path}"
        )

    metrics, _ = evaluate_vr_tolerance(
        truth,
        predictions[["sample_id"] + PREDICTION_COLUMNS],
        method=args.method,
        tolerances_m=args.tolerance or (0.01, 0.02, 0.03, 0.04, 0.05),
    )
    per_column = per_column_error_summary(
        truth,
        predictions[["sample_id"] + PREDICTION_COLUMNS],
        tolerance_m=0.03,
    )
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    per_column.to_csv(per_column_path, index=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {metrics_path}", flush=True)
    print(f"wrote {per_column_path}", flush=True)


def _load_manifest(path: Path, *, limit: int | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _truth_row(record: dict[str, object]) -> dict[str, object]:
    row = {
        "sample_id": record["sample_id"],
        "scenario": record.get("scenario") or record.get("target_type"),
    }
    row.update({column: float(record[column]) for column in REQUIREMENT_COLUMNS})
    return row


def _resolve_paths(run_dir: Path, record: dict[str, object]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key in ("rgb_path", "depth_path", "point_cloud_path"):
        path = Path(str(record[key]))
        if not path.is_absolute():
            path = run_dir / path
        paths[key] = path
    return paths


if __name__ == "__main__":
    main()
