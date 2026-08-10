#!/usr/bin/env python3
"""Cache requirement vectors from a service for a dataset manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mima_vr.schema import REQUIREMENT_KEYS
from mima_vr.service_client import RequirementVectorServiceClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--backend", choices=("internvl", "student"), required=True)
    parser.add_argument("--expected-model")
    parser.add_argument("--sample-ids-file", type=Path)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_manifest(dataset_dir / "manifest.jsonl")
    if args.sample_ids_file:
        selected = {
            line.strip()
            for line in args.sample_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        rows = [row for row in rows if str(row["sample_id"]) in selected]
        missing = selected - {str(row["sample_id"]) for row in rows}
        if missing:
            raise ValueError(f"sample IDs absent from manifest: {sorted(missing)}")

    jsonl_path = output_dir / "requirement_vector_predictions.jsonl"
    cached = _load_successful_rows(jsonl_path) if args.resume else {}
    client = RequirementVectorServiceClient(
        base_url=args.service_url,
        backend=args.backend,
        timeout_s=args.timeout_s,
    )
    results: list[dict[str, Any]] = []
    for index, record in enumerate(rows, start=1):
        sample_id = str(record["sample_id"])
        if sample_id in cached:
            results.append(cached[sample_id])
            continue
        try:
            prediction = client.predict_from_sensor_paths(
                rgb_path=_artifact_path(dataset_dir, record, "rgb_path", "rgb.png"),
                depth_path=_artifact_path(dataset_dir, record, "depth_path", "depth.npy"),
                point_cloud_path=_artifact_path(
                    dataset_dir, record, "point_cloud_path", "point_cloud.npy"
                ),
                sample_id=sample_id,
                scenario=str(record.get("scenario", "open_ground")),
                task_command=str(record.get("task_command", "")),
                model_key=(
                    "full_mima_teacher" if args.backend == "internvl" else "mllm_distilled"
                ),
            )
            if args.expected_model and prediction["model"] != args.expected_model:
                raise RuntimeError(
                    f"returned model {prediction['model']!r}, expected {args.expected_model!r}"
                )
            result = {"success": True, **prediction}
        except Exception as exc:
            result = {
                "sample_id": sample_id,
                "scenario": str(record.get("scenario", "open_ground")),
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append(result)
        print(f"[{index}/{len(rows)}] {sample_id} success={result['success']}", flush=True)

    _write_outputs(output_dir, results, args)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _artifact_path(
    root: Path, record: dict[str, Any], key: str, fallback: str
) -> Path:
    value = record.get(key)
    path = Path(str(value)) if value else Path("samples") / str(record["sample_id"]) / fallback
    return path if path.is_absolute() else root / path


def _load_successful_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {str(row["sample_id"]): row for row in rows if row.get("success") is True}


def _write_outputs(
    output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    jsonl_path = output_dir / "requirement_vector_predictions.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "scenario": row.get("scenario"),
                "success": row.get("success"),
                **{key: (row.get("v_r") or {}).get(key) for key in REQUIREMENT_KEYS},
                "model": row.get("model"),
                "error_type": row.get("error_type", ""),
                "error": row.get("error", ""),
            }
        )
    with (output_dir / "requirement_vector_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]) if flat_rows else [])
        if flat_rows:
            writer.writeheader()
            writer.writerows(flat_rows)
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "service_url": args.service_url,
        "backend": args.backend,
        "expected_model": args.expected_model,
        "total_samples": len(rows),
        "success_count": sum(row.get("success") is True for row in rows),
        "failed_count": sum(row.get("success") is not True for row in rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
