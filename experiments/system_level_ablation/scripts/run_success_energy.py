#!/usr/bin/env python3
"""Run full-chain success and energy evaluations through an explicit backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mima_ablation.batch import run_batch
from mima_ablation.config import load_json, validate_run_assets
from mima_ablation.methods import METHOD_SPECS, selected_specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--sample-ids-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", required=True, help="Python callable: module:function")
    parser.add_argument(
        "--methods",
        default=",".join(METHOD_SPECS),
        help="Comma-separated canonical method keys.",
    )
    parser.add_argument("--seeds", default="1-10")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run_config = load_json(args.run_config)
    specs = selected_specs(_parse_methods(args.methods))
    summary = run_batch(
        mode="success_energy",
        backend_path=args.backend,
        dataset_dir=args.dataset_dir,
        sample_ids_file=args.sample_ids_file,
        output_dir=args.output_dir,
        method_specs=specs,
        seeds=_parse_seeds(args.seeds),
        assets=validate_run_assets(run_config, specs),
        protocol=run_config.get("protocol", {}),
        workers=args.workers,
        progress=args.progress,
    )
    print(json.dumps(summary, indent=2), flush=True)


def _parse_methods(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--methods must not be empty")
    return values


def _parse_seeds(text: str) -> tuple[int, ...]:
    value = text.strip()
    if "-" in value and "," not in value:
        start_text, end_text = value.split("-", 1)
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError("seed range end must be >= start")
        return tuple(range(start, end + 1))
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("--seeds must not be empty")
    return seeds


if __name__ == "__main__":
    main()
