#!/usr/bin/env python3
"""Run command-ready timing evaluations through an explicit backend."""

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
    parser.add_argument("--methods", default=",".join(METHOD_SPECS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run_config = load_json(args.run_config)
    method_keys = [item.strip() for item in args.methods.split(",") if item.strip()]
    specs = selected_specs(method_keys)
    summary = run_batch(
        mode="timing",
        backend_path=args.backend,
        dataset_dir=args.dataset_dir,
        sample_ids_file=args.sample_ids_file,
        output_dir=args.output_dir,
        method_specs=specs,
        seeds=(args.seed,),
        assets=validate_run_assets(run_config, specs),
        protocol=run_config.get("protocol", {}),
        workers=1,
        progress=args.progress,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
