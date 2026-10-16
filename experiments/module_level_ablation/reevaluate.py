#!/usr/bin/env python3
"""Reevaluate the selected module-level ablation checkpoints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all four selected checkpoints on one explicit split."
    )
    parser.add_argument(
        "--val-pairs",
        type=Path,
        default=ROOT / "datasets/processed_dataset/split_seed7/val_pairs.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    command = [
        sys.executable,
        str(ROOT / "scripts/eval/evaluate_all_models.py"),
        "--full-model",
        str(ROOT / "checkpoints/selected/cvae_gnn/best_model.pt"),
        "--cvae-mlp-model",
        str(ROOT / "checkpoints/selected/cvae_mlp/best_model.pt"),
        "--mlp-gnn-model",
        str(ROOT / "checkpoints/selected/mlp_gnn/best_model.pt"),
        "--direct-mlp-model",
        str(ROOT / "checkpoints/selected/direct_mlp/best_model.pt"),
        "--val-pairs",
        str(args.val_pairs.resolve()),
        "--graph-imputation",
        str(ROOT / "configs/graph_imputation.yaml"),
        "--samples-per-query",
        "64",
        "--seed",
        "7",
        "--output-dir",
        str(output_dir),
        "--csv-output",
        str(output_dir / "validation_metrics_table.csv"),
        "--device",
        args.device,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
