#!/usr/bin/env python3
"""Evaluate the full pipeline and three ablations, then export one table."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    key: str
    checkpoint_group: str
    output_name: str
    method_name: str


EXPERIMENTS = (
    Experiment(
        key="full",
        checkpoint_group="structure_generator",
        output_name="validation_metrics.json",
        method_name="Full generator",
    ),
    Experiment(
        key="cvae_mlp",
        checkpoint_group="cvae_mlp_ablation",
        output_name="cvae_mlp_ablation_metrics.json",
        method_name="Scale GNN -> MLP",
    ),
    Experiment(
        key="mlp_gnn",
        checkpoint_group="mlp_gnn_ablation",
        output_name="mlp_gnn_ablation_metrics.json",
        method_name="CVAE -> MLP",
    ),
    Experiment(
        key="direct_mlp",
        checkpoint_group="mlp_ablation",
        output_name="mlp_ablation_metrics.json",
        method_name="CVAE+Scale GNN -> MLP",
    ),
)


def find_project_root(start: Path) -> Path:
    for parent in [start.resolve(), *start.resolve().parents]:
        if (parent / "models" / "gvae").is_dir():
            return parent
    raise RuntimeError("Could not find project root containing models/gvae")


def parse_args() -> argparse.Namespace:
    default_split = os.environ.get(
        "SPLIT_DIR",
        "datasets/processed_dataset/split_seed7_val20",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Run validation for the full GVAE pipeline and ablations, then "
            "merge the resulting metric JSON files into the paper table."
        )
    )
    parser.add_argument(
        "--ckpt-dir",
        default=os.environ.get("CKPT_DIR", "checkpoints"),
        help="Root checkpoint directory containing experiment subfolders.",
    )
    parser.add_argument(
        "--val-pairs",
        default=f"{default_split}/val_pairs.jsonl",
        help="Validation pair JSONL file.",
    )
    parser.add_argument(
        "--graph-imputation",
        default="configs/graph_imputation.yaml",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--csv-output",
        default="outputs/validation_metrics_table.csv",
        help="CSV table output path.",
    )
    parser.add_argument("--samples-per-query", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--coverage-threshold", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Skip model evaluation and only merge existing metric JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )

    for experiment in EXPERIMENTS:
        parser.add_argument(
            f"--{experiment.key.replace('_', '-')}-model",
            dest=f"{experiment.key}_model",
            default=None,
            help=(
                f"Explicit checkpoint for {experiment.method_name}. "
                f"Default: latest {experiment.checkpoint_group}/*/best_model.pt"
            ),
        )
    return parser.parse_args()


def latest_checkpoint(ckpt_dir: Path, group: str) -> Path:
    candidates = sorted((ckpt_dir / group).glob("*/best_model.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under {ckpt_dir / group}/*/best_model.pt"
        )
    return candidates[-1]


def model_path_for(args: argparse.Namespace, experiment: Experiment) -> Path:
    explicit = getattr(args, f"{experiment.key}_model")
    if explicit:
        return Path(explicit)
    return latest_checkpoint(Path(args.ckpt_dir), experiment.checkpoint_group)


def run_command(command: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path(__file__))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluate_script = project_root / "scripts" / "eval" / "evaluate_structure_generator.py"
    table_script = project_root / "scripts" / "eval" / "print_validation_table.py"

    metric_paths = []
    if not args.table_only:
        for experiment in EXPERIMENTS:
            model_path = model_path_for(args, experiment)
            output_path = output_dir / experiment.output_name
            metric_paths.append(output_path)
            command = [
                sys.executable,
                str(evaluate_script),
                "--model",
                str(model_path),
                "--val-pairs",
                str(args.val_pairs),
                "--graph-imputation",
                str(args.graph_imputation),
                "--samples-per-query",
                str(args.samples_per_query),
                "--batch-size",
                str(args.batch_size),
                "--query-batch-size",
                str(args.query_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--max-pairs",
                str(args.max_pairs),
                "--max-queries",
                str(args.max_queries),
                "--temperature",
                str(args.temperature),
                "--coverage-threshold",
                str(args.coverage_threshold),
                "--seed",
                str(args.seed),
                "--device",
                str(args.device),
                "--method-name",
                experiment.method_name,
                "--output",
                str(output_path),
            ]
            run_command(command, args.dry_run)
    else:
        metric_paths = [output_dir / experiment.output_name for experiment in EXPERIMENTS]

    table_command = [
        sys.executable,
        str(table_script),
        *[str(path) for path in metric_paths],
        "--csv-output",
        str(args.csv_output),
    ]
    run_command(table_command, args.dry_run)


if __name__ == "__main__":
    main()
