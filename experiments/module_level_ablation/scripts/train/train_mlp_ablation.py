#!/usr/bin/env python3
"""Train deterministic Vreq-only MLP structure-generation ablations."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

def find_project_root(start: Path) -> Path:
    for parent in start.resolve().parents:
        if (parent / "models" / "gvae").is_dir():
            return parent
    raise RuntimeError("Could not find project root containing models/gvae")


PROJECT_ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(PROJECT_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from gvae.robot.geometry import (
    EdgeAngleRecoverer,
    GraphImputationLayer,
    derive_edge_angle_static_transforms,
)
from gvae.core.io import write_json
from gvae.networks.mlp_baseline import DirectMLPStructureGenerator
from gvae.data.structure_dataset import StructurePairDataset
from gvae.losses.structure_losses import compute_direct_mlp_losses
from gvae.core.training import move_batch_to_device, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Vreq-only MLP ablation.")
    parser.add_argument(
        "--train-pairs",
        default="datasets/processed_dataset/split_seed7/train_pairs.jsonl",
    )
    parser.add_argument(
        "--val-pairs",
        default="datasets/processed_dataset/split_seed7/val_pairs.jsonl",
    )
    parser.add_argument("--graph-imputation", default="configs/graph_imputation.yaml")
    parser.add_argument(
        "--angle-reference",
        default="datasets/structure_config/4-bar/data_0000.json",
    )
    parser.add_argument("--output-dir", default="ckpts/mlp_ablation")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-val-pairs", type=int, default=0)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=(320, 320, 320, 320),
    )
    parser.add_argument(
        "--scale-mode",
        choices=["direct_mlp", "gnn"],
        default="direct_mlp",
        help="Predict scale directly or estimate it from generated structure with GNN.",
    )
    parser.add_argument("--bar-embedding-dim", type=int, default=16)
    parser.add_argument(
        "--scale-gnn-hidden-dims",
        type=int,
        nargs=3,
        default=(128, 256, 256),
        metavar=("H1", "H2", "H3"),
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=5)

    parser.add_argument("--w-nodes", type=float, default=1.0)
    parser.add_argument("--w-edge-pose", type=float, default=1.0)
    parser.add_argument("--w-leg-base", type=float, default=0.5)
    parser.add_argument("--w-leg-angle", type=float, default=1.0)
    parser.add_argument("--w-scale", type=float, default=1.0)
    parser.add_argument("--w-scale-teacher", type=float, default=0.5)
    parser.add_argument("--w-bar", type=float, default=0.2)
    parser.add_argument("--w-spacing", type=float, default=1.0)
    parser.add_argument("--w-geometry", type=float, default=0.5)
    parser.add_argument("--w-angle-equal", type=float, default=0.2)
    parser.add_argument("--w-size", type=float, default=1.0)
    parser.add_argument("--w-task", type=float, default=1.0)
    return parser.parse_args()


def make_weights(args: argparse.Namespace) -> dict[str, float]:
    return {
        "nodes": args.w_nodes,
        "edge_pose": args.w_edge_pose,
        "leg_base": args.w_leg_base,
        "leg_angle": args.w_leg_angle,
        "scale": args.w_scale,
        "scale_teacher": args.w_scale_teacher,
        "bar": args.w_bar,
        "spacing": args.w_spacing,
        "geometry": args.w_geometry,
        "angle_equal": args.w_angle_equal,
        "size": args.w_size,
        "task": args.w_task,
    }


def input_statistics(dataset: StructurePairDataset) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([row["vreq"] for row in dataset.rows], dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def read_reference_edges(path: Path) -> list[list[float]]:
    with path.open("r") as f:
        return json.load(f)["edges"]


def model_config(
    model: DirectMLPStructureGenerator,
) -> dict[str, object]:
    output_format = "nodes[8,7] + leg_angle[3]"
    if model.scale_mode == "direct_mlp":
        output_format += " + scale[3]"
    else:
        output_format += "; Scale GNN -> scale[3]"
    return {
        "model_family": model.model_family,
        "input_dim": 6,
        "hidden_dims": list(model.hidden_dims),
        "num_bar_types": model.num_bar_types,
        "dropout": model.dropout,
        "input_mean": model.input_mean.detach().cpu().tolist(),
        "input_std": model.input_std.detach().cpu().tolist(),
        "output_format": output_format,
        "scale_mode": model.scale_mode,
        "bar_embedding_dim": model.bar_embedding_dim,
        "scale_gnn_hidden_dims": list(model.scale_gnn_hidden_dims),
    }


def run_epoch(
    model: DirectMLPStructureGenerator,
    graph_layer: GraphImputationLayer,
    angle_recoverer: EdgeAngleRecoverer,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    epoch: int,
    phase: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, float], float]:
    training = optimizer is not None
    model.train(training)
    sums: defaultdict[str, float] = defaultdict(float)
    count = 0
    start = time.time()
    iterable = (
        tqdm(
            loader,
            desc=f"MLP epoch {epoch:03d} {phase}",
            unit="batch",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else loader
    )

    for batch in iterable:
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(training):
            prediction = model(
                batch["vreq"],
                batch["bar_v"],
                batch["nodes"],
                batch["leg_angle"],
                batch["scale"],
            )
            graph = graph_layer(prediction["nodes"])
            edge_angles = angle_recoverer(graph["edge_t"])
            loss, metrics = compute_direct_mlp_losses(
                batch,
                prediction,
                graph,
                edge_angles,
                weights,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        for name, value in metrics.items():
            sums[name] += float(value)
        count += 1
        if tqdm is not None:
            iterable.set_postfix(
                total=f"{metrics['total']:.4f}",
                nodes=f"{metrics['nodes']:.4f}",
                scale=f"{metrics['scale']:.4f}",
                bar_acc=f"{metrics['bar_accuracy']:.3f}",
            )

    return (
        {name: value / max(count, 1) for name, value in sums.items()},
        time.time() - start,
    )


def checkpoint_payload(
    model: DirectMLPStructureGenerator,
    config: dict,
    static_transforms: np.ndarray,
    history: list[dict],
    **extra,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "model_config": model_config(model),
        "config": config,
        "edge_angle_static_transforms": static_transforms.tolist(),
        "history": history,
        **extra,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    weights = make_weights(args)

    train_dataset = StructurePairDataset(
        args.train_pairs,
        max_pairs=(args.max_pairs if args.max_pairs > 0 else None),
        seed=args.seed,
        shuffle=True,
    )
    val_dataset = StructurePairDataset(
        args.val_pairs,
        max_pairs=(args.max_val_pairs if args.max_val_pairs > 0 else None),
        seed=args.seed,
        shuffle=False,
    )
    overlap = train_dataset.source_paths & val_dataset.source_paths
    if overlap:
        raise ValueError(
            f"Train/validation structure leakage: {len(overlap)} source paths"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    mean, std = input_statistics(train_dataset)
    model = DirectMLPStructureGenerator(
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        input_mean=mean.tolist(),
        input_std=std.tolist(),
        scale_mode=args.scale_mode,
        bar_embedding_dim=args.bar_embedding_dim,
        scale_gnn_hidden_dims=tuple(args.scale_gnn_hidden_dims),
    ).to(device)
    graph_layer = GraphImputationLayer(args.graph_imputation).to(device)
    static_transforms = derive_edge_angle_static_transforms(
        read_reference_edges(Path(args.angle_reference))
    )
    angle_recoverer = EdgeAngleRecoverer(static_transforms).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_dir = (
        Path(args.output_dir)
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_vreq_{args.scale_mode}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "weights": weights,
            "num_train": len(train_dataset),
            "num_val": len(val_dataset),
            "device": str(device),
            "input_mean": mean.astype(float).tolist(),
            "input_std": std.astype(float).tolist(),
        }
    )
    write_json(run_dir / "config.json", config)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    print(
        f"Training MLP + {args.scale_mode} scale on {len(train_dataset)} rows, "
        f"validating on {len(val_dataset)}, device={device}, output={run_dir}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics, train_time = run_epoch(
            model,
            graph_layer,
            angle_recoverer,
            train_loader,
            device,
            weights,
            epoch,
            "train",
            optimizer,
        )
        val_metrics, val_time = run_epoch(
            model,
            graph_layer,
            angle_recoverer,
            val_loader,
            device,
            weights,
            epoch,
            "val",
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "train_time_seconds": train_time,
                "val_time_seconds": val_time,
            }
        )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                checkpoint_payload(
                    model,
                    config,
                    static_transforms,
                    history,
                    model_state=best_state,
                    best_epoch=epoch,
                    best_val_total=best_val,
                ),
                run_dir / "best_model.pt",
            )
        if epoch % args.save_interval == 0:
            torch.save(
                checkpoint_payload(
                    model,
                    config,
                    static_transforms,
                    history,
                    epoch=epoch,
                ),
                run_dir / f"epoch_{epoch:04d}.pt",
            )

        print(
            f"MLP epoch {epoch:03d} "
            f"train_total={train_metrics['total']:.6f} "
            f"val_total={val_metrics['total']:.6f} "
            f"val_nodes={val_metrics['nodes']:.6f} "
            f"val_scale={val_metrics['scale']:.6f} "
            f"val_bar_acc={val_metrics['bar_accuracy']:.4f} "
            f"train_time={train_time:.1f}s val_time={val_time:.1f}s"
        )

    model.load_state_dict(best_state)
    torch.save(
        checkpoint_payload(
            model,
            config,
            static_transforms,
            history,
            best_val_total=best_val,
        ),
        run_dir / "final_model.pt",
    )
    write_json(
        run_dir / "metrics.json",
        {"history": history, "best_val_total": best_val},
    )
    print(f"Best val total: {best_val:.6f}")
    print(f"Saved MLP baseline to {run_dir}")


if __name__ == "__main__":
    main()
