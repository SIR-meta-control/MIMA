#!/usr/bin/env python3
"""Train the third-stage conditional structure generator."""

from __future__ import annotations

import argparse
import copy
import json
import os
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

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
from gvae.data.structure_dataset import StructurePairDataset
from gvae.networks.structure_generator import ConditionalStructureVAE
from gvae.losses.structure_losses import compute_structure_losses
from gvae.core.training import move_batch_to_device, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train structure generator.")
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
        help="Reference JSON used to derive fixed edge-angle transforms.",
    )
    parser.add_argument("--output-dir", default="ckpts/structure_generator")
    parser.add_argument("--bar-type", choices=["4-bar", "8-bar", "6-bar"], default=None)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Maximum training pairs; 0 means use the full training split.",
    )
    parser.add_argument(
        "--max-val-pairs",
        type=int,
        default=0,
        help="Maximum validation pairs; 0 means use the full validation split.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--condition-dim", type=int, default=128)
    parser.add_argument("--xyz-hidden-dim", type=int, default=32)
    parser.add_argument("--task-hidden-dim", type=int, default=32)
    parser.add_argument("--bar-embedding-dim", type=int, default=16)
    parser.add_argument(
        "--scale-mode",
        choices=["gnn", "mlp"],
        default="gnn",
        help="Estimate scale with topology-aware message passing or a flat MLP.",
    )
    parser.add_argument(
        "--scale-gnn-hidden-dims",
        type=int,
        nargs=3,
        default=(128, 256, 256),
        metavar=("H1", "H2", "H3"),
    )
    parser.add_argument(
        "--scale-mlp-hidden-dims",
        type=int,
        nargs=3,
        default=(320, 352, 256),
        metavar=("H1", "H2", "H3"),
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=5)

    parser.add_argument("--w-nodes", type=float, default=1.0)
    parser.add_argument("--w-edge-pose", type=float, default=1.0)
    parser.add_argument("--w-leg-base", type=float, default=0.5)
    parser.add_argument("--w-leg-angle", type=float, default=1.0)
    parser.add_argument("--w-scale", type=float, default=1.0)
    parser.add_argument("--w-scale-teacher", type=float, default=0.5)
    parser.add_argument("--w-spacing", type=float, default=1.0)
    parser.add_argument("--w-geometry", type=float, default=0.5)
    parser.add_argument("--w-angle-equal", type=float, default=0.2)
    parser.add_argument("--w-size", type=float, default=1.0)
    parser.add_argument("--w-task", type=float, default=1.0)
    parser.add_argument("--w-kl", type=float, default=1e-4)
    return parser.parse_args()


def read_reference_edges(path: Path) -> list[list[float]]:
    with path.open("r") as f:
        return json.load(f)["edges"]


def make_weights(args: argparse.Namespace) -> dict[str, float]:
    return {
        "nodes": args.w_nodes,
        "edge_pose": args.w_edge_pose,
        "leg_base": args.w_leg_base,
        "leg_angle": args.w_leg_angle,
        "scale": args.w_scale,
        "scale_teacher": args.w_scale_teacher,
        "spacing": args.w_spacing,
        "geometry": args.w_geometry,
        "angle_equal": args.w_angle_equal,
        "size": args.w_size,
        "task": args.w_task,
        "kl": args.w_kl,
    }


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    sums = defaultdict(float)
    for row in rows:
        for key, value in row.items():
            sums[key] += float(value)
    return {key: value / max(len(rows), 1) for key, value in sums.items()}


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, torch.device]:
    """Initialize torch.distributed when launched by torchrun."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, select_device(args.device)

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA in this script")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        dist.init_process_group(backend="nccl", device_id=device)
    except TypeError:
        dist.init_process_group(backend="nccl")
    return True, dist.get_rank(), dist.get_world_size(), device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


class TeeStream:
    """Write console output to both the terminal and a run-local log file."""

    def __init__(self, terminal_stream, log_stream) -> None:
        self.terminal_stream = terminal_stream
        self.log_stream = log_stream
        self.encoding = getattr(terminal_stream, "encoding", "utf-8")

    def write(self, data: str) -> int:
        self.terminal_stream.write(data)
        self.log_stream.write(data)
        return len(data)

    def flush(self) -> None:
        self.terminal_stream.flush()
        self.log_stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal_stream, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.terminal_stream.fileno()


def setup_run_logging(
    output_dir: Path,
    rank: int,
) -> tuple[object, object, object] | None:
    if not is_main_process(rank):
        return None

    log_path = output_dir / "train.log"
    log_file = log_path.open("a", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    print(f"Logging to {log_path}")
    return log_file, original_stdout, original_stderr


def close_run_logging(state: tuple[object, object, object] | None) -> None:
    if state is None:
        return

    log_file, original_stdout, original_stderr = state
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


def broadcast_string(value: str, distributed: bool) -> str:
    if not distributed:
        return value
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return str(values[0])


def distributed_barrier(distributed: bool, device: torch.device) -> None:
    if not distributed:
        return
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> ConditionalStructureVAE:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def model_config_dict(model: torch.nn.Module) -> dict[str, object]:
    base_model = unwrap_model(model)
    return {
        "model_family": base_model.model_family,
        "condition_dim": base_model.condition_dim,
        "latent_dim": base_model.latent_dim,
        "hidden_dim": base_model.hidden_dim,
        "xyz_hidden_dim": base_model.xyz_hidden_dim,
        "task_hidden_dim": base_model.task_hidden_dim,
        "bar_embedding_dim": base_model.bar_embedding_dim,
        "num_bar_types": base_model.num_bar_types,
        "dropout": base_model.dropout,
        "scale_mode": base_model.scale_mode,
        "scale_gnn_hidden_dims": list(base_model.scale_gnn_hidden_dims),
        "scale_mlp_hidden_dims": list(base_model.scale_mlp_hidden_dims),
    }


def reduce_metric_sums(
    sums: dict[str, float],
    count: int,
    device: torch.device,
    distributed: bool,
) -> dict[str, float]:
    if not sums:
        return {}

    keys = sorted(sums)
    values = torch.tensor(
        [float(count)] + [float(sums[key]) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)

    total_count = max(float(values[0].item()), 1.0)
    return {
        key: float((values[index + 1] / total_count).detach().cpu().item())
        for index, key in enumerate(keys)
    }


class NoProgress:
    """Progress wrapper used by nonzero DDP ranks to keep logs readable."""

    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        yield from self.iterable

    def set_postfix(self, **kwargs) -> None:
        return None


class SimpleProgress:
    """Small tqdm-like fallback without external dependencies."""

    def __init__(self, iterable, desc: str, unit: str = "batch", mininterval: float = 1.0):
        self.iterable = iterable
        self.desc = desc
        self.unit = unit
        self.total = len(iterable) if hasattr(iterable, "__len__") else None
        self.count = 0
        self.start_time = time.time()
        self.last_print = 0.0
        self.postfix = {}

    def __iter__(self):
        for item in self.iterable:
            self.count += 1
            yield item
        self._print(force=True, final=True)

    def set_postfix(self, **kwargs) -> None:
        self.postfix = kwargs
        now = time.time()
        if now - self.last_print >= 1.0 or self.count == self.total:
            self._print(force=True)
            self.last_print = now

    def _print(self, force: bool = False, final: bool = False) -> None:
        if not force:
            return
        elapsed = max(time.time() - self.start_time, 1e-9)
        rate = self.count / elapsed
        if self.total:
            ratio = min(self.count / self.total, 1.0)
            filled = int(ratio * 30)
            bar = "#" * filled + "." * (30 - filled)
            remaining = max(self.total - self.count, 0)
            eta = remaining / max(rate, 1e-9)
            progress = f"[{bar}] {self.count}/{self.total}"
            eta_text = f"eta={eta:.1f}s"
        else:
            progress = f"{self.count}"
            eta_text = "eta=?"
        postfix = " ".join(f"{key}={value}" for key, value in self.postfix.items())
        message = (
            f"\r{self.desc}: {progress} {self.unit} "
            f"elapsed={elapsed:.1f}s {eta_text} {postfix}"
        )
        print(message, end="\n" if final else "", flush=True)


def make_progress(iterable, desc: str, enabled: bool = True):
    if not enabled:
        return NoProgress(iterable)
    if tqdm is not None:
        return tqdm(
            iterable,
            desc=desc,
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
    return SimpleProgress(iterable, desc=desc, unit="batch")


def run_epoch(
    model: ConditionalStructureVAE,
    graph_layer: GraphImputationLayer,
    angle_recoverer: EdgeAngleRecoverer,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    epoch: int,
    phase: str,
    distributed: bool,
    rank: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, float], float]:
    is_train = optimizer is not None
    model.train(is_train)
    metric_sums = defaultdict(float)
    metric_count = 0
    start_time = time.time()

    progress = make_progress(
        loader,
        desc=f"epoch {epoch:03d} {phase}",
        enabled=is_main_process(rank),
    )
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(is_train):
            pred = model(
                batch["vreq"],
                batch["bar_v"],
                batch["nodes"],
                batch["leg_angle"],
                batch["scale"],
                sample=is_train,
            )
            graph_outputs = graph_layer(pred["nodes"])
            edge_angles_pred = angle_recoverer(graph_outputs["edge_t"])
            loss, metrics = compute_structure_losses(
                batch,
                pred,
                graph_outputs,
                edge_angles_pred,
                weights,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        for key, value in metrics.items():
            metric_sums[key] += float(value)
        metric_count += 1
        progress.set_postfix(
            total=f"{metrics['total']:.4f}",
            nodes=f"{metrics['nodes']:.4f}",
            scale=f"{metrics['scale']:.4f}",
            geom=f"{metrics['geometry']:.4f}",
            angle=f"{metrics['angle_equal']:.4f}",
        )

    return (
        reduce_metric_sums(metric_sums, metric_count, device, distributed),
        time.time() - start_time,
    )


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, device = setup_distributed(args)
    set_seed(args.seed + rank)
    weights = make_weights(args)

    train_dataset = StructurePairDataset(
        args.train_pairs,
        max_pairs=(args.max_pairs if args.max_pairs > 0 else None),
        bar_type=args.bar_type,
        seed=args.seed,
        shuffle=True,
    )
    val_dataset = StructurePairDataset(
        args.val_pairs,
        max_pairs=(args.max_val_pairs if args.max_val_pairs > 0 else None),
        bar_type=args.bar_type,
        seed=args.seed,
        shuffle=False,
    )
    overlap = train_dataset.source_paths & val_dataset.source_paths
    if overlap:
        examples = ", ".join(sorted(overlap)[:3])
        raise ValueError(
            f"Train/validation structure leakage: {len(overlap)} overlapping "
            f"source_path values, including {examples}"
        )
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.seed,
        )
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = ConditionalStructureVAE(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        condition_dim=args.condition_dim,
        xyz_hidden_dim=args.xyz_hidden_dim,
        task_hidden_dim=args.task_hidden_dim,
        bar_embedding_dim=args.bar_embedding_dim,
        dropout=args.dropout,
        scale_mode=args.scale_mode,
        scale_gnn_hidden_dims=tuple(args.scale_gnn_hidden_dims),
        scale_mlp_hidden_dims=tuple(args.scale_mlp_hidden_dims),
    ).to(device)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
        )
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if is_main_process(rank) else ""
    timestamp = broadcast_string(timestamp, distributed)
    suffix = f"{args.bar_type or 'all'}_{args.scale_mode}_scale"
    output_dir = Path(args.output_dir) / f"{timestamp}_{suffix}"
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(distributed, device)
    log_state = setup_run_logging(output_dir, rank)

    config = vars(args).copy()
    config["weights"] = weights
    config["num_train"] = train_size
    config["num_val"] = val_size
    config["device"] = str(device)
    config["distributed"] = distributed
    config["world_size"] = world_size
    config["per_gpu_batch_size"] = args.batch_size
    config["global_batch_size"] = args.batch_size * world_size
    if is_main_process(rank):
        write_json(output_dir / "config.json", config)

    best_state = copy.deepcopy(unwrap_model(model).state_dict())
    best_val = float("inf")
    history = []

    if is_main_process(rank):
        print(
            f"Training structure generator on {train_size} rows, "
            f"validating on {val_size}, device={device}, "
            f"world_size={world_size}, per_gpu_batch={args.batch_size}, "
            f"global_batch={args.batch_size * world_size}, output={output_dir}"
        )

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        train_metrics, train_time = run_epoch(
            model,
            graph_layer,
            angle_recoverer,
            train_loader,
            device,
            weights,
            epoch=epoch,
            phase="train",
            distributed=distributed,
            rank=rank,
            optimizer=optimizer,
        )
        val_metrics, val_time = run_epoch(
            model,
            graph_layer,
            angle_recoverer,
            val_loader,
            device,
            weights,
            epoch=epoch,
            phase="val",
            distributed=distributed,
            rank=rank,
            optimizer=None,
        )
        epoch_time = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "train_time_seconds": train_time,
            "val_time_seconds": val_time,
            "epoch_time_seconds": epoch_time,
        }
        history.append(row)

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_state = copy.deepcopy(unwrap_model(model).state_dict())
            if is_main_process(rank):
                torch.save(
                    {
                        "model_state": best_state,
                        "model_config": model_config_dict(model),
                        "config": config,
                        "edge_angle_static_transforms": static_transforms.tolist(),
                        "history": history,
                        "best_epoch": epoch,
                        "best_val_total": best_val,
                    },
                    output_dir / "best_model.pt",
                )

        if is_main_process(rank) and epoch % args.save_interval == 0:
            torch.save(
                {
                    "model_state": unwrap_model(model).state_dict(),
                    "model_config": model_config_dict(model),
                    "config": config,
                    "edge_angle_static_transforms": static_transforms.tolist(),
                    "history": history,
                    "epoch": epoch,
                },
                output_dir / f"epoch_{epoch:04d}.pt",
            )

        if is_main_process(rank) and (epoch == 1 or epoch % args.log_interval == 0):
            print(
                f"epoch {epoch:03d} "
                f"train_total={train_metrics['total']:.6f} "
                f"val_total={val_metrics['total']:.6f} "
                f"val_nodes={val_metrics['nodes']:.6f} "
                f"val_scale={val_metrics['scale']:.6f} "
                f"val_angle_equal={val_metrics['angle_equal']:.6f} "
                f"val_geometry={val_metrics['geometry']:.6f} "
                f"train_time={train_time:.1f}s "
                f"val_time={val_time:.1f}s "
                f"epoch_time={epoch_time:.1f}s"
            )

    unwrap_model(model).load_state_dict(best_state)
    if is_main_process(rank):
        torch.save(
            {
                "model_state": unwrap_model(model).state_dict(),
                "model_config": model_config_dict(model),
                "config": config,
                "edge_angle_static_transforms": static_transforms.tolist(),
                "history": history,
                "best_val_total": best_val,
            },
            output_dir / "final_model.pt",
        )
        write_json(output_dir / "metrics.json", {"history": history, "best_val_total": best_val})
        print(f"Best val total: {best_val:.6f}")
        print(f"Saved structure generator to {output_dir}")
    close_run_logging(log_state)
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
