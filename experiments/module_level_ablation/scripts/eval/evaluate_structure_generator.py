#!/usr/bin/env python3
"""Evaluate reconstruction, scale estimation, and prior-generation validity."""

from __future__ import annotations

import argparse
import json
import math
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
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from gvae.networks.generator_loader import load_generator
from gvae.robot.geometry import EdgeAngleRecoverer, GraphImputationLayer, normalize_quaternion
from gvae.robot.inference import ConstraintThresholds, HardConstraintDecoder
from gvae.core.io import read_jsonl
from gvae.data.structure_dataset import StructurePairDataset, load_structure_json
from gvae.core.training import move_batch_to_device, select_device, set_seed


AXES = ("x", "y", "z")
METHOD_NAME_ALIASES = {
    "Full pipeline": "Full generator",
    "CVAE + Scale GNN": "Full generator",
    "w/o Scale GNN -> MLP": "Scale GNN -> MLP",
    "CVAE + Scale MLP": "Scale GNN -> MLP",
    "w/o CVAE -> MLP": "CVAE -> MLP",
    "MLP + Scale GNN": "CVAE -> MLP",
    "Direct MLP baseline": "CVAE+Scale GNN -> MLP",
    "Direct MLP": "CVAE+Scale GNN -> MLP",
}


def scale_mae_mm(scale_metrics: dict) -> dict[str, float]:
    mae_cm = scale_metrics["mae_cm"]
    return {axis: float(mae_cm[axis]) * 10.0 for axis in AXES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a structure-generator checkpoint on the fixed validation split."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--val-pairs",
        default="datasets/processed_dataset/split_seed7_val20/val_pairs.jsonl",
    )
    parser.add_argument("--graph-imputation", default="configs/graph_imputation.yaml")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--samples-per-query", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.02,
        help="Maximum normalized RMS feature distance counted as covered.",
    )
    parser.add_argument(
        "--method-name",
        default=None,
        help="Optional display name used in the printed paper table.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument("--skip-prior", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def progress(iterable, description: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=description, dynamic_ncols=True)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_timing(num_batches: int, elapsed_seconds: float) -> dict:
    batches_per_second = (
        num_batches / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    )
    return {
        "num_batches": int(num_batches),
        "elapsed_seconds": float(elapsed_seconds),
        "batches_per_second": float(batches_per_second),
    }


def print_timing(timing: dict) -> None:
    if not timing:
        return

    labels = {
        "validation_reconstruction": "validation reconstruction",
        "prior_generation": "prior generation",
    }
    print("\nTiming")
    for key in ("validation_reconstruction", "prior_generation"):
        if key not in timing:
            continue
        item = timing[key]
        num_batches = int(item["num_batches"])
        elapsed_seconds = float(item["elapsed_seconds"])
        batches_per_second = float(item["batches_per_second"])
        print(f"\n{labels[key]}:")
        print(
            f"{num_batches} batch / {batches_per_second:.2f} batch/s "
            f"≈ {elapsed_seconds:.2f} s"
        )


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def task_name(vreq: torch.Tensor | list[float] | tuple[float, ...]) -> str:
    values = torch.as_tensor(vreq).flatten()
    active = [
        name
        for name, enabled in zip(("load", "inspect", "pack"), values[3:6])
        if enabled > 0.5
    ]
    return "+".join(active) if active else "none"


class ScaleMetrics:
    def __init__(self) -> None:
        self.count = 0
        self.absolute_sum = np.zeros(3, dtype=np.float64)
        self.squared_sum = np.zeros(3, dtype=np.float64)
        self.target_sum = np.zeros(3, dtype=np.float64)
        self.target_squared_sum = np.zeros(3, dtype=np.float64)
        self.within_2cm = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        error = prediction - target
        self.count += int(target.shape[0])
        self.absolute_sum += error.abs().sum(dim=0).double().cpu().numpy()
        self.squared_sum += error.square().sum(dim=0).double().cpu().numpy()
        self.target_sum += target.sum(dim=0).double().cpu().numpy()
        self.target_squared_sum += target.square().sum(dim=0).double().cpu().numpy()
        self.within_2cm += int((error.abs().amax(dim=-1) <= 0.02).sum().item())

    def compute(self) -> dict:
        if self.count == 0:
            return {}
        mae = self.absolute_sum / self.count
        rmse = np.sqrt(self.squared_sum / self.count)
        target_variance_sum = (
            self.target_squared_sum - np.square(self.target_sum) / self.count
        )
        r2 = 1.0 - self.squared_sum / np.maximum(target_variance_sum, 1e-12)
        return {
            "num_examples": self.count,
            "mae_cm": {axis: float(value * 100.0) for axis, value in zip(AXES, mae)},
            "rmse_cm": {
                axis: float(value * 100.0) for axis, value in zip(AXES, rmse)
            },
            "r2": {axis: float(value) for axis, value in zip(AXES, r2)},
            "within_2cm_rate": self.within_2cm / self.count,
        }


class ReconstructionMetrics:
    def __init__(self) -> None:
        self.num_pairs = 0
        self.num_node_values = 0
        self.num_nodes = 0
        self.num_leg_angles = 0
        self.node_position_absolute_sum = 0.0
        self.node_position_squared_sum = 0.0
        self.node_position_distance_sum = 0.0
        self.node_rotation_degree_sum = 0.0
        self.leg_angle_degree_sum = 0.0
        self.end_to_end_scale = ScaleMetrics()
        self.teacher_scale = ScaleMetrics()
        self.check_passes: defaultdict[str, int] = defaultdict(int)
        self.constraint_metric_sums: defaultdict[str, float] = defaultdict(float)
        self.valid_count = 0
        self.bar_prediction_count = 0
        self.bar_prediction_correct = 0
        self.bar_nll_sum = 0.0

    def update(
        self,
        batch: dict,
        prediction: dict,
        decoded: dict,
        teacher_indices: torch.Tensor | None,
    ) -> None:
        nodes_pred = prediction["nodes"]
        nodes_gt = batch["nodes"]
        position_error = nodes_pred[..., :3] - nodes_gt[..., :3]
        self.node_position_absolute_sum += float(position_error.abs().sum().item())
        self.node_position_squared_sum += float(position_error.square().sum().item())
        self.node_position_distance_sum += float(
            torch.linalg.vector_norm(position_error, dim=-1).sum().item()
        )

        quat_pred = normalize_quaternion(nodes_pred[..., 3:])
        quat_gt = normalize_quaternion(nodes_gt[..., 3:])
        dot = (quat_pred * quat_gt).sum(dim=-1).abs().clamp(max=1.0)
        rotation_degrees = 2.0 * torch.acos(dot) * (180.0 / math.pi)
        self.node_rotation_degree_sum += float(rotation_degrees.sum().item())

        leg_error = wrap_angle(prediction["leg_angle"] - batch["leg_angle"]).abs()
        self.leg_angle_degree_sum += float(
            (leg_error * (180.0 / math.pi)).sum().item()
        )

        batch_size = int(nodes_gt.shape[0])
        self.num_pairs += batch_size
        self.num_node_values += int(position_error.numel())
        self.num_nodes += int(rotation_degrees.numel())
        self.num_leg_angles += int(leg_error.numel())
        self.end_to_end_scale.update(prediction["scale"], batch["scale"])
        if "bar_logits" in prediction:
            logits = prediction["bar_logits"]
            self.bar_prediction_count += batch_size
            self.bar_prediction_correct += int(
                (logits.argmax(dim=-1) == batch["bar_v"]).sum().item()
            )
            self.bar_nll_sum += float(
                F.cross_entropy(
                    logits,
                    batch["bar_v"],
                    reduction="sum",
                ).item()
            )

        if teacher_indices is not None and "scale_teacher" in prediction:
            self.teacher_scale.update(
                prediction["scale_teacher"][teacher_indices],
                batch["scale"][teacher_indices],
            )

        self.valid_count += int(decoded["valid"].sum().item())
        for name, values in decoded["checks"].items():
            self.check_passes[name] += int(values.sum().item())
        for name, values in decoded["metrics"].items():
            self.constraint_metric_sums[name] += float(values.sum().item())

    def compute(self) -> dict:
        rotation_mean_deg = self.node_rotation_degree_sum / max(self.num_nodes, 1)
        leg_angle_mae_deg = self.leg_angle_degree_sum / max(self.num_leg_angles, 1)
        result = {
            "num_pairs": self.num_pairs,
            "node_position_mae_mm": 1000.0
            * self.node_position_absolute_sum
            / max(self.num_node_values, 1),
            "node_position_mae_cm": 100.0
            * self.node_position_absolute_sum
            / max(self.num_node_values, 1),
            "node_position_rmse_cm": 100.0
            * math.sqrt(self.node_position_squared_sum / max(self.num_node_values, 1)),
            "node_position_mean_distance_cm": 100.0
            * self.node_position_distance_sum
            / max(self.num_nodes, 1),
            "node_rotation_mean_rad": rotation_mean_deg * math.pi / 180.0,
            "node_rotation_mean_deg": rotation_mean_deg,
            "leg_angle_mae_rad": leg_angle_mae_deg * math.pi / 180.0,
            "leg_angle_mae_deg": leg_angle_mae_deg,
            "scale_end_to_end": self.end_to_end_scale.compute(),
            "scale_teacher": self.teacher_scale.compute(),
            "overall_valid_rate": self.valid_count / max(self.num_pairs, 1),
            "constraint_pass_rates": {
                name: count / max(self.num_pairs, 1)
                for name, count in sorted(self.check_passes.items())
            },
            "constraint_mean_errors": {
                name: total / max(self.num_pairs, 1)
                for name, total in sorted(self.constraint_metric_sums.items())
            },
        }
        if self.bar_prediction_count:
            result["bar_prediction"] = {
                "pair_weighted_accuracy": self.bar_prediction_correct
                / self.bar_prediction_count,
                "pair_weighted_nll": self.bar_nll_sum
                / self.bar_prediction_count,
            }
        return result


class PriorMetrics:
    def __init__(self) -> None:
        self.num_queries = 0
        self.num_samples = 0
        self.valid_samples = 0
        self.successful_queries = 0
        self.valid_count_sum = 0
        self.check_passes: defaultdict[str, int] = defaultdict(int)

    def update(self, decoded: dict, samples_per_query: int) -> None:
        valid = decoded["valid"].reshape(-1, samples_per_query)
        self.num_queries += int(valid.shape[0])
        self.num_samples += int(valid.numel())
        self.valid_samples += int(valid.sum().item())
        valid_per_query = valid.sum(dim=-1)
        self.successful_queries += int((valid_per_query > 0).sum().item())
        self.valid_count_sum += int(valid_per_query.sum().item())
        for name, values in decoded["checks"].items():
            self.check_passes[name] += int(values.sum().item())

    def compute(self, samples_per_query: int) -> dict:
        return {
            "num_queries": self.num_queries,
            "samples_per_query": samples_per_query,
            "num_samples": self.num_samples,
            "valid_sample_rate": self.valid_samples / max(self.num_samples, 1),
            f"success_at_{samples_per_query}": self.successful_queries
            / max(self.num_queries, 1),
            "mean_valid_count_per_query": self.valid_count_sum
            / max(self.num_queries, 1),
            "constraint_pass_rates": {
                name: count / max(self.num_samples, 1)
                for name, count in sorted(self.check_passes.items())
            },
        }


class SetMetrics:
    """Macro/micro coverage and valid-candidate diversity over query sets."""

    def __init__(self) -> None:
        self.num_queries = 0
        self.coverage_sum = 0.0
        self.covered_ground_truth = 0
        self.total_ground_truth = 0
        self.nearest_distance_sum = 0.0
        self.ground_truth_with_valid_candidates = 0
        self.diversity_sum = 0.0
        self.queries_with_two_valid = 0

    def update(
        self,
        coverage: float,
        covered_ground_truth: int,
        total_ground_truth: int,
        nearest_distance_sum: float,
        diversity: float,
        num_valid: int,
    ) -> None:
        self.num_queries += 1
        self.coverage_sum += float(coverage)
        self.covered_ground_truth += int(covered_ground_truth)
        self.total_ground_truth += int(total_ground_truth)
        self.nearest_distance_sum += float(nearest_distance_sum)
        if num_valid > 0:
            self.ground_truth_with_valid_candidates += int(total_ground_truth)
        self.diversity_sum += float(diversity)
        self.queries_with_two_valid += int(num_valid >= 2)

    def compute(self, samples_per_query: int, threshold: float) -> dict:
        return {
            f"coverage_at_{samples_per_query}": self.coverage_sum
            / max(self.num_queries, 1),
            f"coverage_at_{samples_per_query}_micro": self.covered_ground_truth
            / max(self.total_ground_truth, 1),
            "coverage_threshold": threshold,
            "mean_gt_to_generated_distance": self.nearest_distance_sum
            / max(self.ground_truth_with_valid_candidates, 1),
            f"diversity_at_{samples_per_query}": self.diversity_sum
            / max(self.num_queries, 1),
            "queries_with_two_valid_rate": self.queries_with_two_valid
            / max(self.num_queries, 1),
            "num_ground_truth_memberships": self.total_ground_truth,
        }


def structure_features(
    nodes: torch.Tensor,
    leg_angle: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Build the sign-invariant normalized feature used by Top-K diversity."""
    positions = nodes[..., :3] / 0.5
    quaternions = normalize_quaternion(nodes[..., 3:])
    signs = torch.where(
        quaternions[..., :1] < 0.0,
        -torch.ones_like(quaternions[..., :1]),
        torch.ones_like(quaternions[..., :1]),
    )
    quaternions = quaternions * signs
    return torch.cat(
        [
            positions.flatten(start_dim=1),
            quaternions.flatten(start_dim=1),
            scale / 0.5,
            leg_angle / math.pi,
        ],
        dim=-1,
    )


def normalized_rms_distances(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("Feature tensors must have shape [N, D]")
    if first.shape[-1] != second.shape[-1]:
        raise ValueError("Feature tensors must have matching feature dimensions")
    return torch.cdist(first, second) / math.sqrt(first.shape[-1])


def query_set_metrics(
    generated_features: torch.Tensor,
    valid_mask: torch.Tensor,
    ground_truth_features: torch.Tensor,
    coverage_threshold: float,
) -> tuple[float, int, int, float, float, int]:
    valid_features = generated_features[valid_mask]
    total_ground_truth = int(ground_truth_features.shape[0])
    num_valid = int(valid_features.shape[0])
    if num_valid == 0:
        return 0.0, 0, total_ground_truth, 0.0, 0.0, 0

    gt_to_generated = normalized_rms_distances(
        ground_truth_features,
        valid_features,
    )
    nearest = gt_to_generated.min(dim=-1).values
    covered = int((nearest <= coverage_threshold).sum().item())
    coverage = covered / max(total_ground_truth, 1)

    if num_valid < 2:
        diversity = 0.0
    else:
        upper = torch.triu_indices(
            num_valid,
            num_valid,
            offset=1,
            device=valid_features.device,
        )
        differences = valid_features[upper[0]] - valid_features[upper[1]]
        pairwise_rms = torch.sqrt(differences.square().mean(dim=-1))
        diversity = float(pairwise_rms.mean().item())
        if diversity < 1e-8:
            diversity = 0.0
    return (
        coverage,
        covered,
        total_ground_truth,
        float(nearest.sum().item()),
        diversity,
        num_valid,
    )


def make_decoder(
    graph_imputation_path: str,
    checkpoint: dict,
    device: torch.device,
) -> HardConstraintDecoder:
    graph_layer = GraphImputationLayer(graph_imputation_path).to(device)
    angle_recoverer = EdgeAngleRecoverer(
        np.asarray(checkpoint["edge_angle_static_transforms"], dtype=np.float32)
    ).to(device)
    return HardConstraintDecoder(
        graph_layer,
        angle_recoverer,
        ConstraintThresholds(),
    )


def evaluate_reconstruction(
    model: torch.nn.Module,
    decoder: HardConstraintDecoder,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, dict]:
    dataset = StructurePairDataset(
        args.val_pairs,
        max_pairs=(args.max_pairs if args.max_pairs > 0 else None),
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    metrics = ReconstructionMetrics()
    seen_teacher_sources: set[str] = set()

    model.eval()
    synchronize_device(device)
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in progress(loader, "validation reconstruction"):
            source_paths = list(batch["source_path"])
            teacher_indices_list = []
            for index, source_path in enumerate(source_paths):
                if source_path not in seen_teacher_sources:
                    seen_teacher_sources.add(source_path)
                    teacher_indices_list.append(index)

            batch = move_batch_to_device(batch, device)
            prediction = model(
                batch["vreq"],
                batch["bar_v"],
                batch["nodes"],
                batch["leg_angle"],
                batch["scale"],
                sample=False,
            )
            decoded = decoder.decode_and_evaluate(
                prediction["nodes"],
                prediction["leg_angle"],
                prediction["scale"],
                batch["vreq"],
                batch["bar_v"],
                torch.ones(batch["bar_v"].shape[0], device=device),
            )
            teacher_indices = (
                torch.as_tensor(teacher_indices_list, dtype=torch.long, device=device)
                if teacher_indices_list
                else None
            )
            metrics.update(batch, prediction, decoded, teacher_indices)
    synchronize_device(device)
    elapsed_seconds = time.perf_counter() - start_time

    result = metrics.compute()
    result["num_unique_teacher_structures"] = len(seen_teacher_sources)
    return result, make_timing(len(loader), elapsed_seconds)


def load_unique_queries(
    path: str | Path,
    max_queries: int,
) -> list[dict]:
    queries = {}
    for row in read_jsonl(path):
        vreq = tuple(float(value) for value in row["vreq"])
        bar_v = int(row["v"])
        key = (vreq, bar_v)
        query = queries.setdefault(
            key,
            {
                "vreq": vreq,
                "bar_v": bar_v,
                "bar_type": row["bar_type"],
                "task": row.get("task", task_name(vreq)),
                "source_paths": set(),
            },
        )
        query["source_paths"].add(row["source_path"])
    rows = []
    for query in queries.values():
        query["source_paths"] = sorted(query["source_paths"])
        rows.append(query)
    return rows[:max_queries] if max_queries > 0 else rows


def evaluate_prior(
    model: torch.nn.Module,
    decoder: HardConstraintDecoder,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, dict]:
    queries = load_unique_queries(args.val_pairs, args.max_queries)
    overall = PriorMetrics()
    by_task: defaultdict[str, PriorMetrics] = defaultdict(PriorMetrics)
    by_bar: defaultdict[str, PriorMetrics] = defaultdict(PriorMetrics)
    set_overall = SetMetrics()
    set_by_task: defaultdict[str, SetMetrics] = defaultdict(SetMetrics)
    set_by_bar: defaultdict[str, SetMetrics] = defaultdict(SetMetrics)
    structure_feature_cache: dict[str, torch.Tensor] = {}

    model.eval()
    with torch.no_grad():
        batches = range(0, len(queries), args.query_batch_size)
        synchronize_device(device)
        start_time = time.perf_counter()
        for start in progress(batches, "prior generation"):
            rows = queries[start : start + args.query_batch_size]
            vreq = torch.as_tensor(
                [row["vreq"] for row in rows],
                dtype=torch.float32,
                device=device,
            )
            bar_v = torch.as_tensor(
                [row["bar_v"] for row in rows],
                dtype=torch.long,
                device=device,
            )
            prediction = model.sample(
                vreq,
                bar_v,
                num_samples=args.samples_per_query,
                temperature=args.temperature,
            )
            repeated_vreq = vreq.repeat_interleave(args.samples_per_query, dim=0)
            repeated_bar = bar_v.repeat_interleave(args.samples_per_query, dim=0)
            decoded = decoder.decode_and_evaluate(
                prediction["nodes"],
                prediction["leg_angle"],
                prediction["scale"],
                repeated_vreq,
                repeated_bar,
                torch.ones(repeated_bar.shape[0], device=device),
            )
            overall.update(decoded, args.samples_per_query)
            generated_features = structure_features(
                decoded["nodes"],
                decoded["leg_angle"],
                decoded["scale"],
            )

            for index, row in enumerate(rows):
                first = index * args.samples_per_query
                last = first + args.samples_per_query
                sliced = {
                    "valid": decoded["valid"][first:last],
                    "checks": {
                        name: values[first:last]
                        for name, values in decoded["checks"].items()
                    },
                }
                by_task[row["task"]].update(sliced, args.samples_per_query)
                by_bar[row["bar_type"]].update(sliced, args.samples_per_query)
                gt_features = []
                for source_path in row["source_paths"]:
                    if source_path not in structure_feature_cache:
                        structure = load_structure_json(source_path)
                        feature = structure_features(
                            torch.as_tensor(
                                structure["nodes"],
                                dtype=torch.float32,
                                device=device,
                            ).unsqueeze(0),
                            torch.as_tensor(
                                structure["leg_angle"],
                                dtype=torch.float32,
                                device=device,
                            ).unsqueeze(0),
                            torch.as_tensor(
                                structure["scale"],
                                dtype=torch.float32,
                                device=device,
                            ).unsqueeze(0),
                        )[0]
                        structure_feature_cache[source_path] = feature
                    gt_features.append(structure_feature_cache[source_path])

                values = query_set_metrics(
                    generated_features[first:last],
                    decoded["valid"][first:last],
                    torch.stack(gt_features),
                    args.coverage_threshold,
                )
                set_overall.update(*values)
                set_by_task[row["task"]].update(*values)
                set_by_bar[row["bar_type"]].update(*values)
        synchronize_device(device)
        elapsed_seconds = time.perf_counter() - start_time

    overall_result = overall.compute(args.samples_per_query)
    overall_result.update(
        set_overall.compute(args.samples_per_query, args.coverage_threshold)
    )
    task_result = {}
    for name, values in sorted(by_task.items()):
        current = values.compute(args.samples_per_query)
        current.update(
            set_by_task[name].compute(
                args.samples_per_query,
                args.coverage_threshold,
            )
        )
        task_result[name] = current
    bar_result = {}
    for name, values in sorted(by_bar.items()):
        current = values.compute(args.samples_per_query)
        current.update(
            set_by_bar[name].compute(
                args.samples_per_query,
                args.coverage_threshold,
            )
        )
        bar_result[name] = current
    result = {
        "overall": overall_result,
        "by_task": task_result,
        "by_bar_type": bar_result,
    }
    return result, make_timing(len(batches), elapsed_seconds)


def print_summary(result: dict) -> None:
    reconstruction = result.get("reconstruction")
    if reconstruction:
        scale_e2e = reconstruction["scale_end_to_end"]
        scale_teacher = reconstruction["scale_teacher"]
        print("\nReconstruction")
        print(
            f"  node position MAE: "
            f"{reconstruction['node_position_mae_mm']:.2f} mm"
        )
        print(
            f"  node rotation mean: "
            f"{reconstruction['node_rotation_mean_rad']:.2f} rad"
        )
        print(
            f"  leg angle MAE: "
            f"{reconstruction['leg_angle_mae_rad']:.2f} rad"
        )
        if scale_e2e:
            scale_mm = scale_mae_mm(scale_e2e)
            print(
                "  scale end-to-end MAE: "
                + ", ".join(
                    f"{axis}={scale_mm[axis]:.2f} mm" for axis in AXES
                )
            )
        if scale_teacher:
            scale_mm = scale_mae_mm(scale_teacher)
            print(
                "  scale teacher MAE: "
                + ", ".join(
                    f"{axis}={scale_mm[axis]:.2f} mm" for axis in AXES
                )
            )
        print(
            f"  reconstruction valid rate: "
            f"{reconstruction['overall_valid_rate']:.2%}"
        )
        if "bar_prediction" in reconstruction:
            print(
                "  internal bar accuracy: "
                f"{reconstruction['bar_prediction']['pair_weighted_accuracy']:.2%}"
            )

    prior = result.get("prior_generation", {}).get("overall")
    if prior:
        samples = int(prior["samples_per_query"])
        success_key = f"success_at_{samples}"
        coverage_key = f"coverage_at_{samples}"
        diversity_key = f"diversity_at_{samples}"
        print("\nPrior generation")
        print(f"  valid sample rate: {prior['valid_sample_rate']:.2%}")
        print(f"  {success_key}: {prior[success_key]:.2%}")
        print(
            f"  mean valid candidates/query: "
            f"{prior['mean_valid_count_per_query']:.2f}"
        )
        print(
            f"  coverage@{samples}: "
            f"{prior[coverage_key]:.2%}"
        )
        print(
            f"  diversity@{samples}: "
            f"{prior[diversity_key]:.4f}"
        )


def method_display_name(result: dict) -> str:
    if result.get("method_name"):
        name = str(result["method_name"])
        return METHOD_NAME_ALIASES.get(name, name)
    if (
        result.get("model_family") == "direct_mlp"
        and result.get("scale_mode") == "gnn"
    ):
        return "CVAE -> MLP"
    if result.get("model_family") == "direct_mlp":
        return "CVAE+Scale GNN -> MLP"
    if result.get("scale_mode") == "gnn":
        return "Full generator"
    if result.get("scale_mode") == "mlp":
        return "Scale GNN -> MLP"
    return str(result.get("model_family", "Model"))


def format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(values: list[str]) -> str:
        cells = []
        for index, value in enumerate(values):
            if index == 0:
                cells.append(value.ljust(widths[index]))
            else:
                cells.append(value.rjust(widths[index]))
        return "| " + " | ".join(cells) + " |"

    separator = [
        ":" + "-" * max(widths[0] - 1, 2)
    ] + [
        "-" * max(width - 1, 2) + ":"
        for width in widths[1:]
    ]
    return "\n".join(
        [
            format_row(headers),
            "| " + " | ".join(separator) + " |",
            *(format_row(row) for row in rows),
        ]
    )


def print_paper_table(result: dict) -> None:
    reconstruction = result.get("reconstruction")
    prior = result.get("prior_generation", {}).get("overall")
    if not reconstruction or not prior:
        return

    samples = int(prior["samples_per_query"])
    scale = scale_mae_mm(reconstruction["scale_end_to_end"])
    headers = [
        "Method",
        "Orientation (rad) ↓",
        "Location (mm) ↓",
        "Ql (rad) ↓",
        "w x/y/z (mm) ↓",
        "Achievement Rate ↑",
        "Valid Rate ↑",
        f"Coverage@{samples} ↑",
        f"Diversity@{samples} ↑",
    ]
    values = [
        method_display_name(result),
        f"{reconstruction['node_rotation_mean_rad']:.2f}",
        f"{reconstruction['node_position_mae_mm']:.2f}",
        f"{reconstruction['leg_angle_mae_rad']:.2f}",
        f"{scale['x']:.2f}/{scale['y']:.2f}/{scale['z']:.2f}",
        f"{prior[f'success_at_{samples}']:.2%}",
        f"{prior['valid_sample_rate']:.2%}",
        f"{prior[f'coverage_at_{samples}']:.2%}",
        f"{prior[f'diversity_at_{samples}']:.2f}",
    ]
    print("\nPaper table")
    print(format_markdown_table(headers, [values]))


def main() -> None:
    args = parse_args()
    if args.skip_reconstruction and args.skip_prior:
        raise ValueError("At least one evaluation section must be enabled")
    if args.samples_per_query <= 0:
        raise ValueError("--samples-per-query must be positive")
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be positive")
    if args.coverage_threshold <= 0.0:
        raise ValueError("--coverage-threshold must be positive")

    set_seed(args.seed)
    device = select_device(args.device)
    model, checkpoint = load_generator(args.model, device)
    decoder = make_decoder(args.graph_imputation, checkpoint, device)

    result = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": str(args.model),
        "model_family": getattr(model, "model_family", "conditional_structure_vae"),
        "scale_mode": model.scale_mode,
        "method_name": args.method_name,
        "validation_pairs": str(args.val_pairs),
        "device": str(device),
        "thresholds": decoder.thresholds.to_dict(),
    }
    timing = {}
    if not args.skip_reconstruction:
        reconstruction, timing["validation_reconstruction"] = evaluate_reconstruction(
            model, decoder, args, device
        )
        result["reconstruction"] = reconstruction
    if not args.skip_prior:
        prior_generation, timing["prior_generation"] = evaluate_prior(
            model, decoder, args, device
        )
        result["prior_generation"] = prior_generation
    if timing:
        result["timing"] = timing

    output_path = (
        Path(args.output)
        if args.output
        else Path("outputs")
        / f"validation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print_timing(timing)
    print_summary(result)
    print_paper_table(result)
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
