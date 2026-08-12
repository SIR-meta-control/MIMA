#!/usr/bin/env python3
"""Generate, hard-filter, score, and diversify robot structure candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

from gvae.networks.bar_classifier import (
    load_bar_classifier,
    parse_vreq,
    predict_bar_probabilities,
)
from gvae.core.constants import BAR_ORDER, BAR_TO_V
from gvae.networks.generator_loader import load_generator
from gvae.robot.geometry import EdgeAngleRecoverer, GraphImputationLayer
from gvae.robot.inference import (
    ConstraintThresholds,
    HardConstraintDecoder,
    eligible_bar_indices,
    select_diverse_top_k,
)
from gvae.core.training import select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate valid, ranked, diverse robot structures."
    )
    parser.add_argument("--model", required=True, help="Structure generator checkpoint.")
    parser.add_argument(
        "--bar-classifier",
        default="ckpts/bar_classifier/bar_classifier.pt",
    )
    parser.add_argument("--graph-imputation", default="configs/graph_imputation.yaml")
    parser.add_argument("--vreq", required=True)
    parser.add_argument(
        "--bar-types",
        default="auto",
        help='Use "auto", "all", or comma-separated names such as "4-bar,6-bar".',
    )
    parser.add_argument("--samples-per-bar", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--diversity-threshold", type=float, default=0.02)
    parser.add_argument(
        "--min-per-bar",
        type=int,
        default=1,
        help="Minimum valid candidates retained per sampled bar type when Top-K permits.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default=None)

    defaults = ConstraintThresholds()
    parser.add_argument(
        "--max-quaternion-error",
        type=float,
        default=defaults.quaternion_norm_max,
    )
    parser.add_argument(
        "--max-spacing-rmse",
        type=float,
        default=defaults.spacing_rmse_max,
    )
    parser.add_argument(
        "--max-spacing-error",
        type=float,
        default=defaults.spacing_error_max,
    )
    parser.add_argument(
        "--max-geometry-error",
        type=float,
        default=defaults.geometry_error_max,
    )
    parser.add_argument(
        "--max-angle-rmse",
        type=float,
        default=defaults.angle_rmse_max,
    )
    parser.add_argument(
        "--max-size-excess",
        type=float,
        default=defaults.size_excess_max,
    )
    parser.add_argument(
        "--max-task-excess",
        type=float,
        default=defaults.task_excess_max,
    )
    return parser.parse_args()


def choose_bar_indices(text: str, vreq: np.ndarray) -> list[int]:
    value = text.strip().lower()
    if value == "auto":
        return eligible_bar_indices(vreq)
    if value == "all":
        return list(range(len(BAR_ORDER)))

    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in BAR_TO_V]
    if unknown:
        raise ValueError(f"Unknown bar types: {unknown}")
    if not names:
        raise ValueError("--bar-types must not be empty")
    return list(dict.fromkeys(BAR_TO_V[name] for name in names))


def tensor_row(values: torch.Tensor, index: int):
    return values[index].detach().cpu().numpy().astype(float).tolist()


def build_candidate(
    decoded: dict,
    index: int,
    vreq: np.ndarray,
    v: int,
    bar_probability: float,
) -> dict:
    edge_angles = decoded["edge_angles"][index, :, None]
    edges = torch.cat([edge_angles, decoded["edge_pose"][index]], dim=-1)
    checks = {
        name: bool(values[index].detach().cpu().item())
        for name, values in decoded["checks"].items()
    }
    metrics = {
        name: float(values[index].detach().cpu().item())
        for name, values in decoded["metrics"].items()
    }
    scores = {
        name: float(values[index].detach().cpu().item())
        for name, values in decoded["scores"].items()
    }

    return {
        "rank": None,
        "v": v,
        "bar_type": BAR_ORDER[v],
        "bar_probability": bar_probability,
        "valid": bool(decoded["valid"][index].detach().cpu().item()),
        "score": scores["overall"],
        "confidence": scores["confidence"],
        "scores": scores,
        "constraints": {
            "checks": checks,
            "metrics": metrics,
        },
        "vreq": vreq.astype(float).tolist(),
        "structure": {
            "nodes": tensor_row(decoded["nodes"], index),
            "edges": edges.detach().cpu().numpy().astype(float).tolist(),
            "global": {
                "scale": tensor_row(decoded["scale"], index),
                "leg_base": tensor_row(decoded["leg_base"], index),
                "leg_angle": tensor_row(decoded["leg_angle"], index),
            },
        },
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_bar <= 0:
        raise ValueError("--samples-per-bar must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.min_per_bar < 0:
        raise ValueError("--min-per-bar must be non-negative")

    set_seed(args.seed)
    device = select_device(args.device)
    vreq_np = parse_vreq(args.vreq)
    hard_eligible_bar_indices = eligible_bar_indices(vreq_np)
    bar_indices = choose_bar_indices(args.bar_types, vreq_np)

    model, checkpoint = load_generator(args.model, device)
    if hasattr(model, "predict_bar_probabilities"):
        with torch.no_grad():
            raw_bar_probabilities = (
                model.predict_bar_probabilities(
                    torch.as_tensor(
                        vreq_np,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0),
                    temperature=args.temperature,
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
        bar_probability_source = "internal_vreq_mlp"
    else:
        classifier, mean, std, _ = load_bar_classifier(args.bar_classifier)
        raw_bar_probabilities = predict_bar_probabilities(
            classifier,
            vreq_np,
            mean,
            std,
            temperature=args.temperature,
        )
        bar_probability_source = str(args.bar_classifier)
    bar_probabilities = raw_bar_probabilities.copy()
    eligibility_mask = np.zeros_like(bar_probabilities)
    eligibility_mask[hard_eligible_bar_indices] = 1.0
    bar_probabilities *= eligibility_mask
    probability_sum = float(bar_probabilities.sum())
    if probability_sum > 0.0:
        bar_probabilities /= probability_sum
    else:
        bar_probabilities[hard_eligible_bar_indices] = (
            1.0 / len(hard_eligible_bar_indices)
        )

    graph_layer = GraphImputationLayer(args.graph_imputation).to(device)
    angle_recoverer = EdgeAngleRecoverer(
        np.asarray(checkpoint["edge_angle_static_transforms"], dtype=np.float32)
    ).to(device)
    thresholds = ConstraintThresholds(
        quaternion_norm_max=args.max_quaternion_error,
        spacing_rmse_max=args.max_spacing_rmse,
        spacing_error_max=args.max_spacing_error,
        geometry_error_max=args.max_geometry_error,
        angle_rmse_max=args.max_angle_rmse,
        size_excess_max=args.max_size_excess,
        task_excess_max=args.max_task_excess,
    )
    decoder = HardConstraintDecoder(graph_layer, angle_recoverer, thresholds)

    all_candidates = []
    generated_by_bar = Counter()
    valid_by_bar = Counter()
    rejection_counts = Counter()

    with torch.no_grad():
        for v in bar_indices:
            vreq = torch.as_tensor(
                vreq_np,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            bar_v = torch.tensor([v], dtype=torch.long, device=device)
            pred = model.sample(
                vreq,
                bar_v,
                num_samples=args.samples_per_bar,
                temperature=args.temperature,
            )

            repeated_vreq = vreq.repeat(args.samples_per_bar, 1)
            repeated_bar_v = bar_v.repeat(args.samples_per_bar)
            probability = float(bar_probabilities[v])
            repeated_probability = torch.full(
                (args.samples_per_bar,),
                probability,
                dtype=torch.float32,
                device=device,
            )
            decoded = decoder.decode_and_evaluate(
                pred["nodes"],
                pred["leg_angle"],
                pred["scale"],
                repeated_vreq,
                repeated_bar_v,
                repeated_probability,
            )

            for index in range(args.samples_per_bar):
                candidate = build_candidate(
                    decoded,
                    index,
                    vreq_np,
                    v,
                    probability,
                )
                generated_by_bar[BAR_ORDER[v]] += 1
                if candidate["valid"]:
                    valid_by_bar[BAR_ORDER[v]] += 1
                    all_candidates.append(candidate)
                else:
                    for name, passed in candidate["constraints"]["checks"].items():
                        if not passed:
                            rejection_counts[name] += 1

    selected = select_diverse_top_k(
        all_candidates,
        top_k=args.top_k,
        min_distance=args.diversity_threshold,
        min_per_bar=args.min_per_bar,
    )

    result = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "vreq": vreq_np.astype(float).tolist(),
        "model": str(args.model),
        "bar_classifier": bar_probability_source,
        "raw_bar_probabilities": {
            BAR_ORDER[v]: float(raw_bar_probabilities[v])
            for v in range(len(BAR_ORDER))
        },
        "bar_probabilities_after_hard_rules": {
            BAR_ORDER[v]: float(bar_probabilities[v])
            for v in range(len(BAR_ORDER))
        },
        "sampled_bar_types": [BAR_ORDER[v] for v in bar_indices],
        "thresholds": thresholds.to_dict(),
        "summary": {
            "generated": int(sum(generated_by_bar.values())),
            "valid": len(all_candidates),
            "returned": len(selected),
            "generated_by_bar": dict(generated_by_bar),
            "valid_by_bar": dict(valid_by_bar),
            "rejection_counts": dict(rejection_counts),
        },
        "candidates": selected,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("outputs") / f"structure_candidates_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(
        f"Generated {result['summary']['generated']} candidates, "
        f"accepted {result['summary']['valid']}, "
        f"returned {result['summary']['returned']}."
    )
    print(f"Rejections: {result['summary']['rejection_counts']}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
