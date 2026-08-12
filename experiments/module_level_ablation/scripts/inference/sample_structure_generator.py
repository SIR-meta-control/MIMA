#!/usr/bin/env python3
"""Sample structures from the third-stage generator."""

from __future__ import annotations

import argparse
import json
import sys
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
from gvae.core.constants import BAR_ORDER
from gvae.networks.generator_loader import load_generator
from gvae.robot.geometry import (
    EdgeAngleRecoverer,
    GraphImputationLayer,
    transform_to_pose,
)
from gvae.core.training import select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample robot structures.")
    parser.add_argument("--model", required=True, help="Structure generator checkpoint.")
    parser.add_argument("--bar-classifier", default="ckpts/bar_classifier/bar_classifier.pt")
    parser.add_argument("--graph-imputation", default="configs/graph_imputation.yaml")
    parser.add_argument("--vreq", required=True)
    parser.add_argument("--v", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def choose_v(
    args: argparse.Namespace,
    vreq: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[int, list[float]]:
    if args.v is not None:
        probs = [0.0, 0.0, 0.0]
        probs[args.v] = 1.0
        return args.v, probs

    if hasattr(model, "predict_bar_probabilities"):
        with torch.no_grad():
            probs = (
                model.predict_bar_probabilities(
                    torch.as_tensor(
                        vreq,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0),
                    temperature=args.temperature,
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
    else:
        classifier, mean, std, _ = load_bar_classifier(args.bar_classifier)
        probs = predict_bar_probabilities(
            classifier,
            vreq,
            mean,
            std,
            temperature=args.temperature,
        )
    rng = np.random.default_rng(args.seed)
    return int(rng.choice(len(probs), p=probs)), probs.astype(float).tolist()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    vreq_np = parse_vreq(args.vreq)
    model, checkpoint = load_generator(Path(args.model), device)
    v, bar_probs = choose_v(args, vreq_np, model, device)
    graph_layer = GraphImputationLayer(args.graph_imputation).to(device)
    angle_recoverer = EdgeAngleRecoverer(
        np.asarray(checkpoint["edge_angle_static_transforms"], dtype=np.float32)
    ).to(device)

    vreq = torch.as_tensor(vreq_np, dtype=torch.float32, device=device).unsqueeze(0)
    bar_v = torch.tensor([v], dtype=torch.long, device=device)

    with torch.no_grad():
        pred = model.sample(
            vreq,
            bar_v,
            num_samples=args.num_samples,
            temperature=args.temperature,
        )
        graph = graph_layer(pred["nodes"])
        edge_angles = angle_recoverer(graph["edge_t"])
        edge_pose = transform_to_pose(graph["edge_t"])
        leg_base = transform_to_pose(graph["leg_base_t"])

    samples = []
    for i in range(args.num_samples):
        edges = torch.cat(
            [edge_angles[i, :, None], edge_pose[i]],
            dim=-1,
        )
        samples.append(
            {
                "v": v,
                "bar_type": BAR_ORDER[v],
                "vreq": vreq_np.astype(float).tolist(),
                "nodes": pred["nodes"][i].cpu().numpy().astype(float).tolist(),
                "edges": edges.cpu().numpy().astype(float).tolist(),
                "leg_base": leg_base[i].cpu().numpy().astype(float).tolist(),
                "leg_angle": pred["leg_angle"][i].cpu().numpy().astype(float).tolist(),
                "scale": pred["scale"][i].cpu().numpy().astype(float).tolist(),
            }
        )

    result = {
        "selected_v": v,
        "selected_bar_type": BAR_ORDER[v],
        "bar_probabilities": {
            str(i): {"bar_type": BAR_ORDER[i], "probability": float(bar_probs[i])}
            for i in range(3)
        },
        "samples": samples,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2)[:4000])


if __name__ == "__main__":
    main()
