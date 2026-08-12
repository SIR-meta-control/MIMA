#!/usr/bin/env python3
"""Audit graph and angle constraints on raw JSON structures."""

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

import torch

from gvae.robot.constraints import edge_angle_consistency_loss, type_geometry_loss
from gvae.robot.geometry import (
    EdgeAngleRecoverer,
    GraphImputationLayer,
    derive_edge_angle_static_transforms,
    pose_to_transform,
)
from gvae.data.structure_dataset import load_structure_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify raw structure constraints.")
    parser.add_argument("paths", nargs="+", help="Raw structure JSON files.")
    parser.add_argument("--graph-imputation", default="configs/graph_imputation.yaml")
    parser.add_argument(
        "--angle-reference",
        default="datasets/structure_config/4-bar/data_0000.json",
    )
    return parser.parse_args()


def read_edges(path: Path) -> list[list[float]]:
    with path.open("r") as f:
        return json.load(f)["edges"]


def infer_bar_type(path: Path) -> str:
    parts = set(path.parts)
    for bar_type in ("4-bar", "8-bar", "6-bar"):
        if bar_type in parts:
            return bar_type
    raise ValueError(f"Cannot infer bar type from {path}")


def main() -> None:
    args = parse_args()
    graph_layer = GraphImputationLayer(args.graph_imputation)
    static = derive_edge_angle_static_transforms(read_edges(Path(args.angle_reference)))
    angle_recoverer = EdgeAngleRecoverer(static)

    for path_text in args.paths:
        path = Path(path_text)
        data = load_structure_json(path)
        nodes = torch.as_tensor(data["nodes"], dtype=torch.float32).unsqueeze(0)
        edge_gt_t = pose_to_transform(
            torch.as_tensor(data["edge_pose"], dtype=torch.float32).unsqueeze(0)
        )
        out = graph_layer(nodes)
        recovered = angle_recoverer(edge_gt_t)
        angle_equal = edge_angle_consistency_loss(recovered).item()
        geometry = type_geometry_loss(edge_gt_t, [infer_bar_type(path)]).item()
        graph_edge_pose_mse = torch.mean(
            (out["edge_t"][:, :, :3, 3] - edge_gt_t[:, :, :3, 3]).pow(2)
        ).item()
        print(
            f"{path}: angle_equal={angle_equal:.6f} "
            f"geometry={geometry:.6f} graph_edge_pos_mse={graph_edge_pose_mse:.8f} "
            f"angles={[round(x, 6) for x in recovered[0].tolist()]}"
        )


if __name__ == "__main__":
    main()
