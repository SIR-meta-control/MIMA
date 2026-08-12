#!/usr/bin/env python3
"""
Sample bar type v from a trained p(v | vreq) classifier.

Example:
  python scripts/inference/sample_bar_type.py --vreq "0.6,0.6,0.6,0,0,0" --num-samples 8
"""

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

from gvae.networks.bar_classifier import (
    load_bar_classifier,
    parse_vreq,
    predict_bar_probabilities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample bar type from vreq.")
    parser.add_argument("--model", default="ckpts/bar_classifier/bar_classifier.pt")
    parser.add_argument(
        "--vreq",
        required=True,
        help='Six values as JSON list or comma string, e.g. "[0.6,0.6,0.6,0,0,0]".',
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    vreq = parse_vreq(args.vreq)
    model, mean, std, checkpoint = load_bar_classifier(model_path)
    probs = predict_bar_probabilities(model, vreq, mean, std, args.temperature)

    v_mapping = checkpoint.get(
        "v_mapping",
        {"0": "4-bar", "1": "8-bar", "2": "6-bar"},
    )
    rng = np.random.default_rng(args.seed)
    sampled = rng.choice(len(probs), size=args.num_samples, p=probs)

    result = {
        "vreq": vreq.astype(float).tolist(),
        "probabilities": {
            str(i): {
                "bar_type": v_mapping[str(i)],
                "probability": float(probs[i]),
            }
            for i in range(len(probs))
        },
        "samples": [
            {"v": int(v), "bar_type": v_mapping[str(int(v))]} for v in sampled.tolist()
        ],
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"vreq: {result['vreq']}")
    print("probabilities:")
    for i in range(len(probs)):
        print(f"  v={i} {v_mapping[str(i)]:>5s}: {probs[i]:.6f}")
    print("samples:")
    print("  " + ", ".join(f"{item['v']}:{item['bar_type']}" for item in result["samples"]))


if __name__ == "__main__":
    main()
