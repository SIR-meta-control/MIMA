#!/usr/bin/env python3
"""Run the packaged RF, DT, or GBT requirement-vector baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mima_vr.baseline_models import MODEL_FILENAMES, predict_from_sensor_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_FILENAMES), required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "baselines")
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--point-cloud", type=Path, required=True)
    parser.add_argument("--sample-id", default="sensor_sample")
    parser.add_argument("--scenario", default="open_ground")
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_from_sensor_paths(
        model_dir=args.model_dir,
        model_key=args.model,
        rgb_path=args.rgb,
        depth_path=args.depth,
        point_cloud_path=args.point_cloud,
        sample_id=args.sample_id,
        scenario=args.scenario,
    )
    print(json.dumps(result["v_r"] if args.compact else result, indent=2))


if __name__ == "__main__":
    main()
