from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refactored_mllm.internvl.inference import (
    DEFAULT_INTERNVL_SOURCE,
    DEFAULT_FINETUNED_CHECKPOINT,
    InternVLVRPredictor,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict v_r with an authorized paper-specific InternVL checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_FINETUNED_CHECKPOINT)
    parser.add_argument("--internvl-source", type=Path, default=DEFAULT_INTERNVL_SOURCE)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--point-cloud", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    predictor = InternVLVRPredictor(
        checkpoint=args.checkpoint,
        internvl_source=args.internvl_source,
        device=args.device,
    )
    prediction = predictor.predict(
        rgb_path=args.rgb,
        depth_path=args.depth,
        point_cloud_path=args.point_cloud,
    )
    output = prediction["v_r"] if args.compact else prediction
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
