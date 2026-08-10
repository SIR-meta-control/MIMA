#!/usr/bin/env python3
"""Call a Full-MIMA teacher or MLLM-distilled requirement-vector service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mima_vr.service_client import RequirementVectorServiceClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--backend", choices=("internvl", "student"), required=True)
    parser.add_argument("--expected-model")
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--point-cloud", type=Path, required=True)
    parser.add_argument("--sample-id", default="sensor_sample")
    parser.add_argument("--scenario", default="open_ground")
    parser.add_argument("--task-command", default="")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = RequirementVectorServiceClient(
        base_url=args.service_url,
        backend=args.backend,
        timeout_s=args.timeout_s,
    )
    result = client.predict_from_sensor_paths(
        rgb_path=args.rgb,
        depth_path=args.depth,
        point_cloud_path=args.point_cloud,
        sample_id=args.sample_id,
        scenario=args.scenario,
        task_command=args.task_command,
        model_key="full_mima_teacher" if args.backend == "internvl" else "mllm_distilled",
    )
    if args.expected_model and result["model"] != args.expected_model:
        raise RuntimeError(
            f"service returned model {result['model']!r}; expected {args.expected_model!r}"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
