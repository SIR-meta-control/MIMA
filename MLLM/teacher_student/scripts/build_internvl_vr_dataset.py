from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refactored_mllm.internvl.data_builder import build_internvl_vr_dataset


DEFAULT_MANIFEST_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "internvl_vr_data"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build InternVL image/depth/pointcloud JSONL fine-tune data with v_r labels."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        help="A MuJoCo sensor manifest.jsonl. Can be passed multiple times.",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT,
        help="Directory searched recursively for manifest.jsonl when --manifest is omitted.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", default="mujoco_vr")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decimals", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifests = args.manifest
    if not manifests:
        manifests = sorted(args.manifest_root.rglob("manifest.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"No manifest.jsonl found under {args.manifest_root}")

    result = build_internvl_vr_dataset(
        manifests,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        limit=args.limit,
        decimals=args.decimals,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
