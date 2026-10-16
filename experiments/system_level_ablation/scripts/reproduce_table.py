#!/usr/bin/env python3
"""Reconstruct the paper table from archived row-level records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mima_ablation.config import load_json
from mima_ablation.reporting import reproduce_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "paper_protocol.json",
    )
    parser.add_argument(
        "--verify-hashes", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    audit = reproduce_table(
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        protocol=load_json(args.protocol),
        verify_hashes=args.verify_hashes,
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
