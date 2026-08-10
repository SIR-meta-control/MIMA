#!/usr/bin/env python3
"""Train and export RF, DT, and GBT models from a prepared feature table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mima_vr.baseline_models import train_and_export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--source-manifest", action="append", default=[])
    args = parser.parse_args()
    written = train_and_export(
        pd.read_csv(args.feature_table),
        args.output_dir,
        random_state=args.random_state,
        source_manifests=args.source_manifest,
    )
    for key, path in written.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
