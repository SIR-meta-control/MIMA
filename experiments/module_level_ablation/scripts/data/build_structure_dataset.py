#!/usr/bin/env python3
"""
Build a derived training dataset from raw robot structure JSON files.

Raw JSON files are kept immutable. This script creates query/structure pairs:

  vreq = [x, y, z, load, inspect, pack]

where x/y/z are upper size constraints and task bits select the subspace rule.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


BAR_TO_V = {
    "4-bar": 0,
    "8-bar": 1,
    "6-bar": 2,
}

BAR_ORDER = ["4-bar", "8-bar", "6-bar"]

TASK_BITS = {
    "none": [0, 0, 0],
    "load": [1, 0, 0],
    "inspect": [0, 1, 0],
    "pack": [0, 0, 1],
}

LOAD_Q2_RANGE = (1.39, 1.75)
PACK_TARGET_SCALE = (0.32, 0.38, 0.40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build structure metadata, vreq query groups, and train pairs."
    )
    parser.add_argument(
        "--input-dir",
        default="datasets/structure_config",
        help="Raw structure config directory with 4-bar/6-bar/8-bar subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/processed_dataset",
        help="Directory to write derived dataset files.",
    )
    parser.add_argument(
        "--xyz-margin",
        type=float,
        default=1.10,
        help="Generate xyz limits as ceil(scale * margin / round-step) * round-step.",
    )
    parser.add_argument(
        "--round-step",
        type=float,
        default=0.1,
        help="Quantization step for generated xyz limits.",
    )
    parser.add_argument(
        "--max-positives-per-query",
        type=int,
        default=0,
        help="Optional cap for positives per query group. 0 means keep all positives.",
    )
    parser.add_argument(
        "--pack-slack",
        type=float,
        default=0.05,
        help="Pack xyz limit is PACK_TARGET_SCALE + this slack.",
    )
    parser.add_argument(
        "--dedup-metadata-decimals",
        type=int,
        default=4,
        help=(
            "Deduplicate by (bar_type, rounded scale, rounded q2). "
            "Use -1 to disable. Four decimals is a good starting point."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_json(path: Path, obj: dict) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def round_float(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


def make_xyz_limit(scale: list[float], margin: float, step: float) -> list[float]:
    limits = []
    for value in scale:
        limited = math.ceil((value * margin) / step) * step
        limits.append(round_float(limited, 4))
    return limits


def in_range(value: float, bounds: tuple[float, float]) -> bool:
    lo, hi = bounds
    return lo <= value <= hi


def xyz_fits(scale: list[float], xyz_limit: tuple[float, float, float]) -> bool:
    return all(value <= limit + 1e-9 for value, limit in zip(scale, xyz_limit))


def pack_xyz_limit(pack_slack: float) -> tuple[float, float, float]:
    return tuple(round_float(value + pack_slack, 5) for value in PACK_TARGET_SCALE)


def task_ok(structure: dict, task: str, pack_limit: tuple[float, float, float]) -> bool:
    if task == "none":
        return True
    if task == "load":
        return structure["v"] == 2 and in_range(structure["q2"], LOAD_Q2_RANGE)
    if task == "inspect":
        return structure["v"] == 0
    if task == "pack":
        return structure["v"] == 0 and xyz_fits(structure["scale"], pack_limit)
    raise ValueError(f"Unknown task: {task}")


def scan_structures(
    input_dir: Path,
    xyz_margin: float,
    round_step: float,
    pack_limit: tuple[float, float, float],
    dedup_metadata_decimals: int = -1,
) -> tuple[list[dict], dict]:
    structures = []
    seen_metadata = set()
    skipped_duplicates = 0

    for bar_type in BAR_ORDER:
        bar_dir = input_dir / bar_type
        if not bar_dir.is_dir():
            continue

        for path in sorted(bar_dir.glob("*.json")):
            config = read_json(path)
            scale = [round_float(v, 5) for v in config["global"]["scale"]]
            q2 = round_float(config["edges"][2][0], 9)
            v = BAR_TO_V[bar_type]

            if dedup_metadata_decimals >= 0:
                dedup_key = (
                    bar_type,
                    *[round_float(value, dedup_metadata_decimals) for value in scale],
                    round_float(q2, dedup_metadata_decimals),
                )
                if dedup_key in seen_metadata:
                    skipped_duplicates += 1
                    continue
                seen_metadata.add(dedup_key)

            structure_id = len(structures)

            item = {
                "structure_id": structure_id,
                "source_path": str(path),
                "bar_type": bar_type,
                "v": v,
                "scale": scale,
                "xyz_min_limit": make_xyz_limit(scale, xyz_margin, round_step),
                "q2": q2,
            }
            item["task_ok"] = {
                task: task_ok(item, task, pack_limit)
                for task in TASK_BITS
                if task != "none"
            }
            structures.append(item)

    dedup_stats = {
        "dedup_metadata_decimals": dedup_metadata_decimals,
        "skipped_metadata_duplicates": skipped_duplicates,
    }
    return structures, dedup_stats


def make_vreq(xyz_limit: tuple[float, float, float], task: str) -> list[float | int]:
    return [*xyz_limit, *TASK_BITS[task]]


def build_query_groups(
    structures: list[dict],
    pack_limit: tuple[float, float, float],
    max_positives_per_query: int = 0,
) -> tuple[list[dict], list[dict]]:
    query_groups = []
    train_pairs = []

    by_task = {
        task: [s for s in structures if task_ok(s, task, pack_limit)]
        for task in TASK_BITS
    }

    for task in TASK_BITS:
        if task == "pack":
            xyz_grid = [pack_limit]
        else:
            xyz_grid = sorted({tuple(s["xyz_min_limit"]) for s in by_task[task]})

        for xyz_limit in xyz_grid:
            positives = [
                s
                for s in by_task[task]
                if xyz_fits(s["scale"], xyz_limit)
            ]
            if not positives:
                continue

            if max_positives_per_query > 0:
                positives = positives[:max_positives_per_query]

            query_id = len(query_groups)
            vreq = make_vreq(xyz_limit, task)
            positive_ids = [s["structure_id"] for s in positives]

            query_groups.append(
                {
                    "query_id": query_id,
                    "task": task,
                    "vreq": vreq,
                    "positive_structure_ids": positive_ids,
                    "num_positives": len(positive_ids),
                }
            )

            for s in positives:
                train_pairs.append(
                    {
                        "query_id": query_id,
                        "structure_id": s["structure_id"],
                        "source_path": s["source_path"],
                        "bar_type": s["bar_type"],
                        "v": s["v"],
                        "task": task,
                        "vreq": vreq,
                    }
                )

    return query_groups, train_pairs


def summarize(
    structures: list[dict],
    query_groups: list[dict],
    train_pairs: list[dict],
    pack_limit: tuple[float, float, float],
    pack_slack: float,
    dedup_stats: dict,
) -> dict:
    bar_counts = Counter(s["bar_type"] for s in structures)
    task_structure_counts = {
        task: sum(1 for s in structures if task_ok(s, task, pack_limit))
        for task in TASK_BITS
    }
    task_query_counts = Counter(q["task"] for q in query_groups)
    task_pair_counts = Counter(p["task"] for p in train_pairs)

    q2_values = [s["q2"] for s in structures if s["bar_type"] == "6-bar"]
    scale_by_bar = defaultdict(list)
    for s in structures:
        scale_by_bar[s["bar_type"]].append(s["scale"])

    scale_stats = {}
    for bar_type, values in scale_by_bar.items():
        cols = list(zip(*values))
        scale_stats[bar_type] = {
            "min": [round_float(min(col), 5) for col in cols],
            "max": [round_float(max(col), 5) for col in cols],
        }

    return {
        "num_structures": len(structures),
        "num_query_groups": len(query_groups),
        "num_train_pairs": len(train_pairs),
        "bar_counts": dict(bar_counts),
        "task_structure_counts": task_structure_counts,
        "task_query_counts": dict(task_query_counts),
        "task_pair_counts": dict(task_pair_counts),
        "dedup": dedup_stats,
        "v_mapping": BAR_TO_V,
        "vreq_format": ["x", "y", "z", "load", "inspect", "pack"],
        "rules": {
            "all_tasks": "A structure is a positive example only when scale <= vreq[:3].",
            "load": {
                "v": 2,
                "q2_field": "edges[2][0]",
                "q2_range": list(LOAD_Q2_RANGE),
            },
            "inspect": {"v": 0},
            "pack": {
                "v": 0,
                "target_scale": list(PACK_TARGET_SCALE),
                "slack": pack_slack,
                "xyz_limit": list(pack_limit),
                "constraint": "scale <= xyz_limit",
            },
            "none": "Only xyz constraints are applied.",
        },
        "six_bar_q2_min_max": [
            round_float(min(q2_values), 9),
            round_float(max(q2_values), 9),
        ],
        "scale_stats": scale_stats,
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pack_limit = pack_xyz_limit(args.pack_slack)
    structures, dedup_stats = scan_structures(
        input_dir,
        args.xyz_margin,
        args.round_step,
        pack_limit,
        args.dedup_metadata_decimals,
    )
    query_groups, train_pairs = build_query_groups(
        structures, pack_limit, args.max_positives_per_query
    )
    stats = summarize(
        structures,
        query_groups,
        train_pairs,
        pack_limit,
        args.pack_slack,
        dedup_stats,
    )

    write_jsonl(output_dir / "structures.jsonl", structures)
    write_jsonl(output_dir / "query_groups.jsonl", query_groups)
    write_jsonl(output_dir / "train_pairs.jsonl", train_pairs)
    write_json(output_dir / "stats.json", stats)

    print(f"Wrote {len(structures)} structures")
    print(f"Wrote {len(query_groups)} query groups")
    print(f"Wrote {len(train_pairs)} train pairs")
    print(f"Output directory: {output_dir}")
    if dedup_stats["dedup_metadata_decimals"] >= 0:
        print(
            "Skipped "
            f"{dedup_stats['skipped_metadata_duplicates']} metadata duplicates"
        )

    for task, count in stats["task_structure_counts"].items():
        if count == 0:
            print(f"Warning: task '{task}' has no matching structures.")


if __name__ == "__main__":
    main()
