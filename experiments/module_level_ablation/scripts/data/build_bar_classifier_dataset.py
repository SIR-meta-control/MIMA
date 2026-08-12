#!/usr/bin/env python3
"""
Generate a dense bar-classifier dataset.

The sparse query groups are useful for auditing, but too small for training a
classifier. This script samples many vreqs and computes soft labels by applying
the same feasibility rules to the structure metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


TASK_BITS = {
    "none": [0, 0, 0],
    "load": [1, 0, 0],
    "inspect": [0, 1, 0],
    "pack": [0, 0, 1],
}

TASKS = ["none", "load", "inspect", "pack"]
LOAD_Q2_RANGE = (1.39, 1.75)
PACK_TARGET_SCALE = np.array([0.32, 0.38, 0.40], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dense vreq -> p(v) labels for the bar classifier."
    )
    parser.add_argument(
        "--structures",
        default="datasets/processed_dataset/structures.jsonl",
        help="Path to structures.jsonl.",
    )
    parser.add_argument(
        "--query-groups",
        default=None,
        help="Optional query_groups.jsonl to seed exact audited queries.",
    )
    parser.add_argument(
        "--output",
        default="datasets/processed_dataset/bar_classifier_dataset.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--stats-output",
        default="datasets/processed_dataset/bar_classifier_dataset_stats.json",
        help="Output JSON stats path.",
    )
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--round-step", type=float, default=0.02)
    parser.add_argument("--pack-slack", type=float, default=0.05)
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_json(path: Path, obj: dict) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def round_float(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


def quantize_up(values: np.ndarray, step: float) -> np.ndarray:
    return np.ceil(values / step) * step


def pack_limit(pack_slack: float) -> np.ndarray:
    return PACK_TARGET_SCALE + pack_slack


def load_structures(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scales = []
    vs = []
    q2s = []

    for row in read_jsonl(path):
        scales.append(row["scale"])
        vs.append(int(row["v"]))
        q2s.append(float(row["q2"]))

    return (
        np.asarray(scales, dtype=np.float64),
        np.asarray(vs, dtype=np.int64),
        np.asarray(q2s, dtype=np.float64),
    )


def make_task_masks(
    scales: np.ndarray, vs: np.ndarray, q2s: np.ndarray, pack_xyz: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "none": np.ones(len(vs), dtype=bool),
        "load": (vs == 2) & (q2s >= LOAD_Q2_RANGE[0]) & (q2s <= LOAD_Q2_RANGE[1]),
        "inspect": vs == 0,
        "pack": (vs == 0) & np.all(scales <= pack_xyz[None, :] + 1e-9, axis=1),
    }


def compute_label(
    xyz: np.ndarray,
    task: str,
    scales: np.ndarray,
    vs: np.ndarray,
    task_masks: dict[str, np.ndarray],
) -> tuple[list[int], list[float]]:
    fits = np.all(scales <= xyz[None, :] + 1e-9, axis=1)
    mask = task_masks[task] & fits
    counts = np.bincount(vs[mask], minlength=3)[:3].astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return counts.tolist(), [0.0, 0.0, 0.0]
    return counts.tolist(), (counts / total).astype(float).tolist()


def make_vreq(xyz: np.ndarray, task: str) -> list[float | int]:
    xyz_values = [round_float(value, 4) for value in xyz]
    return [*xyz_values, *TASK_BITS[task]]


def vreq_key(vreq: list[float | int]) -> tuple:
    return tuple(round(float(value), 4) for value in vreq[:3]) + tuple(vreq[3:])


def add_row(
    rows: list[dict],
    seen: set[tuple],
    vreq: list[float | int],
    task: str,
    scales: np.ndarray,
    vs: np.ndarray,
    task_masks: dict[str, np.ndarray],
) -> None:
    key = vreq_key(vreq)
    if key in seen:
        return

    counts, probs = compute_label(
        np.asarray(vreq[:3], dtype=np.float64), task, scales, vs, task_masks
    )
    if sum(counts) == 0:
        return

    seen.add(key)
    rows.append(
        {
            "sample_id": len(rows),
            "task": task,
            "vreq": vreq,
            "v_counts": counts,
            "v_probs": probs,
            "num_positives": int(sum(counts)),
        }
    )


def seed_from_query_groups(
    rows: list[dict],
    seen: set[tuple],
    query_groups_path: Path,
    scales: np.ndarray,
    vs: np.ndarray,
    task_masks: dict[str, np.ndarray],
) -> None:
    if not query_groups_path or not query_groups_path.exists():
        return

    for row in read_jsonl(query_groups_path):
        add_row(rows, seen, row["vreq"], row["task"], scales, vs, task_masks)


def sample_xyz_for_task(
    rng: np.random.Generator,
    task: str,
    scales: np.ndarray,
    task_masks: dict[str, np.ndarray],
    pack_xyz: np.ndarray,
    round_step: float,
) -> np.ndarray:
    eligible = np.flatnonzero(task_masks[task])
    if len(eligible) == 0:
        raise ValueError(f"No eligible structures for task {task}")

    base = scales[int(rng.choice(eligible))]
    if task == "pack":
        jitter = rng.uniform(0.0, 0.08, size=3)
        xyz = np.maximum(base, pack_xyz) + jitter
    else:
        multiplier = rng.uniform(1.0, 1.35, size=3)
        xyz = base * multiplier

    return quantize_up(xyz, round_step)


def build_rows(
    scales: np.ndarray,
    vs: np.ndarray,
    q2s: np.ndarray,
    query_groups_path: Path | None,
    num_samples: int,
    seed: int,
    round_step: float,
    pack_slack: float,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    pack_xyz = pack_limit(pack_slack)
    task_masks = make_task_masks(scales, vs, q2s, pack_xyz)
    rows: list[dict] = []
    seen: set[tuple] = set()

    if query_groups_path is not None:
        seed_from_query_groups(rows, seen, query_groups_path, scales, vs, task_masks)

    task_probs = np.asarray([0.45, 0.20, 0.20, 0.15], dtype=np.float64)
    max_attempts = num_samples * 30
    attempts = 0

    while len(rows) < num_samples and attempts < max_attempts:
        attempts += 1
        task = TASKS[int(rng.choice(len(TASKS), p=task_probs))]
        if not np.any(task_masks[task]):
            continue

        xyz = sample_xyz_for_task(rng, task, scales, task_masks, pack_xyz, round_step)
        vreq = make_vreq(xyz, task)
        add_row(rows, seen, vreq, task, scales, vs, task_masks)

    return rows


def summarize(rows: list[dict]) -> dict:
    task_counts = Counter(row["task"] for row in rows)
    hard_counts = Counter()
    mixed_rows = 0
    positive_counts = defaultdict(int)

    for row in rows:
        positive_counts[row["task"]] += row["num_positives"]
        nonzero = sum(1 for prob in row["v_probs"] if prob > 0.0)
        if nonzero > 1:
            mixed_rows += 1
        hard_counts[str(int(np.argmax(row["v_probs"])))] += 1

    return {
        "num_rows": len(rows),
        "task_counts": dict(task_counts),
        "task_positive_counts": dict(positive_counts),
        "hard_label_counts": dict(hard_counts),
        "mixed_distribution_rows": mixed_rows,
        "v_mapping": {"0": "4-bar", "1": "8-bar", "2": "6-bar"},
        "format": {
            "input": ["x", "y", "z", "load", "inspect", "pack"],
            "target": ["P(4-bar)", "P(8-bar)", "P(6-bar)"],
        },
    }


def main() -> None:
    args = parse_args()

    structures_path = Path(args.structures)
    query_groups_path = Path(args.query_groups) if args.query_groups else None
    output_path = Path(args.output)
    stats_output_path = Path(args.stats_output)

    scales, vs, q2s = load_structures(structures_path)
    rows = build_rows(
        scales,
        vs,
        q2s,
        query_groups_path,
        args.num_samples,
        args.seed,
        args.round_step,
        args.pack_slack,
    )
    stats = summarize(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    write_json(stats_output_path, stats)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Wrote stats to {stats_output_path}")


if __name__ == "__main__":
    main()
