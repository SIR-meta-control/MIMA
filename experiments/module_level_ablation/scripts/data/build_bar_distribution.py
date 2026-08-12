#!/usr/bin/env python3
"""
Build soft labels for the bar/subspace classifier.

Input:
  datasets/processed_dataset/structures.jsonl
  datasets/processed_dataset/query_groups.jsonl

Output:
  datasets/processed_dataset/bar_distribution.jsonl
  datasets/processed_dataset/bar_distribution_stats.json

Each row maps one vreq to a distribution over:
  v=0 -> 4-bar
  v=1 -> 8-bar
  v=2 -> 6-bar
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


V_TO_BAR = {
    0: "4-bar",
    1: "8-bar",
    2: "6-bar",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build p(v | vreq) soft labels from processed query groups."
    )
    parser.add_argument(
        "--structures",
        default="datasets/processed_dataset/structures.jsonl",
        help="Path to structures.jsonl.",
    )
    parser.add_argument(
        "--query-groups",
        default="datasets/processed_dataset/query_groups.jsonl",
        help="Path to query_groups.jsonl.",
    )
    parser.add_argument(
        "--output",
        default="datasets/processed_dataset/bar_distribution.jsonl",
        help="Output JSONL path for bar distribution labels.",
    )
    parser.add_argument(
        "--stats-output",
        default="datasets/processed_dataset/bar_distribution_stats.json",
        help="Output JSON path for summary stats.",
    )
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


def load_structure_v_map(structures_path: Path) -> dict[int, int]:
    structure_to_v = {}
    for row in read_jsonl(structures_path):
        structure_to_v[int(row["structure_id"])] = int(row["v"])
    return structure_to_v


def make_distribution_rows(
    query_groups_path: Path, structure_to_v: dict[int, int]
) -> list[dict]:
    rows = []

    for query in read_jsonl(query_groups_path):
        v_counts = [0, 0, 0]
        missing_ids = []

        for structure_id in query["positive_structure_ids"]:
            v = structure_to_v.get(int(structure_id))
            if v is None:
                missing_ids.append(structure_id)
                continue
            v_counts[v] += 1

        total = sum(v_counts)
        if total == 0:
            v_probs = [0.0, 0.0, 0.0]
        else:
            v_probs = [count / total for count in v_counts]

        rows.append(
            {
                "query_id": query["query_id"],
                "task": query["task"],
                "vreq": query["vreq"],
                "v_counts": v_counts,
                "v_probs": v_probs,
                "bar_counts": {
                    V_TO_BAR[v]: v_counts[v] for v in range(len(v_counts))
                },
                "num_positives": total,
                "missing_structure_ids": missing_ids,
            }
        )

    return rows


def summarize(rows: list[dict]) -> dict:
    task_query_counts = Counter(row["task"] for row in rows)
    task_positive_counts = defaultdict(int)
    hard_label_counts = Counter()

    for row in rows:
        task_positive_counts[row["task"]] += row["num_positives"]
        if row["num_positives"] > 0:
            hard_label = max(range(3), key=lambda idx: row["v_probs"][idx])
            hard_label_counts[V_TO_BAR[hard_label]] += 1

    return {
        "num_rows": len(rows),
        "v_mapping": V_TO_BAR,
        "label_format": {
            "input": ["x", "y", "z", "load", "inspect", "pack"],
            "target": ["P(4-bar)", "P(8-bar)", "P(6-bar)"],
        },
        "task_query_counts": dict(task_query_counts),
        "task_positive_counts": dict(task_positive_counts),
        "hard_label_query_counts": dict(hard_label_counts),
    }


def main() -> None:
    args = parse_args()

    structures_path = Path(args.structures)
    query_groups_path = Path(args.query_groups)
    output_path = Path(args.output)
    stats_output_path = Path(args.stats_output)

    structure_to_v = load_structure_v_map(structures_path)
    rows = make_distribution_rows(query_groups_path, structure_to_v)
    stats = summarize(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    write_json(stats_output_path, stats)

    print(f"Wrote {len(rows)} bar distribution rows to {output_path}")
    print(f"Wrote stats to {stats_output_path}")


if __name__ == "__main__":
    main()
