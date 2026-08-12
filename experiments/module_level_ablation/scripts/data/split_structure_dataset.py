#!/usr/bin/env python3
"""Create leakage-free structure-generator train/validation pair files."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StructureGroup:
    bar_type: str
    tasks: set[str] = field(default_factory=set)
    pair_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split pair rows by source structure, stratified by bar type and "
            "the structure's complete task signature."
        )
    )
    parser.add_argument(
        "--input",
        default="datasets/processed_dataset/train_pairs.jsonl",
        help="JSONL containing all positive (vreq, structure) pairs.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/processed_dataset/split_seed7",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing split in the output directory.",
    )
    return parser.parse_args()


def stable_seed(seed: int, key: tuple[str, tuple[str, ...]]) -> int:
    text = f"{seed}|{key[0]}|{','.join(key[1])}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def read_structure_groups(path: Path) -> dict[str, StructureGroup]:
    groups: dict[str, StructureGroup] = {}
    with path.open("r") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_path = str(row["source_path"])
            bar_type = str(row["bar_type"])
            task = str(row["task"])
            group = groups.setdefault(source_path, StructureGroup(bar_type=bar_type))
            if group.bar_type != bar_type:
                raise ValueError(
                    f"{source_path} has inconsistent bar types at line {line_number}"
                )
            group.tasks.add(task)
            group.pair_count += 1
    if not groups:
        raise ValueError(f"No pair rows found in {path}")
    return groups


def choose_validation_structures(
    groups: dict[str, StructureGroup],
    val_ratio: float,
    seed: int,
) -> tuple[set[str], dict[tuple[str, tuple[str, ...]], list[str]]]:
    strata: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for source_path, group in groups.items():
        key = (group.bar_type, tuple(sorted(group.tasks)))
        strata[key].append(source_path)

    validation: set[str] = set()
    for key, source_paths in sorted(strata.items()):
        rng = random.Random(stable_seed(seed, key))
        rng.shuffle(source_paths)
        val_count = int(round(len(source_paths) * val_ratio))
        if len(source_paths) > 1:
            val_count = min(max(val_count, 1), len(source_paths) - 1)
        else:
            val_count = int(val_ratio >= 0.5)
        validation.update(source_paths[:val_count])
    return validation, strata


def prepare_outputs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    paths = {
        "train": output_dir / "train_pairs.jsonl",
        "val": output_dir / "val_pairs.jsonl",
        "train_structures": output_dir / "train_structures.txt",
        "val_structures": output_dir / "val_structures.txt",
        "stats": output_dir / "split_stats.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Split outputs already exist: {names}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    return paths


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return {
        "|".join(map(str, key)) if isinstance(key, tuple) else str(key): int(value)
        for key, value in sorted(counter.items())
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    paths = prepare_outputs(output_dir, args.overwrite)
    groups = read_structure_groups(input_path)
    val_sources, strata = choose_validation_structures(
        groups,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_sources = set(groups) - val_sources

    pair_counts = {"train": 0, "val": 0}
    task_pair_counts = {"train": Counter(), "val": Counter()}
    bar_task_pair_counts = {"train": Counter(), "val": Counter()}
    query_keys = {"train": set(), "val": set()}
    task_query_keys = {"train": defaultdict(set), "val": defaultdict(set)}
    bar_task_query_keys = {"train": defaultdict(set), "val": defaultdict(set)}
    with (
        input_path.open("r") as source,
        paths["train"].open("w") as train_file,
        paths["val"].open("w") as val_file,
    ):
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            split = "val" if row["source_path"] in val_sources else "train"
            output = val_file if split == "val" else train_file
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
            pair_counts[split] += 1
            task_pair_counts[split][row["task"]] += 1
            bar_task_pair_counts[split][(row["bar_type"], row["task"])] += 1
            query_key = (
                tuple(float(value) for value in row["vreq"]),
                int(row["v"]),
            )
            query_keys[split].add(query_key)
            task_query_keys[split][row["task"]].add(query_key)
            bar_task_query_keys[split][(row["bar_type"], row["task"])].add(
                query_key
            )

    paths["train_structures"].write_text(
        "".join(f"{path}\n" for path in sorted(train_sources))
    )
    paths["val_structures"].write_text(
        "".join(f"{path}\n" for path in sorted(val_sources))
    )

    stratum_stats = {}
    for key, source_paths in sorted(strata.items()):
        name = f"{key[0]}|{'+'.join(key[1])}"
        val_count = sum(path in val_sources for path in source_paths)
        stratum_stats[name] = {
            "total_structures": len(source_paths),
            "train_structures": len(source_paths) - val_count,
            "val_structures": val_count,
        }

    stats = {
        "schema_version": 1,
        "source_pairs": str(input_path),
        "split_method": "grouped_by_source_path_stratified_by_bar_and_task_signature",
        "seed": args.seed,
        "requested_val_ratio": args.val_ratio,
        "num_pairs": {
            "total": pair_counts["train"] + pair_counts["val"],
            "train": pair_counts["train"],
            "val": pair_counts["val"],
            "actual_val_ratio": pair_counts["val"]
            / max(pair_counts["train"] + pair_counts["val"], 1),
        },
        "num_structures": {
            "total": len(groups),
            "train": len(train_sources),
            "val": len(val_sources),
            "actual_val_ratio": len(val_sources) / len(groups),
            "overlap": len(train_sources & val_sources),
        },
        "task_pair_counts": {
            split: counter_to_dict(counts)
            for split, counts in task_pair_counts.items()
        },
        "bar_task_pair_counts": {
            split: counter_to_dict(counts)
            for split, counts in bar_task_pair_counts.items()
        },
        "num_queries": {
            "total": len(query_keys["train"] | query_keys["val"]),
            "train": len(query_keys["train"]),
            "val": len(query_keys["val"]),
            "overlap": len(query_keys["train"] & query_keys["val"]),
        },
        "task_query_counts": {
            split: {task: len(keys) for task, keys in sorted(groups.items())}
            for split, groups in task_query_keys.items()
        },
        "bar_task_query_counts": {
            split: {
                "|".join(key): len(keys)
                for key, keys in sorted(groups.items())
            }
            for split, groups in bar_task_query_keys.items()
        },
        "structure_strata": stratum_stats,
    }
    paths["stats"].write_text(json.dumps(stats, indent=2) + "\n")

    print(
        f"Wrote {pair_counts['train']} train pairs from {len(train_sources)} structures"
    )
    print(f"Wrote {pair_counts['val']} val pairs from {len(val_sources)} structures")
    print(f"Structure overlap: {len(train_sources & val_sources)}")
    print(f"Split stats: {paths['stats']}")


if __name__ == "__main__":
    main()
