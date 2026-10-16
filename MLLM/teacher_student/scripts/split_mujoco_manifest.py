from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Literal, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifest.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "manifest_splits"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/test splits for MuJoCo sensor manifests."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=None,
        help="Input manifest.jsonl. Can be passed multiple times.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default="mujoco_hash_80_20")
    parser.add_argument(
        "--strategy",
        choices=("hash", "band", "grid-block"),
        default="hash",
        help="hash splits samples; band holds out full values of one column; grid-block holds out full wp_m x hp_m cells.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", default="refactored_MLLM_vr_split_v1")
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        type=Path,
        default=None,
        help="Manifest whose sample IDs must not appear in this split.",
    )
    parser.add_argument("--band-column", default="wp_m")
    parser.add_argument("--band-start", type=int, default=None)
    parser.add_argument("--band-count", type=int, default=None)
    parser.add_argument("--block-wp-start", type=int, default=None)
    parser.add_argument("--block-wp-count", type=int, default=None)
    parser.add_argument("--block-hp-start", type=int, default=None)
    parser.add_argument("--block-hp-count", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifests = args.manifest or [DEFAULT_MANIFEST]
    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError("--test-ratio must be between 0 and 1")

    rows = _load_rows(manifests)
    excluded_ids = _load_sample_ids(args.exclude_manifest or [])
    rows = [row for row in rows if str(row["sample_id"]) not in excluded_ids]
    if len(rows) < 2:
        raise ValueError("Need at least two records to split")

    test_ids, split_details = _select_test_ids(
        rows,
        strategy=args.strategy,
        test_ratio=args.test_ratio,
        seed=args.seed,
        band_column=args.band_column,
        band_start=args.band_start,
        band_count=args.band_count,
        block_wp_start=args.block_wp_start,
        block_wp_count=args.block_wp_count,
        block_hp_start=args.block_hp_start,
        block_hp_count=args.block_hp_count,
    )

    output_root = args.output_root / args.name
    train_path = output_root / "train" / "manifest.jsonl"
    test_path = output_root / "test" / "manifest.jsonl"
    train_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        split = "test" if row["sample_id"] in test_ids else "train"
        record = dict(row["record"])
        record["split"] = split
        if split == "test":
            test_rows.append(record)
        else:
            train_rows.append(record)

    _write_jsonl(train_path, train_rows)
    _write_jsonl(test_path, test_rows)
    summary = {
        "name": args.name,
        "strategy": args.strategy,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "n_total": len(rows),
        "n_excluded": len(excluded_ids),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "train_manifest": str(train_path),
        "test_manifest": str(test_path),
        "details": split_details,
        "exclude_manifests": [str(path) for path in args.exclude_manifest or []],
    }
    summary_path = output_root / "split_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _load_rows(manifests: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for manifest in manifests:
        base_dir = manifest.parent
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = str(record.get("sample_id") or "")
                if not sample_id:
                    raise ValueError(f"Missing sample_id in {manifest}:{line_number}")
                if sample_id in seen:
                    raise ValueError(
                        f"Duplicate sample_id across manifests: {sample_id}"
                    )
                seen.add(sample_id)
                record = _absolutize_path_fields(record, base_dir)
                record["source_manifest"] = str(manifest.resolve())
                rows.append({"sample_id": sample_id, "record": record})
    return rows


def _load_sample_ids(manifests: Sequence[Path]) -> set[str]:
    sample_ids: set[str] = set()
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = str(record.get("sample_id") or "")
                if not sample_id:
                    raise ValueError(f"Missing sample_id in {manifest}:{line_number}")
                sample_ids.add(sample_id)
    return sample_ids


def _select_test_ids(
    rows: Sequence[dict[str, object]],
    *,
    strategy: Literal["hash", "band", "grid-block"],
    test_ratio: float,
    seed: str,
    band_column: str,
    band_start: int | None,
    band_count: int | None,
    block_wp_start: int | None,
    block_wp_count: int | None,
    block_hp_start: int | None,
    block_hp_count: int | None,
) -> tuple[set[str], dict[str, object]]:
    if strategy == "hash":
        return _hash_test_ids(rows, test_ratio=test_ratio, seed=seed)
    if strategy == "band":
        return _band_test_ids(
            rows,
            column=band_column,
            test_ratio=test_ratio,
            start=band_start,
            count=band_count,
        )
    if strategy == "grid-block":
        return _grid_block_test_ids(
            rows,
            test_ratio=test_ratio,
            wp_start=block_wp_start,
            wp_count=block_wp_count,
            hp_start=block_hp_start,
            hp_count=block_hp_count,
        )
    raise ValueError(f"Unsupported split strategy: {strategy}")


def _hash_test_ids(
    rows: Sequence[dict[str, object]],
    *,
    test_ratio: float,
    seed: str,
) -> tuple[set[str], dict[str, object]]:
    test_count = round(len(rows) * test_ratio)
    test_count = min(max(test_count, 1), len(rows) - 1)
    ranked = sorted(
        rows,
        key=lambda row: (_stable_digest(row["record"], seed), row["sample_id"]),
    )
    test_ids = {str(row["sample_id"]) for row in ranked[:test_count]}
    return test_ids, {"test_count_target": test_count}


def _band_test_ids(
    rows: Sequence[dict[str, object]],
    *,
    column: str,
    test_ratio: float,
    start: int | None,
    count: int | None,
) -> tuple[set[str], dict[str, object]]:
    values = _sorted_unique_values(rows, column)
    if count is None:
        count = round(len(values) * test_ratio)
    count = min(max(count, 1), len(values) - 1)
    if start is None:
        start = (len(values) - count) // 2
    selected = _slice_values(values, start=start, count=count, label=column)
    selected_set = set(selected)
    test_ids = {
        str(row["sample_id"])
        for row in rows
        if _numeric_value(row["record"], column) in selected_set
    }
    if not test_ids or len(test_ids) == len(rows):
        raise ValueError("Band split produced an empty train or test split")
    return test_ids, {
        "band_column": column,
        "band_start": start,
        "band_count": count,
        "band_values": selected,
    }


def _grid_block_test_ids(
    rows: Sequence[dict[str, object]],
    *,
    test_ratio: float,
    wp_start: int | None,
    wp_count: int | None,
    hp_start: int | None,
    hp_count: int | None,
) -> tuple[set[str], dict[str, object]]:
    wp_values = _sorted_unique_values(rows, "wp_m")
    hp_values = _sorted_unique_values(rows, "hp_m")
    if wp_count is None or hp_count is None:
        target_cells = max(1, round(len(wp_values) * len(hp_values) * test_ratio))
        aspect = len(wp_values) / len(hp_values)
        inferred_hp_count = max(1, round((target_cells / aspect) ** 0.5))
        inferred_wp_count = max(1, round(target_cells / inferred_hp_count))
        wp_count = wp_count or inferred_wp_count
        hp_count = hp_count or inferred_hp_count
    wp_count = min(max(wp_count, 1), len(wp_values) - 1)
    hp_count = min(max(hp_count, 1), len(hp_values) - 1)
    if wp_start is None:
        wp_start = (len(wp_values) - wp_count) // 2
    if hp_start is None:
        hp_start = (len(hp_values) - hp_count) // 2
    selected_wp = _slice_values(wp_values, start=wp_start, count=wp_count, label="wp_m")
    selected_hp = _slice_values(hp_values, start=hp_start, count=hp_count, label="hp_m")
    selected_wp_set = set(selected_wp)
    selected_hp_set = set(selected_hp)
    test_ids = {
        str(row["sample_id"])
        for row in rows
        if _numeric_value(row["record"], "wp_m") in selected_wp_set
        and _numeric_value(row["record"], "hp_m") in selected_hp_set
    }
    if not test_ids or len(test_ids) == len(rows):
        raise ValueError("Grid-block split produced an empty train or test split")
    return test_ids, {
        "block_wp_start": wp_start,
        "block_wp_count": wp_count,
        "block_wp_values": selected_wp,
        "block_hp_start": hp_start,
        "block_hp_count": hp_count,
        "block_hp_values": selected_hp,
    }


def _sorted_unique_values(
    rows: Sequence[dict[str, object]],
    column: str,
) -> list[float]:
    values = sorted({_numeric_value(row["record"], column) for row in rows})
    if len(values) < 2:
        raise ValueError(f"Need at least two unique values for {column}")
    return values


def _numeric_value(record: dict[str, object], column: str) -> float:
    if column not in record:
        raise ValueError(f"Missing split column: {column}")
    return round(float(record[column]), 12)


def _slice_values(
    values: Sequence[float],
    *,
    start: int,
    count: int,
    label: str,
) -> list[float]:
    if start < 0 or start + count > len(values):
        raise ValueError(
            f"{label} slice start={start}, count={count} is outside "
            f"{len(values)} unique values"
        )
    return list(values[start : start + count])


def _absolutize_path_fields(
    record: dict[str, object],
    base_dir: Path,
) -> dict[str, object]:
    converted = dict(record)
    for key, value in list(converted.items()):
        if not key.endswith("_path") or not isinstance(value, str):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = base_dir / path
        converted[key] = str(path.resolve())
    return converted


def _stable_digest(record: dict[str, object], seed: str) -> str:
    sample_id = str(record["sample_id"])
    payload = f"{seed}:{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
