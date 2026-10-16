from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit outer/inner/teacher/ODD sample-ID sets for leakage."
    )
    parser.add_argument("--outer-train", type=Path, required=True)
    parser.add_argument("--outer-test", type=Path, required=True)
    parser.add_argument("--inner-train", type=Path, required=True)
    parser.add_argument("--inner-validation", type=Path, required=True)
    parser.add_argument("--teacher-annotation", type=Path, required=True)
    parser.add_argument("--odd-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = audit_split_leakage(
        outer_train=args.outer_train,
        outer_test=args.outer_test,
        inner_train=args.inner_train,
        inner_validation=args.inner_validation,
        teacher_annotation=args.teacher_annotation,
        odd_ids=args.odd_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    if not payload["leakage_free"]:
        raise SystemExit(2)


def audit_split_leakage(
    *,
    outer_train: Path,
    outer_test: Path,
    inner_train: Path,
    inner_validation: Path,
    teacher_annotation: Path,
    odd_ids: Path,
) -> dict[str, object]:
    paths = {
        "outer_train": outer_train,
        "outer_test": outer_test,
        "inner_train": inner_train,
        "inner_validation": inner_validation,
        "teacher_annotation": teacher_annotation,
        "odd107": odd_ids,
    }
    sets = {
        "outer_train": _manifest_ids(outer_train),
        "outer_test": _manifest_ids(outer_test),
        "inner_train": _manifest_ids(inner_train),
        "inner_validation": _manifest_ids(inner_validation),
        "teacher_annotation": _annotation_ids(teacher_annotation),
        "odd107": _line_ids(odd_ids),
    }
    intersections = {
        "outer_train_vs_outer_test": len(sets["outer_train"] & sets["outer_test"]),
        "inner_train_vs_inner_validation": len(
            sets["inner_train"] & sets["inner_validation"]
        ),
        "odd107_vs_outer_train": len(sets["odd107"] & sets["outer_train"]),
        "odd107_vs_inner_train": len(sets["odd107"] & sets["inner_train"]),
        "odd107_vs_inner_validation": len(sets["odd107"] & sets["inner_validation"]),
        "odd107_vs_teacher_annotation": len(
            sets["odd107"] & sets["teacher_annotation"]
        ),
    }
    invariants = {
        "inner_union_equals_outer_train": (
            sets["inner_train"] | sets["inner_validation"]
        )
        == sets["outer_train"],
        "teacher_annotation_equals_inner_train": (
            sets["teacher_annotation"] == sets["inner_train"]
        ),
        "odd107_is_subset_of_outer_test": sets["odd107"] <= sets["outer_test"],
    }
    return {
        "counts": {key: len(value) for key, value in sets.items()},
        "intersections": intersections,
        **invariants,
        "odd107_sha256": hashlib.sha256(odd_ids.read_bytes()).hexdigest(),
        "leakage_free": (
            all(value == 0 for value in intersections.values())
            and all(invariants.values())
        ),
        "paths": {key: str(value) for key, value in paths.items()},
    }


def _manifest_ids(path: Path) -> set[str]:
    return _jsonl_ids(path, field="sample_id")


def _annotation_ids(path: Path) -> set[str]:
    return _jsonl_ids(path, field="id")


def _jsonl_ids(path: Path, *, field: str) -> set[str]:
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            sample_id = str(payload.get(field) or "")
            if not sample_id:
                raise ValueError(f"Missing {field} in {path}:{line_number}")
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate {field}={sample_id!r} in {path}")
            sample_ids.add(sample_id)
    return sample_ids


def _line_ids(path: Path) -> set[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    sample_ids = [value for value in values if value]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return set(sample_ids)


if __name__ == "__main__":
    main()
