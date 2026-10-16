"""Small JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path):
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | Path, obj: dict) -> None:
    with Path(path).open("w") as f:
        json.dump(obj, f, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
