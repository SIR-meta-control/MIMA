"""Datasets for third-stage structure generation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from gvae.core.constants import BAR_TO_V, V_TO_BAR
from gvae.core.io import read_jsonl


def load_structure_json(path: str | Path) -> dict[str, np.ndarray]:
    with Path(path).open("r") as f:
        raw = json.load(f)

    edges = np.asarray(raw["edges"], dtype=np.float32)
    return {
        "nodes": np.asarray(raw["nodes"], dtype=np.float32),
        "edges": edges,
        "edge_pose": edges[:, 1:].astype(np.float32),
        "edge_angles": edges[:, 0].astype(np.float32),
        "leg_base": np.asarray(raw["global"]["leg_base"], dtype=np.float32),
        "leg_angle": np.asarray(raw["global"]["leg_angle"], dtype=np.float32),
        "scale": np.asarray(raw["global"]["scale"], dtype=np.float32),
    }


class StructurePairDataset(Dataset):
    """Dataset of (vreq, bar type, structure) pairs."""

    def __init__(
        self,
        pairs_path: str | Path,
        max_pairs: int | None = None,
        bar_type: str | None = None,
        seed: int = 7,
        shuffle: bool = True,
    ) -> None:
        self.rows = []
        for row in read_jsonl(pairs_path):
            if bar_type is not None and row["bar_type"] != bar_type:
                continue
            self.rows.append(row)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.rows)

        if max_pairs is not None and max_pairs > 0:
            self.rows = self.rows[:max_pairs]

        if not self.rows:
            raise ValueError(f"No structure pair rows found in {pairs_path}")

        self.structure_cache: dict[str, dict[str, np.ndarray]] = {}
        self.source_paths = frozenset(row["source_path"] for row in self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_cached(self, source_path: str) -> dict[str, np.ndarray]:
        if source_path not in self.structure_cache:
            self.structure_cache[source_path] = load_structure_json(source_path)
        return self.structure_cache[source_path]

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        structure = self._load_cached(row["source_path"])
        v = int(row.get("v", BAR_TO_V[row["bar_type"]]))
        item = {
            "vreq": torch.as_tensor(row["vreq"], dtype=torch.float32),
            "bar_v": torch.tensor(v, dtype=torch.long),
            "bar_type": V_TO_BAR[v],
            "source_path": row["source_path"],
        }
        for key, value in structure.items():
            item[key] = torch.as_tensor(value, dtype=torch.float32)
        return item
