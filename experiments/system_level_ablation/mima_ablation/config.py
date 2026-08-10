"""Configuration loading with explicit-path validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .methods import MethodSpec


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return value


def require_nonempty_path(config: Mapping[str, Any], key: str) -> Path:
    value = str(config.get(key, "")).strip()
    if not value:
        raise ValueError(f"configuration field {key!r} must be set explicitly")
    return Path(value)


def resolve_under(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"protocol data paths must be bundle-relative: {relative_path}")
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"path escapes bundle root: {relative_path}")
    return resolved


def validate_run_assets(
    config: Mapping[str, Any], method_specs: Sequence[MethodSpec]
) -> dict[str, Any]:
    assets = config.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("run configuration must contain an assets object")
    requirement_sources = assets.get("requirement_sources")
    structure_models = assets.get("structure_models")
    if not isinstance(requirement_sources, dict):
        raise ValueError("assets.requirement_sources must be an object")
    if not isinstance(structure_models, dict):
        raise ValueError("assets.structure_models must be an object")

    missing: list[str] = []
    for spec in method_specs:
        if not str(requirement_sources.get(spec.requirement_source, "")).strip():
            missing.append(f"assets.requirement_sources.{spec.requirement_source}")
        if not str(structure_models.get(spec.structure_generator, "")).strip():
            missing.append(f"assets.structure_models.{spec.structure_generator}")
    for key in ("energy_model", "robot_model"):
        if not str(assets.get(key, "")).strip():
            missing.append(f"assets.{key}")
    if missing:
        raise ValueError(
            "run configuration has unset required assets: " + ", ".join(sorted(set(missing)))
        )
    return dict(assets)
