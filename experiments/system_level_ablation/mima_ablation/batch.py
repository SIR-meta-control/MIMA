"""Backend-neutral batch execution for success/energy and timing records."""

from __future__ import annotations

import csv
import importlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .methods import MethodSpec
from .records import SUCCESS_ENERGY_COLUMNS, TIMING_COLUMNS, finite_float


BackendCallable = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def load_backend(import_path: str) -> BackendCallable:
    module_name, separator, attribute = import_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("backend must use the form 'python.module:function'")
    module = importlib.import_module(module_name)
    backend = getattr(module, attribute)
    if not callable(backend):
        raise TypeError(f"configured backend is not callable: {import_path}")
    return backend


def load_manifest(dataset_dir: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(dataset_dir).resolve()
    manifest_path = root / "manifest.jsonl"
    records: dict[str, dict[str, Any]] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            if sample_id in records:
                raise ValueError(f"duplicate sample ID in manifest: {sample_id}")
            record["rgb_path"] = str(_artifact_path(root, record, "rgb_path", "rgb.png"))
            record["depth_path"] = str(_artifact_path(root, record, "depth_path", "depth.npy"))
            record["point_cloud_path"] = str(
                _artifact_path(root, record, "point_cloud_path", "point_cloud.npy")
            )
            records[sample_id] = record
    return records


def read_sample_ids(path: str | Path) -> tuple[str, ...]:
    values = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(values) != len(set(values)):
        raise ValueError("sample ID file contains duplicates")
    if not values:
        raise ValueError("sample ID file is empty")
    return values


def run_batch(
    *,
    mode: str,
    backend_path: str,
    dataset_dir: str | Path,
    sample_ids_file: str | Path,
    output_dir: str | Path,
    method_specs: Sequence[MethodSpec],
    seeds: Sequence[int],
    assets: Mapping[str, Any],
    protocol: Mapping[str, Any],
    workers: int = 1,
    progress: bool = True,
) -> dict[str, Any]:
    if mode not in {"success_energy", "timing"}:
        raise ValueError("mode must be 'success_energy' or 'timing'")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if mode == "timing" and workers != 1:
        raise ValueError("timing runs require workers=1")
    if not seeds:
        raise ValueError("at least one seed is required")

    records = load_manifest(dataset_dir)
    sample_ids = read_sample_ids(sample_ids_file)
    missing = sorted(set(sample_ids) - set(records))
    if missing:
        raise ValueError(f"sample IDs absent from dataset manifest: {missing}")

    tasks: list[dict[str, Any]] = []
    for method in method_specs:
        for seed in seeds:
            for sample_id in sample_ids:
                tasks.append(
                    {
                        "mode": mode,
                        "method": {
                            "key": method.key,
                            "label": method.label,
                            "requirement_source": method.requirement_source,
                            "structure_generator": method.structure_generator,
                            "use_energy_optimizer": method.use_energy_optimizer,
                            "candidate_selection_policy": method.candidate_selection_policy,
                            "posthoc_energy_audit": method.posthoc_energy_audit,
                        },
                        "seed": int(seed),
                        "sample": records[sample_id],
                        "assets": dict(assets),
                        "protocol": dict(protocol),
                    }
                )

    if workers == 1:
        raw_results = [
            _execute_task(backend_path, task, index, len(tasks), progress)
            for index, task in enumerate(tasks, start=1)
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            raw_results = list(
                executor.map(
                    _execute_task_star,
                    (
                        (backend_path, task, index, len(tasks), progress)
                        for index, task in enumerate(tasks, start=1)
                    ),
                )
            )

    rows = [
        _success_energy_row(task, result)
        if mode == "success_energy"
        else _timing_row(task, result)
        for task, result in zip(tasks, raw_results)
    ]
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    filename = "details.csv" if mode == "success_energy" else "execution_time_details.csv"
    columns = SUCCESS_ENERGY_COLUMNS if mode == "success_energy" else TIMING_COLUMNS
    _write_csv(destination / filename, rows, columns)
    summary = {
        "mode": mode,
        "backend": backend_path,
        "method_keys": [spec.key for spec in method_specs],
        "sample_count": len(sample_ids),
        "seeds": list(seeds),
        "row_count": len(rows),
        "failed_count": sum(bool(row.get("error_type")) for row in rows),
        "output_file": filename,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _execute_task_star(arguments: tuple[str, dict[str, Any], int, int, bool]) -> dict[str, Any]:
    return _execute_task(*arguments)


def _execute_task(
    backend_path: str,
    task: dict[str, Any],
    index: int,
    total: int,
    progress: bool,
) -> dict[str, Any]:
    try:
        result = load_backend(backend_path)(task)
        if not isinstance(result, Mapping):
            raise TypeError("backend must return a mapping")
        normalized = dict(result)
    except Exception as exc:
        normalized = {
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    if progress:
        print(
            f"[{index}/{total}] method={task['method']['label']} "
            f"sample={task['sample']['sample_id']} seed={task['seed']} "
            f"success={normalized.get('success', False)}",
            flush=True,
        )
    return normalized


def _success_energy_row(task: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    predicted = result.get("predicted_v_r") or {}
    actual = result.get("actual_v_r") or {}
    return {
        "method": task["method"]["label"],
        "sample_id": task["sample"]["sample_id"],
        "seed": task["seed"],
        "success": bool(result.get("success", False)),
        "actual_wp_m": result.get("actual_wp_m", actual.get("wp_m")),
        "actual_hp_m": result.get("actual_hp_m", actual.get("hp_m")),
        "pred_wp_m": result.get("pred_wp_m", predicted.get("wp_m")),
        "pred_hp_m": result.get("pred_hp_m", predicted.get("hp_m")),
        "energy_j": result.get("energy_j"),
        "error_type": result.get("error_type", ""),
        "error": result.get("error", ""),
    }


def _timing_row(task: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    timing = finite_float(result.get("execution_time_s"))
    error_type = str(result.get("error_type", ""))
    error = str(result.get("error", ""))
    if timing is None and not error_type:
        error_type = "MissingTiming"
        error = "backend did not return a finite execution_time_s"
    return {
        "method": task["method"]["label"],
        "sample_id": task["sample"]["sample_id"],
        "execution_time_s": timing,
        "success": bool(result.get("success", False)),
        "error_type": error_type,
        "error": error,
    }


def _artifact_path(root: Path, record: Mapping[str, Any], key: str, fallback: str) -> Path:
    value = record.get(key)
    path = Path(str(value)) if value else Path("samples") / str(record["sample_id"]) / fallback
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
