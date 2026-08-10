"""Reconstruct and audit the reported system-level ablation table."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import resolve_under
from .methods import METHOD_ORDER, canonical_method
from .records import finite_float, parse_gt, strict_success


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_success_energy(
    rows: Iterable[Mapping[str, str]],
    sample_ids: set[str],
    tolerance_m: float,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, str]]] = {
        method: [] for method in METHOD_ORDER
    }
    for row in rows:
        if row.get("sample_id") in sample_ids:
            grouped[canonical_method(str(row.get("method", "")))].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for method, method_rows in grouped.items():
        if not method_rows:
            raise ValueError(f"no success/energy rows found for {method}")
        energies = [
            value
            for row in method_rows
            if (value := finite_float(row.get("energy_j"))) is not None
        ]
        if len(energies) < 2:
            raise ValueError(f"fewer than two finite energy rows found for {method}")
        success_count = sum(strict_success(row, tolerance_m) for row in method_rows)
        summaries[method] = {
            "n": len(method_rows),
            "sample_ids": {str(row["sample_id"]) for row in method_rows},
            "seeds": {int(float(str(row["seed"]))) for row in method_rows},
            "strict_success_count": success_count,
            "success_rate_pct": 100.0 * success_count / len(method_rows),
            "energy_count": len(energies),
            "energy_mean_j": statistics.mean(energies),
            "energy_std_j": statistics.stdev(energies),
        }
    return summaries


def summarize_timing(
    rows: Iterable[Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, str]]] = {
        method: [] for method in METHOD_ORDER
    }
    for row in rows:
        grouped[canonical_method(str(row.get("method", "")))].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for method, method_rows in grouped.items():
        if not method_rows:
            raise ValueError(f"no timing rows found for {method}")
        values = [
            value
            for row in method_rows
            if (value := finite_float(row.get("execution_time_s"))) is not None
        ]
        if len(values) < 2:
            raise ValueError(f"fewer than two finite timing rows found for {method}")
        summaries[method] = {
            "n": len(method_rows),
            "timed_samples": len(values),
            "sample_ids": {str(row["sample_id"]) for row in method_rows},
            "execution_time_mean_s": statistics.mean(values),
            "execution_time_std_s": statistics.stdev(values),
        }
    return summaries


def reproduce_table(
    *,
    bundle_dir: str | Path,
    output_dir: str | Path,
    protocol: Mapping[str, Any],
    verify_hashes: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    configured_order = tuple(protocol["method_order"])
    if configured_order != METHOD_ORDER:
        raise ValueError(
            f"protocol method order differs from implementation: {configured_order}"
        )

    ids_path = resolve_under(root, str(protocol["sample_ids_path"]))
    sample_ids = read_ids(ids_path)
    coverage = protocol["coverage"]
    expected_sample_count = int(coverage["evaluation_samples"])
    if len(sample_ids) != expected_sample_count:
        raise AssertionError(
            f"expected {expected_sample_count} sample IDs, found {len(sample_ids)}"
        )

    tolerance_config = protocol["tolerance"]
    heights = [parse_gt(sample_id)[1] for sample_id in sample_ids]
    height_min_m = min(heights)
    height_max_m = max(heights)
    height_midpoint_m = (height_min_m + height_max_m) / 2.0
    tolerance_fraction = float(tolerance_config["fraction"])
    tolerance_m = round(height_midpoint_m * tolerance_fraction, 12)
    expected_tolerance_m = float(tolerance_config["expected_tolerance_m"])
    if abs(tolerance_m - expected_tolerance_m) > 1e-12:
        raise AssertionError(
            f"derived tolerance {tolerance_m} differs from {expected_tolerance_m}"
        )

    input_paths = [ids_path]
    success_energy_rows: list[dict[str, str]] = []
    for source in protocol["success_energy_sources"]:
        path = resolve_under(root, str(source["path"]))
        input_paths.append(path)
        selected = set(source["methods"])
        for row in read_csv(path):
            if canonical_method(str(row.get("method", ""))) in selected:
                success_energy_rows.append(row)

    timing_rows: list[dict[str, str]] = []
    for source in protocol["timing_sources"]:
        path = resolve_under(root, str(source["path"]))
        input_paths.append(path)
        selected = set(source["methods"])
        for row in read_csv(path):
            if canonical_method(str(row.get("method", ""))) in selected:
                timing_rows.append(row)

    actual_hashes = {
        path.relative_to(root).as_posix(): sha256(path) for path in input_paths
    }
    expected_hashes = dict(protocol["input_sha256"])
    if verify_hashes and actual_hashes != expected_hashes:
        missing = sorted(set(expected_hashes) - set(actual_hashes))
        unexpected = sorted(set(actual_hashes) - set(expected_hashes))
        mismatched = sorted(
            path
            for path in set(actual_hashes) & set(expected_hashes)
            if actual_hashes[path] != expected_hashes[path]
        )
        raise AssertionError(
            "input checksum mismatch: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )

    energy = summarize_success_energy(success_energy_rows, sample_ids, tolerance_m)
    timing = summarize_timing(timing_rows)
    reference_energy_j = energy["Full-MIMA"]["energy_mean_j"]
    timing_reference_ids = timing["Full-MIMA"]["sample_ids"]
    expected_rounded = protocol["expected_rounded"]
    energy_seeds = set(range(int(coverage["seed_min"]), int(coverage["seed_max"]) + 1))

    output_rows: list[dict[str, Any]] = []
    method_audit: dict[str, Any] = {}
    for method in METHOD_ORDER:
        energy_row = energy[method]
        timing_row = timing[method]
        normalized_mean = 100.0 * energy_row["energy_mean_j"] / reference_energy_j
        normalized_std = 100.0 * energy_row["energy_std_j"] / reference_energy_j
        rounded = (
            round(energy_row["success_rate_pct"], 2),
            round(normalized_mean, 2),
            round(normalized_std, 2),
            round(timing_row["execution_time_mean_s"], 2),
            round(timing_row["execution_time_std_s"], 2),
        )
        expected = tuple(float(value) for value in expected_rounded[method])
        if rounded != expected:
            raise AssertionError(f"reported values differ for {method}: {rounded} != {expected}")
        if (
            energy_row["n"] != int(coverage["execution_rows_per_method"])
            or energy_row["sample_ids"] != sample_ids
            or energy_row["seeds"] != energy_seeds
        ):
            raise AssertionError(f"success/energy coverage mismatch for {method}")
        if (
            timing_row["n"] != int(coverage["timing_rows_per_method"])
            or timing_row["sample_ids"] != timing_reference_ids
            or not timing_row["sample_ids"].issubset(sample_ids)
        ):
            raise AssertionError(f"timing coverage mismatch for {method}")

        output_rows.append(
            {
                "method": method,
                "success_rate_pct": energy_row["success_rate_pct"],
                "strict_success_count": energy_row["strict_success_count"],
                "execution_rows": energy_row["n"],
                "energy_mean_j": energy_row["energy_mean_j"],
                "energy_std_j": energy_row["energy_std_j"],
                "energy_count": energy_row["energy_count"],
                "normalized_energy_mean_pct": normalized_mean,
                "normalized_energy_std_pct": normalized_std,
                "execution_time_mean_s": timing_row["execution_time_mean_s"],
                "execution_time_std_s": timing_row["execution_time_std_s"],
                "timing_rows": timing_row["n"],
                "timed_samples": timing_row["timed_samples"],
            }
        )
        method_audit[method] = {
            "rounded_values": rounded,
            "expected_rounded_values": expected,
            "strict_success_count": energy_row["strict_success_count"],
            "execution_rows": energy_row["n"],
            "finite_energy_count": energy_row["energy_count"],
            "timing_rows": timing_row["n"],
            "timed_samples": timing_row["timed_samples"],
        }

    _write_csv(destination / "system_level_ablation_table_full_precision.csv", output_rows)
    _write_markdown(destination / "system_level_ablation_table.md", output_rows)
    audit = {
        "status": "passed",
        "target": "System-level ablation table",
        "tolerance_definition": tolerance_config["definition"],
        "height_min_m": height_min_m,
        "height_max_m": height_max_m,
        "height_midpoint_m": height_midpoint_m,
        "tolerance_fraction": tolerance_fraction,
        "strict_tolerance_m": tolerance_m,
        "evaluation_sample_count": len(sample_ids),
        "energy_seed_count": len(energy_seeds),
        "execution_rows_per_method": int(coverage["execution_rows_per_method"]),
        "timing_sample_count": len(timing_reference_ids),
        "timing_sample_sets_identical": True,
        "full_mima_energy_reference_j": reference_energy_j,
        "normalized_energy_mean_formula": (
            "100 * method mean finite energy / Full-MIMA mean finite energy"
        ),
        "normalized_energy_std_formula": (
            "100 * method finite-energy sample SD / Full-MIMA mean finite energy"
        ),
        "input_sha256": actual_hashes,
        "methods": method_audit,
    }
    (destination / "audit_report.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Method | Success Rate (%) | Normalized Energy (%) | Execution Time (s) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['success_rate_pct']:.2f} | "
            f"{row['normalized_energy_mean_pct']:.2f} +/- "
            f"{row['normalized_energy_std_pct']:.2f} | "
            f"{row['execution_time_mean_s']:.2f} +/- "
            f"{row['execution_time_std_s']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
