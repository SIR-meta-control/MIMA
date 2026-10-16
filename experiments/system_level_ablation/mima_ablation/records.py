"""Row-level schemas and success rules for system-level ablation records."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


GT_PATTERN = re.compile(r"_w(-?\d+(?:\.\d+)?)_h(-?\d+(?:\.\d+)?)(?:_|$)")
SUCCESS_ENERGY_COLUMNS = (
    "method",
    "sample_id",
    "seed",
    "success",
    "actual_wp_m",
    "actual_hp_m",
    "pred_wp_m",
    "pred_hp_m",
    "energy_j",
    "error_type",
    "error",
)
TIMING_COLUMNS = (
    "method",
    "sample_id",
    "execution_time_s",
    "success",
    "error_type",
    "error",
)


def finite_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_gt(sample_id: str) -> tuple[float, float]:
    match = GT_PATTERN.search(sample_id)
    if match is None:
        raise ValueError(f"cannot parse width and height from sample ID: {sample_id}")
    return float(match.group(1)), float(match.group(2))


def strict_success(row: Mapping[str, Any], tolerance_m: float) -> bool:
    """Apply chain, physical-fit, and requirement-prediction conditions."""

    if tolerance_m < 0:
        raise ValueError("tolerance_m must be non-negative")
    if not bool_value(row.get("success")):
        return False
    gt_wp, gt_hp = parse_gt(str(row["sample_id"]))
    actual_wp = finite_float(row.get("actual_wp_m"))
    actual_hp = finite_float(row.get("actual_hp_m"))
    pred_wp = finite_float(row.get("pred_wp_m"))
    pred_hp = finite_float(row.get("pred_hp_m"))
    if None in (actual_wp, actual_hp, pred_wp, pred_hp):
        return False
    physical_ok = actual_wp <= gt_wp + 1e-12 and actual_hp <= gt_hp + 1e-12
    prediction_ok = (
        abs(pred_wp - gt_wp) <= tolerance_m + 1e-12
        and abs(pred_hp - gt_hp) <= tolerance_m + 1e-12
    )
    return physical_ok and prediction_ok
