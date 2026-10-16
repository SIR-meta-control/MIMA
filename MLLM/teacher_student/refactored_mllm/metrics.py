from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from refactored_mllm.schema import REQUIREMENT_COLUMNS


DEFAULT_TOLERANCES_M = (0.01, 0.02, 0.03, 0.04, 0.05)


def evaluate_vr_tolerance(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    method: str,
    requirement_columns: Sequence[str] = REQUIREMENT_COLUMNS,
    tolerances_m: Sequence[float] = DEFAULT_TOLERANCES_M,
) -> tuple[dict[str, str | int | float], pd.DataFrame]:
    """Evaluate v_r predictions using bidirectional absolute-error tolerances."""
    prediction_columns = [f"{column}_pred" for column in requirement_columns]
    _require_columns(truth, ["sample_id", *requirement_columns], "truth")
    _require_columns(predictions, ["sample_id", *prediction_columns], "prediction")
    if truth.empty or predictions.empty:
        raise ValueError("Cannot evaluate predictions with no samples")
    _duplicate_ids(truth, "truth")
    _duplicate_ids(predictions, "prediction")

    truth_ids = set(truth["sample_id"])
    prediction_ids = set(predictions["sample_id"])
    if truth_ids != prediction_ids:
        missing = _format_ids(truth_ids - prediction_ids)
        extra = _format_ids(prediction_ids - truth_ids)
        raise ValueError(
            "Prediction sample ids do not match truth. "
            f"Missing: {missing}; Extra: {extra}"
        )

    merged = truth[["sample_id", *requirement_columns]].merge(
        predictions[["sample_id", *prediction_columns]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    errors = compute_vr_errors(merged, requirement_columns=requirement_columns)

    metrics: dict[str, str | int | float] = {
        "method": method,
        "n_test": len(merged),
        "mae_vr": _rounded(errors[requirement_columns].to_numpy().mean()),
        "rmse_vr": _rounded(
            (errors[requirement_columns].to_numpy() ** 2).mean() ** 0.5
        ),
    }
    for tolerance_m in tolerances_m:
        success = errors[requirement_columns].le(tolerance_m + 1e-12).all(axis=1)
        suffix = _format_tolerance(tolerance_m)
        metrics[f"vr_success_rate@{suffix}"] = _percentage(success.mean())
        metrics[f"vr_success_count@{suffix}"] = int(success.sum())
    if 0.03 in {round(float(value), 12) for value in tolerances_m}:
        metrics["vr_success_rate"] = metrics["vr_success_rate@0.03m"]
    return metrics, errors


def compute_vr_errors(
    merged: pd.DataFrame,
    *,
    requirement_columns: Sequence[str] = REQUIREMENT_COLUMNS,
) -> pd.DataFrame:
    errors = pd.DataFrame(index=merged.index)
    for truth_column in requirement_columns:
        prediction_column = f"{truth_column}_pred"
        _require_columns(merged, [truth_column, prediction_column], "merged")
        errors[truth_column] = (
            pd.to_numeric(merged[truth_column], errors="coerce")
            - pd.to_numeric(merged[prediction_column], errors="coerce")
        ).abs()
    if errors.isna().any().any():
        bad_columns = errors.columns[errors.isna().any()].tolist()
        raise ValueError(f"Non-numeric v_r values in columns: {bad_columns}")
    return errors


def per_column_error_summary(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    requirement_columns: Sequence[str] = REQUIREMENT_COLUMNS,
    tolerance_m: float = 0.03,
) -> pd.DataFrame:
    prediction_columns = [f"{column}_pred" for column in requirement_columns]
    merged = truth[["sample_id", *requirement_columns]].merge(
        predictions[["sample_id", *prediction_columns]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    errors = compute_vr_errors(merged, requirement_columns=requirement_columns)
    rows: list[dict[str, int | float | str]] = []
    for column in requirement_columns:
        prediction_column = f"{column}_pred"
        signed_error = pd.to_numeric(
            merged[prediction_column], errors="coerce"
        ) - pd.to_numeric(merged[column], errors="coerce")
        absolute_error = errors[column]
        rows.append(
            {
                "column": column,
                "success_rate": _percentage(
                    absolute_error.le(tolerance_m + 1e-12).mean()
                ),
                "success_count": int(absolute_error.le(tolerance_m + 1e-12).sum()),
                "fail_count": int(absolute_error.gt(tolerance_m + 1e-12).sum()),
                "mae": _rounded(absolute_error.mean()),
                "bias_pred_minus_gt": _rounded(signed_error.mean()),
                "p50_abs_error": _rounded(absolute_error.quantile(0.50)),
                "p90_abs_error": _rounded(absolute_error.quantile(0.90)),
                "p95_abs_error": _rounded(absolute_error.quantile(0.95)),
                "max_abs_error": _rounded(absolute_error.max()),
                "unique_gt": int(merged[column].nunique()),
                "unique_pred": int(merged[prediction_column].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _percentage(value: float) -> float:
    return round(float(value) * 100.0, 2)


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _format_tolerance(tolerance_m: float) -> str:
    return f"{float(tolerance_m):.2f}m"


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], kind: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing {kind} columns: {missing}")


def _format_ids(values: Iterable[object]) -> str:
    return f"[{', '.join(sorted(repr(value) for value in values))}]"


def _duplicate_ids(frame: pd.DataFrame, kind: str) -> None:
    duplicated = frame["sample_id"].duplicated(keep=False)
    sample_ids = frame.loc[duplicated, "sample_id"].unique()
    if len(sample_ids):
        raise ValueError(f"Duplicate {kind} sample ids: {_format_ids(sample_ids)}")
