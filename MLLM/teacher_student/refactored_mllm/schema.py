from __future__ import annotations

from pathlib import Path

import pandas as pd


META_COLUMNS = ["sample_id", "scenario", "split"]
RGB_FEATURE_COLUMNS = [
    "rgb_tunnel_score",
    "rgb_step_score",
    "rgb_obstacle_score",
    "rgb_person_score",
    "rgb_open_ground_score",
]
DEPTH_FEATURE_COLUMNS = [
    "depth_min_clearance_m",
    "depth_obstacle_height_m",
    "depth_step_height_m",
    "depth_slope_deg",
]
POINT_CLOUD_FEATURE_COLUMNS = [
    "pc_corridor_width_m",
    "pc_ground_roughness",
    "pc_turn_angle_deg",
    "pc_free_space_area_m2",
]
COMMAND_FEATURE_COLUMNS = [
    "cmd_load",
    "cmd_inspect",
    "cmd_pack",
]
FEATURE_COLUMNS = (
    RGB_FEATURE_COLUMNS
    + DEPTH_FEATURE_COLUMNS
    + POINT_CLOUD_FEATURE_COLUMNS
    + COMMAND_FEATURE_COLUMNS
)
ENV_REQUIREMENT_COLUMNS = ["wp_m", "hp_m", "dp_m", "hs_m"]
TASK_REQUIREMENT_COLUMNS = ["fl", "fi", "fp"]
REQUIREMENT_COLUMNS = ENV_REQUIREMENT_COLUMNS + TASK_REQUIREMENT_COLUMNS
PREDICTION_COLUMNS = [f"{column}_pred" for column in REQUIREMENT_COLUMNS]

VALID_SCENARIOS = {
    "tunnel",
    "midbox",
    "high_obstacle",
    "step",
    "sharp_turn",
    "open_ground",
    "person_following",
}
VALID_SPLITS = {"train", "test", "inference"}


def validate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [
        column
        for column in META_COLUMNS + FEATURE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    invalid_scenarios = _invalid_values(frame, "scenario", VALID_SCENARIOS)
    if invalid_scenarios:
        raise ValueError(f"Invalid scenario values: {invalid_scenarios}")

    invalid_splits = _invalid_values(frame, "split", VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"Invalid split values: {invalid_splits}")

    validated = frame.copy()
    numeric = validated[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad_columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Feature columns contain non-numeric values: {bad_columns}")
    validated[FEATURE_COLUMNS] = numeric
    return validated[META_COLUMNS + FEATURE_COLUMNS]


def load_feature_frame(path: str | Path) -> pd.DataFrame:
    return validate_feature_frame(pd.read_csv(path))


def _invalid_values(
    frame: pd.DataFrame,
    column: str,
    valid_values: set[str],
) -> list[str]:
    invalid = frame.loc[~frame[column].isin(valid_values), column]
    return sorted({_format_invalid_value(value) for value in invalid})


def _format_invalid_value(value: object) -> str:
    if pd.isna(value):
        return "<missing>"
    return str(value)
