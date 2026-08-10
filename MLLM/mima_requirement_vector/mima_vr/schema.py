"""Shared schemas for the seven-dimensional requirement vector and baselines."""

REQUIREMENT_KEYS = ("wp_m", "hp_m", "dp_m", "hs_m", "fl", "fi", "fp")
ENVIRONMENT_REQUIREMENT_KEYS = REQUIREMENT_KEYS[:4]
TASK_REQUIREMENT_KEYS = REQUIREMENT_KEYS[4:]

RGB_FEATURE_COLUMNS = (
    "rgb_tunnel_score",
    "rgb_step_score",
    "rgb_obstacle_score",
    "rgb_person_score",
    "rgb_open_ground_score",
)
DEPTH_FEATURE_COLUMNS = (
    "depth_min_clearance_m",
    "depth_obstacle_height_m",
    "depth_step_height_m",
    "depth_slope_deg",
)
POINT_CLOUD_FEATURE_COLUMNS = (
    "pc_corridor_width_m",
    "pc_ground_roughness",
    "pc_turn_angle_deg",
    "pc_free_space_area_m2",
)
COMMAND_FEATURE_COLUMNS = ("cmd_load", "cmd_inspect", "cmd_pack")
FEATURE_COLUMNS = (
    RGB_FEATURE_COLUMNS
    + DEPTH_FEATURE_COLUMNS
    + POINT_CLOUD_FEATURE_COLUMNS
    + COMMAND_FEATURE_COLUMNS
)
PREDICTION_COLUMNS = tuple(f"{key}_pred" for key in REQUIREMENT_KEYS)
