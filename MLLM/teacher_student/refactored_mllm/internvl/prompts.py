from __future__ import annotations


VR_REQUIREMENT_TASK_HEADER = """You are the requirement-vector generator for a multifunctional robot.
The RGB image, depth map, and point-cloud features describe the local
environment perceived by the robot sensors. Predict the complete requirement
vector v_r for downstream structure generation.

Return exactly one JSON object with these numeric keys:
{"wp_m": width_passable_m, "hp_m": height_passable_m, "dp_m": depth_passable_m, "hs_m": step_height_m, "fl": load_factor, "fi": inspection_factor, "fp": packing_factor}"""

VR_REQUIREMENT_OUTPUT_RULE = "Do not output cfg, mode, h, or w. Do not add explanation."


CHECKPOINT_NATIVE_TASK_PROMPT = f"""{VR_REQUIREMENT_TASK_HEADER}

{VR_REQUIREMENT_OUTPUT_RULE}"""


GRID_TASK_PROMPT = f"""{VR_REQUIREMENT_TASK_HEADER}

For this sensor domain, wp_m and hp_m are measured on a 0.01 m grid. Infer the
nearest observed grid value from the sensors and metric summary. Do not add a
safety margin, clearance margin, or conservative offset to either dimension.

{VR_REQUIREMENT_OUTPUT_RULE}"""


CONTINUOUS_AUDIT_TASK_PROMPT = f"""{VR_REQUIREMENT_TASK_HEADER}

Return wp_m and hp_m in meters as JSON numbers. Preserve the model's native
continuous estimate and output at least 4 decimal places when such precision
is available. Do not round, snap, or quantize values to a centimeter grid. Do
not add trailing zeros solely to imply precision that is not available.
The JSON object must still contain all seven required numeric keys. Do not omit
dp_m, hs_m, fl, fi, or fp when auditing wp_m and hp_m precision.

{VR_REQUIREMENT_OUTPUT_RULE}"""


PROMPT_PROFILES = {
    "checkpoint_native": CHECKPOINT_NATIVE_TASK_PROMPT,
    "continuous_audit": CONTINUOUS_AUDIT_TASK_PROMPT,
    "continuous_scale_audit": CONTINUOUS_AUDIT_TASK_PROMPT,
    "grid": GRID_TASK_PROMPT,
}

# Keep the existing default for the stricter teacher training and evaluation path.
VR_REQUIREMENT_TASK_PROMPT = GRID_TASK_PROMPT


VR_REQUIREMENT_PROMPT = f"""<image>
<depth>
<pointcloud>
{VR_REQUIREMENT_TASK_PROMPT}"""


def format_vr_answer(record: dict[str, object], decimals: int = 4) -> str:
    geometry = {column: f"{float(record[column]):.2f}" for column in ("wp_m", "hp_m")}
    continuous = {
        column: _format_float(record[column], decimals) for column in ("dp_m", "hs_m")
    }
    task = {column: _format_float(record[column], 1) for column in ("fl", "fi", "fp")}
    return (
        "{"
        f'"wp_m": {geometry["wp_m"]}, '
        f'"hp_m": {geometry["hp_m"]}, '
        f'"dp_m": {continuous["dp_m"]}, '
        f'"hs_m": {continuous["hs_m"]}, '
        f'"fl": {task["fl"]}, '
        f'"fi": {task["fi"]}, '
        f'"fp": {task["fp"]}'
        "}"
    )


def _format_float(value: object, decimals: int) -> str:
    return f"{float(value):.{decimals}f}"


def internvl_prompt_for_profile(profile: str) -> str:
    try:
        return PROMPT_PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(PROMPT_PROFILES))
        raise ValueError(
            f"Unknown InternVL prompt profile {profile!r}; choose from {choices}"
        ) from exc


def prompt_profile_uses_sensor_scale(profile: str) -> bool:
    if profile not in PROMPT_PROFILES:
        internvl_prompt_for_profile(profile)
    return profile in {"grid", "continuous_scale_audit"}
