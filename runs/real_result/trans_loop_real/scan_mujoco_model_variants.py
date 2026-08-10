from __future__ import annotations

import json
import math
import re
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "src" / "models" / "crimson" / "mjcf" / "crimson_scene.xml"
INCLUDED_MODEL_PATH = (
    ROOT
    / "src"
    / "models"
    / "crimson"
    / "mjcf"
    / "crimson_stand_legInit_forSimOnly.xml"
)
JSON_PATH = ROOT / "runs" / "mujoco_experiment_energy_full10.json"
REAL_CSV_DIR = ROOT / "runs" / "real_result" / "trans_loop_real" / "csv"
OUT_DIR = ROOT / "runs" / "real_result" / "trans_loop_real" / "model_variant_scan"

SIGN_MATRIX = np.diag(
    [-1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1]
)
BIAS_MATRIX = np.array(
    [90, 90, 180, 180, 180, 180, 158, 248, 180, 158, 248, 180, 158, 248, 180, 158, 248],
    dtype=np.float64,
)
RULES = {
    0: 0,
    1: 2,
    2: 4,
    3: 1,
    4: 3,
    5: 8,
    6: 9,
    7: 10,
    8: 14,
    9: 15,
    10: 16,
    11: 11,
    12: 12,
    13: 13,
    14: 5,
    15: 6,
    16: 7,
}
REAL_TO_MUJOCO = {real_index: mujoco_index for mujoco_index, real_index in RULES.items()}
TRANSITIONS = [
    "mu1->mu2",
    "mu2->mu3",
    "mu3->mu4",
    "mu4->mu5",
    "mu5->mu6",
    "mu6->mu7",
    "mu7->mu8",
    "mu8->mu9",
    "mu9->mu8",
    "mu8->mu7",
    "mu7->mu6",
    "mu6->mu5",
    "mu5->mu4",
    "mu4->mu3",
    "mu3->mu2",
    "mu2->mu1",
]


def torque_to_current(torque: np.ndarray) -> np.ndarray:
    return (
        (0.000130565974) * torque**4
        + (-0.00188139351) * torque**3
        + (0.0216771226) * torque**2
        + (0.410017411) * torque
        + 0.0357777777778
    )


def frame_power_w(torque: np.ndarray) -> float:
    return float(np.sum(np.abs(torque_to_current(torque)) * 12.0))


def encoder_tick_to_radians(value: float, real_index: int) -> float:
    degrees = SIGN_MATRIX[real_index, real_index] * (
        float(value) * 180.0 / 2048.0 - BIAS_MATRIX[real_index]
    )
    return math.radians(degrees)


def control_from_sync_write(msg: dict, current_control: np.ndarray) -> np.ndarray:
    control = np.array(current_control, dtype=np.float64).copy()
    for motor_id, encoder_value in zip(msg["motorID"], msg["params"]):
        real_index = int(motor_id) - 1
        control[REAL_TO_MUJOCO[real_index]] = encoder_tick_to_radians(
            encoder_value, real_index
        )
    return control


def newest_real_trial_dirs() -> list[Path]:
    pattern = re.compile(r"^trans_exp_(?P<experiment>\d{2})_(?P<timestamp>\d{8}_\d{6})$")
    newest: dict[int, tuple[str, Path]] = {}
    for path in REAL_CSV_DIR.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        experiment = int(match.group("experiment"))
        if not 1 <= experiment <= 10:
            continue
        timestamp = match.group("timestamp")
        current = newest.get(experiment)
        if current is None or timestamp > current[0]:
            newest[experiment] = (timestamp, path)
    return [newest[index][1] for index in range(1, 11)]


def integrate_interval(time_s: np.ndarray, values: np.ndarray, start_s: float, end_s: float) -> float:
    inside = (time_s > start_s) & (time_s < end_s)
    clipped_time = np.concatenate(([start_s], time_s[inside], [end_s]))
    clipped_values = np.concatenate(
        (
            [np.interp(start_s, time_s, values)],
            values[inside],
            [np.interp(end_s, time_s, values)],
        )
    )
    return float(
        np.sum(0.5 * (clipped_values[:-1] + clipped_values[1:]) * np.diff(clipped_time))
    )


def load_real_interval_power() -> pd.DataFrame:
    rows = []
    for experiment, path in enumerate(newest_real_trial_dirs(), start=1):
        transforms = pd.read_csv(path / "crimson_control_transform.csv")
        log = pd.read_csv(path / "dynamixel_control_log.csv")
        log_time = log["bag_time_sec"].to_numpy(dtype=float)
        total_power = np.zeros(len(log), dtype=float)
        for motor_id in range(17):
            total_power += (
                log[f"U[{motor_id}]"].to_numpy(dtype=float)
                * np.abs(log[f"I[{motor_id}]"].to_numpy(dtype=float))
            )
        event_time = transforms["bag_time_sec"].to_numpy(dtype=float)
        for step in range(len(event_time) - 1):
            energy = integrate_interval(
                log_time, total_power, float(event_time[step]), float(event_time[step + 1])
            )
            duration = float(event_time[step + 1] - event_time[step])
            rows.append(
                {
                    "experiment": experiment,
                    "step": step,
                    "transition": TRANSITIONS[step],
                    "real_power_w": energy / duration,
                }
            )
    return (
        pd.DataFrame(rows)
        .groupby(["step", "transition"], as_index=False)
        .agg(real_power_w=("real_power_w", "mean"))
    )


def xml_variant(scene_xml: str, included_xml: str, variant: str) -> tuple[str, str]:
    scene = scene_xml
    included = included_xml
    if variant.startswith("floor_friction_"):
        value = variant.rsplit("_", 1)[1]
        scene = re.sub(r'friction="[^"]+"', f'friction="{value}"', scene, count=1)
    elif variant == "contact_disabled":
        included = included.replace('contact="enable"', 'contact="disable"')
    elif variant == "gravity_disabled":
        included = included.replace('gravity="0 0 -9.81"', 'gravity="0 0 0"')
    elif variant == "kp15_kv2p5":
        included = re.sub(r'kp="30"', 'kp="15"', included)
        included = re.sub(r'kv="5"', 'kv="2.5"', included)
    elif variant == "kp20_kv3":
        included = re.sub(r'kp="30"', 'kp="20"', included)
        included = re.sub(r'kv="5"', 'kv="3"', included)
    elif variant.startswith("allkp_"):
        kp = float(variant.split("_", 1)[1])
        kv = kp / 6.0
        included = re.sub(r'kp="30"', f'kp="{kp:g}"', included)
        included = re.sub(r'kv="5"', f'kv="{kv:g}"', included)
    elif variant.startswith("framekp_"):
        match = re.fullmatch(r"framekp_([0-9.]+)_legkp_([0-9.]+)", variant)
        if match is None:
            raise ValueError(f"Invalid group gain variant {variant}")
        frame_kp = float(match.group(1))
        leg_kp = float(match.group(2))

        def replace_position_gain(match_obj: re.Match[str]) -> str:
            line = match_obj.group(0)
            is_frame = 'name="frame' in line
            kp = frame_kp if is_frame else leg_kp
            kv = kp / 6.0
            line = re.sub(r'kp="[^"]+"', f'kp="{kp:g}"', line)
            line = re.sub(r'kv="[^"]+"', f'kv="{kv:g}"', line)
            return line

        included = re.sub(r"<position[^>]+/>", replace_position_gain, included)
    elif variant != "baseline":
        raise ValueError(f"Unknown variant {variant}")
    return scene, included


def load_variant_model(scene_xml: str, included_xml: str, variant: str) -> mujoco.MjModel:
    # MuJoCo resolves <include> and meshdir relative to the XML file path, not
    # reliably from_xml_string. Keep the temporary scene beside the real model.
    temp_scene_path = MODEL_PATH.with_name(f".codex_tmp_scene_{variant}.xml")
    temp_include_path = INCLUDED_MODEL_PATH.with_name(
        f".codex_tmp_crimson_{variant}.xml"
    )
    variant_scene, variant_included = xml_variant(scene_xml, included_xml, variant)
    variant_scene = variant_scene.replace(
        'include file="crimson_stand_legInit_forSimOnly.xml"',
        f'include file="{temp_include_path.name}"',
    )
    temp_include_path.write_text(variant_included, encoding="utf-8")
    temp_scene_path.write_text(variant_scene, encoding="utf-8")
    try:
        return mujoco.MjModel.from_xml_path(temp_scene_path.as_posix())
    finally:
        temp_scene_path.unlink(missing_ok=True)
        temp_include_path.unlink(missing_ok=True)


def simulate_variant(
    data: dict, scene_xml: str, included_xml: str, variant: str
) -> pd.DataFrame:
    model = load_variant_model(scene_xml, included_xml, variant)
    rows = []
    for experiment in data["experiments"]:
        sim_data = mujoco.MjData(model)
        experiment_id = int(experiment["experiment"])
        commands = experiment["sync_write_commands"]
        segments = [
            row for row in data["segments"] if int(row["experiment"]) == experiment_id
        ]
        for segment in segments:
            final_control = None
            for command_index in range(
                int(segment["sync_write_command_start"]),
                int(segment["sync_write_command_end"]),
            ):
                final_control = control_from_sync_write(
                    commands[command_index]["msg"], sim_data.ctrl
                )
                sim_data.ctrl[:] = final_control
                mujoco.mj_step(model, sim_data, nstep=10)

            powers = []
            contacts = []
            for _ in range(int(segment["settle_steps"])):
                sim_data.ctrl[:] = final_control
                mujoco.mj_step(model, sim_data, nstep=10)
                powers.append(frame_power_w(sim_data.actuator_force.copy()))
                contacts.append(int(sim_data.ncon))

            rows.append(
                {
                    "experiment": experiment_id,
                    "step": int(segment["step"]),
                    "transition": TRANSITIONS[int(segment["step"])],
                    "variant": variant,
                    "sim_settle_power_w": float(np.mean(powers)),
                    "sim_contact_count": float(np.mean(contacts)),
                }
            )

    return (
        pd.DataFrame(rows)
        .groupby(["step", "transition", "variant"], as_index=False)
        .agg(
            sim_settle_power_w=("sim_settle_power_w", "mean"),
            sim_contact_count=("sim_contact_count", "mean"),
        )
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    scene_xml = MODEL_PATH.read_text(encoding="utf-8")
    included_xml = INCLUDED_MODEL_PATH.read_text(encoding="utf-8")
    real_power = load_real_interval_power()
    variants = [
        "baseline",
        "floor_friction_0.05",
        "floor_friction_0.2",
        "floor_friction_1.5",
        "floor_friction_3.0",
        "contact_disabled",
        "gravity_disabled",
        "allkp_10",
        "allkp_12",
        "allkp_15",
        "allkp_18",
        "allkp_20",
        "framekp_20_legkp_10",
        "framekp_30_legkp_10",
        "framekp_40_legkp_10",
        "framekp_50_legkp_10",
        "framekp_20_legkp_15",
        "framekp_30_legkp_15",
        "framekp_40_legkp_15",
        "framekp_50_legkp_15",
        "framekp_20_legkp_20",
        "framekp_30_legkp_20",
        "framekp_40_legkp_20",
    ]

    frames = []
    summary = []
    for variant in variants:
        sim = simulate_variant(data, scene_xml, included_xml, variant)
        merged = real_power.merge(sim, on=["step", "transition"])
        merged["error_w"] = merged["sim_settle_power_w"] - merged["real_power_w"]
        merged["abs_error_w"] = np.abs(merged["error_w"])
        merged["squared_error_w2"] = merged["error_w"] ** 2
        frames.append(merged)
        summary.append(
            {
                "variant": variant,
                "mae_w": float(merged["abs_error_w"].mean()),
                "rmse_w": float(np.sqrt(merged["squared_error_w2"].mean())),
                "bias_w": float(merged["error_w"].mean()),
                "corr": float(
                    np.corrcoef(
                        merged["real_power_w"], merged["sim_settle_power_w"]
                    )[0, 1]
                ),
                "mean_contact_count": float(merged["sim_contact_count"].mean()),
            }
        )

    all_rows = pd.concat(frames, ignore_index=True)
    summary_df = pd.DataFrame(summary).sort_values("rmse_w")
    all_rows.to_csv(OUT_DIR / "variant_interval_power.csv", index=False)
    summary_df.to_csv(OUT_DIR / "variant_summary.csv", index=False)

    print("Variant summary sorted by RMSE:")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nBaseline interval errors:")
    baseline = all_rows[all_rows["variant"] == "baseline"]
    print(
        baseline[
            [
                "step",
                "transition",
                "real_power_w",
                "sim_settle_power_w",
                "error_w",
                "sim_contact_count",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )


if __name__ == "__main__":
    main()
