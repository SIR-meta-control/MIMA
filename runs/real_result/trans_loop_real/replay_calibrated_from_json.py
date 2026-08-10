from __future__ import annotations

import json
import math
import re
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SRC_JSON = ROOT / "runs" / "mujoco_experiment_energy_full10.json"
OUT_JSON = ROOT / "runs" / "mujoco_experiment_energy_calibrated_full10.json"
MODEL_PATH = ROOT / "src" / "models" / "crimson" / "mjcf" / "crimson_scene.xml"
REAL_CSV_DIR = ROOT / "runs" / "real_result" / "trans_loop_real" / "csv"
SCRIPTS_DIR = ROOT / "src" / "ros_mujoco" / "scripts"

import sys

sys.path.insert(0, str(SCRIPTS_DIR))

from ros_mujoco_utils.energy_calibration import (  # noqa: E402
    calibrated_total_power_w,
    frame_energy_j,
    power_to_current_vector,
)


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


def real_interval_power() -> pd.DataFrame:
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


def replay() -> dict:
    source = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(MODEL_PATH.as_posix())
    segments_out = []
    experiments_out = []

    for experiment in source["experiments"]:
        sim_data = mujoco.MjData(model)
        experiment_id = int(experiment["experiment"])
        commands = experiment["sync_write_commands"]
        experiment_copy = json.loads(json.dumps(experiment))
        logs = experiment_copy["records"]["/dynamixel_control/log"]
        next_log_index = 0

        for segment in [row for row in source["segments"] if int(row["experiment"]) == experiment_id]:
            transition_label = TRANSITIONS[int(segment["step"])]
            final_control = None
            transition_energy = 0.0
            for command_index in range(
                int(segment["sync_write_command_start"]),
                int(segment["sync_write_command_end"]),
            ):
                final_control = control_from_sync_write(
                    commands[command_index]["msg"], sim_data.ctrl
                )
                sim_data.ctrl[:] = final_control
                mujoco.mj_step(model, sim_data, nstep=10)
                transition_energy += frame_energy_j(
                    sim_data.actuator_force.copy(),
                    fps=float(source["fps"]),
                    mode="calibrated",
                    transition=transition_label,
                )

            settle_energy = 0.0
            for _ in range(int(segment["settle_steps"])):
                sim_data.ctrl[:] = final_control
                mujoco.mj_step(model, sim_data, nstep=10)
                torque = sim_data.actuator_force.copy()
                power = calibrated_total_power_w(torque, transition=transition_label)
                settle_energy += power / float(source["fps"])
                while (
                    next_log_index < len(logs)
                    and float(logs[next_log_index]["relative_time_s"])
                    <= float(segment["transformed_time_s"]) + 3.0
                ):
                    logs[next_log_index]["msg"]["I"] = [
                        float(value)
                        for value in power_to_current_vector(
                            power,
                            voltage=float(source["voltage"]),
                            weights=np.abs(torque),
                            motor_count=17,
                        )
                    ]
                    next_log_index += 1

            segment_copy = dict(segment)
            segment_copy["transition_energy_j"] = transition_energy
            segment_copy["settle_energy_j"] = settle_energy
            segment_copy["total_energy_j"] = transition_energy + settle_energy
            segments_out.append(segment_copy)
        experiments_out.append(experiment_copy)

    source["energy_mode"] = "calibrated"
    source["segments"] = segments_out
    source["experiments"] = experiments_out
    source["total_energy_j"] = float(sum(row["total_energy_j"] for row in segments_out))
    OUT_JSON.write_text(json.dumps(source, indent=2), encoding="utf-8")
    return source


def main() -> None:
    data = replay()
    real = real_interval_power()
    sim_rows = []
    for segment in data["segments"]:
        if int(segment["step"]) >= 15:
            continue
        sim_rows.append(
            {
                "step": int(segment["step"]),
                "transition": TRANSITIONS[int(segment["step"])],
                "sim_power_w": float(segment["total_energy_j"]) / 3.1,
            }
        )
    sim = (
        pd.DataFrame(sim_rows)
        .groupby(["step", "transition"], as_index=False)
        .agg(sim_power_w=("sim_power_w", "mean"))
    )
    merged = real.merge(sim, on=["step", "transition"])
    merged["error_w"] = merged["sim_power_w"] - merged["real_power_w"]
    print(f"Wrote {OUT_JSON}")
    print(
        "MAE={:.3f} W RMSE={:.3f} W bias={:.3f} W corr={:.3f}".format(
            float(np.mean(np.abs(merged["error_w"]))),
            float(np.sqrt(np.mean(merged["error_w"] ** 2))),
            float(np.mean(merged["error_w"])),
            float(np.corrcoef(merged["real_power_w"], merged["sim_power_w"])[0, 1]),
        )
    )
    print(
        merged[["step", "transition", "real_power_w", "sim_power_w", "error_w"]]
        .to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )


if __name__ == "__main__":
    main()
