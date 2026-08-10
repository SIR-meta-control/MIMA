from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "src" / "models" / "crimson" / "mjcf" / "crimson_scene.xml"
JSON_PATH = ROOT / "runs" / "mujoco_experiment_energy_full10.json"

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


def torque_to_current_signed(torque: np.ndarray) -> np.ndarray:
    return (
        (0.000130565974) * torque**4
        + (-0.00188139351) * torque**3
        + (0.0216771226) * torque**2
        + (0.410017411) * torque
        + 0.0357777777778
    )


def frame_energy_j(torque: np.ndarray, mode: str) -> float:
    if mode == "signed":
        current = torque_to_current_signed(torque)
    elif mode == "abs_torque":
        current = torque_to_current_signed(np.abs(torque))
    else:
        raise ValueError(mode)
    return float(np.sum(np.abs(current * 12.0 * (1.0 / 50.0))))


def replay_experiment(data: dict, experiment_id: int, mode: str) -> list[dict[str, float]]:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH.as_posix())
    sim_data = mujoco.MjData(model)
    experiment = next(
        row for row in data["experiments"] if int(row["experiment"]) == experiment_id
    )
    commands = experiment["sync_write_commands"]
    rows = []

    for segment in data["segments"]:
        if int(segment["experiment"]) != experiment_id:
            continue

        transition_energy = 0.0
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
            transition_energy += frame_energy_j(sim_data.actuator_force.copy(), mode)

        settle_energy = 0.0
        for _ in range(int(segment["settle_steps"])):
            sim_data.ctrl[:] = final_control
            mujoco.mj_step(model, sim_data, nstep=10)
            settle_energy += frame_energy_j(sim_data.actuator_force.copy(), mode)

        rows.append(
            {
                "step": int(segment["step"]),
                "transition_energy_j": transition_energy,
                "settle_energy_j": settle_energy,
                "total_energy_j": transition_energy + settle_energy,
            }
        )

    return rows


def main() -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    reference = [
        row for row in data["segments"] if int(row["experiment"]) == 1
    ]

    for mode in ("signed", "abs_torque"):
        rows = replay_experiment(data, experiment_id=1, mode=mode)
        print(f"\nmode={mode}")
        print("step  transition_j   settle_j     total_j")
        for row in rows:
            print(
                f"{row['step']:>4d}"
                f" {row['transition_energy_j']:>13.3f}"
                f" {row['settle_energy_j']:>10.3f}"
                f" {row['total_energy_j']:>11.3f}"
            )
        print(f"total={sum(row['total_energy_j'] for row in rows):.3f}")

    print("\nreference_json_experiment_1")
    print("step  transition_j   settle_j     total_j")
    for row in reference:
        print(
            f"{int(row['step']):>4d}"
            f" {float(row['transition_energy_j']):>13.3f}"
            f" {float(row['settle_energy_j']):>10.3f}"
            f" {float(row['total_energy_j']):>11.3f}"
        )
    print(f"total={sum(float(row['total_energy_j']) for row in reference):.3f}")


if __name__ == "__main__":
    main()
