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
OUT_DIR = ROOT / "runs" / "real_result" / "trans_loop_real" / "energy_calibration"

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


def make_model(leg_kp: float = 10.0, frame_kp: float = 30.0) -> mujoco.MjModel:
    scene = MODEL_PATH.read_text(encoding="utf-8")
    included = INCLUDED_MODEL_PATH.read_text(encoding="utf-8")

    def replace_position_gain(match_obj: re.Match[str]) -> str:
        line = match_obj.group(0)
        is_frame = 'name="frame' in line
        kp = frame_kp if is_frame else leg_kp
        kv = kp / 6.0
        line = re.sub(r'kp="[^"]+"', f'kp="{kp:g}"', line)
        line = re.sub(r'kv="[^"]+"', f'kv="{kv:g}"', line)
        return line

    included = re.sub(r"<position[^>]+/>", replace_position_gain, included)
    temp_scene = MODEL_PATH.with_name(".codex_tmp_calibration_scene.xml")
    temp_include = INCLUDED_MODEL_PATH.with_name(".codex_tmp_calibration_crimson.xml")
    scene = scene.replace(
        'include file="crimson_stand_legInit_forSimOnly.xml"',
        f'include file="{temp_include.name}"',
    )
    temp_include.write_text(included, encoding="utf-8")
    temp_scene.write_text(scene, encoding="utf-8")
    try:
        return mujoco.MjModel.from_xml_path(temp_scene.as_posix())
    finally:
        temp_scene.unlink(missing_ok=True)
        temp_include.unlink(missing_ok=True)


def collect_torque_features() -> pd.DataFrame:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    model = make_model(leg_kp=10.0, frame_kp=30.0)
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

            features = []
            for _ in range(int(segment["settle_steps"])):
                sim_data.ctrl[:] = final_control
                mujoco.mj_step(model, sim_data, nstep=10)
                torque = sim_data.actuator_force.copy()
                abs_torque = np.abs(torque)
                features.append(
                    [
                        float(len(torque)),
                        float(np.sum(abs_torque)),
                        float(np.sum(abs_torque**2)),
                        float(np.sum(abs_torque**3)),
                        float(np.sum(abs_torque**4)),
                        float(np.sum(abs_torque[:5])),
                        float(np.sum(abs_torque[5:])),
                    ]
                )
            mean_features = np.mean(np.asarray(features), axis=0)
            rows.append(
                {
                    "experiment": experiment_id,
                    "step": int(segment["step"]),
                    "transition": TRANSITIONS[int(segment["step"])],
                    "n_act": mean_features[0],
                    "sum_abs_tau": mean_features[1],
                    "sum_abs_tau2": mean_features[2],
                    "sum_abs_tau3": mean_features[3],
                    "sum_abs_tau4": mean_features[4],
                    "sum_frame_abs_tau": mean_features[5],
                    "sum_leg_abs_tau": mean_features[6],
                }
            )
    raw = pd.DataFrame(rows)
    return (
        raw.groupby(["step", "transition"], as_index=False)
        .agg(
            n_act=("n_act", "mean"),
            sum_abs_tau=("sum_abs_tau", "mean"),
            sum_abs_tau2=("sum_abs_tau2", "mean"),
            sum_abs_tau3=("sum_abs_tau3", "mean"),
            sum_abs_tau4=("sum_abs_tau4", "mean"),
            sum_frame_abs_tau=("sum_frame_abs_tau", "mean"),
            sum_leg_abs_tau=("sum_leg_abs_tau", "mean"),
        )
    )


def fit_nonnegative_least_squares(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Small projected-gradient NNLS; avoids requiring scipy in the MuJoCo env.
    coef = np.zeros(x.shape[1], dtype=float)
    lipschitz = float(np.linalg.norm(x, ord=2) ** 2)
    step = 1.0 / max(lipschitz, 1e-12)
    for _ in range(200000):
        grad = x.T @ (x @ coef - y)
        new_coef = np.maximum(0.0, coef - step * grad)
        if np.linalg.norm(new_coef - coef) < 1e-10:
            coef = new_coef
            break
        coef = new_coef
    return coef


def evaluate_fit(name: str, features: pd.DataFrame, columns: list[str]) -> dict:
    x = features[columns].to_numpy(dtype=float)
    y = features["real_power_w"].to_numpy(dtype=float)
    coef = fit_nonnegative_least_squares(x, y)
    pred = x @ coef
    error = pred - y
    return {
        "model": name,
        "columns": ",".join(columns),
        "coefficients": ",".join(f"{value:.8g}" for value in coef),
        "mae_w": float(np.mean(np.abs(error))),
        "rmse_w": float(np.sqrt(np.mean(error**2))),
        "bias_w": float(np.mean(error)),
        "corr": float(np.corrcoef(y, pred)[0, 1]),
        "pred": pred,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    real = load_real_interval_power()
    torque = collect_torque_features()
    features = real.merge(torque, on=["step", "transition"])

    fits = [
        evaluate_fit("constant_plus_abs_tau", features, ["n_act", "sum_abs_tau"]),
        evaluate_fit(
            "constant_abs_tau_abs_tau2",
            features,
            ["n_act", "sum_abs_tau", "sum_abs_tau2"],
        ),
        evaluate_fit(
            "constant_frame_leg_abs_tau",
            features,
            ["n_act", "sum_frame_abs_tau", "sum_leg_abs_tau"],
        ),
        evaluate_fit(
            "poly_abs_tau_1_to_4",
            features,
            ["n_act", "sum_abs_tau", "sum_abs_tau2", "sum_abs_tau3", "sum_abs_tau4"],
        ),
    ]
    summary = pd.DataFrame([{k: v for k, v in fit.items() if k != "pred"} for fit in fits])
    summary = summary.sort_values("rmse_w")
    rows = features[["step", "transition", "real_power_w"]].copy()
    for fit in fits:
        rows[f"pred_{fit['model']}"] = fit["pred"]
        rows[f"err_{fit['model']}"] = fit["pred"] - rows["real_power_w"]

    features.to_csv(OUT_DIR / "torque_features.csv", index=False)
    rows.to_csv(OUT_DIR / "fit_predictions.csv", index=False)
    summary.to_csv(OUT_DIR / "fit_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    best = summary.iloc[0]["model"]
    print(f"\nBest model: {best}")
    print(
        rows[
            ["step", "transition", "real_power_w", f"pred_{best}", f"err_{best}"]
        ].to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )


if __name__ == "__main__":
    main()
