from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


"""Generate the calibrated real-vs-MuJoCo energy comparison figure.

The transform mode in the CSV/JSON is 0-based. In the paper figure it is
displayed as 1-based morphology labels, so mode 0 is mu1 and mode 8 is mu9.
"""

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[2]
DEFAULT_REAL_CSV_DIR = SCRIPT_DIR / "csv"
DEFAULT_SIM_JSON = (
    REPO_DIR
    / "runs"
    / "mujoco_experiment_energy_dynamic_calibrated_full10_50hz.json"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "dynamic_energy_comparison"
A4_WIDTH_IN = 8.2677165354
FIGURE_HEIGHT_IN = 6.65
DEFAULT_FINAL_INTERVAL_DURATION_S = 2.80
TEXT_SIZE_PT = 12
ARIAL = "Arial"
TIMES = "Times New Roman"
REAL_COLOR = "#C76561"
SIM_COLOR = "#5B84B1"
TRANSFORM_COLOR = "#F2A65A"
SUSTAIN_COLOR = "#7AA6C2"
REAL_TRANSFORM_BAR_COLOR = "#F2A65A"
REAL_SUSTAIN_BAR_COLOR = "#7AA6C2"
SIM_TRANSFORM_BAR_COLOR = "#F8C58A"
SIM_SUSTAIN_BAR_COLOR = "#B8D0DF"

EXPECTED_TARGET_MODES = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    dtype=int,
)
MOTOR_IDS = tuple(range(17))
GRID_POINTS = 900
DEFAULT_FALLBACK_TRANSITION_WINDOW_S = 0.8
KINEMATIC_SMOOTH_WINDOW_S = 0.12
KINEMATIC_LOCAL_RANGE_WINDOW_S = 0.30
KINEMATIC_HOLD_WINDOW_S = 0.30
KINEMATIC_PLATEAU_WINDOW_S = 0.45
KINEMATIC_POSITION_TOL_FRAC = 0.03
KINEMATIC_LOCAL_RANGE_TOL_FRAC = 0.02
KINEMATIC_SPEED_TOL_FRAC_PER_S = 0.12
KINEMATIC_POSITION_ABS_TOL = {
    "encoder_ticks": 18.0,
    "qpos": 0.025,
}
KINEMATIC_LOCAL_RANGE_ABS_TOL = {
    "encoder_ticks": 8.0,
    "qpos": 0.012,
}
KINEMATIC_SPEED_ABS_TOL = {
    "encoder_ticks": 45.0,
    "qpos": 0.08,
}
KINEMATIC_MIN_TRANSITION_WINDOW_S = 0.35
MU = "\u03bc"
ARROW = "\u2192"
PHASE_STEP_INDICES = np.arange(len(EXPECTED_TARGET_MODES))
TRANSITIONS = [
    f"{MU}{(0 if index == 0 else EXPECTED_TARGET_MODES[index - 1]) + 1}"
    f"{ARROW}{MU}{EXPECTED_TARGET_MODES[index] + 1}"
    for index in PHASE_STEP_INDICES
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot real-vs-calibrated-MuJoCo transformation-loop power and "
            "phase-separated energy."
        )
    )
    parser.add_argument(
        "--real-csv-dir",
        type=Path,
        default=DEFAULT_REAL_CSV_DIR,
        help="Directory containing trans_exp_XX_*/ CSV folders.",
    )
    parser.add_argument(
        "--sim-json",
        type=Path,
        default=DEFAULT_SIM_JSON,
        help="Calibrated MuJoCo JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--fallback-transition-window-s",
        type=float,
        default=DEFAULT_FALLBACK_TRANSITION_WINDOW_S,
        help=(
            "Fallback duration after a transform command when no power "
            "change is detected in the log."
        ),
    )
    parser.add_argument(
        "--final-interval-duration-s",
        type=float,
        default=DEFAULT_FINAL_INTERVAL_DURATION_S,
        help=(
            "Fixed observation window after the final transform command. "
            "This replaces the old log-end truncation for the final interval."
        ),
    )
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def select_latest_trial_dirs(csv_dir: Path) -> list[Path]:
    pattern = re.compile(
        r"^trans_exp_(?P<experiment>\d{2})_(?P<timestamp>\d{8}_\d{6})$"
    )
    newest: dict[int, tuple[str, Path]] = {}
    for path in csv_dir.iterdir():
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

    missing = sorted(set(range(1, 11)) - set(newest))
    if missing:
        raise RuntimeError(f"Missing real experiment CSV folders: {missing}")
    return [newest[index][1] for index in range(1, 11)]


def total_power_from_real_log(log: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    time_s = log["bag_time_sec"].to_numpy(dtype=float)
    power_w = np.zeros(len(log), dtype=float)
    for motor_id in MOTOR_IDS:
        voltage = log[f"U[{motor_id}]"].to_numpy(dtype=float)
        current = log[f"I[{motor_id}]"].to_numpy(dtype=float)
        power_w += voltage * np.abs(current)
    return time_s, power_w


def total_power_from_json_logs(log_records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.array(
        [float(record["relative_time_s"]) for record in log_records],
        dtype=float,
    )
    power_w = []
    for record in log_records:
        voltage = np.asarray(record["msg"]["U"], dtype=float)
        current = np.asarray(record["msg"]["I"], dtype=float)
        power_w.append(float(np.sum(voltage * np.abs(current))))
    return time_s, np.asarray(power_w, dtype=float)


def motor_position_from_real_log(log: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    time_s = log["bag_time_sec"].to_numpy(dtype=float)
    positions = np.column_stack(
        [log[f"P[{motor_id}]"].to_numpy(dtype=float) for motor_id in MOTOR_IDS]
    )
    return time_s, positions


def joint_position_from_json_logs(log_records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.array(
        [float(record["relative_time_s"]) for record in log_records],
        dtype=float,
    )
    missing_qpos = [
        index for index, record in enumerate(log_records) if "qpos" not in record["msg"]
    ]
    if missing_qpos:
        raise RuntimeError(
            "MuJoCo joint convergence requires true qpos in the JSON log; "
            f"missing qpos in {len(missing_qpos)} / {len(log_records)} records."
        )
    positions = np.asarray(
        [record["msg"]["qpos"] for record in log_records],
        dtype=float,
    )
    return time_s, positions


def rolling_median(values: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return values
    if window_samples % 2 == 0:
        window_samples += 1
    return (
        pd.Series(values)
        .rolling(window_samples, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def rolling_median_matrix(values: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return values
    if window_samples % 2 == 0:
        window_samples += 1
    return (
        pd.DataFrame(values)
        .rolling(window_samples, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def rolling_range(values: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return np.zeros_like(values)
    if window_samples % 2 == 0:
        window_samples += 1
    series = pd.Series(values)
    high = series.rolling(window_samples, center=True, min_periods=1).max()
    low = series.rolling(window_samples, center=True, min_periods=1).min()
    return (high - low).to_numpy(dtype=float)


def rolling_range_matrix(values: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return np.zeros_like(values)
    if window_samples % 2 == 0:
        window_samples += 1
    frame = pd.DataFrame(values)
    high = frame.rolling(window_samples, center=True, min_periods=1).max()
    low = frame.rolling(window_samples, center=True, min_periods=1).min()
    return (high - low).to_numpy(dtype=float)


def rms(values: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=axis))


def final_interval_end(
    command_time_s: np.ndarray,
    log_end_s: float,
    final_interval_duration_s: float,
) -> float:
    requested_end = float(command_time_s[-1]) + final_interval_duration_s
    if requested_end > log_end_s + 1e-6:
        raise RuntimeError(
            "The requested final interval duration exceeds the available log "
            f"tail: requested {final_interval_duration_s:.3f} s, available "
            f"{log_end_s - float(command_time_s[-1]):.3f} s."
        )
    return requested_end


def joint_convergence_phase_windows(
    position_time_s: np.ndarray,
    positions: np.ndarray,
    command_time_s: np.ndarray,
    final_end_s: float,
    fallback_window_s: float,
    position_kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split phases from motor-position convergence rather than power shape."""

    if position_kind not in KINEMATIC_POSITION_ABS_TOL:
        raise ValueError(f"Unsupported position kind: {position_kind}")

    dt = np.diff(position_time_s)
    median_dt = float(np.median(dt))
    if median_dt <= 0:
        raise ValueError("Position log timestamps must be strictly increasing.")

    smooth_samples = max(3, int(round(KINEMATIC_SMOOTH_WINDOW_S / median_dt)))
    range_samples = max(3, int(round(KINEMATIC_LOCAL_RANGE_WINDOW_S / median_dt)))
    hold_samples = max(3, int(np.ceil(KINEMATIC_HOLD_WINDOW_S / median_dt)))

    smoothed_all = rolling_median_matrix(positions, smooth_samples)
    speed_all = np.gradient(smoothed_all, position_time_s, axis=0)
    range_all = rolling_range_matrix(smoothed_all, range_samples)

    transition_start = []
    transition_end = []
    sustain_end = []

    for step in PHASE_STEP_INDICES:
        interval_start = float(command_time_s[step])
        interval_end = (
            float(command_time_s[step + 1])
            if step + 1 < len(command_time_s)
            else final_end_s
        )
        interval_end = min(interval_end, final_end_s)
        if interval_end <= interval_start:
            transition_start.append(interval_start)
            transition_end.append(interval_start)
            sustain_end.append(interval_start)
            continue

        inside = (position_time_s >= interval_start) & (position_time_s <= interval_end)
        interval_time = position_time_s[inside]
        if len(interval_time) < 3:
            phase_end = min(interval_start + fallback_window_s, interval_end)
            transition_start.append(interval_start)
            transition_end.append(phase_end)
            sustain_end.append(interval_end)
            continue

        interval_pos = smoothed_all[inside]
        interval_speed = speed_all[inside]
        interval_range = range_all[inside]

        interval_duration = interval_end - interval_start
        plateau_window_s = min(
            KINEMATIC_PLATEAU_WINDOW_S,
            max(0.20, 0.25 * interval_duration),
        )
        plateau_mask = interval_time >= (interval_end - plateau_window_s)
        if np.count_nonzero(plateau_mask) < 2:
            plateau_mask = np.zeros_like(interval_time, dtype=bool)
            plateau_mask[-min(3, len(interval_time)) :] = True
        plateau_pos = np.median(interval_pos[plateau_mask], axis=0)

        position_error = rms(interval_pos - plateau_pos)
        local_range = rms(interval_range)
        speed = rms(interval_speed)
        movement_scale = max(
            float(np.percentile(position_error, 95)),
            float(rms(interval_pos[0:1] - plateau_pos)[0]),
            np.finfo(float).eps,
        )
        position_tolerance = max(
            KINEMATIC_POSITION_ABS_TOL[position_kind],
            KINEMATIC_POSITION_TOL_FRAC * movement_scale,
        )
        local_range_tolerance = max(
            KINEMATIC_LOCAL_RANGE_ABS_TOL[position_kind],
            KINEMATIC_LOCAL_RANGE_TOL_FRAC * movement_scale,
        )
        speed_tolerance = max(
            KINEMATIC_SPEED_ABS_TOL[position_kind],
            KINEMATIC_SPEED_TOL_FRAC_PER_S
            * movement_scale
            / max(interval_duration, np.finfo(float).eps),
        )

        search_start = interval_start + min(
            KINEMATIC_MIN_TRANSITION_WINDOW_S,
            0.50 * interval_duration,
        )
        settled_index = None
        for index in range(len(interval_time)):
            if interval_time[index] < search_start:
                continue
            end_index = min(index + hold_samples, len(interval_time))
            if end_index - index < max(3, hold_samples // 2):
                break
            if (
                float(np.max(position_error[index:end_index]))
                <= position_tolerance
                and float(np.max(local_range[index:end_index]))
                <= local_range_tolerance
                and float(np.percentile(speed[index:end_index], 75))
                <= speed_tolerance
            ):
                settled_index = index
                break

        if settled_index is not None:
            phase_end = float(interval_time[settled_index])
        else:
            unsettled = (
                (position_error > position_tolerance)
                | (local_range > local_range_tolerance)
                | (speed > speed_tolerance)
            )
            unsettled_indices = np.flatnonzero(
                unsettled & (interval_time >= search_start)
            )
            if len(unsettled_indices):
                done_index = min(int(unsettled_indices[-1]) + 1, len(interval_time) - 1)
                phase_end = float(interval_time[done_index])
            else:
                phase_end = min(interval_start + fallback_window_s, interval_end)

        phase_end = max(
            phase_end,
            min(interval_start + fallback_window_s, interval_end),
        )
        phase_end = min(max(phase_end, interval_start), interval_end)
        transition_start.append(interval_start)
        transition_end.append(phase_end)
        sustain_end.append(interval_end)

    return (
        np.asarray(transition_start, dtype=float),
        np.asarray(transition_end, dtype=float),
        np.asarray(sustain_end, dtype=float),
    )


def clipped_series(
    time_s: np.ndarray,
    values: np.ndarray,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if start_s >= end_s:
        raise ValueError(f"Invalid clip window [{start_s}, {end_s}]")
    inside = (time_s > start_s) & (time_s < end_s)
    clipped_time = np.concatenate(([start_s], time_s[inside], [end_s]))
    clipped_values = np.concatenate(
        (
            [np.interp(start_s, time_s, values)],
            values[inside],
            [np.interp(end_s, time_s, values)],
        )
    )
    return clipped_time, clipped_values


def integrate_window(
    time_s: np.ndarray,
    values: np.ndarray,
    start_s: float,
    end_s: float,
) -> float:
    if end_s <= start_s:
        return 0.0
    clipped_time, clipped_values = clipped_series(time_s, values, start_s, end_s)
    return float(
        np.sum(
            0.5
            * (clipped_values[:-1] + clipped_values[1:])
            * np.diff(clipped_time)
        )
    )


def cumulative_trapezoid(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    increments = 0.5 * (values[:-1] + values[1:]) * np.diff(time_s)
    return np.concatenate(([0.0], np.cumsum(increments)))


def load_real_trial(
    path: Path,
    grid: np.ndarray,
    fallback_window_s: float,
    final_interval_duration_s: float,
) -> dict[str, object]:
    transform = pd.read_csv(path / "crimson_control_transform.csv")
    log = pd.read_csv(path / "dynamixel_control_log.csv")

    modes = transform["mode"].to_numpy(dtype=int)
    if not np.array_equal(modes, EXPECTED_TARGET_MODES):
        raise RuntimeError(f"{path.name}: unexpected transform modes {modes}")

    log_time, power_w = total_power_from_real_log(log)
    position_time, positions_tick = motor_position_from_real_log(log)
    command_time = transform["bag_time_sec"].to_numpy(dtype=float)
    start_s = float(command_time[PHASE_STEP_INDICES[0]])
    end_s = final_interval_end(
        command_time,
        min(float(log_time[-1]), float(position_time[-1])),
        final_interval_duration_s,
    )
    duration_s = end_s - start_s
    clipped_time, clipped_power = clipped_series(log_time, power_w, start_s, end_s)
    elapsed = clipped_time - start_s
    normalized_time = elapsed / duration_s

    transition_energy = []
    sustain_energy = []
    interval_energy = []
    interval_duration = []
    phase_start_time, done_time, sustain_end_time = joint_convergence_phase_windows(
        position_time,
        positions_tick,
        command_time,
        end_s,
        fallback_window_s,
        "encoder_ticks",
    )
    interval_start_time = command_time[PHASE_STEP_INDICES]
    for interval_start, phase_start, done, next_command in zip(
        interval_start_time,
        phase_start_time,
        done_time,
        sustain_end_time,
        strict=True,
    ):
        transition_start = max(float(phase_start), float(interval_start))
        transition_end = min(float(done), float(next_command))
        total = integrate_window(log_time, power_w, interval_start, next_command)
        transition = integrate_window(log_time, power_w, transition_start, transition_end)
        sustain = total - transition
        transition_energy.append(transition)
        sustain_energy.append(sustain)
        interval_energy.append(total)
        interval_duration.append(next_command - interval_start)

    return {
        "name": path.name,
        "duration_s": duration_s,
        "power_grid": np.interp(grid, normalized_time, clipped_power),
        "energy_grid": np.interp(
            grid,
            normalized_time,
            cumulative_trapezoid(clipped_time, clipped_power),
        ),
        "command_norm": (interval_start_time - start_s) / duration_s,
        "phase_start_norm": (phase_start_time - start_s) / duration_s,
        "done_norm": (done_time - start_s) / duration_s,
        "sustain_end_norm": (sustain_end_time - start_s) / duration_s,
        "transition_energy_j": np.asarray(transition_energy),
        "sustain_energy_j": np.asarray(sustain_energy),
        "interval_energy_j": np.asarray(interval_energy),
        "interval_duration_s": np.asarray(interval_duration),
    }


def load_sim_trials(
    json_path: Path,
    grid: np.ndarray,
    fallback_window_s: float,
    final_interval_duration_s: float,
) -> list[dict[str, object]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    segment_rows = data["segments"]
    trials = []
    for experiment in data["experiments"]:
        experiment_id = int(experiment["experiment"])
        records = experiment["records"]
        transforms = records["/crimson_control/transform"]
        logs = records["/dynamixel_control/log"]
        target_modes = np.array(
            [int(record["msg"]["mode"]) for record in transforms],
            dtype=int,
        )
        if not np.array_equal(target_modes, EXPECTED_TARGET_MODES):
            raise RuntimeError(
                f"Sim experiment {experiment_id}: unexpected target modes"
            )

        log_time, power_w = total_power_from_json_logs(logs)
        position_time, joint_qpos = joint_position_from_json_logs(logs)
        command_time = np.array(
            [float(record["relative_time_s"]) for record in transforms],
            dtype=float,
        )
        start_s = float(command_time[PHASE_STEP_INDICES[0]])
        end_s = final_interval_end(
            command_time,
            min(float(log_time[-1]), float(position_time[-1])),
            final_interval_duration_s,
        )
        duration_s = end_s - start_s
        clipped_time, clipped_power = clipped_series(log_time, power_w, start_s, end_s)
        elapsed = clipped_time - start_s
        normalized_time = elapsed / duration_s

        transition_energy = []
        sustain_energy = []
        interval_energy = []
        interval_duration = []
        rows = sorted(
            [
                row
                for row in segment_rows
                if int(row["experiment"]) == experiment_id
                and int(row["step"]) in PHASE_STEP_INDICES
            ],
            key=lambda row: int(row["step"]),
        )
        if len(rows) != len(PHASE_STEP_INDICES):
            raise RuntimeError(
                f"Sim experiment {experiment_id}: expected "
                f"{len(PHASE_STEP_INDICES)} segment rows, got {len(rows)}"
            )
        phase_start_time, done_time, sustain_end_time = joint_convergence_phase_windows(
            position_time,
            joint_qpos,
            command_time,
            end_s,
            fallback_window_s,
            "qpos",
        )
        interval_start_time = command_time[PHASE_STEP_INDICES]
        for interval_start, phase_start, done, next_command in zip(
            interval_start_time,
            phase_start_time,
            done_time,
            sustain_end_time,
            strict=True,
        ):
            transition_start = max(float(phase_start), float(interval_start))
            transition_end = min(float(done), float(next_command))
            total = integrate_window(log_time, power_w, interval_start, next_command)
            transition = integrate_window(log_time, power_w, transition_start, transition_end)
            sustain = total - transition
            transition_energy.append(transition)
            sustain_energy.append(sustain)
            interval_energy.append(total)
            interval_duration.append(next_command - interval_start)

        trials.append(
            {
                "name": experiment["bag_name"],
                "duration_s": duration_s,
                "power_grid": np.interp(grid, normalized_time, clipped_power),
                "energy_grid": np.interp(
                    grid,
                    normalized_time,
                    cumulative_trapezoid(clipped_time, clipped_power),
                ),
                "command_norm": (interval_start_time - start_s) / duration_s,
                "phase_start_norm": (phase_start_time - start_s) / duration_s,
                "done_norm": (done_time - start_s) / duration_s,
                "sustain_end_norm": (sustain_end_time - start_s) / duration_s,
                "transition_energy_j": np.asarray(transition_energy),
                "sustain_energy_j": np.asarray(sustain_energy),
                "interval_energy_j": np.asarray(interval_energy),
                "interval_duration_s": np.asarray(interval_duration),
            }
        )
    return trials


def stack(trials: list[dict[str, object]], key: str) -> np.ndarray:
    return np.vstack([np.asarray(trial[key], dtype=float) for trial in trials])


def summarize_trials(
    trials: list[dict[str, object]],
    grid: np.ndarray,
    common_duration_s: float,
) -> dict[str, object]:
    power = stack(trials, "power_grid")
    energy = stack(trials, "energy_grid")
    command_norm = stack(trials, "command_norm")
    phase_start_norm = stack(trials, "phase_start_norm")
    done_norm = stack(trials, "done_norm")
    sustain_end_norm = stack(trials, "sustain_end_norm")
    transition_energy = stack(trials, "transition_energy_j")
    sustain_energy = stack(trials, "sustain_energy_j")
    interval_energy = stack(trials, "interval_energy_j")
    interval_duration = stack(trials, "interval_duration_s")
    durations = np.asarray([float(trial["duration_s"]) for trial in trials])

    return {
        "n": len(trials),
        "duration_mean_s": float(durations.mean()),
        "duration_sd_s": float(durations.std(ddof=1)) if len(trials) > 1 else 0.0,
        "time_s": grid * common_duration_s,
        "power_mean_w": power.mean(axis=0),
        "power_sd_w": power.std(axis=0, ddof=1) if len(trials) > 1 else np.zeros(power.shape[1]),
        "energy_mean_j": energy.mean(axis=0),
        "energy_sd_j": energy.std(axis=0, ddof=1) if len(trials) > 1 else np.zeros(energy.shape[1]),
        "command_time_s": command_norm.mean(axis=0) * common_duration_s,
        "phase_start_time_s": phase_start_norm.mean(axis=0) * common_duration_s,
        "done_time_s": done_norm.mean(axis=0) * common_duration_s,
        "sustain_end_time_s": sustain_end_norm.mean(axis=0) * common_duration_s,
        "transition_duration_mean_s": (
            (done_norm - command_norm) * common_duration_s
        ).mean(axis=0),
        "transition_energy_mean_j": transition_energy.mean(axis=0),
        "transition_energy_sd_j": transition_energy.std(axis=0, ddof=1)
        if len(trials) > 1
        else np.zeros(transition_energy.shape[1]),
        "sustain_energy_mean_j": sustain_energy.mean(axis=0),
        "sustain_energy_sd_j": sustain_energy.std(axis=0, ddof=1)
        if len(trials) > 1
        else np.zeros(sustain_energy.shape[1]),
        "interval_energy_mean_j": interval_energy.mean(axis=0),
        "interval_energy_sd_j": interval_energy.std(axis=0, ddof=1)
        if len(trials) > 1
        else np.zeros(interval_energy.shape[1]),
        "interval_duration_mean_s": interval_duration.mean(axis=0),
    }


def metric_summary(real: dict[str, object], sim: dict[str, object]) -> dict[str, float]:
    real_power = np.asarray(real["interval_energy_mean_j"]) / np.asarray(
        real["interval_duration_mean_s"]
    )
    sim_power = np.asarray(sim["interval_energy_mean_j"]) / np.asarray(
        sim["interval_duration_mean_s"]
    )
    delta = sim_power - real_power
    return {
        "mae_w": float(np.mean(np.abs(delta))),
        "rmse_w": float(np.sqrt(np.mean(delta**2))),
        "bias_w": float(np.mean(delta)),
        "corr": float(np.corrcoef(real_power, sim_power)[0, 1]),
        "real_total_kj": float(np.sum(real["interval_energy_mean_j"]) / 1000.0),
        "sim_total_kj": float(np.sum(sim["interval_energy_mean_j"]) / 1000.0),
    }


def write_interval_summary(
    out_dir: Path,
    real: dict[str, object],
    sim: dict[str, object],
) -> pd.DataFrame:
    interval_count = len(TRANSITIONS)
    real_total = np.asarray(real["interval_energy_mean_j"])
    sim_total = np.asarray(sim["interval_energy_mean_j"])
    real_power = real_total / np.asarray(real["interval_duration_mean_s"])
    sim_power = sim_total / np.asarray(sim["interval_duration_mean_s"])
    df = pd.DataFrame(
        {
            "step": np.arange(interval_count),
            "transition": TRANSITIONS[:interval_count],
            "real_transition_energy_j": real["transition_energy_mean_j"],
            "real_sustain_energy_j": real["sustain_energy_mean_j"],
            "real_total_energy_j": real_total,
            "real_interval_power_w": real_power,
            "real_transition_duration_s": real["transition_duration_mean_s"],
            "sim_transition_energy_j": sim["transition_energy_mean_j"],
            "sim_sustain_energy_j": sim["sustain_energy_mean_j"],
            "sim_total_energy_j": sim_total,
            "sim_interval_power_w": sim_power,
            "sim_transition_duration_s": sim["transition_duration_mean_s"],
            "power_error_w": sim_power - real_power,
        }
    )
    df.to_csv(out_dir / "real_sim_interval_energy_summary.csv", index=False)
    return df


def shade_transition_regions(
    ax: plt.Axes,
    summary: dict[str, object],
) -> None:
    phase_start = np.asarray(summary["phase_start_time_s"], dtype=float)
    done = np.asarray(summary["done_time_s"], dtype=float)
    for start, transition_end in zip(
        phase_start,
        done,
        strict=True,
    ):
        if transition_end <= start:
            continue
        ax.axvspan(
            start,
            transition_end,
            color=TRANSFORM_COLOR,
            alpha=0.13,
            linewidth=0,
            zorder=0,
        )


def style_legend(legend) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.78)
    frame.set_edgecolor("#D6D6D6")
    frame.set_linewidth(0.45)
    frame.set_boxstyle("round,pad=0.25,rounding_size=0.12")
    legend.set_zorder(20)


def set_tick_font(
    ax: plt.Axes,
    x_family: str | None = TIMES,
    y_family: str | None = TIMES,
) -> None:
    if x_family is not None:
        for label in ax.get_xticklabels():
            label.set_fontfamily(x_family)
            label.set_fontsize(TEXT_SIZE_PT)
    if y_family is not None:
        for label in ax.get_yticklabels():
            label.set_fontfamily(y_family)
            label.set_fontsize(TEXT_SIZE_PT)


def postprocess_svg_font_units(svg_path: Path) -> None:
    svg = svg_path.read_text(encoding="utf-8")
    svg = re.sub(r"font: 700 12px", "font: 700 12pt", svg)
    svg = re.sub(r"font: 12px", "font: 12pt", svg)
    svg_path.write_text(svg, encoding="utf-8")


def plot_comparison(
    out_dir: Path,
    real: dict[str, object],
    sim: dict[str, object],
    metrics: dict[str, float],
    fallback_window_s: float,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": ARIAL,
            "font.sans-serif": [ARIAL],
            "font.size": TEXT_SIZE_PT,
            "axes.labelsize": TEXT_SIZE_PT,
            "xtick.labelsize": TEXT_SIZE_PT,
            "ytick.labelsize": TEXT_SIZE_PT,
            "legend.fontsize": TEXT_SIZE_PT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": dpi,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(A4_WIDTH_IN, FIGURE_HEIGHT_IN))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.32)
    left = gs[0, 0].subgridspec(2, 1, hspace=0.09)
    ax_real = fig.add_subplot(left[0, 0])
    ax_sim = fig.add_subplot(left[1, 0], sharex=ax_real)
    ax_bar = fig.add_subplot(gs[0, 1])

    time_s = np.asarray(real["time_s"], dtype=float)
    real_mean = np.asarray(real["power_mean_w"], dtype=float)
    real_sd = np.asarray(real["power_sd_w"], dtype=float)
    sim_mean = np.asarray(sim["power_mean_w"], dtype=float)
    sim_sd = np.asarray(sim["power_sd_w"], dtype=float)

    shade_transition_regions(ax_real, real)
    ax_real.fill_between(
        time_s,
        np.maximum(0, real_mean - real_sd),
        real_mean + real_sd,
        color=REAL_COLOR,
        alpha=0.12,
        linewidth=0,
    )
    ax_real.plot(time_s, real_mean, color=REAL_COLOR, lw=1.8, label="Real")

    shade_transition_regions(ax_sim, sim)
    if np.max(sim_sd) > 1e-9:
        ax_sim.fill_between(
            time_s,
            np.maximum(0, sim_mean - sim_sd),
            sim_mean + sim_sd,
            color=SIM_COLOR,
            alpha=0.12,
            linewidth=0,
        )
    ax_sim.plot(
        time_s,
        sim_mean,
        color=SIM_COLOR,
        lw=1.65,
        label="Calibrated MuJoCo",
    )
    ax_real.set_ylabel("Real power (W)")
    ax_sim.set_ylabel("MuJoCo power (W)")
    ax_sim.set_xlabel("Elapsed time mapped to mean real-loop duration (s)")
    for ax in (ax_real, ax_sim):
        ax.set_xlim(time_s[0], time_s[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, color="#AAB4C3", alpha=0.20, linewidth=0.65)
    ax_real.tick_params(axis="x", labelbottom=False)
    ax_real.text(
        -0.18,
        1.04,
        "a",
        transform=ax_real.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=TEXT_SIZE_PT,
        fontfamily=ARIAL,
        clip_on=False,
    )

    phase_handles = [
        Patch(facecolor=TRANSFORM_COLOR, alpha=0.18, label="Transformation phase"),
    ]
    legend_real = ax_real.legend(
        handles=[
            ax_real.lines[0],
            *phase_handles,
        ],
        labels=["Real", "Transformation phase"],
        loc="upper right",
        frameon=True,
        fancybox=True,
        fontsize=TEXT_SIZE_PT,
        handlelength=1.25,
        handletextpad=0.45,
        borderpad=0.28,
        labelspacing=0.28,
        borderaxespad=0.25,
    )
    style_legend(legend_real)
    legend_sim = ax_sim.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        fontsize=TEXT_SIZE_PT,
        handlelength=1.25,
        handletextpad=0.45,
        borderpad=0.28,
        labelspacing=0.28,
        borderaxespad=0.25,
    )
    style_legend(legend_sim)

    x = np.arange(len(TRANSITIONS))
    width = 0.36
    real_trans = np.asarray(real["transition_energy_mean_j"]) / 1000.0
    real_sustain = np.asarray(real["sustain_energy_mean_j"]) / 1000.0
    sim_trans = np.asarray(sim["transition_energy_mean_j"]) / 1000.0
    sim_sustain = np.asarray(sim["sustain_energy_mean_j"]) / 1000.0
    ax_bar.bar(
        x - width / 2,
        real_trans,
        width,
        color=REAL_TRANSFORM_BAR_COLOR,
        edgecolor="#5b6470",
        linewidth=0.45,
        label="Real transform",
    )
    ax_bar.bar(
        x - width / 2,
        real_sustain,
        width,
        bottom=real_trans,
        color=REAL_SUSTAIN_BAR_COLOR,
        edgecolor="#5b6470",
        linewidth=0.45,
        label="Real sustain",
    )
    ax_bar.bar(
        x + width / 2,
        sim_trans,
        width,
        color=SIM_TRANSFORM_BAR_COLOR,
        edgecolor="#5b6470",
        linewidth=0.65,
        label="Sim transform",
    )
    ax_bar.bar(
        x + width / 2,
        sim_sustain,
        width,
        bottom=sim_trans,
        color=SIM_SUSTAIN_BAR_COLOR,
        edgecolor="#5b6470",
        linewidth=0.65,
        label="Sim sustain",
    )
    ax_bar.set_ylabel("Energy per command interval (kJ)")
    ax_bar.set_xlabel("Command interval")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [str(index) for index in range(1, len(x) + 1)],
        rotation=0,
        ha="center",
        fontfamily=TIMES,
        fontsize=TEXT_SIZE_PT,
    )
    ax_bar.grid(True, axis="y", color="#AAB4C3", alpha=0.22, linewidth=0.65)
    ax_bar.set_axisbelow(True)
    ax_bar.text(
        -0.18,
        1.04,
        "b",
        transform=ax_bar.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=TEXT_SIZE_PT,
        fontfamily=ARIAL,
        clip_on=False,
    )
    legend_bar = ax_bar.legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        fancybox=True,
        ncol=1,
        fontsize=TEXT_SIZE_PT,
        handlelength=1.25,
        handletextpad=0.45,
        borderpad=0.28,
        labelspacing=0.18,
        columnspacing=0.60,
        borderaxespad=0.0,
    )
    style_legend(legend_bar)

    set_tick_font(ax_real, x_family=None, y_family=TIMES)
    set_tick_font(ax_sim, x_family=TIMES, y_family=TIMES)
    set_tick_font(ax_bar, x_family=TIMES, y_family=TIMES)

    fig.subplots_adjust(top=0.94, bottom=0.115, left=0.095, right=0.985)

    for suffix in ("png", "pdf", "svg"):
        out_path = out_dir / f"real_sim_energy_comparison.{suffix}"
        fig.savefig(
            out_path,
            dpi=dpi if suffix == "png" else None,
        )
        if suffix == "svg":
            postprocess_svg_font_units(out_path)
    plt.close(fig)


def write_curve_summary(
    out_dir: Path,
    grid: np.ndarray,
    real: dict[str, object],
    sim: dict[str, object],
) -> None:
    pd.DataFrame(
        {
            "normalized_cycle_time": grid,
            "time_s": real["time_s"],
            "real_power_mean_w": real["power_mean_w"],
            "real_power_sd_w": real["power_sd_w"],
            "sim_power_mean_w": sim["power_mean_w"],
            "sim_power_sd_w": sim["power_sd_w"],
            "real_cumulative_energy_mean_j": real["energy_mean_j"],
            "sim_cumulative_energy_mean_j": sim["energy_mean_j"],
        }
    ).to_csv(out_dir / "real_sim_power_curve_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, GRID_POINTS)

    real_trials = [
        load_real_trial(
            path,
            grid,
            args.fallback_transition_window_s,
            args.final_interval_duration_s,
        )
        for path in select_latest_trial_dirs(args.real_csv_dir)
    ]
    sim_trials = load_sim_trials(
        args.sim_json,
        grid,
        args.fallback_transition_window_s,
        args.final_interval_duration_s,
    )
    common_duration = float(np.mean([trial["duration_s"] for trial in real_trials]))
    real = summarize_trials(real_trials, grid, common_duration)
    sim = summarize_trials(sim_trials, grid, common_duration)
    metrics = metric_summary(real, sim)
    interval = write_interval_summary(args.out_dir, real, sim)
    write_curve_summary(args.out_dir, grid, real, sim)
    plot_comparison(
        args.out_dir,
        real,
        sim,
        metrics,
        args.fallback_transition_window_s,
        args.dpi,
    )

    print(f"Real trials: {len(real_trials)} from {args.real_csv_dir}")
    print(f"Sim trials: {len(sim_trials)} from {args.sim_json}")
    print(
        "Joint-position convergence phase detection: "
        f"fallback={args.fallback_transition_window_s:.3f} s, "
        f"final_interval={args.final_interval_duration_s:.3f} s"
    )
    print(f"Output directory: {args.out_dir}")
    print(
        "Interval power metrics: "
        f"MAE={metrics['mae_w']:.3f} W, "
        f"RMSE={metrics['rmse_w']:.3f} W, "
        f"bias={metrics['bias_w']:.3f} W, "
        f"corr={metrics['corr']:.3f}"
    )
    print(interval.head().to_string(index=False))


if __name__ == "__main__":
    main()
