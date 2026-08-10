from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import plot_real_sim_energy_comparison as fig


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[2]
DEFAULT_SOURCE_JSON = (
    SCRIPT_DIR
    / "dynamic_scan"
    / "json"
    / "mujoco_frame30_leg10_kvdiv11p5.json"
)
DEFAULT_OUTPUT_JSON = (
    REPO_DIR
    / "runs"
    / "mujoco_experiment_energy_dynamic_calibrated_full10_50hz.json"
)
DEFAULT_OFFSET_CSV = (
    SCRIPT_DIR
    / "dynamic_energy_calibration"
    / "mujoco_dynamic_interval_residual_offsets.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply one constant power residual per command interval to a 50 Hz "
            "MuJoCo replay so the plotted 16 interval energies match the real "
            "mean interval energies. The dynamic qpos trace and phase windows "
            "are not changed."
        )
    )
    parser.add_argument("--source-json", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--offset-csv", type=Path, default=DEFAULT_OFFSET_CSV)
    parser.add_argument(
        "--real-csv-dir",
        type=Path,
        default=fig.DEFAULT_REAL_CSV_DIR,
        help="Real transformation-loop CSV directory.",
    )
    return parser.parse_args()


def load_real_summary(real_csv_dir: Path) -> tuple[dict[str, object], float]:
    grid = np.linspace(0.0, 1.0, fig.GRID_POINTS)
    trials = [
        fig.load_real_trial(
            path,
            grid,
            fig.DEFAULT_FALLBACK_TRANSITION_WINDOW_S,
            fig.DEFAULT_FINAL_INTERVAL_DURATION_S,
        )
        for path in fig.select_latest_trial_dirs(real_csv_dir)
    ]
    common_duration = float(np.mean([trial["duration_s"] for trial in trials]))
    return fig.summarize_trials(trials, grid, common_duration), common_duration


def load_sim_summary(
    source_json: Path,
    common_duration_s: float,
) -> dict[str, object]:
    grid = np.linspace(0.0, 1.0, fig.GRID_POINTS)
    trials = fig.load_sim_trials(
        source_json,
        grid,
        fig.DEFAULT_FALLBACK_TRANSITION_WINDOW_S,
        fig.DEFAULT_FINAL_INTERVAL_DURATION_S,
    )
    return fig.summarize_trials(trials, grid, common_duration_s)


def fit_interval_offsets(
    real: dict[str, object],
    sim: dict[str, object],
) -> pd.DataFrame:
    real_energy = np.asarray(real["interval_energy_mean_j"], dtype=float)
    sim_energy = np.asarray(sim["interval_energy_mean_j"], dtype=float)
    sim_duration = np.asarray(sim["interval_duration_mean_s"], dtype=float)
    offset_w = (real_energy - sim_energy) / sim_duration
    return pd.DataFrame(
        {
            "step": np.arange(len(offset_w)),
            "transition": fig.TRANSITIONS,
            "power_offset_w": offset_w,
            "real_energy_j": real_energy,
            "source_sim_energy_j": sim_energy,
            "sim_duration_s": sim_duration,
        }
    )


def scale_current_to_power(msg: dict, new_power_w: float) -> None:
    voltage = np.asarray(msg["U"], dtype=float)
    current = np.asarray(msg["I"], dtype=float)
    old_power_w = float(np.sum(voltage * np.abs(current)))
    new_power_w = max(0.0, float(new_power_w))
    if old_power_w > 1e-12:
        msg["I"] = [float(value) for value in current * (new_power_w / old_power_w)]
        return

    voltage_sum = float(np.sum(voltage))
    if voltage_sum <= 0.0:
        voltage_sum = 12.0 * len(current)
    equal_current = new_power_w / voltage_sum
    msg["I"] = [float(equal_current) for _ in current]


def apply_offsets(
    source_json: Path,
    output_json: Path,
    offset_table: pd.DataFrame,
    offset_csv: Path,
) -> None:
    data = json.loads(source_json.read_text(encoding="utf-8"))
    offsets = offset_table["power_offset_w"].to_numpy(dtype=float)
    data["energy_calibration_note"] = {
        "source_json": str(source_json),
        "method": (
            "One constant power residual was fitted for each of the 16 plotted "
            "command intervals. Offsets were fitted to real mean interval "
            "energy using the same 50 Hz trace integration and fixed final "
            "interval window as the figure. The qpos trace and phase-window "
            "detection were not changed."
        ),
        "offset_csv": str(offset_csv),
    }

    for experiment in data["experiments"]:
        logs = experiment["records"]["/dynamixel_control/log"]
        transforms = experiment["records"]["/crimson_control/transform"]
        command_time = np.asarray(
            [float(record["relative_time_s"]) for record in transforms],
            dtype=float,
        )
        final_end = fig.final_interval_end(
            command_time,
            float(logs[-1]["relative_time_s"]),
            fig.DEFAULT_FINAL_INTERVAL_DURATION_S,
        )
        interval_end = np.concatenate([command_time[1:], [final_end]])

        for step, (start_s, end_s, offset_w) in enumerate(
            zip(command_time, interval_end, offsets, strict=True)
        ):
            for record in logs:
                time_s = float(record["relative_time_s"])
                in_window = time_s >= start_s and (
                    time_s < end_s or (step == len(offsets) - 1 and time_s <= end_s)
                )
                if not in_window:
                    continue
                msg = record["msg"]
                old_power_w = float(
                    np.sum(
                        np.asarray(msg["U"], dtype=float)
                        * np.abs(np.asarray(msg["I"], dtype=float))
                    )
                )
                scale_current_to_power(msg, old_power_w + float(offset_w))

    output_json.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    real, common_duration_s = load_real_summary(args.real_csv_dir)
    sim = load_sim_summary(args.source_json, common_duration_s)
    offset_table = fit_interval_offsets(real, sim)
    args.offset_csv.parent.mkdir(parents=True, exist_ok=True)
    offset_table.to_csv(args.offset_csv, index=False)
    apply_offsets(args.source_json, args.output_json, offset_table, args.offset_csv)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.offset_csv}")
    print(offset_table.to_string(index=False))


if __name__ == "__main__":
    main()
