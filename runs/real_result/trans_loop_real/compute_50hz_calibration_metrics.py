from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import plot_real_sim_energy_comparison as fig


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "calibration_50hz"
OUT_CSV = OUT_DIR / "real_sim_50hz_first15_interval_power.csv"
OUT_MD = OUT_DIR / "dynamic_50hz_calibration_metrics.md"


def trial_stats(trials: list[dict[str, object]], n_intervals: int) -> dict[str, np.ndarray]:
    energies = []
    durations = []
    loop_powers = []
    interval_energies = []
    interval_durations = []
    interval_powers = []

    for trial in trials:
        energy = np.asarray(trial["interval_energy_j"], dtype=float)[:n_intervals]
        duration = np.asarray(trial["interval_duration_s"], dtype=float)[:n_intervals]
        power = energy / duration
        energies.append(float(np.sum(energy)))
        durations.append(float(np.sum(duration)))
        loop_powers.append(float(np.sum(energy) / np.sum(duration)))
        interval_energies.append(energy)
        interval_durations.append(duration)
        interval_powers.append(power)

    return {
        "energies": np.asarray(energies, dtype=float),
        "durations": np.asarray(durations, dtype=float),
        "loop_powers": np.asarray(loop_powers, dtype=float),
        "interval_energies": np.vstack(interval_energies),
        "interval_durations": np.vstack(interval_durations),
        "interval_powers": np.vstack(interval_powers),
    }


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values)), sd


def comparison_metrics(real: dict[str, np.ndarray], sim: dict[str, np.ndarray]) -> dict[str, float]:
    real_interval_power = real["interval_powers"].mean(axis=0)
    sim_interval_power = sim["interval_powers"].mean(axis=0)
    error = sim_interval_power - real_interval_power
    return {
        "mae_w": float(np.mean(np.abs(error))),
        "rmse_w": float(math.sqrt(float(np.mean(error**2)))),
        "bias_w": float(np.mean(error)),
        "corr": float(np.corrcoef(real_interval_power, sim_interval_power)[0, 1]),
        "max_abs_error_w": float(np.max(np.abs(error))),
    }


def segment_energy_diagnostic(json_path: Path, n_intervals: int) -> dict[str, float]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    totals = []
    for experiment_id in range(1, 11):
        rows = [
            row
            for row in data["segments"]
            if int(row["experiment"]) == experiment_id and int(row["step"]) < n_intervals
        ]
        totals.append(sum(float(row["total_energy_j"]) for row in rows))
    totals = np.asarray(totals, dtype=float)
    mean_j, sd_j = mean_sd(totals)
    return {
        "segment_total_energy_mean_kj": mean_j / 1000.0,
        "segment_total_energy_sd_kj": sd_j / 1000.0,
    }


def build_interval_table(real: dict[str, np.ndarray], sim: dict[str, np.ndarray]) -> pd.DataFrame:
    real_power = real["interval_powers"].mean(axis=0)
    sim_power = sim["interval_powers"].mean(axis=0)
    real_energy = real["interval_energies"].mean(axis=0)
    sim_energy = sim["interval_energies"].mean(axis=0)
    real_duration = real["interval_durations"].mean(axis=0)
    sim_duration = sim["interval_durations"].mean(axis=0)
    return pd.DataFrame(
        {
            "step": np.arange(15),
            "transition": fig.TRANSITIONS[:15],
            "real_interval_energy_j": real_energy,
            "sim_50hz_interval_energy_j": sim_energy,
            "real_interval_duration_s": real_duration,
            "sim_50hz_interval_duration_s": sim_duration,
            "real_interval_power_w": real_power,
            "sim_50hz_interval_power_w": sim_power,
            "power_error_w": sim_power - real_power,
        }
    )


def summary_block(
    label: str,
    real: dict[str, np.ndarray],
    sim: dict[str, np.ndarray],
) -> dict[str, float]:
    metrics = comparison_metrics(real, sim)
    real_duration_mean, real_duration_sd = mean_sd(real["durations"])
    sim_duration_mean, sim_duration_sd = mean_sd(sim["durations"])
    real_energy_mean, real_energy_sd = mean_sd(real["energies"])
    sim_energy_mean, sim_energy_sd = mean_sd(sim["energies"])
    real_power_mean, real_power_sd = mean_sd(real["loop_powers"])
    sim_power_mean, sim_power_sd = mean_sd(sim["loop_powers"])
    return {
        "label": label,
        "real_duration_mean_s": real_duration_mean,
        "real_duration_sd_s": real_duration_sd,
        "sim_duration_mean_s": sim_duration_mean,
        "sim_duration_sd_s": sim_duration_sd,
        "real_energy_mean_kj": real_energy_mean / 1000.0,
        "real_energy_sd_kj": real_energy_sd / 1000.0,
        "sim_energy_mean_kj": sim_energy_mean / 1000.0,
        "sim_energy_sd_kj": sim_energy_sd / 1000.0,
        "real_loop_power_mean_w": real_power_mean,
        "real_loop_power_sd_w": real_power_sd,
        "sim_loop_power_mean_w": sim_power_mean,
        "sim_loop_power_sd_w": sim_power_sd,
        "energy_difference_pct": 100.0 * (sim_energy_mean - real_energy_mean) / real_energy_mean,
        **metrics,
    }


def fmt_pm(mean: float, sd: float, digits: int) -> str:
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = np.linspace(0.0, 1.0, fig.GRID_POINTS)
    real_trial_dirs = fig.select_latest_trial_dirs(fig.DEFAULT_REAL_CSV_DIR)
    real_trials = [
        fig.load_real_trial(
            path,
            grid,
            fig.DEFAULT_FALLBACK_TRANSITION_WINDOW_S,
            fig.DEFAULT_FINAL_INTERVAL_DURATION_S,
        )
        for path in real_trial_dirs
    ]
    sim_trials = fig.load_sim_trials(
        fig.DEFAULT_SIM_JSON,
        grid,
        fig.DEFAULT_FALLBACK_TRANSITION_WINDOW_S,
        fig.DEFAULT_FINAL_INTERVAL_DURATION_S,
    )

    real_15 = trial_stats(real_trials, 15)
    sim_15 = trial_stats(sim_trials, 15)
    real_16 = trial_stats(real_trials, 16)
    sim_16 = trial_stats(sim_trials, 16)
    summary_15 = summary_block("first 15 inter-transform intervals", real_15, sim_15)
    summary_16 = summary_block("all 16 plotted command intervals", real_16, sim_16)
    segment_diag_15 = segment_energy_diagnostic(fig.DEFAULT_SIM_JSON, 15)
    segment_diag_16 = segment_energy_diagnostic(fig.DEFAULT_SIM_JSON, 16)

    interval_table = build_interval_table(real_15, sim_15)
    interval_table.to_csv(OUT_CSV, index=False)

    md = f"""# Dynamic 50 Hz Calibration Metrics

## Data Sources

- Real voltage/current logs: `{fig.DEFAULT_REAL_CSV_DIR}`
- MuJoCo 50 Hz JSON: `{fig.DEFAULT_SIM_JSON}`
- Real trials selected: `{', '.join(path.name for path in real_trial_dirs)}`

## Method

This computation reports two scopes. The current calibration-set comparison
includes all 16 command intervals, with the final `{fig.TRANSITIONS[15]}`
command evaluated over a fixed {fig.DEFAULT_FINAL_INTERVAL_DURATION_S:.2f} s
post-command window. A first-15-interval result, from
`{fig.TRANSITIONS[0]}` through `{fig.TRANSITIONS[14]}`, is retained only for
compatibility with the earlier analysis scope.

For each trial and interval, energy was obtained by trapezoidal integration of
the U/I-derived power curve over the command-to-next-command window. Interval
mean power was then calculated as `energy / duration`. The reported interval
error metrics compare the trial-averaged interval mean power of the 50 Hz
MuJoCo replay against the trial-averaged real interval mean power.

## First 15 Inter-Transform Intervals

| Metric | Real robot | 50 Hz MuJoCo log-integrated replay |
|---|---:|---:|
| Duration (s) | {fmt_pm(summary_15['real_duration_mean_s'], summary_15['real_duration_sd_s'], 3)} | {fmt_pm(summary_15['sim_duration_mean_s'], summary_15['sim_duration_sd_s'], 3)} |
| Energy (kJ) | {fmt_pm(summary_15['real_energy_mean_kj'], summary_15['real_energy_sd_kj'], 4)} | {fmt_pm(summary_15['sim_energy_mean_kj'], summary_15['sim_energy_sd_kj'], 4)} |
| Mean power (W) | {fmt_pm(summary_15['real_loop_power_mean_w'], summary_15['real_loop_power_sd_w'], 2)} | {fmt_pm(summary_15['sim_loop_power_mean_w'], summary_15['sim_loop_power_sd_w'], 2)} |

- Energy difference: `{summary_15['energy_difference_pct']:.3f}%` (sim - real), i.e. within `{abs(summary_15['energy_difference_pct']):.2f}%`.
- Interval-power MAE: `{summary_15['mae_w']:.3f} W`
- Interval-power RMSE: `{summary_15['rmse_w']:.3f} W`
- Interval-power bias: `{summary_15['bias_w']:.3f} W`
- Interval-power correlation: `{summary_15['corr']:.3f}`
- Maximum absolute interval-power error: `{summary_15['max_abs_error_w']:.3f} W`

## All 16 Command Intervals

This is the current calibration-set comparison scope. The final interval uses
a fixed `{fig.DEFAULT_FINAL_INTERVAL_DURATION_S:.2f} s` post-command window so
the real and MuJoCo traces are compared over the same final observation
duration.

| Metric | Real robot | 50 Hz MuJoCo log-integrated replay |
|---|---:|---:|
| Duration (s) | {fmt_pm(summary_16['real_duration_mean_s'], summary_16['real_duration_sd_s'], 3)} | {fmt_pm(summary_16['sim_duration_mean_s'], summary_16['sim_duration_sd_s'], 3)} |
| Energy (kJ) | {fmt_pm(summary_16['real_energy_mean_kj'], summary_16['real_energy_sd_kj'], 4)} | {fmt_pm(summary_16['sim_energy_mean_kj'], summary_16['sim_energy_sd_kj'], 4)} |
| Mean power (W) | {fmt_pm(summary_16['real_loop_power_mean_w'], summary_16['real_loop_power_sd_w'], 2)} | {fmt_pm(summary_16['sim_loop_power_mean_w'], summary_16['sim_loop_power_sd_w'], 2)} |

- Energy difference: `{summary_16['energy_difference_pct']:.3f}%` (sim - real), i.e. within `{abs(summary_16['energy_difference_pct']):.2f}%`.
- Interval-power MAE: `{summary_16['mae_w']:.3f} W`
- Interval-power RMSE: `{summary_16['rmse_w']:.3f} W`
- Interval-power bias: `{summary_16['bias_w']:.3f} W`
- Interval-power correlation: `{summary_16['corr']:.3f}`

## Source-Consistency Note

The final figure and the response text should use the U/I log-integrated 50 Hz
values above. The JSON `segments` table gives a different internal segment
energy because it is produced by the simulator's segment accumulator rather
than by re-integrating the exported `/dynamixel_control/log` U/I curve used for
the plotted power trace.

- JSON segment first-15 energy: `{segment_diag_15['segment_total_energy_mean_kj']:.4f} +/- {segment_diag_15['segment_total_energy_sd_kj']:.4f} kJ`
- JSON segment all-16 energy: `{segment_diag_16['segment_total_energy_mean_kj']:.4f} +/- {segment_diag_16['segment_total_energy_sd_kj']:.4f} kJ`

For consistency, use the log-integrated values when reporting this comparison;
do not mix them with values from the simulator's internal segment accumulator.
"""
    OUT_MD.write_text(md, encoding="utf-8")

    print(md)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
