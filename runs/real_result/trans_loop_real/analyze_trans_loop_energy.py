from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "csv"
OUT_DIR = BASE_DIR / "analysis_energy"

EXPECTED_MODES = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    dtype=int,
)
MOTOR_IDS = tuple(range(17))
GRID_POINTS = 500


def select_bag_directories() -> list[Path]:
    pattern = re.compile(
        r"^trans_exp_(?P<experiment>\d{2})_(?P<timestamp>\d{8}_\d{6})$"
    )
    newest_by_experiment: dict[int, tuple[str, Path]] = {}

    for path in CSV_DIR.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        experiment = int(match.group("experiment"))
        if not 1 <= experiment <= 10:
            continue
        timestamp = match.group("timestamp")
        current = newest_by_experiment.get(experiment)
        if current is None or timestamp > current[0]:
            newest_by_experiment[experiment] = (timestamp, path)

    missing = sorted(set(range(1, 11)) - set(newest_by_experiment))
    if missing:
        raise RuntimeError(f"Missing experiment directories: {missing}")

    return [newest_by_experiment[i][1] for i in range(1, 11)]


def insert_window_boundaries(
    time_sec: np.ndarray,
    values: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    median_interval = float(np.median(np.diff(time_sec)))
    tolerance_sec = 1.5 * median_interval
    if (
        start_sec < time_sec[0] - tolerance_sec
        or end_sec > time_sec[-1] + tolerance_sec
    ):
        raise ValueError(
            "Transform window extends beyond the log by more than "
            "1.5 sampling intervals"
        )

    inside = (time_sec > start_sec) & (time_sec < end_sec)
    clipped_time = np.concatenate(([start_sec], time_sec[inside], [end_sec]))
    clipped_values = np.concatenate(
        (
            [np.interp(start_sec, time_sec, values)],
            values[inside],
            [np.interp(end_sec, time_sec, values)],
        )
    )
    return clipped_time, clipped_values


def cumulative_trapezoid(time_sec: np.ndarray, power_w: np.ndarray) -> np.ndarray:
    increments = 0.5 * (power_w[:-1] + power_w[1:]) * np.diff(time_sec)
    return np.concatenate(([0.0], np.cumsum(increments)))


def load_trial(path: Path) -> dict[str, object]:
    transform = pd.read_csv(path / "crimson_control_transform.csv")
    log = pd.read_csv(path / "dynamixel_control_log.csv")

    modes = transform["mode"].to_numpy(dtype=int)
    if not np.array_equal(modes, EXPECTED_MODES):
        raise RuntimeError(
            f"{path.name}: unexpected mode sequence {modes.tolist()}"
        )

    required_columns = ["bag_time_sec"] + [
        column
        for motor_id in MOTOR_IDS
        for column in (f"U[{motor_id}]", f"I[{motor_id}]")
    ]
    missing_columns = [column for column in required_columns if column not in log]
    if missing_columns:
        raise RuntimeError(f"{path.name}: missing columns {missing_columns}")

    log_time = log["bag_time_sec"].to_numpy(dtype=float)
    transform_time = transform["bag_time_sec"].to_numpy(dtype=float)
    start_sec = float(transform_time[0])
    end_sec = float(transform_time[-1])
    duration_sec = end_sec - start_sec

    total_power_w = np.zeros(len(log), dtype=float)
    for motor_id in MOTOR_IDS:
        voltage_v = log[f"U[{motor_id}]"].to_numpy(dtype=float)
        current_a = log[f"I[{motor_id}]"].to_numpy(dtype=float)
        total_power_w += voltage_v * np.abs(current_a)

    clipped_time, clipped_power = insert_window_boundaries(
        log_time, total_power_w, start_sec, end_sec
    )
    elapsed_sec = clipped_time - start_sec
    cumulative_energy_j = cumulative_trapezoid(clipped_time, clipped_power)

    normalized_grid = np.linspace(0.0, 1.0, GRID_POINTS)
    normalized_time = elapsed_sec / duration_sec
    aligned_power_w = np.interp(normalized_grid, normalized_time, clipped_power)
    aligned_energy_j = np.interp(
        normalized_grid, normalized_time, cumulative_energy_j
    )

    event_elapsed_sec = transform_time - start_sec
    event_normalized_time = event_elapsed_sec / duration_sec
    sample_intervals = np.diff(log_time)

    return {
        "bag_name": path.name,
        "duration_sec": duration_sec,
        "energy_j": float(cumulative_energy_j[-1]),
        "energy_wh": float(cumulative_energy_j[-1] / 3600.0),
        "mean_power_w": float(cumulative_energy_j[-1] / duration_sec),
        "peak_power_w": float(np.max(clipped_power)),
        "log_rows": int(len(log)),
        "sample_rate_hz": float(1.0 / np.median(sample_intervals)),
        "aligned_power_w": aligned_power_w,
        "aligned_energy_j": aligned_energy_j,
        "event_elapsed_sec": event_elapsed_sec,
        "event_normalized_time": event_normalized_time,
    }


def write_summary(trials: list[dict[str, object]]) -> pd.DataFrame:
    summary_columns = [
        "bag_name",
        "duration_sec",
        "energy_j",
        "energy_wh",
        "mean_power_w",
        "peak_power_w",
        "log_rows",
        "sample_rate_hz",
    ]
    summary = pd.DataFrame(
        [{column: trial[column] for column in summary_columns} for trial in trials]
    )
    summary.to_csv(
        OUT_DIR / "trans_loop_trial_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def write_event_statistics(
    trials: list[dict[str, object]], mean_duration_sec: float
) -> pd.DataFrame:
    event_elapsed = np.vstack(
        [trial["event_elapsed_sec"] for trial in trials]
    ).astype(float)
    event_normalized = np.vstack(
        [trial["event_normalized_time"] for trial in trials]
    ).astype(float)

    source_modes = np.array([0, *EXPECTED_MODES[:-1]], dtype=int)
    target_modes = EXPECTED_MODES
    transition_labels = [
        f"μ{source + 1}→μ{target + 1}"
        for source, target in zip(source_modes, target_modes)
    ]

    event_stats = pd.DataFrame(
        {
            "event_index": np.arange(1, len(EXPECTED_MODES) + 1),
            "transition": transition_labels,
            "target_mode": target_modes,
            "target_mu": [f"μ{mode + 1}" for mode in target_modes],
            "direction": ["forward"] * 8 + ["reverse"] * 8,
            "elapsed_time_mean_sec": event_elapsed.mean(axis=0),
            "elapsed_time_std_sec": event_elapsed.std(axis=0, ddof=1),
            "normalized_time_mean": event_normalized.mean(axis=0),
            "normalized_time_std": event_normalized.std(axis=0, ddof=1),
            "plot_time_sec": event_normalized.mean(axis=0) * mean_duration_sec,
        }
    )
    event_stats["interval_from_previous_mean_sec"] = np.concatenate(
        ([np.nan], np.diff(event_elapsed, axis=1).mean(axis=0))
    )
    event_stats["interval_from_previous_std_sec"] = np.concatenate(
        ([np.nan], np.diff(event_elapsed, axis=1).std(axis=0, ddof=1))
    )
    event_stats.to_csv(
        OUT_DIR / "trans_loop_transform_event_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return event_stats


def write_curve_statistics(
    trials: list[dict[str, object]], mean_duration_sec: float
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    normalized_grid = np.linspace(0.0, 1.0, GRID_POINTS)
    power = np.vstack([trial["aligned_power_w"] for trial in trials]).astype(float)
    energy = np.vstack([trial["aligned_energy_j"] for trial in trials]).astype(float)

    power_mean = power.mean(axis=0)
    power_std = power.std(axis=0, ddof=1)
    energy_mean = energy.mean(axis=0)
    energy_std = energy.std(axis=0, ddof=1)

    curve = pd.DataFrame(
        {
            "normalized_cycle_time": normalized_grid,
            "elapsed_time_at_mean_duration_sec": normalized_grid
            * mean_duration_sec,
            "power_mean_w": power_mean,
            "power_std_w": power_std,
            "power_mean_minus_std_w": power_mean - power_std,
            "power_mean_plus_std_w": power_mean + power_std,
            "cumulative_energy_mean_j": energy_mean,
            "cumulative_energy_std_j": energy_std,
            "cumulative_energy_mean_minus_std_j": energy_mean - energy_std,
            "cumulative_energy_mean_plus_std_j": energy_mean + energy_std,
            "trial_count": len(trials),
        }
    )
    curve.to_csv(
        OUT_DIR / "trans_loop_mean_std_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return curve, power, energy


def make_plot(curve: pd.DataFrame, event_stats: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    time_sec = curve["elapsed_time_at_mean_duration_sec"].to_numpy()
    power_mean = curve["power_mean_w"].to_numpy()
    power_std = curve["power_std_w"].to_numpy()
    energy_mean_kj = curve["cumulative_energy_mean_j"].to_numpy() / 1000.0
    energy_std_kj = curve["cumulative_energy_std_j"].to_numpy() / 1000.0
    event_time = event_stats["plot_time_sec"].to_numpy()

    fig, (ax_power, ax_energy) = plt.subplots(
        2,
        1,
        figsize=(11.2, 7.2),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.12},
    )

    for ax in (ax_power, ax_energy):
        for timestamp in event_time:
            ax.axvline(
                timestamp,
                color="#6F7887",
                linewidth=0.9,
                linestyle=(0, (4, 3)),
                alpha=0.55,
                zorder=0,
            )
        ax.grid(axis="y", color="#D7DCE3", linewidth=0.7, alpha=0.65)

    power_color = "#C83E4D"
    ax_power.fill_between(
        time_sec,
        power_mean - power_std,
        power_mean + power_std,
        color=power_color,
        alpha=0.22,
        linewidth=0,
        label="Mean ± 1 SD",
    )
    ax_power.plot(
        time_sec,
        power_mean,
        color=power_color,
        linewidth=1.8,
        label="Mean total electrical power",
    )
    ax_power.set_ylabel("Power (W)")
    ax_power.legend(loc="upper right", frameon=False)
    ax_power.set_title(
        "Robot transformation loop: mean electrical power and cumulative energy (n=10)",
        loc="left",
        fontsize=13,
        pad=14,
    )

    state_axis = ax_power.secondary_xaxis("top")
    state_axis.set_xticks(np.concatenate(([time_sec[0]], event_time)))
    state_axis.set_xticklabels(
        ["μ1", *event_stats["transition"].tolist()],
        fontsize=7.2,
        rotation=38,
        ha="left",
        rotation_mode="anchor",
    )
    state_axis.set_xlabel("Initial state and transform commands", labelpad=32)
    state_axis.spines["top"].set_color("#AAB1BC")
    state_axis.tick_params(axis="x", length=3, color="#AAB1BC", pad=3)

    energy_color = "#2364AA"
    ax_energy.fill_between(
        time_sec,
        energy_mean_kj - energy_std_kj,
        energy_mean_kj + energy_std_kj,
        color=energy_color,
        alpha=0.20,
        linewidth=0,
        label="Mean ± 1 SD",
    )
    ax_energy.plot(
        time_sec,
        energy_mean_kj,
        color=energy_color,
        linewidth=2.0,
        label="Mean cumulative energy",
    )
    ax_energy.set_ylabel("Cumulative energy (kJ)")
    ax_energy.set_xlabel("Elapsed time mapped to mean cycle duration (s)")
    ax_energy.legend(loc="upper left", frameon=False)
    ax_energy.set_xlim(time_sec[0], time_sec[-1])

    note = (
        "Power = Σ Uᵢ|Iᵢ| across 17 motors; energy uses trapezoidal integration. "
        "Curves are phase-aligned from the first to last transform timestamp."
    )
    fig.text(0.075, 0.018, note, fontsize=8.4, color="#4C5667")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.865, bottom=0.09)

    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            OUT_DIR / f"trans_loop_energy_mean_std.{suffix}",
            bbox_inches="tight",
        )
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    duration_mean = float(summary["duration_sec"].mean())
    duration_std = float(summary["duration_sec"].std(ddof=1))
    energy_mean = float(summary["energy_j"].mean())
    energy_std = float(summary["energy_j"].std(ddof=1))
    power_mean = float(summary["mean_power_w"].mean())
    power_std = float(summary["mean_power_w"].std(ddof=1))

    report = f"""# Transformation loop energy analysis

- Trials: 10 (latest directory for each experiment number 01–10)
- Analysis window: first to last `/crimson_control/transform` timestamp
- Total electrical power: `P(t) = sum(U_i * abs(I_i))` for motors 0–16
- Energy: trapezoidal integration of total electrical power
- Curve alignment: normalized cycle time, displayed using the mean duration
- Shading: sample standard deviation (`ddof=1`)

## Main results

- Transformation-loop duration: **{duration_mean:.3f} ± {duration_std:.3f} s**
- Electrical energy: **{energy_mean / 1000.0:.4f} ± {energy_std / 1000.0:.4f} kJ**
- Electrical energy: **{energy_mean / 3600.0:.4f} ± {energy_std / 3600.0:.4f} Wh**
- Mean electrical power: **{power_mean:.2f} ± {power_std:.2f} W**

Values are mean ± sample standard deviation across the 10 trials.
"""
    (OUT_DIR / "README.md").write_text(report, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bag_directories = select_bag_directories()
    trials = [load_trial(path) for path in bag_directories]

    summary = write_summary(trials)
    mean_duration_sec = float(summary["duration_sec"].mean())
    event_stats = write_event_statistics(trials, mean_duration_sec)
    curve, _, _ = write_curve_statistics(trials, mean_duration_sec)
    make_plot(curve, event_stats)
    write_report(summary)

    print(f"Analyzed {len(trials)} trials")
    print(
        "Duration: "
        f"{summary['duration_sec'].mean():.3f} +/- "
        f"{summary['duration_sec'].std(ddof=1):.3f} s"
    )
    print(
        "Energy: "
        f"{summary['energy_j'].mean() / 1000.0:.4f} +/- "
        f"{summary['energy_j'].std(ddof=1) / 1000.0:.4f} kJ"
    )
    print(
        "Mean power: "
        f"{summary['mean_power_w'].mean():.2f} +/- "
        f"{summary['mean_power_w'].std(ddof=1):.2f} W"
    )
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
