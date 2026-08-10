from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import plot_real_sim_energy_comparison as fig


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "phase_aware_calibration"


def stack(trials: list[dict[str, object]], key: str) -> np.ndarray:
    return np.vstack([np.asarray(trial[key], dtype=float) for trial in trials])


def phase_durations(trials: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    trans_duration = []
    sustain_duration = []
    for trial in trials:
        duration = float(trial["duration_s"])
        command = np.asarray(trial["command_norm"], dtype=float)
        done = np.asarray(trial["done_norm"], dtype=float)
        sustain_end = np.asarray(trial["sustain_end_norm"], dtype=float)
        trans_duration.append((done - command) * duration)
        sustain_duration.append((sustain_end - done) * duration)
    return (
        np.vstack(trans_duration),
        np.vstack(sustain_duration),
    )


def cumulative_energy_from_grid(
    normalized_grid: np.ndarray,
    power_grid: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    time_s = normalized_grid * duration_s
    increments = 0.5 * (power_grid[:-1] + power_grid[1:]) * np.diff(time_s)
    return np.concatenate(([0.0], np.cumsum(increments)))


def fit_phase_gains(
    real_trials: list[dict[str, object]],
    sim_trials: list[dict[str, object]],
) -> pd.DataFrame:
    real_trans_energy = stack(real_trials, "transition_energy_j")
    real_sustain_energy = stack(real_trials, "sustain_energy_j")
    sim_trans_energy = stack(sim_trials, "transition_energy_j")
    sim_sustain_energy = stack(sim_trials, "sustain_energy_j")
    sim_trans_duration, sim_sustain_duration = phase_durations(sim_trials)

    rows = []
    for step, transition in enumerate(fig.TRANSITIONS):
        for phase, real_energy, sim_energy, sim_duration in (
            (
                "transform",
                real_trans_energy[:, step],
                sim_trans_energy[:, step],
                sim_trans_duration[:, step],
            ),
            (
                "sustain",
                real_sustain_energy[:, step],
                sim_sustain_energy[:, step],
                sim_sustain_duration[:, step],
            ),
        ):
            real_mean = float(np.mean(real_energy))
            sim_mean = float(np.mean(sim_energy))
            duration_mean = float(np.mean(sim_duration))
            residual = (real_mean - sim_mean) / duration_mean
            rows.append(
                {
                    "step": step,
                    "transition": transition,
                    "phase": phase,
                    "real_energy_mean_j": real_mean,
                    "sim_energy_mean_j": sim_mean,
                    "sim_phase_duration_mean_s": duration_mean,
                    "equivalent_additive_residual_w": residual,
                    "phase_gain": real_mean / sim_mean if sim_mean else np.nan,
                    "adjusted_sim_energy_mean_j": real_mean,
                    "adjusted_energy_error_j": 0.0,
                }
            )
    return pd.DataFrame(rows)


def apply_phase_gains(
    sim_trials: list[dict[str, object]],
    gains: pd.DataFrame,
    grid: np.ndarray,
) -> list[dict[str, object]]:
    gain_lookup = {
        (int(row.step), str(row.phase)): float(row.phase_gain)
        for row in gains.itertuples(index=False)
    }
    adjusted = []

    for trial in sim_trials:
        new_trial = dict(trial)
        duration = float(trial["duration_s"])
        command = np.asarray(trial["command_norm"], dtype=float)
        done = np.asarray(trial["done_norm"], dtype=float)
        sustain_end = np.asarray(trial["sustain_end_norm"], dtype=float)
        power = np.asarray(trial["power_grid"], dtype=float).copy()
        transition_energy = np.asarray(trial["transition_energy_j"], dtype=float).copy()
        sustain_energy = np.asarray(trial["sustain_energy_j"], dtype=float).copy()
        interval_energy = np.asarray(trial["interval_energy_j"], dtype=float).copy()

        for step in fig.PHASE_STEP_INDICES:
            trans_gain = gain_lookup[(int(step), "transform")]
            sustain_gain = gain_lookup[(int(step), "sustain")]

            transition_energy[step] *= trans_gain
            sustain_energy[step] *= sustain_gain
            interval_energy[step] = transition_energy[step] + sustain_energy[step]

            trans_mask = (grid >= command[step]) & (grid <= done[step])
            sustain_mask = (grid > done[step]) & (grid <= sustain_end[step])
            power[trans_mask] *= trans_gain
            power[sustain_mask] *= sustain_gain

        new_trial["power_grid"] = power
        new_trial["energy_grid"] = cumulative_energy_from_grid(grid, power, duration)
        new_trial["transition_energy_j"] = transition_energy
        new_trial["sustain_energy_j"] = sustain_energy
        new_trial["interval_energy_j"] = interval_energy
        adjusted.append(new_trial)

    return adjusted


def phase_energy_metrics(
    real: dict[str, object],
    sim: dict[str, object],
) -> dict[str, float]:
    real_phase = np.concatenate(
        (
            np.asarray(real["transition_energy_mean_j"], dtype=float),
            np.asarray(real["sustain_energy_mean_j"], dtype=float),
        )
    )
    sim_phase = np.concatenate(
        (
            np.asarray(sim["transition_energy_mean_j"], dtype=float),
            np.asarray(sim["sustain_energy_mean_j"], dtype=float),
        )
    )
    error = sim_phase - real_phase
    return {
        "phase_energy_mae_j": float(np.mean(np.abs(error))),
        "phase_energy_rmse_j": float(np.sqrt(np.mean(error**2))),
        "phase_energy_max_abs_j": float(np.max(np.abs(error))),
        "phase_energy_mape_pct": float(
            np.mean(np.abs(error) / np.maximum(real_phase, 1e-9)) * 100.0
        ),
    }


def write_report(
    gains: pd.DataFrame,
    original_metrics: dict[str, float],
    adjusted_metrics: dict[str, float],
    original_interval_metrics: dict[str, float],
    adjusted_interval_metrics: dict[str, float],
    real: dict[str, object],
    sim_original: dict[str, object],
    sim_adjusted: dict[str, object],
) -> None:
    lines = [
        "# Phase-Aware Energy Calibration",
        "",
        "## Method",
        "",
        "The current qpos-based phase windows are kept fixed. For each of the",
        "16 command intervals and for each phase (`transform` and `sustain`),",
        "a multiplicative phase gain is fitted:",
        "",
        "```text",
        "P_new(t, k, phase) = g[k, phase] * P_old(t, k)",
        "```",
        "",
        "The gain is fitted from the mean real and MuJoCo phase energies:",
        "",
        "```text",
        "g[k, phase] = E_real_mean[k, phase] / E_sim_mean[k, phase]",
        "```",
        "",
        "This multiplicative form preserves the non-negativity and within-phase",
        "shape of the simulated power curve. The table also reports the",
        "equivalent additive residual for reference, but the preview figure uses",
        "the multiplicative gain.",
        "",
        "This is a dataset-scoped calibration table. It should be described as",
        "phase-aware sim-to-real calibration, not as an independent validation.",
        "",
        "## Metrics",
        "",
        "| Metric | Before phase-aware gain | After phase-aware gain |",
        "|---|---:|---:|",
        (
            "| Phase-energy MAE (J) | "
            f"{original_metrics['phase_energy_mae_j']:.3f} | "
            f"{adjusted_metrics['phase_energy_mae_j']:.3f} |"
        ),
        (
            "| Phase-energy RMSE (J) | "
            f"{original_metrics['phase_energy_rmse_j']:.3f} | "
            f"{adjusted_metrics['phase_energy_rmse_j']:.3f} |"
        ),
        (
            "| Max phase-energy error (J) | "
            f"{original_metrics['phase_energy_max_abs_j']:.3f} | "
            f"{adjusted_metrics['phase_energy_max_abs_j']:.3f} |"
        ),
        (
            "| Phase-energy MAPE (%) | "
            f"{original_metrics['phase_energy_mape_pct']:.3f} | "
            f"{adjusted_metrics['phase_energy_mape_pct']:.3f} |"
        ),
        (
            "| Interval-power MAE (W) | "
            f"{original_interval_metrics['mae_w']:.3f} | "
            f"{adjusted_interval_metrics['mae_w']:.3f} |"
        ),
        (
            "| Interval-power RMSE (W) | "
            f"{original_interval_metrics['rmse_w']:.3f} | "
            f"{adjusted_interval_metrics['rmse_w']:.3f} |"
        ),
        "",
        "## Energy Totals",
        "",
        "| Scope | Real robot | Original MuJoCo | Phase-aware MuJoCo |",
        "|---|---:|---:|---:|",
        (
            "| Transform energy (kJ) | "
            f"{np.sum(real['transition_energy_mean_j']) / 1000.0:.4f} | "
            f"{np.sum(sim_original['transition_energy_mean_j']) / 1000.0:.4f} | "
            f"{np.sum(sim_adjusted['transition_energy_mean_j']) / 1000.0:.4f} |"
        ),
        (
            "| Sustain energy (kJ) | "
            f"{np.sum(real['sustain_energy_mean_j']) / 1000.0:.4f} | "
            f"{np.sum(sim_original['sustain_energy_mean_j']) / 1000.0:.4f} | "
            f"{np.sum(sim_adjusted['sustain_energy_mean_j']) / 1000.0:.4f} |"
        ),
        (
            "| Total energy (kJ) | "
            f"{np.sum(real['interval_energy_mean_j']) / 1000.0:.4f} | "
            f"{np.sum(sim_original['interval_energy_mean_j']) / 1000.0:.4f} | "
            f"{np.sum(sim_adjusted['interval_energy_mean_j']) / 1000.0:.4f} |"
        ),
        "",
        "## Outputs",
        "",
        "- `phase_aware_energy_gains.csv`: fitted phase-gain table.",
        "- `real_sim_interval_energy_summary.csv`: adjusted interval summary.",
        "- `real_sim_energy_comparison.svg`: adjusted preview figure.",
        "",
        "## Gain Range",
        "",
        (
            f"- Transform gain range: "
            f"{gains[gains.phase == 'transform'].phase_gain.min():.3f}"
            " to "
            f"{gains[gains.phase == 'transform'].phase_gain.max():.3f}"
        ),
        (
            f"- Sustain gain range: "
            f"{gains[gains.phase == 'sustain'].phase_gain.min():.3f}"
            " to "
            f"{gains[gains.phase == 'sustain'].phase_gain.max():.3f}"
        ),
    ]
    (OUT_DIR / "phase_aware_calibration_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, fig.GRID_POINTS)
    real_trials = [
        fig.load_real_trial(
            path,
            grid,
            fig.DEFAULT_FALLBACK_TRANSITION_WINDOW_S,
            fig.DEFAULT_FINAL_INTERVAL_DURATION_S,
        )
        for path in fig.select_latest_trial_dirs(fig.DEFAULT_REAL_CSV_DIR)
    ]
    sim_trials = fig.load_sim_trials(
        fig.DEFAULT_SIM_JSON,
        grid,
        fig.DEFAULT_FALLBACK_TRANSITION_WINDOW_S,
        fig.DEFAULT_FINAL_INTERVAL_DURATION_S,
    )
    common_duration = float(np.mean([trial["duration_s"] for trial in real_trials]))
    real = fig.summarize_trials(real_trials, grid, common_duration)
    sim_original = fig.summarize_trials(sim_trials, grid, common_duration)

    gains = fit_phase_gains(real_trials, sim_trials)
    gains.to_csv(OUT_DIR / "phase_aware_energy_gains.csv", index=False)
    adjusted_trials = apply_phase_gains(sim_trials, gains, grid)
    sim_adjusted = fig.summarize_trials(adjusted_trials, grid, common_duration)

    original_phase_metrics = phase_energy_metrics(real, sim_original)
    adjusted_phase_metrics = phase_energy_metrics(real, sim_adjusted)
    original_interval_metrics = fig.metric_summary(real, sim_original)
    adjusted_interval_metrics = fig.metric_summary(real, sim_adjusted)

    fig.write_interval_summary(OUT_DIR, real, sim_adjusted)
    fig.write_curve_summary(OUT_DIR, grid, real, sim_adjusted)
    fig.plot_comparison(
        OUT_DIR,
        real,
        sim_adjusted,
        adjusted_interval_metrics,
        fig.DEFAULT_FALLBACK_TRANSITION_WINDOW_S,
        dpi=260,
    )
    write_report(
        gains,
        original_phase_metrics,
        adjusted_phase_metrics,
        original_interval_metrics,
        adjusted_interval_metrics,
        real,
        sim_original,
        sim_adjusted,
    )

    print(f"Wrote {OUT_DIR}")
    print(
        "Phase-energy MAE: "
        f"{original_phase_metrics['phase_energy_mae_j']:.3f} J -> "
        f"{adjusted_phase_metrics['phase_energy_mae_j']:.3f} J"
    )
    print(
        "Interval-power MAE: "
        f"{original_interval_metrics['mae_w']:.3f} W -> "
        f"{adjusted_interval_metrics['mae_w']:.3f} W"
    )
    print(gains.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
