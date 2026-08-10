from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import plot_real_sim_energy_comparison as fig


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[2]
GENERATOR = REPO_DIR / "src" / "ros_mujoco" / "scripts" / "mujoco_experiment_energy.py"
ACTUATOR_XML = (
    REPO_DIR
    / "src"
    / "models"
    / "crimson"
    / "mjcf"
    / "crimson_stand_legInit_forSimOnly.xml"
)
PYTHON = Path(sys.executable)
OUT_DIR = SCRIPT_DIR / "dynamic_scan"
JSON_DIR = OUT_DIR / "json"


@dataclass(frozen=True)
class Variant:
    frame_kp: float
    leg_kp: float
    kv_divisor: float

    @property
    def label(self) -> str:
        return (
            f"frame{self.frame_kp:g}_leg{self.leg_kp:g}"
            f"_kvdiv{self.kv_divisor:g}"
        ).replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan MuJoCo actuator gains against real power/phase metrics."
    )
    parser.add_argument("--num-experiments", type=int, default=1)
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--keep-json", action="store_true")
    parser.add_argument("--frame-kp", type=float, default=None)
    parser.add_argument("--leg-kp", type=float, default=None)
    parser.add_argument("--kv-divisor", type=float, default=None)
    return parser.parse_args()


def default_variants() -> list[Variant]:
    variants = []
    # Current calibrated setting is frame=30, leg=10, kv=kp/6.
    variants.append(Variant(frame_kp=30, leg_kp=10, kv_divisor=6))
    for leg_kp in (12, 15, 18, 22, 26, 30):
        variants.append(Variant(frame_kp=30, leg_kp=leg_kp, kv_divisor=6))
    for leg_kp in (15, 18, 22):
        variants.append(Variant(frame_kp=40, leg_kp=leg_kp, kv_divisor=6))
    for leg_kp in (18, 22, 26):
        variants.append(Variant(frame_kp=30, leg_kp=leg_kp, kv_divisor=4))
    for leg_kp in (18, 22, 26):
        variants.append(Variant(frame_kp=30, leg_kp=leg_kp, kv_divisor=8))
    for kv_divisor in (10, 12, 16):
        for leg_kp in (10, 12, 15, 18, 22):
            variants.append(
                Variant(frame_kp=30, leg_kp=leg_kp, kv_divisor=kv_divisor)
            )
    return variants


def patched_xml(original_xml: str, variant: Variant) -> str:
    def replace_position(match: re.Match[str]) -> str:
        line = match.group(0)
        if "frameJoint4_motor" in line or "frameJoint7_motor" in line:
            return line
        is_frame = 'name="frame' in line
        kp = variant.frame_kp if is_frame else variant.leg_kp
        kv = kp / variant.kv_divisor
        line = re.sub(r'kp="[^"]+"', f'kp="{kp:g}"', line)
        line = re.sub(r'kv="[^"]+"', f'kv="{kv:g}"', line)
        return line

    return re.sub(r"<position[^>]+/>", replace_position, original_xml)


def generate_variant_json(variant: Variant, num_experiments: int) -> Path:
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    output = JSON_DIR / f"mujoco_{variant.label}.json"
    log_path = JSON_DIR / f"mujoco_{variant.label}.log"
    command = [
        str(PYTHON),
        str(GENERATOR),
        "--num-experiments",
        str(num_experiments),
        "--energy-mode",
        "calibrated",
        "--log-rate",
        "50",
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"MuJoCo generation failed for {variant.label}; see {log_path}"
        )
    return output


def real_reference(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object]]:
    real_trials = [
        fig.load_real_trial(
            path,
            grid,
            fig.DEFAULT_FALLBACK_TRANSITION_WINDOW_S,
            fig.DEFAULT_FINAL_INTERVAL_DURATION_S,
        )
        for path in fig.select_latest_trial_dirs(fig.DEFAULT_REAL_CSV_DIR)
    ]
    common_duration = float(np.mean([trial["duration_s"] for trial in real_trials]))
    return real_trials, fig.summarize_trials(real_trials, grid, common_duration)


def phase_energy_metrics(real: dict[str, object], sim: dict[str, object]) -> dict[str, float]:
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
        "phase_energy_mape_pct": float(
            np.mean(np.abs(error) / np.maximum(real_phase, 1e-9)) * 100.0
        ),
    }


def evaluate_variant(
    variant: Variant,
    json_path: Path,
    grid: np.ndarray,
    real: dict[str, object],
) -> dict[str, float | str]:
    sim_trials = fig.load_sim_trials(
        json_path,
        grid,
        fig.DEFAULT_FALLBACK_TRANSITION_WINDOW_S,
        fig.DEFAULT_FINAL_INTERVAL_DURATION_S,
    )
    common_duration = float(np.max(np.asarray(real["time_s"], dtype=float)))
    sim = fig.summarize_trials(sim_trials, grid, common_duration)

    real_duration = np.asarray(real["transition_duration_mean_s"], dtype=float)
    sim_duration = np.asarray(sim["transition_duration_mean_s"], dtype=float)
    duration_error = sim_duration - real_duration

    real_power = np.asarray(real["power_mean_w"], dtype=float)
    sim_power = np.asarray(sim["power_mean_w"], dtype=float)
    power_error = sim_power - real_power

    interval_metrics = fig.metric_summary(real, sim)
    phase_metrics = phase_energy_metrics(real, sim)
    return {
        "label": variant.label,
        "frame_kp": variant.frame_kp,
        "leg_kp": variant.leg_kp,
        "kv_divisor": variant.kv_divisor,
        "sim_json": str(json_path),
        "trans_duration_mae_s": float(np.mean(np.abs(duration_error))),
        "trans_duration_rmse_s": float(np.sqrt(np.mean(duration_error**2))),
        "trans_duration_bias_s": float(np.mean(duration_error)),
        "trans_duration_ratio_mean": float(np.mean(sim_duration / real_duration)),
        "sim_trans_duration_mean_s": float(np.mean(sim_duration)),
        "real_trans_duration_mean_s": float(np.mean(real_duration)),
        "power_curve_rmse_w": float(np.sqrt(np.mean(power_error**2))),
        "power_curve_mae_w": float(np.mean(np.abs(power_error))),
        **interval_metrics,
        **phase_metrics,
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, fig.GRID_POINTS)
    _, real = real_reference(grid)
    variants = default_variants()
    if (
        args.frame_kp is not None
        or args.leg_kp is not None
        or args.kv_divisor is not None
    ):
        if args.frame_kp is None or args.leg_kp is None or args.kv_divisor is None:
            raise ValueError(
                "--frame-kp, --leg-kp and --kv-divisor must be provided together."
            )
        variants = [Variant(args.frame_kp, args.leg_kp, args.kv_divisor)]
    if args.max_variants is not None:
        variants = variants[: args.max_variants]

    original_xml = ACTUATOR_XML.read_text(encoding="utf-8")
    rows = []
    try:
        for index, variant in enumerate(variants, start=1):
            print(f"[{index}/{len(variants)}] {variant.label}", flush=True)
            ACTUATOR_XML.write_text(patched_xml(original_xml, variant), encoding="utf-8")
            json_path = generate_variant_json(variant, args.num_experiments)
            rows.append(evaluate_variant(variant, json_path, grid, real))
            pd.DataFrame(rows).to_csv(OUT_DIR / "dynamic_scan_summary.csv", index=False)
            if not args.keep_json:
                json_path.unlink(missing_ok=True)
                (JSON_DIR / f"mujoco_{variant.label}.log").unlink(missing_ok=True)
    finally:
        ACTUATOR_XML.write_text(original_xml, encoding="utf-8")

    summary = pd.DataFrame(rows).sort_values(
        ["trans_duration_mae_s", "power_curve_rmse_w", "phase_energy_mae_j"]
    )
    summary.to_csv(OUT_DIR / "dynamic_scan_summary.csv", index=False)
    print(summary.head(10).to_string(index=False))
    print(f"Wrote {OUT_DIR / 'dynamic_scan_summary.csv'}")


if __name__ == "__main__":
    main()
