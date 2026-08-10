from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "csv"
OUT_DIR = BASE_DIR / "analysis_energy_json_no_startup"
GRID_POINTS = 800
EXPECTED_TARGET_MODES = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    dtype=int,
)
TARGET_LABELS = [
    "μ2",
    "μ3",
    "μ4",
    "μ5",
    "μ6",
    "μ7",
    "μ8",
    "μ9",
    "μ8",
    "μ7",
    "μ6",
    "μ5",
    "μ4",
    "μ3",
    "μ2",
    "μ1",
]


def select_source_json() -> Path:
    workspace_runs = BASE_DIR.parents[1]
    candidates = sorted(
        [
            *CSV_DIR.glob("mujoco_experiment_energy_full10*.json"),
            *CSV_DIR.glob("mujoco_experiment_energy_calibrated_full10*.json"),
            *workspace_runs.glob("mujoco_experiment_energy_full10*.json"),
            *workspace_runs.glob("mujoco_experiment_energy_calibrated_full10*.json"),
        ],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No full10 JSON file found below {CSV_DIR}"
        )
    return candidates[-1]


def cumulative_trapezoid(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    increments = 0.5 * (values[:-1] + values[1:]) * np.diff(time_s)
    return np.concatenate(([0.0], np.cumsum(increments)))


def clip_with_interpolated_boundaries(
    time_s: np.ndarray,
    values: np.ndarray,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if start_s < time_s[0] or end_s > time_s[-1] or start_s >= end_s:
        raise ValueError(
            f"Invalid analysis window [{start_s}, {end_s}] for "
            f"data range [{time_s[0]}, {time_s[-1]}]"
        )
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


def calculate_total_power(log_records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.array(
        [float(record["relative_time_s"]) for record in log_records],
        dtype=float,
    )
    power_w: list[float] = []
    motor_count: int | None = None

    for record in log_records:
        voltage_v = np.asarray(record["msg"]["U"], dtype=float)
        current_a = np.asarray(record["msg"]["I"], dtype=float)
        if voltage_v.shape != current_a.shape:
            raise ValueError("U and I arrays have different lengths")
        if motor_count is None:
            motor_count = len(voltage_v)
        if len(voltage_v) != motor_count:
            raise ValueError("Motor count changes within one experiment")
        power_w.append(float(np.sum(voltage_v * np.abs(current_a))))

    if motor_count != 17:
        raise ValueError(f"Expected 17 motors, found {motor_count}")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("Dynamixel log timestamps are not strictly increasing")
    return time_s, np.asarray(power_w, dtype=float)


def parse_trial(
    experiment: dict,
    segment_rows: list[dict],
    normalized_grid: np.ndarray,
    startup_replacement: tuple[float, float] | None = None,
) -> dict[str, object]:
    records = experiment["records"]
    transforms = records["/crimson_control/transform"]
    transformed = records["/crimson_control/transformed"]
    logs = records["/dynamixel_control/log"]

    target_modes = np.array(
        [int(record["msg"]["mode"]) for record in transforms],
        dtype=int,
    )
    if not np.array_equal(target_modes, EXPECTED_TARGET_MODES):
        raise ValueError(
            f"Experiment {experiment['experiment']} has unexpected target "
            f"sequence: {target_modes.tolist()}"
        )

    log_time_s, total_power_w = calculate_total_power(logs)
    removed_startup_transient = False
    startup_original_power_w = np.nan
    startup_replacement_power_w = np.nan
    if startup_replacement is not None:
        replacement_time_s, replacement_power_w = startup_replacement
        replacement_index = int(
            np.argmin(np.abs(log_time_s - replacement_time_s))
        )
        if not np.isclose(
            log_time_s[replacement_index],
            replacement_time_s,
            atol=1e-9,
        ):
            raise ValueError(
                "Startup-transient replacement time does not match a log "
                "sample"
            )
        startup_original_power_w = float(total_power_w[replacement_index])
        total_power_w[replacement_index] = replacement_power_w
        startup_replacement_power_w = float(replacement_power_w)
        removed_startup_transient = True
    event_time_s = np.array(
        [float(record["relative_time_s"]) for record in transforms],
        dtype=float,
    )
    transformed_time_s = np.array(
        [float(record["relative_time_s"]) for record in transformed],
        dtype=float,
    )

    # Match the supplied example: first transform command (μ1→μ2) through the
    # final transform command (μ2→μ1). The final 0.1 s movement is also
    # reported separately through the JSON segment-energy reference columns.
    start_s = float(event_time_s[0])
    end_s = float(event_time_s[-1])
    duration_s = end_s - start_s
    clipped_time_s, clipped_power_w = clip_with_interpolated_boundaries(
        log_time_s, total_power_w, start_s, end_s
    )
    elapsed_s = clipped_time_s - start_s
    cumulative_energy_j = cumulative_trapezoid(
        clipped_time_s, clipped_power_w
    )
    normalized_time = elapsed_s / duration_s

    aligned_power_w = np.interp(
        normalized_grid, normalized_time, clipped_power_w
    )
    aligned_energy_j = np.interp(
        normalized_grid, normalized_time, cumulative_energy_j
    )
    event_elapsed_s = event_time_s - start_s
    event_normalized = event_elapsed_s / duration_s

    sorted_segments = sorted(segment_rows, key=lambda row: int(row["step"]))
    if len(sorted_segments) != len(EXPECTED_TARGET_MODES):
        raise ValueError(
            f"Experiment {experiment['experiment']} has "
            f"{len(sorted_segments)} segments, expected 16"
        )
    segment_energy_to_final_state_j = float(
        sum(float(row["total_energy_j"]) for row in sorted_segments[:-1])
        + float(sorted_segments[-1]["transition_energy_j"])
    )
    segment_energy_with_final_settle_j = float(
        sum(float(row["total_energy_j"]) for row in sorted_segments)
    )

    energy_j = float(cumulative_energy_j[-1])
    sample_rate_hz = float(1.0 / np.median(np.diff(log_time_s)))
    return {
        "experiment": int(experiment["experiment"]),
        "bag_name": experiment["bag_name"],
        "duration_sec": duration_s,
        "energy_j": energy_j,
        "energy_kj": energy_j / 1000.0,
        "energy_wh": energy_j / 3600.0,
        "mean_power_w": energy_j / duration_s,
        "peak_sampled_power_w": float(np.max(clipped_power_w)),
        "sample_rate_hz": sample_rate_hz,
        "motor_count": len(logs[0]["msg"]["U"]),
        "segment_energy_to_final_state_j": segment_energy_to_final_state_j,
        "segment_energy_with_final_settle_j": (
            segment_energy_with_final_settle_j
        ),
        "final_transform_completion_sec": float(
            transformed_time_s[-1] - start_s
        ),
        "startup_transient_removed": removed_startup_transient,
        "startup_original_power_w": startup_original_power_w,
        "startup_replacement_power_w": startup_replacement_power_w,
        "aligned_power_w": aligned_power_w,
        "aligned_energy_j": aligned_energy_j,
        "event_elapsed_s": event_elapsed_s,
        "event_normalized": event_normalized,
    }


def write_trial_summary(trials: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "experiment",
        "bag_name",
        "duration_sec",
        "energy_j",
        "energy_kj",
        "energy_wh",
        "mean_power_w",
        "peak_sampled_power_w",
        "sample_rate_hz",
        "motor_count",
        "segment_energy_to_final_state_j",
        "segment_energy_with_final_settle_j",
        "final_transform_completion_sec",
        "startup_transient_removed",
        "startup_original_power_w",
        "startup_replacement_power_w",
    ]
    summary = pd.DataFrame(
        [{column: trial[column] for column in columns} for trial in trials]
    )
    summary.to_csv(
        OUT_DIR / "trans_loop_json_trial_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def write_curve_statistics(
    trials: list[dict[str, object]],
    normalized_grid: np.ndarray,
    mean_duration_s: float,
) -> pd.DataFrame:
    power = np.vstack([trial["aligned_power_w"] for trial in trials]).astype(
        float
    )
    energy = np.vstack([trial["aligned_energy_j"] for trial in trials]).astype(
        float
    )
    curve = pd.DataFrame(
        {
            "normalized_cycle_time": normalized_grid,
            "elapsed_time_at_mean_duration_sec": (
                normalized_grid * mean_duration_s
            ),
            "power_mean_w": power.mean(axis=0),
            "power_std_w": power.std(axis=0, ddof=1),
            "power_mean_minus_std_w": (
                power.mean(axis=0) - power.std(axis=0, ddof=1)
            ),
            "power_mean_plus_std_w": (
                power.mean(axis=0) + power.std(axis=0, ddof=1)
            ),
            "cumulative_energy_mean_j": energy.mean(axis=0),
            "cumulative_energy_std_j": energy.std(axis=0, ddof=1),
            "cumulative_energy_mean_minus_std_j": (
                energy.mean(axis=0) - energy.std(axis=0, ddof=1)
            ),
            "cumulative_energy_mean_plus_std_j": (
                energy.mean(axis=0) + energy.std(axis=0, ddof=1)
            ),
            "trial_count": len(trials),
        }
    )
    curve.to_csv(
        OUT_DIR / "trans_loop_json_mean_std_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return curve


def write_event_statistics(
    trials: list[dict[str, object]],
    segment_rows: list[dict],
    mean_duration_s: float,
) -> pd.DataFrame:
    event_elapsed = np.vstack(
        [trial["event_elapsed_s"] for trial in trials]
    ).astype(float)
    event_normalized = np.vstack(
        [trial["event_normalized"] for trial in trials]
    ).astype(float)
    source_modes = np.array([0, *EXPECTED_TARGET_MODES[:-1]], dtype=int)

    transition_energy_by_step: list[list[float]] = []
    settle_energy_by_step: list[list[float]] = []
    for step in range(len(EXPECTED_TARGET_MODES)):
        rows = sorted(
            [row for row in segment_rows if int(row["step"]) == step],
            key=lambda row: int(row["experiment"]),
        )
        transition_energy_by_step.append(
            [float(row["transition_energy_j"]) for row in rows]
        )
        settle_energy_by_step.append(
            [float(row["settle_energy_j"]) for row in rows]
        )

    transition_energy = np.asarray(transition_energy_by_step, dtype=float)
    settle_energy = np.asarray(settle_energy_by_step, dtype=float)
    interval_s = np.diff(event_elapsed, axis=1)
    event_stats = pd.DataFrame(
        {
            "event_index": np.arange(1, 17),
            "transition": [
                f"μ{source + 1}→μ{target + 1}"
                for source, target in zip(
                    source_modes, EXPECTED_TARGET_MODES
                )
            ],
            "target_mu": TARGET_LABELS,
            "direction": ["forward"] * 8 + ["reverse"] * 8,
            "command_time_mean_sec": event_elapsed.mean(axis=0),
            "command_time_std_sec": event_elapsed.std(axis=0, ddof=1),
            "plot_time_sec": (
                event_normalized.mean(axis=0) * mean_duration_s
            ),
            "transition_energy_mean_j": transition_energy.mean(axis=1),
            "transition_energy_std_j": transition_energy.std(
                axis=1, ddof=1
            ),
            "settle_energy_mean_j": settle_energy.mean(axis=1),
            "settle_energy_std_j": settle_energy.std(axis=1, ddof=1),
        }
    )
    event_stats["interval_to_next_command_mean_sec"] = np.concatenate(
        (interval_s.mean(axis=0), [np.nan])
    )
    event_stats["interval_to_next_command_std_sec"] = np.concatenate(
        (interval_s.std(axis=0, ddof=1), [np.nan])
    )
    event_stats.to_csv(
        OUT_DIR / "trans_loop_json_event_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return event_stats


def make_main_plot(
    curve: pd.DataFrame,
    event_stats: pd.DataFrame,
    trial_count: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    time_s = curve["elapsed_time_at_mean_duration_sec"].to_numpy()
    power_mean = curve["power_mean_w"].to_numpy()
    power_std = curve["power_std_w"].to_numpy()
    energy_mean_kj = curve["cumulative_energy_mean_j"].to_numpy() / 1000.0
    energy_std_kj = curve["cumulative_energy_std_j"].to_numpy() / 1000.0
    event_time_s = event_stats["plot_time_sec"].to_numpy()

    fig, (ax_power, ax_energy) = plt.subplots(
        2,
        1,
        figsize=(15.5, 9.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12},
    )
    red = "#c83d4d"
    blue = "#2463a6"

    ax_power.fill_between(
        time_s,
        np.maximum(0.0, power_mean - power_std),
        power_mean + power_std,
        color=red,
        alpha=0.20,
        linewidth=0,
        label="Mean ± 1 SD",
    )
    ax_power.plot(
        time_s,
        power_mean,
        color=red,
        linewidth=2.4,
        label="Mean total electrical power",
    )
    ax_power.set_ylabel("Power (W)")
    ax_power.grid(True, color="#9aa9bd", alpha=0.28)
    ax_power.set_ylim(bottom=0)
    ax_power.set_xlim(time_s[0], time_s[-1])

    target_axis = ax_power.twiny()
    target_axis.set_xlim(ax_power.get_xlim())
    target_axis.set_xticks(event_time_s)
    target_axis.set_xticklabels(TARGET_LABELS)
    target_axis.set_xlabel("Transform target state", labelpad=8)
    target_axis.tick_params(axis="x", length=4, color="#718096")
    target_axis.spines["top"].set_visible(True)
    target_axis.spines["top"].set_color("#9aa9bd")
    target_axis.spines["right"].set_visible(False)
    target_axis.spines["left"].set_visible(False)
    for event_index, event_time in enumerate(event_time_s):
        ax_power.axvline(
            event_time,
            color="#64748b",
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
            label=(
                "Transform command time" if event_index == 0 else None
            ),
        )
        ax_energy.axvline(
            event_time,
            color="#64748b",
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
        )
    ax_power.legend(loc="upper right", frameon=False)

    ax_energy.fill_between(
        time_s,
        np.maximum(0.0, energy_mean_kj - energy_std_kj),
        energy_mean_kj + energy_std_kj,
        color=blue,
        alpha=0.18,
        linewidth=0,
        label="Mean ± 1 SD",
    )
    ax_energy.plot(
        time_s,
        energy_mean_kj,
        color=blue,
        linewidth=2.4,
        label="Mean cumulative energy",
    )
    ax_energy.set_ylabel("Cumulative energy (kJ)")
    ax_energy.set_xlabel("Elapsed time mapped to mean cycle duration (s)")
    ax_energy.grid(True, color="#9aa9bd", alpha=0.28)
    ax_energy.legend(loc="upper left", frameon=False)
    ax_energy.set_ylim(bottom=0)
    ax_energy.set_xlim(time_s[0], time_s[-1])

    fig.suptitle(
        "Robot transformation loop without startup transient: "
        f"mean electrical power and cumulative energy (n={trial_count})",
        fontsize=17,
        y=0.985,
    )
    fig.text(
        0.08,
        0.018,
        "Power = Σ Uᵢ|Iᵢ| across 17 motors; energy uses trapezoidal "
        "integration. Curves are normalized to cycle phase and shown at "
        "the mean duration.",
        fontsize=9.5,
        color="#4a5568",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.09)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            OUT_DIR / f"trans_loop_json_energy_mean_std_no_startup.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_readme(
    source_path: Path,
    summary: pd.DataFrame,
    metadata: dict,
) -> None:
    def mean_sd(column: str) -> tuple[float, float]:
        values = summary[column].to_numpy(dtype=float)
        return float(values.mean()), float(values.std(ddof=1))

    duration_mean, duration_sd = mean_sd("duration_sec")
    energy_mean, energy_sd = mean_sd("energy_j")
    energy_wh_mean, energy_wh_sd = mean_sd("energy_wh")
    power_mean, power_sd = mean_sd("mean_power_w")
    segment_mean, segment_sd = mean_sd("segment_energy_to_final_state_j")
    full_segment_mean, full_segment_sd = mean_sd(
        "segment_energy_with_final_settle_j"
    )
    removed_startup = bool(summary["startup_transient_removed"].any())
    if removed_startup:
        peak_first = float(
            summary.loc[summary["experiment"] == 1, "peak_sampled_power_w"].iloc[0]
        )
        peak_rest = float(
            summary.loc[summary["experiment"] != 1, "peak_sampled_power_w"].max()
        )
        startup_text = f"""- 启动瞬态处理：第 1 次实验 0.667 s 的异常功率采样点，以其余 9 次实验
  相同相位功率的中位数替换；不删除整次实验，也不裁掉 μ1→μ2 阶段"""
        quality_text = f"""## 数据质量提示

原始第 1 次实验在 0.667 s 的采样功率为 **1543.9 W**。该异常点已替换为
其余 9 次实验相同相位的中位功率；第 1 次实验仍完整参与平均值和标准差计算。
处理后第 1 次实验的循环内采样峰值为 **{peak_first:.1f} W**，其余实验最大值
不超过 **{peak_rest:.1f} W**。
"""
    else:
        startup_text = "- 启动瞬态处理：未检测到需要替换的启动异常采样点"
        quality_text = """## 数据质量提示

当前源 JSON 未检测到启动异常采样点，所有实验按原始 U/I 日志参与统计。
"""

    text = f"""# JSON 机器人变形循环能耗分析

## 数据与方法

- 源文件：`{source_path.name}`
- 重复次数：{len(summary)}
- 变形顺序：μ1→μ2→…→μ9→μ8→…→μ1
- 主分析时间窗：第一次 `/transform` 指令至最后一次 `/transform` 指令，与示例图一致
- 总电功率：`P(t) = Σ U_i × |I_i|`，共 17 个电机
- 累计能耗：对功率按时间进行梯形积分
- 曲线对齐：每次循环归一化至 0–1，再映射到平均循环时长
- 阴影带：10 次实验的样本标准差（mean ± 1 SD，`ddof=1`）
{startup_text}

## 主结果（去除启动瞬态后，由 JSON 的 U/I 日志计算）

- 变形循环时间：**{duration_mean:.3f} ± {duration_sd:.3f} s**
- 变形循环能耗：**{energy_mean / 1000:.4f} ± {energy_sd / 1000:.4f} kJ**
- 变形循环能耗：**{energy_wh_mean:.4f} ± {energy_wh_sd:.4f} Wh**
- 循环平均功率：**{power_mean:.2f} ± {power_sd:.2f} W**

以上均为 10 次实验的平均值 ± 样本标准差。

## JSON 分段能耗校核

JSON 还提供了生成端按高时间分辨率计算的 `transition_energy_j` 和
`settle_energy_j`。从第一次变形开始，到最终 μ1 变形完成（不含最终稳定等待）
的分段能耗为 **{segment_mean / 1000:.4f} ± {segment_sd / 1000:.4f} kJ**；
若包含最终稳定等待，则为
**{full_segment_mean / 1000:.4f} ± {full_segment_sd / 1000:.4f} kJ**。

该数值比 3 Hz 的 U/I 日志梯形积分略高，原因是 0.1 s 左右的短时变形电流峰值
可能落在日志采样点之间。因此，主图反映“记录到的 U/I 曲线”，分段字段适合用作
完整短时能耗的校核。

{quality_text}
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    source_path = select_source_json()
    data = json.loads(source_path.read_text(encoding="utf-8"))
    experiments = data["experiments"]
    if len(experiments) != 10 or int(data.get("num_experiments", -1)) != 10:
        raise ValueError("Expected exactly 10 experiments in the JSON file")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    normalized_grid = np.linspace(0.0, 1.0, GRID_POINTS)
    all_segments = data["segments"]
    startup_samples: list[tuple[int, float, float]] = []
    for experiment in experiments:
        transform_start_s = float(
            experiment["records"]["/crimson_control/transform"][0][
                "relative_time_s"
            ]
        )
        log_time_s, total_power_w = calculate_total_power(
            experiment["records"]["/dynamixel_control/log"]
        )
        first_active_index = int(np.searchsorted(log_time_s, transform_start_s))
        startup_samples.append(
            (
                int(experiment["experiment"]),
                float(log_time_s[first_active_index]),
                float(total_power_w[first_active_index]),
            )
        )

    startup_power = np.array(
        [sample[2] for sample in startup_samples], dtype=float
    )
    startup_median_w = float(np.median(startup_power))
    startup_mad_w = float(
        np.median(np.abs(startup_power - startup_median_w))
    )
    startup_threshold_w = startup_median_w + 8.0 * startup_mad_w
    startup_replacements = {
        experiment_id: (sample_time_s, startup_median_w)
        for experiment_id, sample_time_s, sample_power_w in startup_samples
        if sample_power_w > startup_threshold_w
    }
    if startup_replacements and set(startup_replacements) != {1}:
        raise ValueError(
            "Expected only experiment 1 to contain the startup transient, "
            f"detected experiments {sorted(startup_replacements)}"
        )

    trials: list[dict[str, object]] = []
    for experiment in experiments:
        experiment_id = int(experiment["experiment"])
        segment_rows = [
            row
            for row in all_segments
            if int(row["experiment"]) == experiment_id
        ]
        trials.append(
            parse_trial(
                experiment,
                segment_rows,
                normalized_grid,
                startup_replacements.get(experiment_id),
            )
        )

    summary = write_trial_summary(trials)
    mean_duration_s = float(summary["duration_sec"].mean())
    curve = write_curve_statistics(
        trials, normalized_grid, mean_duration_s
    )
    event_stats = write_event_statistics(
        trials, all_segments, mean_duration_s
    )
    make_main_plot(curve, event_stats, len(trials))
    write_readme(source_path, summary, data)

    print(f"Source: {source_path}")
    print(f"Output: {OUT_DIR}")
    print(
        "Duration: "
        f"{summary['duration_sec'].mean():.3f} ± "
        f"{summary['duration_sec'].std(ddof=1):.3f} s"
    )
    print(
        "Energy from logged U/I after startup-transient removal: "
        f"{summary['energy_j'].mean() / 1000:.4f} ± "
        f"{summary['energy_j'].std(ddof=1) / 1000:.4f} kJ"
    )
    if startup_replacements:
        print(
            "Startup sample replacement: "
            f"{startup_power[0]:.1f} W -> {startup_median_w:.1f} W"
        )
    else:
        print("Startup sample replacement: none")
    print(
        "Mean power: "
        f"{summary['mean_power_w'].mean():.2f} ± "
        f"{summary['mean_power_w'].std(ddof=1):.2f} W"
    )


if __name__ == "__main__":
    main()
