#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 MuJoCo 中复现 trans_planner 的 experiment.yaml 变形实验并估算能耗。

这是一个独立测试脚本，不会修改或发布到
trans_planner/scripts/experiment.py 使用的实物机器人控制链路。
它只复用原实验脚本中的数据转换和插值流程：

实物编码器构型 -> real2sim -> Interpolation -> sim2real -> MuJoCo 控制量

输出 JSON 同时保存两类数据：
  1. experiment.py 默认 rosbag 三个话题的等价记录；
  2. experiment.py 每帧实际发布的 /dynamixel_control/sync_write 指令。
"""

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from mujoco import viewer as mujoco_viewer

from ros_mujoco_utils.energy_calibration import (
    calibrated_total_power_w,
    dynamic_calibrated_total_power_w,
    frame_energy_j,
    legacy_torque_to_current,
    power_to_current_vector,
)
from ros_mujoco_utils.conversion import update_mujoco_control_from_real


MOTOR_IDS = list(range(1, 18))
PARAM_GOAL_POSITION = 1
DEFAULT_BAG_TOPICS = [
    "/crimson_control/transform",
    "/crimson_control/transformed",
    "/dynamixel_control/log",
]
DEFAULT_DYNAMIXEL_LOG_RATE = 3.0
DEFAULT_BAG_START_DELAY = 0.5
ENERGY_MODES = (
    "legacy",
    "calibrated",
    "calibrated_no_residual",
    "dynamic_calibrated",
)


def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo_paths():
    src_dir = Path(__file__).resolve().parents[2]
    repo_root = src_dir.parent
    return {
        "src": src_dir,
        "repo": repo_root,
        "trans_scripts": src_dir / "trans_planner" / "scripts",
        "optimizer_scripts": src_dir / "optimizer" / "scripts",
        "default_config": src_dir / "trans_planner" / "config" / "experiment.yaml",
        "dynamixel_config": (
            src_dir
            / "crimson_control"
            / "dynamixel"
            / "dynamixel_control"
            / "config"
            / "dynamixel.yaml"
        ),
        "default_output_dir": repo_root / "runs",
    }


def _import_trans_planner_flow(paths):
    sys.path.insert(0, str(paths["trans_scripts"]))
    from interpolation import Interpolation
    from utils.sim2real import real2sim, sim2real

    return Interpolation, real2sim, sim2real


def _import_mujoco_env(paths):
    sim_env_path = paths["optimizer_scripts"] / "sim_env.py"
    sim_env = _load_module("optimizer_sim_env_for_experiment_energy", sim_env_path)
    return sim_env.CrimsonMujocoEnv


def _load_dynamixel_log_rate(paths):
    """读取底层 Dynamixel 节点的 looprate，失败时退回源码默认值。"""
    try:
        with Path(paths["dynamixel_config"]).open("r") as f:
            config = yaml.safe_load(f) or {}
        return float(config.get("looprate", DEFAULT_DYNAMIXEL_LOG_RATE))
    except Exception:
        return DEFAULT_DYNAMIXEL_LOG_RATE


def _resolve_bag_dir(bag_dir, workspace_root):
    """复刻 experiment.py 的 bag_dir 规则：相对路径挂到工作空间根目录。"""
    path = Path(str(bag_dir)).expanduser()
    if path.is_absolute():
        return path
    return Path(workspace_root) / path


def _stamp_from_seconds(seconds):
    """用 JSON 表达 ROS Header.stamp 的 secs/nsecs 结构。"""
    stamp = max(0.0, float(seconds))
    secs = int(stamp)
    nsecs = int(round((stamp - secs) * 1_000_000_000))
    if nsecs >= 1_000_000_000:
        secs += 1
        nsecs -= 1_000_000_000
    return {"secs": secs, "nsecs": nsecs}


def _format_timestamp(dt):
    """统一输出带时区的 ISO 时间戳。"""
    return dt.isoformat(timespec="microseconds")


def _parse_timestamp(timestamp):
    """解析用户给定的 ISO 时间；没有时区时按本机本地时区处理。"""
    if timestamp is None:
        return datetime.now().astimezone()

    dt = datetime.fromisoformat(timestamp)
    return dt.astimezone()


def _ros_header_from_datetime(dt):
    """复刻 experiment.py 和 DynamixelControl 只填写 stamp 的 Header。"""
    return {"seq": 0, "stamp": _stamp_from_seconds(dt.timestamp()), "frame_id": ""}


def torque_to_current(torque):
    """沿用 optimizer/scripts/pareto_optimizer_node.py 的扭矩到电流拟合。"""
    return legacy_torque_to_current(torque)


def torque_to_energy(torque, voltage=12.0, fps=50.0):
    """沿用 optimizer/scripts/pareto_optimizer_node.py 的每帧能耗计算。"""
    current = torque_to_current(torque)
    return np.abs(current * voltage * (1.0 / fps))


class MujocoExperimentEnergyRunner:
    def __init__(
        self,
        config,
        interpolation_cls,
        real2sim,
        sim2real,
        env_cls,
        output_path,
        render=False,
        include_settle=True,
        fps=50.0,
        voltage=12.0,
        num_experiments=None,
        log_rate=None,
        bag_start_delay=DEFAULT_BAG_START_DELAY,
        record_commands=True,
        realtime_render=False,
        render_hold=0.0,
        workspace_root=None,
        timestamp_start=None,
        energy_mode="legacy",
    ):
        self.configs = config["configs"]
        self.sequence = config["sequence"]
        self.config_names = list(self.configs.keys())
        self.steps = int(config.get("steps", 100))
        self.step_time = float(config.get("step_time", 0.05))
        self.settle_time = float(config.get("settle_time", 1.0))
        self.flag_sim = bool(config.get("flag_sim", False))
        self.num_experiments = int(
            config.get("num_experiments", 1) if num_experiments is None else num_experiments
        )
        self.workspace_root = Path(workspace_root or Path.cwd())
        self.bag_dir = _resolve_bag_dir(config.get("bag_dir", "bags"), self.workspace_root)
        self.bag_prefix = config.get("bag_prefix", "trans_exp")
        self.bag_topics = list(config.get("bag_topics", DEFAULT_BAG_TOPICS))

        self.interpolation_cls = interpolation_cls
        self.real2sim = real2sim
        self.sim2real = sim2real
        self.env = env_cls(render_mode=None)
        self.output_path = Path(output_path)
        self.include_settle = include_settle
        self.fps = float(fps)
        self.voltage = float(voltage)
        self.log_rate = float(
            DEFAULT_DYNAMIXEL_LOG_RATE if log_rate is None else log_rate
        )
        self.bag_start_delay = float(bag_start_delay)
        self.record_commands = bool(record_commands)
        self.render = bool(render)
        self.realtime_render = bool(realtime_render)
        self.render_hold = float(render_hold)
        self.timestamp_start = timestamp_start or datetime.now().astimezone()
        self.energy_mode = str(energy_mode)
        self._timeline_elapsed = 0.0
        self._experiment_start_elapsed = 0.0
        self.viewer = (
            mujoco_viewer.launch_passive(self.env.model, self.env.data)
            if self.render
            else None
        )

        self._active_experiment = None
        self._sim_time = 0.0
        self._next_log_time = 0.0
        self._last_log_encoder = [int(v) for v in self.configs[self.sequence[0]]]
        self._last_log_current = np.zeros(17, dtype=np.float64)

        self._validate_config()

    def _validate_config(self):
        for name in self.sequence:
            if name not in self.configs:
                raise KeyError(f"sequence config '{name}' is missing from configs.")
            if len(self.configs[name]) != 17:
                raise ValueError(f"config '{name}' must contain 17 encoder values.")

        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if self.log_rate <= 0:
            raise ValueError("log_rate must be positive.")
        if self.bag_start_delay < 0:
            raise ValueError("bag_start_delay must be non-negative.")
        if self.render_hold < 0:
            raise ValueError("render_hold must be non-negative.")
        if self.energy_mode not in ENERGY_MODES:
            raise ValueError(f"energy_mode must be one of: {', '.join(ENERGY_MODES)}.")

    def _topic_types(self):
        """记录 experiment.py 默认 rosbag 话题对应的消息类型。"""
        return {
            "/crimson_control/transform": "crimson_msgs/Trans",
            "/crimson_control/transformed": "crimson_msgs/Trans",
            "/dynamixel_control/log": "dynamixel_msgs/LogData",
        }

    def _interpolate_segment(self, start_enc, end_enc):
        """完全沿用 trans_planner/scripts/experiment.py 的插值数据通路。"""
        q0 = self.real2sim(start_enc)
        qt = self.real2sim(end_enc)
        interp = self.interpolation_cls(self.flag_sim, q0, qt, self.steps)
        traj_sim = interp.gen_trajectory()
        return [self.sim2real(row) for row in traj_sim]

    def _encoder_frame_to_mujoco_control(self, encoder_frame):
        # experiment.py 发布的是 SetParam(motorID=1..17, params=encoder_frame)。
        return update_mujoco_control_from_real(
            encoder_frame,
            current_control=self.env.data.ctrl.copy(),
            motor_ids=MOTOR_IDS,
        )

    def _render_frame(self, playback_dt=0.0):
        """在 MuJoCo human viewer 中刷新一帧，必要时按真实时间停顿。"""
        if not self.render:
            return

        self.viewer.sync()
        if self.realtime_render and playback_dt > 0:
            time.sleep(playback_dt)

    def _simulate_control(
        self, control, encoder_frame=None, playback_dt=0.0, transition_label=None
    ):
        self.env.data.ctrl[:] = control
        self.env.do_simulation(self.env.data.ctrl, self.env.frame_skip)
        torque = self.env.get_joint_torque()
        if self.energy_mode == "legacy":
            self._last_log_current = np.array(torque_to_current(torque), dtype=np.float64)
        elif self.energy_mode == "dynamic_calibrated":
            calibrated_power = dynamic_calibrated_total_power_w(
                torque,
                transition=transition_label,
            )
            self._last_log_current = power_to_current_vector(
                calibrated_power,
                voltage=self.voltage,
                weights=np.abs(torque),
                motor_count=17,
            )
        else:
            calibrated_power = calibrated_total_power_w(
                torque,
                transition=transition_label,
                apply_residual=self.energy_mode == "calibrated",
            )
            self._last_log_current = power_to_current_vector(
                calibrated_power,
                voltage=self.voltage,
                weights=np.abs(torque),
                motor_count=17,
            )
        if encoder_frame is not None:
            self._last_log_encoder = [int(v) for v in encoder_frame]
        self._render_frame(playback_dt)
        if self.energy_mode == "legacy":
            return float(
                np.sum(torque_to_energy(torque, voltage=self.voltage, fps=self.fps))
            )
        return frame_energy_j(
            torque, fps=self.fps, mode=self.energy_mode, transition=transition_label
        )

    def _simulate_settle(self, final_control, transition_label=None):
        if not self.include_settle or self.settle_time <= 0:
            return 0.0, 0

        settle_steps = int(round(self.settle_time * self.fps))
        energy = 0.0
        for _ in range(settle_steps):
            energy += self._simulate_control(
                final_control,
                playback_dt=1.0 / self.fps,
                transition_label=transition_label,
            )
            self._advance_time(1.0 / self.fps)
        return energy, settle_steps

    def _hold_render_window(self):
        """实验结束后保持窗口一小段时间，便于观察最终是否落地。"""
        if not self.render or self.render_hold <= 0:
            return

        end_time = time.time() + self.render_hold
        while time.time() < end_time:
            self.viewer.sync()
            time.sleep(1.0 / 30.0)

    def _close_viewer(self):
        """关闭 MuJoCo passive viewer。"""
        if self.viewer is None:
            return

        self.viewer.close()
        self.viewer = None

    def _timestamp_at(self, local_stamp):
        """按实验计划时间轴计算绝对时间，避免 MuJoCo 快跑压缩时间戳。"""
        elapsed = self._experiment_start_elapsed + float(local_stamp)
        return self.timestamp_start + timedelta(seconds=elapsed)

    def _time_fields(self, local_stamp):
        """生成记录项共用的相对时间与真实时间戳字段。"""
        timestamp = self._timestamp_at(local_stamp)
        elapsed = self._experiment_start_elapsed + float(local_stamp)
        return {
            "time": float(local_stamp),
            "relative_time_s": float(local_stamp),
            "elapsed_real_s": elapsed,
            "timestamp": _format_timestamp(timestamp),
            "unix_time": timestamp.timestamp(),
        }

    def _new_experiment_record(self, exp_idx):
        """创建一份与 experiment.py 单次 rosbag 录制等价的 JSON 容器。"""
        start_time = self._timestamp_at(0.0)
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        bag_name = "{}_{:02d}_{}".format(self.bag_prefix, exp_idx, timestamp)
        return {
            "experiment": exp_idx,
            "started_at": _format_timestamp(start_time),
            "scheduled_start_elapsed_s": self._experiment_start_elapsed,
            "bag_name": bag_name,
            "bag_output_prefix": str(self.bag_dir / bag_name),
            "bag_path": str(self.bag_dir / f"{bag_name}.bag"),
            "bag_topics": self.bag_topics,
            "topic_types": self._topic_types(),
            "records": {topic: [] for topic in self.bag_topics},
            "messages": [],
            "sync_write_commands": [],
            "duration_s": 0.0,
        }

    def _append_topic_record(self, topic, stamp, msg_type, msg):
        """只记录 experiment.py 默认 rosbag 会录到的话题。"""
        if self._active_experiment is None or topic not in self.bag_topics:
            return

        entry = {
            "topic": topic,
            "type": msg_type,
            "msg": msg,
        }
        entry.update(self._time_fields(stamp))
        self._active_experiment["records"].setdefault(topic, []).append(entry)
        self._active_experiment["messages"].append(entry)

    def _append_sync_write_command(self, stamp, encoder_frame):
        """保存 experiment.py 每帧实际发布的 SetParam 指令。"""
        if self._active_experiment is None or not self.record_commands:
            return

        self._active_experiment["sync_write_commands"].append(
            {
                "topic": "/dynamixel_control/sync_write",
                "type": "dynamixel_msgs/SetParam",
                "msg": {
                    "paramType": PARAM_GOAL_POSITION,
                    "motorID": MOTOR_IDS,
                    "params": [int(v) for v in encoder_frame],
                },
                **self._time_fields(stamp),
            }
        )

    def _append_marker(self, topic, step_idx, cfg_name):
        """复刻 experiment.py::_publish_marker 的 Trans 消息字段。"""
        try:
            mode = self.config_names.index(cfg_name) & 0xFF
        except ValueError:
            mode = 0

        msg = {
            "header": _ros_header_from_datetime(self._timestamp_at(self._sim_time)),
            "cfg": step_idx & 0xFF,
            "mode": mode,
            "w": 0,
            "h": 0,
        }
        self._append_topic_record(topic, self._sim_time, "crimson_msgs/Trans", msg)

    def _current_joint_position(self):
        """Return the 17 actuated MuJoCo joint positions used for qpos phase timing."""
        qpos = np.asarray(self.env.data.qpos, dtype=np.float64).reshape(-1)
        actuated = qpos[7:] if qpos.shape[0] >= 24 else qpos[-17:]
        if actuated.shape[0] != 17:
            raise ValueError(
                f"Expected 17 actuated qpos values, got {actuated.shape[0]}."
            )
        return [float(v) for v in actuated]

    def _append_log(self, stamp):
        """复刻 dynamixel_control 发布的 LogData 字段形状。"""
        msg = {
            "header": _ros_header_from_datetime(self._timestamp_at(stamp)),
            "I": [float(v) for v in self._last_log_current],
            "V": [],
            "P": [float(v) for v in self._last_log_encoder],
            "U": [float(self.voltage) for _ in MOTOR_IDS],
            "T": [],
            "qpos": self._current_joint_position(),
        }
        self._append_topic_record(stamp=stamp, topic="/dynamixel_control/log",
                                  msg_type="dynamixel_msgs/LogData", msg=msg)

    def _emit_logs_until(self, end_time):
        """按 DynamixelControl::Run 的 looprate 生成 /dynamixel_control/log。"""
        epsilon = 1e-12
        while self._next_log_time <= end_time + epsilon:
            self._append_log(self._next_log_time)
            self._next_log_time += 1.0 / self.log_rate

    def _advance_time(self, delta):
        """推进实验逻辑时间，并补齐这段时间内的 Dynamixel log。"""
        end_time = self._sim_time + float(delta)
        self._emit_logs_until(end_time)
        self._sim_time = end_time

    def _run_segment(self, exp_idx, step_idx, start_name, end_name):
        start_enc = self.configs[start_name]
        end_enc = self.configs[end_name]
        encoder_frames = self._interpolate_segment(start_enc, end_enc)

        transition_energy = 0.0
        final_control = None
        transition_label = f"mu{int(start_name[1:])}->mu{int(end_name[1:])}"
        transform_time = self._sim_time
        self._append_marker("/crimson_control/transform", step_idx, end_name)
        command_start = 0
        if self._active_experiment is not None:
            command_start = len(self._active_experiment["sync_write_commands"])

        for encoder_frame in encoder_frames:
            self._append_sync_write_command(self._sim_time, encoder_frame)
            final_control = self._encoder_frame_to_mujoco_control(encoder_frame)
            transition_energy += self._simulate_control(
                final_control,
                encoder_frame,
                playback_dt=self.step_time,
                transition_label=transition_label,
            )
            self._advance_time(self.step_time)

        transformed_time = self._sim_time
        self._append_marker("/crimson_control/transformed", step_idx, end_name)
        settle_energy, settle_steps = self._simulate_settle(
            final_control, transition_label=transition_label
        )
        command_end = command_start
        if self._active_experiment is not None:
            command_end = len(self._active_experiment["sync_write_commands"])

        return {
            "experiment": exp_idx,
            "step": step_idx,
            "from": start_name,
            "to": end_name,
            "frames": len(encoder_frames),
            "settle_steps": settle_steps,
            "transform_time_s": transform_time,
            "transformed_time_s": transformed_time,
            "transform_timestamp": _format_timestamp(self._timestamp_at(transform_time)),
            "transformed_timestamp": _format_timestamp(
                self._timestamp_at(transformed_time)
            ),
            "sync_write_command_start": command_start,
            "sync_write_command_end": command_end,
            "transition_energy_j": transition_energy,
            "settle_energy_j": settle_energy,
            "total_energy_j": transition_energy + settle_energy,
        }

    def run(self):
        self._timeline_elapsed = 0.0
        started_at = self.timestamp_start
        results = {
            "created_at": _format_timestamp(datetime.now().astimezone()),
            "timeline_started_at": _format_timestamp(started_at),
            "sequence": self.sequence,
            "steps": self.steps,
            "step_time": self.step_time,
            "settle_time": self.settle_time,
            "flag_sim": self.flag_sim,
            "fps": self.fps,
            "voltage": self.voltage,
            "energy_mode": self.energy_mode,
            "include_settle": self.include_settle,
            "num_experiments": self.num_experiments,
            "recording": {
                "source_script": "src/trans_planner/scripts/experiment.py",
                "bag_start_delay_s": self.bag_start_delay,
                "dynamixel_log_rate_hz": self.log_rate,
                "bag_dir": str(self.bag_dir),
                "bag_prefix": self.bag_prefix,
                "bag_topics": self.bag_topics,
                "record_commands": self.record_commands,
                "timestamp_policy": (
                    "所有 timestamp 按 experiment.py 的真实节奏推进："
                    "step_time、settle_time、fps 和 log_rate 决定时间间隔；"
                    "不会使用 MuJoCo 非可视化快速执行所消耗的实际运行时间。"
                ),
                "message_shapes": {
                    "/crimson_control/transform": "crimson_msgs/Trans",
                    "/crimson_control/transformed": "crimson_msgs/Trans",
                    "/dynamixel_control/log": "dynamixel_msgs/LogData",
                    "/dynamixel_control/sync_write": "dynamixel_msgs/SetParam",
                },
            },
            "segments": [],
            "experiments": [],
        }

        try:
            for exp_idx in range(1, self.num_experiments + 1):
                self._experiment_start_elapsed = self._timeline_elapsed
                self._active_experiment = self._new_experiment_record(exp_idx)
                self._sim_time = 0.0
                self._next_log_time = 0.0
                self._last_log_encoder = [
                    int(v) for v in self.configs[self.sequence[0]]
                ]
                self._last_log_current = np.zeros(17, dtype=np.float64)
                self._advance_time(self.bag_start_delay)

                for step_idx in range(len(self.sequence) - 1):
                    segment = self._run_segment(
                        exp_idx,
                        step_idx,
                        self.sequence[step_idx],
                        self.sequence[step_idx + 1],
                    )
                    results["segments"].append(segment)
                    print(
                        "[mujoco_energy] exp={experiment} step={step} "
                        "{from}->{to} total={total_energy_j:.4f} J".format(**segment)
                    )
                self._active_experiment["duration_s"] = self._sim_time
                end_time = self._timestamp_at(self._sim_time)
                self._active_experiment["ended_at"] = _format_timestamp(end_time)
                self._active_experiment["scheduled_end_elapsed_s"] = (
                    self._experiment_start_elapsed + self._sim_time
                )
                results["experiments"].append(self._active_experiment)
                self._active_experiment = None
                self._timeline_elapsed += self._sim_time
        finally:
            self._hold_render_window()
            self._close_viewer()
            self.env.close()

        results["timeline_ended_at"] = _format_timestamp(
            started_at + timedelta(seconds=self._timeline_elapsed)
        )
        results["timeline_duration_s"] = self._timeline_elapsed
        results["total_energy_j"] = float(
            sum(segment["total_energy_j"] for segment in results["segments"])
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w") as f:
            json.dump(results, f, indent=2)

        return results


def parse_args():
    paths = _repo_paths()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = paths["default_output_dir"] / f"mujoco_experiment_energy_{timestamp}.json"

    parser = argparse.ArgumentParser(
        description="在 MuJoCo 中复现实物变形实验，保存同口径记录并计算能耗。"
    )
    parser.add_argument("--config", default=str(paths["default_config"]))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--num-experiments", type=int, default=None)
    parser.add_argument(
        "--timestamp-start",
        default=None,
        help="真实时间轴起点 ISO 时间；默认使用程序启动时本机时间。",
    )
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--voltage", type=float, default=12.0)
    parser.add_argument(
        "--energy-mode",
        choices=ENERGY_MODES,
        default="legacy",
        help=(
            "Energy calculation model. legacy keeps the original torque-current "
            "polynomial; calibrated uses the first-pass real-loop power "
            "calibration; dynamic_calibrated adds the final 50 Hz interval "
            "offsets used for the calibration-set trace comparison."
        ),
    )
    parser.add_argument(
        "--log-rate",
        type=float,
        default=None,
        help="Dynamixel /log 采样频率；默认读取 dynamixel.yaml 的 looprate。",
    )
    parser.add_argument(
        "--bag-start-delay",
        type=float,
        default=DEFAULT_BAG_START_DELAY,
        help="匹配 experiment.py 在第一个 marker 前等待 rosbag 建立订阅的 0.5s。",
    )
    parser.add_argument(
        "--no-command-record",
        action="store_true",
        help="不保存 experiment.py 每帧生成的 SetParam 指令。",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--fast-render",
        action="store_true",
        help="打开可视化但不按 step_time/fps 等待，用最快速度播放。",
    )
    parser.add_argument(
        "--render-hold",
        type=float,
        default=10.0,
        help="可视化运行结束后保持窗口的秒数。",
    )
    parser.add_argument(
        "--no-settle",
        action="store_true",
        help="跳过每段结束后的 settle_time 保持阶段。",
    )
    return parser.parse_args()


def main():
    paths = _repo_paths()
    args = parse_args()

    Interpolation, real2sim, sim2real = _import_trans_planner_flow(paths)
    env_cls = _import_mujoco_env(paths)

    with Path(args.config).open("r") as f:
        config = yaml.safe_load(f)
    log_rate = _load_dynamixel_log_rate(paths) if args.log_rate is None else args.log_rate
    timestamp_start = _parse_timestamp(args.timestamp_start)

    runner = MujocoExperimentEnergyRunner(
        config=config,
        interpolation_cls=Interpolation,
        real2sim=real2sim,
        sim2real=sim2real,
        env_cls=env_cls,
        output_path=args.output,
        render=args.render,
        include_settle=not args.no_settle,
        fps=args.fps,
        voltage=args.voltage,
        num_experiments=args.num_experiments,
        log_rate=log_rate,
        bag_start_delay=args.bag_start_delay,
        record_commands=not args.no_command_record,
        realtime_render=args.render and not args.fast_render,
        render_hold=args.render_hold if args.render else 0.0,
        workspace_root=paths["repo"],
        timestamp_start=timestamp_start,
        energy_mode=args.energy_mode,
    )
    results = runner.run()
    print(f"[mujoco_energy] total={results['total_energy_j']:.4f} J")
    print(f"[mujoco_energy] saved={args.output}")


if __name__ == "__main__":
    main()
