#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变形重复实验编排脚本 (ROS1)。

流程：
  对每次实验 (共 num_experiments 次)：
    1. 启动一个 rosbag record 子进程，录制 transform/transformed/log 三个话题；
    2. 按 sequence (c1->...->c9->c1) 依次变形，相邻构型之间用 sim 空间线性插值，
       逐帧通过 /dynamixel_control/sync_write (sim2real) 下发；
    3. 每段开始前发布 /crimson_control/transform 标记，到达后发布 /crimson_control/transformed；
    4. 一次实验结束后停止 rosbag，等待用户回车再开始下一次。

数据空间：配置文件里 c1..c9 填的是 real 编码器值；脚本读 real2sim 转到 sim 空间，
在 sim 空间做带躯干轴对齐约束的线性插值，再 sim2real 回编码器值下发，
与 trans_planner/planner.py 的实物路径一致。
"""

import os
import sys
import signal
import subprocess
from datetime import datetime

import rospy

from crimson_msgs.msg import Trans
from dynamixel_msgs.msg import SetParam

# catkin 安装后 rosrun 经 devel/lib 的 wrapper 运行本脚本，scripts/ 目录不在
# sys.path 上，导致同目录的 interpolation / utils 无法导入。这里显式补上。
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from interpolation import Interpolation
from utils.sim2real import sim2real, real2sim

MOTOR_IDS = list(range(1, 18))   # 电机 ID 1..17
PARAM_GOAL_POSITION = 1          # SetParam.paramType=1 -> GoalPosition


class ExperimentRunner:
    def __init__(self):
        # ---- 读取参数 ----
        self.configs = rospy.get_param("~configs")          # dict: name -> 17 编码器值
        self.sequence = rospy.get_param("~sequence")        # list of config names
        self.steps = int(rospy.get_param("~steps", 100))
        self.step_time = float(rospy.get_param("~step_time", 0.05))
        self.settle_time = float(rospy.get_param("~settle_time", 1.0))
        self.flag_sim = bool(rospy.get_param("~flag_sim", False))
        self.num_experiments = int(rospy.get_param("~num_experiments", 10))
        self.wait_enter = bool(rospy.get_param("~wait_enter", True))

        bag_dir = rospy.get_param("~bag_dir", "bags")
        self.bag_dir = self._resolve_bag_dir(bag_dir)
        self.bag_prefix = rospy.get_param("~bag_prefix", "trans_exp")
        self.bag_topics = rospy.get_param(
            "~bag_topics",
            ["/crimson_control/transform",
             "/crimson_control/transformed",
             "/dynamixel_control/log"],
        )
        os.makedirs(self.bag_dir, exist_ok=True)

        # ---- 校验配置 ----
        for name in self.sequence:
            if name not in self.configs:
                rospy.logfatal("[experiment] sequence 中的 '%s' 不在 configs 里", name)
                raise KeyError(name)
            if len(self.configs[name]) != 17:
                rospy.logfatal("[experiment] 构型 '%s' 的关节数不是 17", name)
                raise ValueError(name)

        # ---- 发布器 ----
        self.transform_pub = rospy.Publisher(
            "/crimson_control/transform", Trans, queue_size=1, latch=True)
        self.transformed_pub = rospy.Publisher(
            "/crimson_control/transformed", Trans, queue_size=1, latch=True)
        self.actuator_pub = rospy.Publisher(
            "/dynamixel_control/sync_write", SetParam, queue_size=10)

        self._bag_proc = None
        self._bag_path = None
        rospy.loginfo("[experiment] 初始化完成：%d 次实验，序列 %s",
                      self.num_experiments, self.sequence)
        rospy.loginfo("[experiment] rosbag 保存目录: %s", self.bag_dir)

    # ---------------------------------------------------------------
    # 解析 bag 保存目录：
    #   - 绝对路径 或 ~ 开头 -> 直接展开使用
    #   - 相对路径 (如 "bags") -> 挂到 catkin 工作空间根目录下
    #     (从脚本自身路径向上回溯，找到含 src/ 的目录即为工作空间根；
    #      找不到则退回当前工作目录)
    # ---------------------------------------------------------------
    def _resolve_bag_dir(self, bag_dir):
        bag_dir = os.path.expanduser(bag_dir)
        if os.path.isabs(bag_dir):
            return bag_dir
        here = os.path.dirname(os.path.realpath(__file__))
        ws_root = None
        d = here
        while True:
            if os.path.isdir(os.path.join(d, "src")):
                ws_root = d
                break
            parent = os.path.dirname(d)
            if parent == d:        # 到根了仍没找到
                break
            d = parent
        base = ws_root if ws_root is not None else os.getcwd()
        return os.path.join(base, bag_dir)

    # ---------------------------------------------------------------
    # 标记话题：用 Trans.cfg 字段携带「序列中的第几步」，header.stamp 记时间。
    # mode 字段复用为「该构型在 configs 中的索引」(可选)，w/h 留 0。
    # ---------------------------------------------------------------
    def _publish_marker(self, pub, step_idx, cfg_name):
        msg = Trans()
        msg.header.stamp = rospy.Time.now()
        msg.cfg = step_idx & 0xFF          # 序列步号 (0-based)
        try:
            msg.mode = list(self.configs.keys()).index(cfg_name) & 0xFF
        except ValueError:
            msg.mode = 0
        msg.w = 0
        msg.h = 0
        pub.publish(msg)

    def _send_encoder(self, encoder_vals):
        """把一帧 17 维编码器值通过 sync_write 下发 (GoalPosition)。"""
        sp = SetParam()
        sp.paramType = PARAM_GOAL_POSITION
        sp.motorID = MOTOR_IDS
        sp.params = [int(v) for v in encoder_vals]
        self.actuator_pub.publish(sp)

    def _interpolate_segment(self, start_enc, end_enc):
        """在 sim 空间对一段 (start->end) 做插值，返回 real 编码器帧列表。"""
        q0 = real2sim(start_enc)       # real 编码器 -> sim 弧度 (17 维)
        qt = real2sim(end_enc)
        interp = Interpolation(self.flag_sim, q0, qt, self.steps)
        traj_sim = interp.gen_trajectory()        # 2D: 每行 sim 关节角
        return [sim2real(row) for row in traj_sim]  # 每行 -> real 编码器值

    # ---------------------------------------------------------------
    # rosbag 录制：每次实验一个独立 bag，用进程组 + SIGINT 优雅收尾。
    # ---------------------------------------------------------------
    def _start_bag(self, exp_idx):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = "{}_{:02d}_{}".format(self.bag_prefix, exp_idx, ts)
        out_path = os.path.join(self.bag_dir, name)
        self._bag_path = out_path + ".bag"
        cmd = ["rosbag", "record", "-O", out_path] + list(self.bag_topics)
        rospy.loginfo("[experiment] 开始录制: %s", self._bag_path)
        # setsid 使 rosbag 成为新进程组组长，便于整体发 SIGINT。
        self._bag_proc = subprocess.Popen(cmd, preexec_fn=os.setsid)

    def _stop_bag(self):
        if self._bag_proc is None:
            return
        bag_path = getattr(self, "_bag_path", None)
        try:
            # 向 rosbag 进程组发 SIGINT，触发其正常 flush + 重命名 .active->.bag
            os.killpg(os.getpgid(self._bag_proc.pid), signal.SIGINT)
            # 给足时间 flush 缓冲（大 bag 可能较久）；先等 30s
            try:
                self._bag_proc.wait(timeout=30)
                rospy.loginfo("[experiment] rosbag 已正常关闭: %s", bag_path)
            except subprocess.TimeoutExpired:
                # 还没退出，再发一次 SIGINT 并多等，尽量避免强杀留下 .active
                rospy.logwarn("[experiment] rosbag 收尾较慢，再等待...")
                os.killpg(os.getpgid(self._bag_proc.pid), signal.SIGINT)
                self._bag_proc.wait(timeout=30)
                rospy.loginfo("[experiment] rosbag 已关闭: %s", bag_path)
        except Exception as e:
            rospy.logwarn("[experiment] 关闭 rosbag 异常: %s，强制结束（可能留下 "
                          ".bag.active，可用 rosbag reindex 修复）", e)
            try:
                os.killpg(os.getpgid(self._bag_proc.pid), signal.SIGKILL)
            except Exception:
                pass
        finally:
            self._bag_proc = None

    # ---------------------------------------------------------------
    # 单次实验：按 sequence 逐段变形。
    # ---------------------------------------------------------------
    def _run_once(self, exp_idx):
        rate = rospy.Rate(1.0 / self.step_time) if self.step_time > 0 else None
        self._start_bag(exp_idx)
        rospy.sleep(0.5)   # 给 rosbag 订阅留出建立时间

        for step in range(len(self.sequence) - 1):
            if rospy.is_shutdown():
                break
            cur_name = self.sequence[step]
            nxt_name = self.sequence[step + 1]
            start_enc = self.configs[cur_name]
            end_enc = self.configs[nxt_name]
            rospy.loginfo("[experiment] 实验%d 段%d: %s -> %s",
                          exp_idx, step, cur_name, nxt_name)

            # 标记：开始变到 nxt
            self._publish_marker(self.transform_pub, step, nxt_name)

            frames = self._interpolate_segment(start_enc, end_enc)
            for frame in frames:
                if rospy.is_shutdown():
                    break
                self._send_encoder(frame)
                if rate is not None:
                    rate.sleep()
                else:
                    rospy.sleep(self.step_time)

            # 标记：该段变形完成 (已到达 nxt)
            self._publish_marker(self.transformed_pub, step, nxt_name)
            rospy.sleep(self.settle_time)

        self._stop_bag()
        rospy.loginfo("[experiment] 实验%d 完成", exp_idx)

    # ---------------------------------------------------------------
    # 等待回车，但可被 Ctrl-C / 节点关闭打断。
    # rospy 的 SIGINT handler 只设 shutdown 标志、不会中断阻塞的
    # sys.stdin.readline()，所以这里用 select 轮询 stdin，
    # 每 0.2s 检查一次 is_shutdown，保证 Ctrl-C 能及时退出。
    # 返回 True 表示用户按了回车，False 表示因关闭而中断。
    # ---------------------------------------------------------------
    def _wait_for_enter(self):
        import select
        while not rospy.is_shutdown():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            except KeyboardInterrupt:
                return False
            except Exception:
                return False
            if ready:
                sys.stdin.readline()
                return True
        return False

    # ---------------------------------------------------------------
    # 顶层循环：重复 num_experiments 次，每次之间等回车。
    # ---------------------------------------------------------------
    def run(self):
        try:
            for exp_idx in range(1, self.num_experiments + 1):
                if rospy.is_shutdown():
                    break
                rospy.loginfo("========== 实验 %d / %d ==========",
                              exp_idx, self.num_experiments)
                self._run_once(exp_idx)

                if exp_idx < self.num_experiments and self.wait_enter:
                    rospy.loginfo("按回车开始下一次实验 (Ctrl-C 退出)...")
                    if not self._wait_for_enter():
                        rospy.loginfo("[experiment] 收到退出信号，结束实验")
                        break
            rospy.loginfo("[experiment] 全部 %d 次实验结束", self.num_experiments)
        except (KeyboardInterrupt, rospy.ROSInterruptException):
            rospy.loginfo("[experiment] 被中断，正在收尾")
        finally:
            # 任何情况下都确保 bag 被关闭，避免留下损坏的 .bag.active
            self._stop_bag()


def main():
    rospy.init_node("trans_experiment")
    runner = ExperimentRunner()
    # 确保节点被关闭时也收尾 rosbag
    rospy.on_shutdown(runner._stop_bag)
    runner.run()


if __name__ == "__main__":
    main()
