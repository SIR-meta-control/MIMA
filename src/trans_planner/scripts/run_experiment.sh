#!/usr/bin/env bash
# ============================================================
# 变形重复实验一键启动 (交互式：每次实验后按回车继续)
# ------------------------------------------------------------
# 与 experiment.launch 的区别：
#   本脚本用 rosrun 在前台运行 experiment.py，stdin 连终端，
#   因此支持「每完成一次实验，按回车开始下一次」。
#
# 用法 (在机器人 jetson 上，已 source 工作空间)：
#   rosrun trans_planner run_experiment.sh
#   或:  ./run_experiment.sh
#
# 可选环境变量：
#   START_DYNAMIXEL=0   不自动启动底层电机驱动 (已单独起好时用)
# ============================================================
set -u

CONFIG="$(rospack find trans_planner)/config/experiment.yaml"

cleanup() {
  echo ""
  echo "[run_experiment] 正在清理后台进程..."
  [ -n "${DXL_PID:-}" ] && kill -INT "$DXL_PID" 2>/dev/null
  [ -n "${CORE_PID:-}" ] && kill -INT "$CORE_PID" 2>/dev/null
  wait 2>/dev/null
  echo "[run_experiment] 已退出"
}
trap cleanup EXIT INT TERM

# --- 确保 roscore 在运行 ---
if ! rostopic list >/dev/null 2>&1; then
  echo "[run_experiment] 未检测到 roscore，正在启动..."
  roscore >/tmp/roscore_experiment.log 2>&1 &
  CORE_PID=$!
  until rostopic list >/dev/null 2>&1; do sleep 0.3; done
  echo "[run_experiment] roscore 已就绪"
fi

# --- 可选：启动底层电机驱动 ---
if [ "${START_DYNAMIXEL:-1}" = "1" ]; then
  echo "[run_experiment] 启动 dynamixel_control..."
  rosparam set /dynamixel_control/dyn_yaml_path \
    "$(rospack find dynamixel_control)/config/dynamixel.yaml"
  rosrun dynamixel_control dynamixel_control_node __name:=dynamixel_control_node \
    >/tmp/dynamixel_experiment.log 2>&1 &
  DXL_PID=$!
  sleep 2
fi

# --- 加载实验参数到 experiment.py 的私有命名空间 ---
echo "[run_experiment] 加载配置: $CONFIG"
rosparam load "$CONFIG" /trans_experiment

# --- 前台运行实验节点 (stdin 连终端，可按回车) ---
echo "[run_experiment] 启动实验节点 (前台，可按回车推进)"
rosrun trans_planner experiment.py __name:=trans_experiment
