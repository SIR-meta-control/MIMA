#!/usr/bin/env python3
"""
实物 Dynamixel 编码器指令与 MuJoCo 控制向量之间的转换工具。

这里保留 crimson_sim/ros_interface 分支中的标定参数和映射规则：
先按符号与零位偏置把编码器值转换为弧度，再重排到 MuJoCo
模型的 actuator 顺序。
"""

import math

import numpy as np


SIGN_MATRIX = np.diag(
    [-1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1]
)
BIAS_MATRIX = np.array(
    [90, 90, 180, 180, 180, 180, 158, 248, 180, 158, 248, 180, 158, 248, 180, 158, 248],
    dtype=np.float64,
)

# Maps MuJoCo control index -> real Dynamixel index.
RULES = {
    0: 0,
    1: 2,
    2: 4,
    3: 1,
    4: 3,
    5: 8,
    6: 9,
    7: 10,
    8: 14,
    9: 15,
    10: 16,
    11: 11,
    12: 12,
    13: 13,
    14: 5,
    15: 6,
    16: 7,
}
REAL_TO_MUJOCO = {real_index: mujoco_index for mujoco_index, real_index in RULES.items()}


def encoder_ticks_to_radians(encoder_values, start_index=0):
    """Convert raw Dynamixel encoder ticks to signed, biased radians."""
    values = np.array(encoder_values, dtype=np.float64)
    end_index = start_index + len(values)
    signs = SIGN_MATRIX[start_index:end_index, start_index:end_index]
    bias = BIAS_MATRIX[start_index:end_index]

    degrees = np.dot(signs, values * 180.0 / 2048.0 - bias)
    return np.array([math.radians(angle) for angle in degrees], dtype=np.float64)


def real_to_mujoco_control(encoder_values):
    """Convert a full 17-motor real command to MuJoCo's 17D actuator order."""
    radians = encoder_ticks_to_radians(encoder_values)
    control = np.zeros(17, dtype=np.float64)

    for mujoco_index, real_index in RULES.items():
        control[mujoco_index] = radians[real_index]

    return control


def update_mujoco_control_from_motor_ids(encoder_values, motor_ids, current_control):
    """Update a 17D MuJoCo control vector using SetParam.motorID/params pairs."""
    control = np.array(current_control, dtype=np.float64).reshape(-1)
    if control.shape[0] != 17:
        raise ValueError(f"current_control must be 17D, got {control.shape[0]}.")
    if len(encoder_values) != len(motor_ids):
        raise ValueError(
            f"motorID/params length mismatch: {len(motor_ids)} ids, "
            f"{len(encoder_values)} params."
        )

    for motor_id, encoder_value in zip(motor_ids, encoder_values):
        real_index = int(motor_id) - 1
        if real_index not in REAL_TO_MUJOCO:
            raise ValueError(f"Unsupported motorID {motor_id}. Expected 1..17.")
        mujoco_index = REAL_TO_MUJOCO[real_index]
        control[mujoco_index] = encoder_ticks_to_radians(
            [encoder_value], start_index=real_index
        )[0]

    return control


def update_mujoco_control_from_real(encoder_values, current_control=None, motor_ids=None):
    """
    Convert a real command to MuJoCo actuator order.

    Supports SetParam.motorID when present, plus the message shapes used by the
    original ros_interface.py:
    - 17 params: full robot command
    - 5 params: frame/body motors only, updates indices 0..4
    - 12 params: leg motors only, updates indices 5..16
    """
    values = list(encoder_values)
    ids = list(motor_ids or [])

    if ids:
        if current_control is None:
            raise ValueError("motorID-based commands require an existing 17D control vector.")
        return update_mujoco_control_from_motor_ids(values, ids, current_control)

    if len(values) == 17:
        return real_to_mujoco_control(values)

    if current_control is None:
        raise ValueError("Partial real commands require an existing 17D control vector.")

    control = np.array(current_control, dtype=np.float64).reshape(-1)
    if control.shape[0] != 17:
        raise ValueError(f"current_control must be 17D, got {control.shape[0]}.")

    if len(values) == 5:
        body_radians = encoder_ticks_to_radians(values, start_index=0)
        for mujoco_index in range(5):
            control[mujoco_index] = body_radians[RULES[mujoco_index]]
        return control

    if len(values) == 12:
        leg_radians = encoder_ticks_to_radians(values, start_index=5)
        for mujoco_index in range(5, 17):
            control[mujoco_index] = leg_radians[RULES[mujoco_index] - 5]
        return control

    raise ValueError(f"Expected 17, 5, or 12 encoder values, got {len(values)}.")
