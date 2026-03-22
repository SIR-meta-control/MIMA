#!/usr/bin/env python3

import numpy as np
import math
from utils.rules import sign_matrix, bias_matrix, rules, inv_rules

def real2sim(encoder_values):
    # 输入: data.params (len=17)
    data17 = np.array(encoder_values, dtype=np.float64)

    # 步骤1: 编码器值转角度（度）
    # 2048 刻度 → 180 度，公式：(val * 180 / 2048)
    # 减去 bias，再乘 sign
    data17 = np.dot(sign_matrix, data17 * 180 / 2048 - bias_matrix)

    # 步骤2: 角度(度) → 弧度
    data17 = [math.radians(angle) for angle in data17]

    # 步骤3: 按 rules 映射到 mujoco 需要的顺序
    dd = np.zeros(17, dtype=np.float64)
    for i in range(17):
        if i in rules:
            dd[i] = data17[rules[i]]
    return dd.tolist()  # 返回 mujoco 需要的关节角度列表

def sim2real(rad):
    # Step1: 弧度 → data17(弧度)
    data17 = np.zeros(17, dtype=np.float64)
    for k in range(17):
        if k in inv_rules:
            data17[k] = rad[inv_rules[k]]

    # Step2: 弧度 → 度
    data17_deg = np.array([math.degrees(a) for a in data17], dtype=np.float64)

    # Step3: 恢复 sign & bias
    angle_before_bias = np.dot(np.linalg.inv(sign_matrix), data17_deg) + bias_matrix

    # Step4: 度 → 编码器值
    encoder_vals = angle_before_bias * 2048 / 180

    # Step5: 转换为整数
    encoder_vals = np.round(encoder_vals).astype(np.int32).tolist()
    return encoder_vals