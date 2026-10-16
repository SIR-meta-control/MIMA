#!/usr/bin/env python3
"""Energy calibration helpers for Crimson MuJoCo replay.

The legacy optimizer path estimates electrical current from MuJoCo actuator
torque. The first sim-to-real pass added an aggregate torque-power model and
transition residuals for the first 15 command intervals. The dynamic 50 Hz
calibration keeps that model available and adds a trace-level
calibration: one constant power offset per plotted command interval, including
the final ``mu2->mu1`` interval.
"""

from __future__ import annotations

import numpy as np


CALIBRATED_BASE_POWER_W = 61.2116943
CALIBRATED_FRAME_ABS_TORQUE_W_PER_NM = 0.0
CALIBRATED_LEG_ABS_TORQUE_W_PER_NM = 1.3421639

TRANSITION_POWER_RESIDUAL_W = {
    "mu1->mu2": -13.7708180,
    "mu2->mu3": -5.1653288,
    "mu3->mu4": 2.4187454,
    "mu4->mu5": 13.0809215,
    "mu5->mu6": 9.1699464,
    "mu6->mu7": 41.3410958,
    "mu7->mu8": 21.7334435,
    "mu8->mu9": -24.6387578,
    "mu9->mu8": -12.8179745,
    "mu8->mu7": 4.7344551,
    "mu7->mu6": -11.8743418,
    "mu6->mu5": -10.0481093,
    "mu5->mu4": -11.8350742,
    "mu4->mu3": -4.3525612,
    "mu3->mu2": 2.0283587,
}

DYNAMIC_INTERVAL_POWER_OFFSET_W = {
    "mu1->mu2": 3.2586327407447246,
    "mu2->mu3": 2.284629898006543,
    "mu3->mu4": 0.10055628208387082,
    "mu4->mu5": 2.614266956362949,
    "mu5->mu6": 2.095915482632857,
    "mu6->mu7": -0.7124498184012245,
    "mu7->mu8": 1.2280883696026628,
    "mu8->mu9": 2.64470202455634,
    "mu9->mu8": 1.8879370761304664,
    "mu8->mu7": 2.924879393748947,
    "mu7->mu6": 1.080222406339399,
    "mu6->mu5": -0.8841816225402332,
    "mu5->mu4": -0.93152228749585,
    "mu4->mu3": -1.2458148740957125,
    "mu3->mu2": 2.605161162156859,
    "mu2->mu1": -17.483995129104606,
}


def legacy_torque_to_current(torque):
    """Return the historical torque-to-current polynomial in amperes."""
    torque = np.asarray(torque, dtype=np.float64)
    return (
        (0.000130565974) * torque**4
        + (-0.00188139351) * torque**3
        + (0.0216771226) * torque**2
        + (0.410017411) * torque
        + 0.0357777777778
    )


def legacy_total_power_w(torque, voltage=12.0):
    """Estimate total electrical power using the original polynomial model."""
    current = legacy_torque_to_current(torque)
    return float(np.sum(np.abs(current * float(voltage))))


def calibrated_total_power_w(torque, transition=None, apply_residual=True):
    """Estimate calibrated total electrical power for a 17D torque vector.

    The calibrated model was fitted against the real transformation-loop CSVs
    using ``frame kp=30`` and ``leg kp=10`` MuJoCo actuators:

    ``P = 61.2116943 + 1.3421639 * sum(abs(tau_leg))``.

    The optional transition residual is deliberately explicit. It captures
    path-specific frame-motor load that is visible in real current logs but not
    represented by the current ideal MuJoCo mechanism.
    """
    torque = np.asarray(torque, dtype=np.float64).reshape(-1)
    if torque.shape[0] != 17:
        raise ValueError(f"Expected 17 actuator torques, got {torque.shape[0]}.")

    power = (
        CALIBRATED_BASE_POWER_W
        + CALIBRATED_FRAME_ABS_TORQUE_W_PER_NM * float(np.sum(np.abs(torque[:5])))
        + CALIBRATED_LEG_ABS_TORQUE_W_PER_NM * float(np.sum(np.abs(torque[5:])))
    )
    if apply_residual and transition:
        power += TRANSITION_POWER_RESIDUAL_W.get(str(transition), 0.0)
    return float(max(0.0, power))


def dynamic_calibrated_total_power_w(torque, transition=None):
    """Return power from the dynamic 50 Hz calibration model.

    This applies the existing aggregate calibrated power model, then adds the
    final interval-level 50 Hz trace residual. The offsets were fitted against
    real mean interval energy for all 16 plotted command intervals while using
    the dynamic actuator gains ``frame kp=30, frame kv=30/11.5`` and
    ``leg kp=10, leg kv=10/11.5``.
    """
    power = calibrated_total_power_w(torque, transition=transition)
    if transition:
        power += DYNAMIC_INTERVAL_POWER_OFFSET_W.get(str(transition), 0.0)
    return float(max(0.0, power))


def power_to_current_vector(power_w, voltage=12.0, weights=None, motor_count=17):
    """Distribute aggregate power into a current vector for JSON log replay."""
    voltage = float(voltage)
    if voltage <= 0:
        raise ValueError("voltage must be positive.")
    motor_count = int(motor_count)
    if motor_count <= 0:
        raise ValueError("motor_count must be positive.")

    if weights is None:
        weights = np.ones(motor_count, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != motor_count:
            raise ValueError(
                f"weights must have {motor_count} values, got {weights.shape[0]}."
            )

    weights = np.abs(weights)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        weights = np.ones(motor_count, dtype=np.float64)
        weight_sum = float(motor_count)
    return (float(power_w) / voltage) * (weights / weight_sum)


def frame_energy_j(torque, fps=50.0, mode="calibrated", transition=None):
    """Return one frame of electrical energy in joules."""
    if mode == "legacy":
        power_w = legacy_total_power_w(torque)
    elif mode == "calibrated":
        power_w = calibrated_total_power_w(torque, transition=transition)
    elif mode == "calibrated_no_residual":
        power_w = calibrated_total_power_w(
            torque, transition=transition, apply_residual=False
        )
    elif mode == "dynamic_calibrated":
        power_w = dynamic_calibrated_total_power_w(torque, transition=transition)
    else:
        raise ValueError(f"Unknown energy mode {mode!r}.")
    return float(power_w / float(fps))
