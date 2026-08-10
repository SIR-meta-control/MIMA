import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ros_mujoco_utils.energy_calibration import (
    DYNAMIC_INTERVAL_POWER_OFFSET_W,
    calibrated_total_power_w,
    dynamic_calibrated_total_power_w,
    frame_energy_j,
    legacy_torque_to_current,
    power_to_current_vector,
)


class EnergyCalibrationTest(unittest.TestCase):
    def test_legacy_torque_to_current_preserves_optimizer_polynomial(self):
        current = legacy_torque_to_current(np.array([0.0, 1.0], dtype=float))

        self.assertAlmostEqual(float(current[0]), 0.0357777777778)
        self.assertGreater(float(current[1]), float(current[0]))

    def test_calibrated_power_uses_base_and_leg_torque(self):
        torque = np.zeros(17, dtype=float)
        torque[5:] = 1.0

        power = calibrated_total_power_w(torque, transition=None, apply_residual=False)

        self.assertAlmostEqual(power, 61.2116943 + 12.0 * 1.3421639, places=5)

    def test_transition_residual_corrects_mu6_to_mu7_hold_power(self):
        torque = np.zeros(17, dtype=float)
        torque[5:] = 1.0

        uncorrected = calibrated_total_power_w(
            torque, transition="mu6->mu7", apply_residual=False
        )
        corrected = calibrated_total_power_w(
            torque, transition="mu6->mu7", apply_residual=True
        )

        self.assertAlmostEqual(corrected - uncorrected, 41.3410958, places=5)

    def test_dynamic_offsets_cover_all_command_intervals(self):
        self.assertEqual(len(DYNAMIC_INTERVAL_POWER_OFFSET_W), 16)
        self.assertIn("mu2->mu1", DYNAMIC_INTERVAL_POWER_OFFSET_W)
        self.assertAlmostEqual(
            DYNAMIC_INTERVAL_POWER_OFFSET_W["mu2->mu1"],
            -17.483995129104606,
            places=9,
        )

    def test_dynamic_calibration_adds_final_50hz_interval_offset(self):
        torque = np.zeros(17, dtype=float)
        torque[5:] = 1.0

        first_pass = calibrated_total_power_w(torque, transition="mu2->mu1")
        dynamic = dynamic_calibrated_total_power_w(torque, transition="mu2->mu1")

        self.assertAlmostEqual(
            dynamic - first_pass,
            DYNAMIC_INTERVAL_POWER_OFFSET_W["mu2->mu1"],
            places=9,
        )

    def test_frame_energy_supports_dynamic_calibrated_mode(self):
        torque = np.zeros(17, dtype=float)
        torque[5:] = 1.0

        energy = frame_energy_j(
            torque,
            fps=50.0,
            mode="dynamic_calibrated",
            transition="mu1->mu2",
        )
        power = dynamic_calibrated_total_power_w(torque, transition="mu1->mu2")

        self.assertAlmostEqual(energy, power / 50.0, places=9)

    def test_power_to_current_vector_preserves_total_power(self):
        current = power_to_current_vector(
            120.0,
            voltage=12.0,
            weights=np.array([0.0, 1.0, 3.0], dtype=float),
            motor_count=3,
        )

        self.assertAlmostEqual(float(np.sum(12.0 * np.abs(current))), 120.0)
        self.assertAlmostEqual(float(current[0]), 0.0)
        self.assertGreater(float(current[2]), float(current[1]))


if __name__ == "__main__":
    unittest.main()
