# Dynamic 50 Hz Energy Calibration

This directory contains the analysis code and calibration record for the
MuJoCo-to-real transformation-loop energy comparison.

## Included

- `calibration_report.md`: calibration procedure, final metrics, and known
  limitations;
- `plot_real_sim_energy_comparison.py`: calibrated 50 Hz trace comparison;
- interval-residual, phase-aware fitting, response-scan, and compatibility
  analysis scripts; and
- compact fitted-feature and parameter-scan tables used by the diagnostics.

## External inputs

The raw Dynamixel CSV logs, generated MuJoCo replay JSON, rosbag files, and
generated plots are intentionally excluded from the code repository. The
comparison script accepts explicit input paths:

```bash
python3 runs/real_result/trans_loop_real/plot_real_sim_energy_comparison.py \
  --real-csv-dir /path/to/real/csv \
  --sim-json runs/mujoco_experiment_energy_dynamic_calibrated_full10_50hz.json
```

The simulation JSON must be generated at 50 Hz with `dynamic_calibrated` mode
and must contain MuJoCo `qpos` in each `/dynamixel_control/log` message. Older
JSON files without `qpos` cannot be used for the final phase-aware comparison.

## Scope

The interval offsets were fitted for this transformation-loop dataset. They
should be described as a dataset-specific calibration layer rather than a
universal actuator-energy model.
