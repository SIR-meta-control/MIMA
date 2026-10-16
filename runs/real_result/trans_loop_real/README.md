# Dynamic 50 Hz Energy Calibration

[🏠 MIMA](../../../README.md) · [📚 Documentation](../../../docs/README.md)

This directory contains the analysis code and calibration record for the
MuJoCo-to-real transformation-loop energy comparison.

<a id="included"></a>

## 📦 Included

- `calibration_report.md`: calibration procedure and final metrics;
- `plot_real_sim_energy_comparison.py`: calibrated 50 Hz trace comparison;
- interval-residual, phase-aware fitting, response-scan, and compatibility
  analysis scripts; and
- compact fitted-feature and parameter-scan tables used by the diagnostics.

<a id="data-inputs"></a>

## 📥 Data inputs

Exported real and simulated motor measurements are provided in
`data/Supplementary Method 6.3.zip`. The comparison script uses per-trial
Dynamixel CSV logs and a MuJoCo replay JSON, specified by:

```bash
python3 runs/real_result/trans_loop_real/plot_real_sim_energy_comparison.py \
  --real-csv-dir /path/to/real/csv \
  --sim-json runs/mujoco_experiment_energy_dynamic_calibrated_full10_50hz.json
```

> [!IMPORTANT]
> The simulation JSON must be generated at 50 Hz with `dynamic_calibrated` mode
> and must contain MuJoCo `qpos` in each `/dynamixel_control/log` message. Older
> JSON files without `qpos` cannot be used for the final phase-aware comparison.

<a id="scope"></a>

## 🎯 Scope

> [!NOTE]
> The interval offsets were fitted to this transformation-loop dataset.
> Error metrics describe agreement on the calibration data.
