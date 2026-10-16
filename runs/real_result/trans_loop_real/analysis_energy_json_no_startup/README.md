# Historical JSON Transformation-Loop Energy Analysis

[🏠 MIMA](../../../../README.md) · [📚 Documentation](../../../../docs/README.md)

> [!NOTE]
> This directory preserves the earlier analysis of low-frequency logs. For the
> 50 Hz, 16-interval results used in the current manuscript, see
> [`../calibration_report.md`](../calibration_report.md).

<a id="data-and-methods"></a>

## 🗂️ Data and methods

- Source file: `mujoco_experiment_energy_calibrated_full10.json`
- Number of runs: 10
- Transformation sequence: μ1→μ2→…→μ9→μ8→…→μ1
- Main analysis window: from the first `/transform` command to the last `/transform` command, consistent with the example figure
- Total electrical power: `P(t) = Σ U_i × |I_i|`, summed over 17 motors
- Cumulative energy: trapezoidal integration of power over time
- Curve alignment: normalize each cycle to 0–1, then map it to the mean cycle duration
- Shaded band: sample standard deviation across 10 runs (mean ± 1 SD, `ddof=1`)
- Startup-transient handling: no anomalous startup samples requiring replacement were detected

<a id="main-results-from-json-voltage-current-logs-after-startup-transient-removal"></a>

## 📊 Main results (from JSON voltage/current logs after startup-transient removal)

- Transformation-cycle duration: **46.460 ± 0.000 s**
- Transformation-cycle energy: **3.7910 ± 0.0000 kJ**
- Transformation-cycle energy: **1.0531 ± 0.0000 Wh**
- Mean cycle power: **81.60 ± 0.00 W**

All values are the mean ± sample standard deviation across 10 runs.

<a id="json-segment-energy-cross-check"></a>

## 📋 JSON segment-energy cross-check

The JSON also provides `transition_energy_j` and `settle_energy_j`, calculated
at a higher temporal resolution when the logs were generated. The segment
energy from the first transformation to completion of the final μ1
transformation, excluding the final settling period, is
**3.7916 ± 0.0000 kJ**. Including the final settling period gives
**4.0057 ± 0.0000 kJ**.

This value is slightly higher than the trapezoidal integral of the 3 Hz
voltage/current logs because brief transformation-current peaks lasting
approximately 0.1 s may fall between log samples. The main figure therefore
represents the recorded voltage/current traces, while the segment fields
provide a cross-check of the full energy contribution of short transients.

<a id="data-quality-notes"></a>

## 🗂️ Data quality notes

No anomalous startup samples were detected in the source JSON. Statistics for
all runs use the original voltage/current logs.
