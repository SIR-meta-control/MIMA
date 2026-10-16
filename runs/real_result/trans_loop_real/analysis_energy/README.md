# Historical First 15 Interval Energy Analysis

[🏠 MIMA](../../../../README.md) · [📚 Documentation](../../../../docs/README.md)

> [!NOTE]
> This earlier analysis ends at the final command timestamp. The current
> 16-interval comparison includes the final 2.80 s observation window and is
> reported in [`../calibration_report.md`](../calibration_report.md).

- Trials: 10 (latest directory for each experiment number 01–10)
- Analysis window: first to last `/crimson_control/transform` timestamp
- Total electrical power: `P(t) = sum(U_i * abs(I_i))` for motors 0–16
- Energy: trapezoidal integration of total electrical power
- Curve alignment: normalized cycle time, displayed using the mean duration
- Shading: sample standard deviation (`ddof=1`)

<a id="main-results"></a>

## 📊 Main results

- Transformation-loop duration: **46.201 ± 0.033 s**
- Electrical energy: **3.6356 ± 0.0237 kJ**
- Electrical energy: **1.0099 ± 0.0066 Wh**
- Mean electrical power: **78.69 ± 0.52 W**

Values are mean ± sample standard deviation across the 10 trials.
