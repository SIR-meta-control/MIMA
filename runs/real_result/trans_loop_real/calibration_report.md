# Sim-to-real energy calibration report

This report records the calibration used to align the Crimson MuJoCo
transformation-loop replay with the real robot Dynamixel voltage/current logs.
It is intentionally tied to the `trans_loop_real` dataset and should be treated
as an experiment calibration record, not as a universal actuator model.

## 0. Dynamic 50 Hz calibration

The dynamic 50 Hz calibration supersedes the older first-15-interval numbers
that are retained later in this report as historical evidence. The final
analysis uses the exported 50 Hz voltage/current trace, includes all 16 command
intervals, uses a fixed `2.80 s` post-command observation window for the final
`mu2->mu1` interval, and separates transformation and sustainment phases by
joint-position convergence. Real phase timing uses Dynamixel `P[]`; simulated
phase timing uses MuJoCo `qpos` exported from `data.qpos[7:]`.

The final dynamic actuator setting is:

```text
frame actuators: kp=30, kv=30/11.5 = 2.6087
leg actuators:   kp=10, kv=10/11.5 = 0.869565
```

The final replay power model is:

```text
P_dynamic(t,k) =
  max(0,
      61.2116943
    + 1.3421639 * sum_{j in L} |tau_j(t)|
    + r_15(k)
    + delta_16(k))
```

where `r_15(k)` is the first-pass transition residual table retained in
`TRANSITION_POWER_RESIDUAL_W`, and `delta_16(k)` is the final 50 Hz interval
offset table retained in `DYNAMIC_INTERVAL_POWER_OFFSET_W`. The second table
contains all 16 intervals:

| Interval | Final 50 Hz power offset (W) |
|---|---:|
| `mu1 -> mu2` | 3.259 |
| `mu2 -> mu3` | 2.285 |
| `mu3 -> mu4` | 0.101 |
| `mu4 -> mu5` | 2.614 |
| `mu5 -> mu6` | 2.096 |
| `mu6 -> mu7` | -0.712 |
| `mu7 -> mu8` | 1.228 |
| `mu8 -> mu9` | 2.645 |
| `mu9 -> mu8` | 1.888 |
| `mu8 -> mu7` | 2.925 |
| `mu7 -> mu6` | 1.080 |
| `mu6 -> mu5` | -0.884 |
| `mu5 -> mu4` | -0.932 |
| `mu4 -> mu3` | -1.246 |
| `mu3 -> mu2` | 2.605 |
| `mu2 -> mu1` | -17.484 |

Calibration-set comparison metrics, computed from trace-integrated 50 Hz
voltage/current logs, are:

| Metric | Real robot | Dynamic-calibrated 50 Hz MuJoCo replay |
|---|---:|---:|
| Total energy, 16 intervals (kJ) | 3.776 +/- 0.026 | 3.776 +/- 0.006 |
| Transformation energy (kJ) | 1.681 +/- 0.021 | 1.629 +/- 0.008 |
| Sustainment energy (kJ) | 2.095 +/- 0.025 | 2.147 +/- 0.010 |
| Mean transformation duration (s) | 1.371 | 1.332 |
| Transformation-duration MAE (s) | - | 0.168 |
| Interval-power MAE / RMSE (W) | - | 0.555 / 1.827 |
| Interval-power bias (W) | - | -0.357 |
| Interval-power correlation | - | 0.995 |

Energy values are reported as the mean plus or minus the sample standard
deviation across ten real-robot trials and ten MuJoCo replay records,
respectively. The sample standard deviation is calculated with `ddof=1`.
Interval-power error metrics compare trial-averaged interval mean powers on
this calibration dataset; they are not held-out validation metrics.

For compatibility with the old first-15-interval scope, the same final 50 Hz
pipeline gives `3.636 +/- 0.024 kJ` real versus `3.635 +/- 0.006 kJ` sim,
with interval-power `MAE=0.587 W`, `RMSE=1.887 W`, and `r=0.995`.

### 0.1 Method description

**Supplementary Note X. Dynamic sim-to-real calibration of transformation-loop
energy.** We calibrated the simulated electrical energy of the metamorphic
transformation loop by replaying the same 17-motor command sequence in MuJoCo
and comparing the exported voltage/current trace with the Dynamixel logs from
ten repeated real-robot trials. For motor \(j\), the Dynamixel command tick
\(c_j\) was converted to a MuJoCo target angle as
\[
\theta_j^{\mathrm{cmd}}
=
s_j\left(\frac{180}{2048}c_j-b_j\right)\frac{\pi}{180},
\]
where \(s_j\in\{-1,1\}\) is the motor-direction sign and \(b_j\) is the
encoder bias in degrees. The reordered 17-dimensional command vector was then
applied to the MuJoCo position actuators. In the current MJCF implementation,
these position actuators act as proportional-derivative servos,
\[
\tau_j^{\mathrm{sim}}(t)
=
k_{p,j}\left(q_j^{\mathrm{cmd}}(t)-q_j(t)\right)
-
k_{v,j}\dot q_j(t),
\]
and the resulting actuator forces are propagated through MuJoCo constrained
multibody dynamics.

Real electrical power was computed directly from the logged motor voltage and
current,
\[
P_r^{\mathrm{real}}(t)
=
\sum_{j=1}^{17} U_{r,j}(t)\left|I_{r,j}(t)\right|,
\]
and interval energy was evaluated by trapezoidal integration over each command
window. The final analysis used the same trace-integration procedure for the
simulated 50 Hz voltage/current log. All 16 command intervals were included;
because the final \(\mu_2\rightarrow\mu_1\) command has no subsequent command
timestamp, it was evaluated over a fixed 2.80 s post-command window that is
covered by both the real logs and the simulation.

Dynamic response was calibrated by scanning MuJoCo position-servo gains and
using joint-position convergence to define the transformation phase. The final
setting used \(k_p=30, k_v=30/11.5=2.6087\) for the frame actuators and
\(k_p=10, k_v=10/11.5=0.869565\) for the leg actuators. On this dynamic replay,
we fitted one additive power residual \(\delta_k\) per command interval,
\[
\delta_k
=
\frac{
  \bar E_k^{\mathrm{real}}
  -
  \bar E_k^{\mathrm{sim,base}}
}{
  \Delta T_k^{\mathrm{sim}}
},
\quad
P^{\mathrm{dynamic}}(t,k)
=
\max\left(0,\,
P^{\mathrm{base}}(t,k)+\delta_k
\right).
\]
This correction aligns the mean trace-integrated interval energy while
preserving the within-interval simulated power waveform. Calibration error was
computed from interval-mean power,
\[
e_k=\bar P_k^{\mathrm{sim}}-\bar P_k^{\mathrm{real}},\quad
\mathrm{MAE}=\frac{1}{K}\sum_{k=1}^{K}|e_k|,\quad
\mathrm{RMSE}=\sqrt{\frac{1}{K}\sum_{k=1}^{K}e_k^2}.
\]

Across the 16 analyzed intervals, the real robot consumed
\(3.776\pm0.026\) kJ and the dynamic-calibrated MuJoCo replay consumed
\(3.776\pm0.006\) kJ. Transformation-phase energy was
\(1.681\pm0.021\) kJ in the real robot and \(1.629\pm0.008\) kJ in simulation,
whereas sustainment-phase energy was \(2.095\pm0.025\) kJ and
\(2.147\pm0.010\) kJ, respectively. Mean transformation duration was
1.371 s in hardware and 1.332 s in simulation, with a transformation-duration
MAE of 0.168 s. The interval-power comparison yielded MAE \(=0.555\) W,
RMSE \(=1.827\) W, bias \(=-0.357\) W, and correlation \(r=0.995\). These
results support the MuJoCo energy estimate as a dataset-specific calibrated
comparative signal for the transformation-loop optimizer, rather than as a
universal actuator-energy model.

**中文对应表述。** 我们通过在 MuJoCo 中回放与实物机器人完全一致的 17 电机指令序列，
并将导出的 50 Hz 电压/电流轨迹与 10 次实物 Dynamixel 日志进行比较，对变形循环
能耗进行了动态仿真-实物标定。实物功率由
\(P_r^{\mathrm{real}}(t)=\sum_{j=1}^{17}U_{r,j}(t)|I_{r,j}(t)|\) 直接计算，
区间能量通过梯形积分得到。最终分析纳入全部 16 个 command intervals；最后的
\(\mu_2\rightarrow\mu_1\) 区间没有后续命令作为截断点，因此使用实物和仿真均覆盖的
固定 2.80 s post-command window。相位划分不再依赖功率曲线形状，而是由关节位置
收敛确定：实物使用 Dynamixel `P[]`，仿真使用 MuJoCo `qpos`。

最终动态参数为机身执行器 \(k_p=30,k_v=2.6087\)，腿部执行器
\(k_p=10,k_v=0.869565\)。在该动态 replay 基础上，我们为每个 command interval
拟合一个常数功率残差 \(\delta_k\)，使 trace-integrated interval energy 的均值与实物
均值对齐，同时保留 interval 内的仿真功率波形。最终 16 个区间上，实物总能耗为
\(3.776\pm0.026\) kJ，动态标定 MuJoCo replay 为 \(3.776\pm0.006\) kJ；变形阶段能耗
为 \(1.681\pm0.021\) kJ 与 \(1.629\pm0.008\) kJ，维持阶段能耗为
\(2.095\pm0.025\) kJ 与 \(2.147\pm0.010\) kJ。平均变形时间分别为 1.371 s 和
1.332 s，变形时长 MAE 为 0.168 s；区间平均功率误差为 MAE \(=0.555\) W、
RMSE \(=1.827\) W、bias \(=-0.357\) W，相关系数 \(r=0.995\)。因此，该结果可作为当前
变形循环优化器能耗估计的 dataset-specific calibrated comparative signal，而不应表述为
通用执行器能耗模型。

## 1. Scope

The calibrated workflow covers the repeated transformation loop:

```text
mu1 -> mu2 -> mu3 -> mu4 -> mu5 -> mu6 -> mu7 -> mu8 -> mu9
mu9 -> mu8 -> mu7 -> mu6 -> mu5 -> mu4 -> mu3 -> mu2 -> mu1
```

The current metric uses all 16 command intervals. Earlier sections and
tables below that mention the first 15 intervals describe the historical
first-pass calibration retained for traceability.

## 2. Data and artifacts

- Real reference logs:
  `runs/real_result/trans_loop_real/csv`
- Real CSV analysis outputs:
  `runs/real_result/trans_loop_real/analysis_energy`
- Legacy MuJoCo full replay input:
  `runs/mujoco_experiment_energy_full10.json`
- Final dynamic calibrated MuJoCo replay output:
  `runs/mujoco_experiment_energy_dynamic_calibrated_full10_50hz.json`
- Calibrated JSON analysis outputs:
  `runs/real_result/trans_loop_real/analysis_energy_json_no_startup`
- Final 50 Hz dynamic comparison outputs:
  `runs/real_result/trans_loop_real/dynamic_energy_comparison`
- Model-variant scan outputs:
  `runs/real_result/trans_loop_real/model_variant_scan`
- Fitted power-model outputs:
  `runs/real_result/trans_loop_real/energy_calibration`

The real CSV analysis selected the newest run directory for each experiment
number `01..10`. Real electrical power was computed from the logged voltage and
current columns:

```text
P(t) = sum(U_i(t) * abs(I_i(t))) for motors 0..16
```

Energy was then integrated by trapezoidal integration over the transform-loop
time window. The historical first-15 summary is:

```text
duration = 46.201 +/- 0.033 s
energy   = 3.6356 +/- 0.0237 kJ
energy   = 1.0099 +/- 0.0066 Wh
power    = 78.69 +/- 0.52 W
```

The final all-16 dynamic 50 Hz trace summary is:

```text
duration = 49.001 +/- 0.033 s real, 49.260 s sim
energy   = 3.776 +/- 0.026 kJ real, 3.776 +/- 0.006 kJ sim
power    = 77.061 +/- 0.523 W real, 76.656 +/- 0.124 W sim
```

The final total-loop trace-integrated energy is aligned to within rounding.
The calibrated JSON replay is deterministic in command timing; the reported
MuJoCo energy SD comes from the exported voltage/current trace processing.

## 3. Mathematical formulation

This section states the calibration process in a form suitable for a paper
methods section. Let \(j \in \{1,\ldots,17\}\) index the actuators. The first
five actuators are frame/body actuators, denoted by set \(\mathcal{F}\), and
the remaining twelve actuators are leg actuators, denoted by set
\(\mathcal{L}\). Let \(r\) index a repeated real or simulated trial, and let
\(k\) index the transformation interval.

### 3.1 Command replay and interval definition

Each transformation interval starts at the ROS marker timestamp
\(T_{r,k}^{\mathrm{start}}\) and ends at the next transform marker
\(T_{r,k}^{\mathrm{end}}\). The interval duration is:

```text
Delta T_{r,k} = T_{r,k}^{end} - T_{r,k}^{start}.
```

In the replay experiment, the same `sync_write` commands produced by the real
experiment are converted from Dynamixel encoder ticks to MuJoCo joint targets.
For motor \(j\), the real command tick \(c_j\) is converted to a target angle:

```text
theta_j^cmd = s_j * (c_j * 180 / 2048 - b_j) * pi / 180,
```

where \(s_j \in \{-1,1\}\) is the sign entry from the conversion matrix and
\(b_j\) is the calibrated encoder bias in degrees. The command vector is then
reordered from the real motor order to the MuJoCo actuator order using the
same `RULES` mapping used by `ros_mujoco`.

### 3.2 MuJoCo position actuator and simulated torque

Each MuJoCo position actuator is a PD servo around the commanded joint target.
Although this is sometimes described informally as PID control, the current
MJCF position actuator has no integral term; it uses proportional tracking
and velocity damping.
For a unit-gear hinge actuator, the actuator force logged by MuJoCo can be
written conceptually as:

```text
tau_j^{sim}(t) = k_{p,j} * (q_j^cmd(t) - q_j(t)) - k_{v,j} * qdot_j(t),
```

where \(q_j(t)\) and \(\dot{q}_j(t)\) are the simulated joint position and
velocity. MuJoCo then advances the constrained multibody dynamics:

```text
M(q) qddot + h(q, qdot) = S^T tau^{sim} + J_c(q)^T lambda_c,
```

where \(M(q)\) is the mass matrix, \(h(q,\dot q)\) collects gravity, Coriolis,
and passive terms, \(S\) maps actuator forces to generalized coordinates, and
\(J_c^T \lambda_c\) represents contact and constraint forces. In the scripts,
\(\tau_j^{sim}\) is read from `data.actuator_force` or
`env.get_joint_torque()` after each MuJoCo step.

The final dynamic XML setting is:

```text
k_p = 30, k_v = 30 / 11.5 = 2.6087   for frame actuators j in F
k_p = 10, k_v = 10 / 11.5 = 0.869565 for leg actuators   j in L
```

This gain change is not itself an energy model. It changes the simulated
tracking torque \(\tau^{sim}\), which is then mapped to electrical power.

### 3.3 Legacy torque-to-current energy model

The legacy model follows the optimizer's empirical torque-to-current
polynomial. For each simulated actuator torque \(\tau_j^{sim}\), current is:

```text
I_j^{legacy}(t) =
  0.000130565974 * tau_j(t)^4
- 0.001881393510 * tau_j(t)^3
+ 0.021677122600 * tau_j(t)^2
+ 0.410017411000 * tau_j(t)
+ 0.0357777777778.
```

Given a nominal voltage \(V=12\) V, the legacy total electrical power is:

```text
P^{legacy}(t) = sum_j V * |I_j^{legacy}(t)|.
```

For replay frame rate \(f_s=50\) Hz, the per-frame energy increment is:

```text
Delta E^{legacy}(t_n) = P^{legacy}(t_n) / f_s.
```

The segment energy is accumulated over both the short interpolation phase and
the post-transform settle phase:

```text
E_{r,k}^{legacy} = sum_{n in segment k} P^{legacy}(t_n) / f_s.
```

This legacy model was useful as an initial actuator-effort proxy, but it
overestimated the real transform-loop interval power.

### 3.4 Real electrical power and energy integration

For the real robot, power is computed directly from the logged Dynamixel
voltage/current arrays. At each log sample \(t_n\):

```text
P_r^{real}(t_n) = sum_{j=1}^{17} U_{r,j}(t_n) * |I_{r,j}(t_n)|.
```

The real interval energy is the continuous-time integral over the interval:

```text
E_{r,k}^{real} = integral_{T_{r,k}^{start}}^{T_{r,k}^{end}}
                 P_r^{real}(t) dt.
```

The implementation evaluates this integral with trapezoidal integration and
interpolated boundary samples:

```text
E_{r,k}^{real} approx
sum_{n=0}^{N_k-1}
  0.5 * (P_r^{real}(t_n) + P_r^{real}(t_{n+1}))
      * (t_{n+1} - t_n).
```

The corresponding interval mean power is:

```text
pbar_{r,k}^{real} = E_{r,k}^{real} / Delta T_{r,k}.
```

The real reference used for fitting is the trial average:

```text
y_k = pbar_k^{real} = (1 / R) * sum_{r=1}^R pbar_{r,k}^{real},
```

where \(R=10\) for the current transform-loop dataset.

### 3.5 Calibrated aggregate power model

After scanning actuator gains, the calibrated model uses MuJoCo torque
features from the selected XML. For each interval, the frame and leg torque
features are:

```text
x_{F,k} = mean_{t in k} sum_{j in F} |tau_j^{sim}(t)|,
x_{L,k} = mean_{t in k} sum_{j in L} |tau_j^{sim}(t)|.
```

The fitted non-residual model is a nonnegative least-squares regression:

```text
min_beta sum_k (x_k^T beta - y_k)^2
subject to beta >= 0,
```

with:

```text
x_k = [17, x_{F,k}, x_{L,k}].
```

The fitted coefficient on \(x_{F,k}\) was zero, so the base calibrated power
model reduces to:

```text
P^{base}(t) = beta_0 + beta_L * sum_{j in L} |tau_j^{sim}(t)|,
```

with:

```text
beta_0 = 61.2116943 W,
beta_L = 1.3421639 W/Nm.
```

The final dataset-specific residual is defined per transition:

```text
delta_k = y_k - pbar_k^{base}.
```

During calibrated replay, the instantaneous power for transition \(k\) is:

```text
P^{cal}(t, k) =
max(0, 61.2116943
        + 1.3421639 * sum_{j in L} |tau_j^{sim}(t)|
        + delta_k).
```

The mode `calibrated_no_residual` uses the same equation with
\(\delta_k=0\). The mode `calibrated` uses the explicit residual table in
`TRANSITION_POWER_RESIDUAL_W`.

### 3.6 Simulated energy after calibration

For a simulated interval, calibrated segment energy is:

```text
E_{r,k}^{cal} = sum_{n in segment k} P^{cal}(t_n, k) / f_s.
```

The simulated interval mean power used in the error table is:

```text
pbar_{r,k}^{cal} = E_{r,k}^{cal} / Delta T_k,
```

where \(\Delta T_k\) is the planned interval duration. In the final 50 Hz
workflow, all 16 command intervals are integrated from the exported U/I trace;
the final interval uses a fixed 2.80 s post-command window.

For JSON compatibility, the calibrated aggregate power is converted back to a
synthetic current vector. The torque magnitude defines the distribution
weights:

```text
w_j(t) = |tau_j^{sim}(t)| / sum_l |tau_l^{sim}(t)|.
```

The synthetic current is:

```text
I_j^{cal}(t) = (P^{cal}(t,k) / V) * w_j(t).
```

Therefore the logged JSON power is conserved by construction:

```text
sum_j V * |I_j^{cal}(t)| = P^{cal}(t,k).
```

If all torque weights are zero, the implementation uses uniform weights to
avoid division by zero.

### 3.7 Error metrics

For each evaluated interval \(k\), define the power error:

```text
e_k = pbar_k^{sim} - pbar_k^{real}.
```

The reported metrics are:

```text
MAE  = (1 / K) * sum_k |e_k|,
RMSE = sqrt((1 / K) * sum_k e_k^2),
bias = (1 / K) * sum_k e_k,
```

with \(K=16\) intervals for the final dynamic 50 Hz comparison. The historical
first-pass comparison used \(K=15\). Correlation is computed as:

```text
corr =
sum_k (pbar_k^{sim} - mean(pbar^{sim}))
      (pbar_k^{real} - mean(pbar^{real}))
/
sqrt(
  sum_k (pbar_k^{sim} - mean(pbar^{sim}))^2
  sum_k (pbar_k^{real} - mean(pbar^{real}))^2
).
```

The total-loop relative energy error is:

```text
epsilon_E =
100 * (E_loop^{sim} - E_loop^{real}) / E_loop^{real}.
```

For the final dynamic calibrated 50 Hz U/I integration, the all-16 total
energy is aligned to the real trace within rounding:

```text
epsilon_E approx 100 * (3.776 - 3.776) / 3.776 approx 0.0%.
```

### 3.8 Concise method summary

In paper form, the method can be summarized as follows. The real robot
electrical energy was computed by integrating the measured total motor power
\(\sum_j U_j |I_j|\) over each transformation interval. The MuJoCo replay used
the same command sequence and converted each Dynamixel encoder command into a
MuJoCo joint target. Position actuators generated simulated torques through a
PD law, and these torques were first evaluated with the legacy empirical
torque-current polynomial. Because the legacy model overestimated the real
settle-state power, the leg actuator gains were reduced and an aggregate
sim-to-real power model was fitted from the absolute simulated leg torque. A
transition-specific residual was then added to account for frame-motor loads
observed in the real current traces but not represented by the ideal MuJoCo
mechanism. Calibration quality was evaluated by MAE, RMSE, signed bias, and
Pearson correlation between simulated and real interval mean powers.

### 3.9 Relationship to the full method description

Section 0.1 contains the current English and Chinese method descriptions for
the dynamic 50 Hz calibration. Earlier first-pass wording is not repeated here
to avoid reuse of superseded 15-interval values.

## 4. Baseline mismatch

The legacy replay path used:

- `src/models/crimson/mjcf/crimson_stand_legInit_forSimOnly.xml` leg position
  actuators with `kp=30, kv=5`.
- The historical torque-to-current polynomial from the optimizer path.
- `sum(abs(I_i * U_i))` as total electrical power.

Against the real interval power, the legacy baseline had:

```text
MAE  = 41.12 W
RMSE = 51.42 W
bias = +32.17 W
corr = 0.197
```

The positive bias means legacy simulation generally overestimated the real
interval power. Inspection showed that most of the error came from the 3 s
hold/settle portion after each transform command. The 0.1 s interpolation
command itself was too short to explain the total mismatch.

## 5. Hypotheses checked

### 5.1 Floor friction

Floor friction values from `0.05` to `3.0` were scanned. They did not provide a
useful reduction in interval RMSE:

```text
variant              MAE W   RMSE W   bias W
floor_friction_0.05  41.12   51.42    +32.17
floor_friction_0.2   41.12   51.42    +32.17
floor_friction_1.5   44.10   53.91    +35.66
floor_friction_3.0   41.56   51.51    +34.07
```

This ruled out floor friction as the primary correction.

### 5.2 Contact and gravity toggles

Contact-disabled and gravity-disabled variants made the replay less useful:

```text
variant           MAE W   RMSE W   bias W
gravity_disabled  50.76   56.68    -40.22
contact_disabled  67.80   70.06    -67.80
```

The sign and magnitude of these errors show that removing major physical
effects does not explain the real current profile.

### 5.3 First-pass actuator gain variants

The strongest XML-only improvement came from lowering leg actuator stiffness
while leaving frame actuators at the original gain:

```text
variant                 MAE W   RMSE W   bias W   corr
framekp_30_legkp_10     17.42   23.61    +1.49    0.297
allkp_10                16.81   23.69    -3.84    0.144
framekp_20_legkp_10     17.12   23.69    +0.01    0.230
framekp_40_legkp_10     18.37   24.24    +3.21    0.302
framekp_30_legkp_20     30.72   38.33    +19.45   0.234
baseline                41.12   51.42    +32.17   0.197
```

The selected first-pass XML setting was:

```text
frame actuators: kp=30, kv=5
leg actuators:   kp=10, kv=1.66667
```

The XML-only change removed most of the global bias, but the remaining
per-transition pattern still did not match the real current logs. This setting
was superseded by the final dynamic `kv_divisor=11.5` scan summarized in
Section 0.

## 6. First-pass calibrated power model

After fixing the MJCF leg gain, several aggregate power models were fitted
using real interval power and MuJoCo actuator torque features. The best
non-residual model separated frame and leg absolute torque:

```text
model                         columns                                  MAE W   RMSE W
constant_frame_leg_abs_tau    n_act,sum_frame_abs_tau,sum_leg_abs_tau   12.60   16.04
constant_plus_abs_tau         n_act,sum_abs_tau                         13.46   17.00
poly_abs_tau_1_to_4           n_act,sum_abs_tau,...                     12.99   17.31
```

Because the fitted frame torque coefficient was zero, the base model became:

```text
P = 61.2116943 + 1.3421639 * sum(abs(tau_leg))
```

This base model captured the average hold power better than the legacy
torque-to-current polynomial, but it still missed transition-specific real
frame-motor loads. The real logs show elevated current around the unfolding
path through `mu6->mu7`, `mu7->mu8`, and `mu8->mu9` while the ideal MuJoCo
mechanism produces low frame torque/contact load for those states.

The first-pass calibrated model therefore added an explicit per-transition
residual:

```text
P = 61.2116943 + 1.3421639 * sum(abs(tau_leg)) + transition_residual
```

The residual table is implemented in
`src/ros_mujoco/scripts/ros_mujoco_utils/energy_calibration.py` as
`TRANSITION_POWER_RESIDUAL_W`. The dynamic mode keeps this table
as the base residual and adds `DYNAMIC_INTERVAL_POWER_OFFSET_W` for the 16
interval 50 Hz calibration-set comparison.

## 7. Code changes

The calibration is represented by these code and model changes:

- `src/models/crimson/mjcf/crimson_stand_legInit_forSimOnly.xml`
  uses the final dynamic gains: frame actuators `kp=30, kv=2.6087` and leg
  actuators `kp=10, kv=0.869565`.
- `src/ros_mujoco/scripts/ros_mujoco_utils/energy_calibration.py`
  provides the legacy polynomial, the first-pass aggregate power model, the
  first-pass residual table, the final 16-interval dynamic offset table, and
  JSON-current reconstruction helpers.
- `src/ros_mujoco/scripts/mujoco_experiment_energy.py`
  supports `--energy-mode legacy|calibrated|calibrated_no_residual|dynamic_calibrated`
  and exports `qpos` in each `/dynamixel_control/log` record for phase timing.
- `src/ros_mujoco/tests/test_energy_calibration.py`
  covers the legacy polynomial, calibrated model constants, residual behavior,
  final dynamic interval offsets, and aggregate-power-to-current conservation.
- `runs/real_result/trans_loop_real/plot_real_sim_energy_comparison.py`
  compares the calibrated 50 Hz real/sim traces and writes the figure summaries.
- `runs/real_result/trans_loop_real/apply_interval_residual_calibration.py`,
  `scan_mujoco_dynamic_response.py`, `fit_phase_aware_calibration.py`, and
  `compute_50hz_calibration_metrics.py` preserve the dynamic 50 Hz calibration
  and its cross-check workflow inside this repository.

The default mode remains `legacy` for backward compatibility. Use
`--energy-mode dynamic_calibrated --log-rate 50` when generating the final
calibration replay for this real transformation-loop dataset.

## 8. Reproduction commands

Generate a legacy 10-run replay:

```bash
python3 src/ros_mujoco/scripts/mujoco_experiment_energy.py \
  --num-experiments 10 \
  --output runs/mujoco_experiment_energy_full10.json
```

Scan model variants:

```bash
python3 runs/real_result/trans_loop_real/scan_mujoco_model_variants.py
```

Fit the aggregate power model:

```bash
python3 runs/real_result/trans_loop_real/fit_energy_calibration.py
```

Replay the saved full10 JSON with the current calibrated model:

```bash
python3 runs/real_result/trans_loop_real/replay_calibrated_from_json.py
```

Generate the final dynamic calibrated 50 Hz JSON directly:

```bash
python3 src/ros_mujoco/scripts/mujoco_experiment_energy.py \
  --num-experiments 10 \
  --energy-mode dynamic_calibrated \
  --log-rate 50 \
  --output runs/mujoco_experiment_energy_dynamic_calibrated_full10_50hz.json
```

Compare the calibrated 50 Hz trace with the real CSV logs:

```bash
python3 runs/real_result/trans_loop_real/plot_real_sim_energy_comparison.py \
  --real-csv-dir runs/real_result/trans_loop_real/csv \
  --sim-json runs/mujoco_experiment_energy_dynamic_calibrated_full10_50hz.json \
  --out-dir runs/real_result/trans_loop_real/dynamic_energy_comparison
```

## 9. Dynamic 50 Hz calibration-set comparison

The calibration-set comparison is produced by
`runs/real_result/trans_loop_real/plot_real_sim_energy_comparison.py`. It
compares real CSV U/I traces with the dynamic calibrated 50 Hz MuJoCo JSON,
uses joint-position convergence for phase boundaries, and writes
`runs/real_result/trans_loop_real/dynamic_energy_comparison/real_sim_interval_energy_summary.csv`.

The all-16-interval calibration-set result is:

```text
total energy real = 3.776 +/- 0.026 kJ
total energy sim  = 3.776 +/- 0.006 kJ
transform energy real/sim = 1.681 +/- 0.021 / 1.629 +/- 0.008 kJ
sustain energy real/sim   = 2.095 +/- 0.025 / 2.147 +/- 0.010 kJ
transform duration real/sim = 1.371 / 1.332 s
transform-duration MAE = 0.168 s
interval-power MAE = 0.555 W
interval-power RMSE = 1.827 W
interval-power bias = -0.357 W
interval-power corr = 0.995
```

The compatibility first-15-interval result from the same final pipeline is:

```text
energy real = 3.636 +/- 0.024 kJ
energy sim  = 3.635 +/- 0.006 kJ
mean power real = 78.690 +/- 0.520 W
mean power sim  = 78.248 +/- 0.127 W
interval-power MAE = 0.587 W
interval-power RMSE = 1.887 W
interval-power corr = 0.995
```

## 10. Verification commands used

The following checks were used after the implementation:

```bash
python -m unittest src\ros_mujoco\tests\test_energy_calibration.py
python -m py_compile \
  src\ros_mujoco\scripts\mujoco_experiment_energy.py \
  src\ros_mujoco\scripts\ros_mujoco_utils\energy_calibration.py \
  runs\real_result\trans_loop_real\apply_interval_residual_calibration.py \
  runs\real_result\trans_loop_real\compute_50hz_calibration_metrics.py \
  runs\real_result\trans_loop_real\fit_phase_aware_calibration.py \
  runs\real_result\trans_loop_real\plot_real_sim_energy_comparison.py \
  runs\real_result\trans_loop_real\scan_mujoco_dynamic_response.py
python3 -c "import mujoco; mujoco.MjModel.from_xml_path('src/models/crimson/mjcf/crimson_scene.xml'); print('xml ok')"
python runs\real_result\trans_loop_real\plot_real_sim_energy_comparison.py \
  --real-csv-dir runs\real_result\trans_loop_real\csv \
  --sim-json runs/mujoco_experiment_energy_dynamic_calibrated_full10_50hz.json \
  --out-dir runs\real_result\trans_loop_real\dynamic_energy_comparison
```

## 11. Limitations and next steps

The residual term is intentionally explicit and dataset-scoped. It makes this
transform-loop experiment match the real logs well, but it is not a general
physical friction or servo-efficiency model.

The remaining physical gap is likely in effects that the current MJCF does not
represent directly:

- frame internal friction or cable load,
- servo preload during closed-chain transformations,
- backlash and stiction,
- unmodeled structural deformation,
- real-controller current draw that is not proportional to MuJoCo actuator
  torque.

For a more general model, collect additional real logs for different motion
families and fit a validation-split model that predicts unseen transitions
without a per-transition residual table.
