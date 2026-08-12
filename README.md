# MIMA

This repository contains the ROS, multimodal requirement-inference, structure-generation,
energy-estimation, and experiment code for **Metamorphous adaptability in robotic
systems through intelligent structural evolvement**.

<div align="center">
  <img src="figure/MIMA.png" width="75%" alt="MIMA system overview">
</div>

MIMA maps multimodal observations and task information to an executable robot
metamorphosis. The released code separates model inference, configuration generation,
candidate selection, command generation, physical control, simulation replay, and
paper-table reconstruction so that each experimental boundary is explicit.

## Complete pipeline

The evaluated full pipeline consists of five stages:

1. **Multimodal input.** RGB, depth, point-cloud, and task inputs describe the passage
   and the requested traversal.
2. **Requirement-vector inference.** Full-MIMA uses the frozen MLLM teacher to infer
   the seven-dimensional requirement vector
   `v_r = [w_p, h_p, d_p, h_s, f_l, f_i, f_p]`.
3. **Configuration generation.** The cVAE generates candidate metamorphosis
   configurations conditioned on the requirement vector.
4. **Feasibility and energy selection.** Candidates are screened using information
   available to the system, ranked by calibrated energy, and tried in ranked order when
   a lower-energy candidate is not feasible. The no-energy-optimizer ablation instead
   uses the first generated candidate; its energy is computed only for post-hoc audit.
5. **Command generation and execution.** The selected topology is converted to joint
   targets, interpolated into a command sequence, and sent to simulation or the robot
   control layer.

The system-level ablations change only the named component while retaining the same
downstream pipeline and evaluation protocol.

## Repository contents

| Location | Purpose |
| --- | --- |
| [`src/`](src/) | ROS packages for generation, optimization, kinematics, planning, sensing, simulation, and robot control |
| [`MLLM/`](MLLM/) | Generic InternVL code and paper-specific requirement-vector inference interfaces |
| [`MLLM/mima_requirement_vector/`](MLLM/mima_requirement_vector/README.md) | Full-MIMA and distilled service clients, RF/DT/GBT baselines, model documentation, and tests |
| [`experiments/system_level_ablation/`](experiments/system_level_ablation/README.md) | Seven-method batch protocols, backend contract, data-bundle table reconstruction, and tests |
| [`experiments/module_level_ablation/`](experiments/module_level_ablation/README.md) | Configuration-generator component ablations, selected checkpoints, archived metrics, and evaluation code |
| [`src/ros_mujoco/`](src/ros_mujoco/README.md) | MuJoCo replay and calibrated energy estimation |
| [`runs/`](runs/) | Released calibration reports and supporting analysis code |
| [`figure/`](figure/) | Repository figures |
| [`data/`](data/README.md) | Git LFS archives for Figure 4, module-level ablation, and system-level ablation evidence |

The repository includes the experimental data archives under `data/`, while the
hosted Full-MIMA teacher and adopted distilled services are not distributed as local
services. An exact full-chain rerun therefore requires compatible requirement-vector
services and a configured full-chain backend with the dependencies identified by the
system-level experiment interface. The conventional RF, DT, and GBT baseline weights
are included in the requirement-vector module. Deterministic reconstruction of the
reported tables does not require those services. See the data and module READMEs for
the precise release boundary.

## Installation

The ROS packages target ROS 1 with Catkin. Install the ROS packages required by the
individual package manifests together with Eigen, PCL, OpenCV, libcurl, ZBar,
yaml-cpp, nlohmann/json, and Python development headers. Live LiDAR operation also
requires the appropriate vendor driver in a sourced workspace.

Place `src/` in a Catkin workspace, install dependencies with `rosdep`, and build:

```bash
cd "$CATKIN_WORKSPACE"
rosdep install --from-paths src --ignore-src -r -y
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

Install the standalone Python dependencies needed by the energy tools with:

```bash
python3 -m pip install -r requirements-energy.txt
```

Requirement-vector and system-level-ablation dependencies and focused tests are
documented in their respective module READMEs.

## MLLM-distilled

MLLM-distilled was constructed from the frozen Full-MIMA teacher to provide a
resource-efficient requirement-vector backend. The adopted Teacher-only ET32 student
uses a multi-output ensemble of 32 extremely randomized trees. It converts RGB, depth,
point-cloud, and task inputs into an 85-dimensional deterministic sensor summary and
jointly predicts all seven requirement-vector components. Feature computation samples
at most 32,768 sensor elements; this is a feature-extraction limit, not the number of
training examples.

The student was trained on teacher-generated requirement-vector targets for 978
outer-training samples. Original targets were withheld from the student-training
objective and retained for offline evaluation. Because the student preserves the
Full-MIMA requirement-vector interface, the cVAE generator, energy optimizer, and
controller remain unchanged.

The RF, DT, and GBT rows are direct conventional replacements rather than distilled
students. They use a separate deterministic 16-feature adapter and are therefore not
equivalent to the 85-dimensional Teacher-only ET32 pipeline. Model interfaces, the
available local assets, and unsupported claims are documented in the
[requirement-vector model card](MLLM/mima_requirement_vector/MODEL_CARD.md).

## System-level ablation

The released protocol covers the following seven methods:

| Method | Requirement source | Generator | Candidate selection |
| --- | --- | --- | --- |
| Full-MIMA | Frozen Full-MIMA teacher | cVAE | Energy-ranked feasible fallback |
| MLLM-distilled | Teacher-only ET32 | cVAE | Energy-ranked feasible fallback |
| MLLM -> RF | Random forest | cVAE | Energy-ranked feasible fallback |
| MLLM -> DT | Decision tree | cVAE | Energy-ranked feasible fallback |
| MLLM -> GBT | Gradient-boosted tree | cVAE | Energy-ranked feasible fallback |
| cVAE -> MLP | Frozen Full-MIMA teacher | Deterministic MLP | Energy-ranked feasible fallback |
| w/o Energy optimizer | Frozen Full-MIMA teacher | cVAE | First generated candidate; post-hoc energy audit only |

The paper-table reconstruction produces:

| Method | Success rate (%) | Normalized energy (%) | Command-ready latency (s) |
| --- | ---: | ---: | ---: |
| Full-MIMA | 95.23 | 100.00 +/- 1.79 | 3.76 +/- 0.21 |
| MLLM-distilled | 87.38 | 100.04 +/- 1.82 | 0.74 +/- 0.21 |
| MLLM -> RF | 62.06 | 99.93 +/- 1.70 | 0.76 +/- 0.10 |
| MLLM -> DT | 15.98 | 99.88 +/- 1.69 | 0.73 +/- 0.09 |
| MLLM -> GBT | 28.32 | 99.80 +/- 1.61 | 0.75 +/- 0.10 |
| cVAE -> MLP | 34.58 | 103.17 +/- 7.57 | 4.32 +/- 0.54 |
| w/o Energy optimizer | 94.95 | 104.87 +/- 14.77 | 3.13 +/- 0.03 |

### Success

An execution is successful only when all three conditions hold:

1. full-chain command generation succeeds;
2. the generated width and height fit the ground-truth passage; and
3. predicted width and height are each within the geometric tolerance.

The geometric tolerance is 3% of the midpoint of the evaluated height range:

```text
height range = [0.27, 0.43] m
midpoint = 0.35 m
tolerance = 0.03 x 0.35 m = 0.0105 m
```

This tolerance is an evaluation rule. It does not alter inference, generation,
candidate selection, planning, replay, geometry measurement, or energy estimation.

### Normalized energy

Energy is the finite estimate for moving the robot from its home configuration to the
generated configuration after the command is issued. It is summarized over the same
evaluation scenarios and configuration-generation seeds. Full-MIMA's mean finite
energy is the 100% reference:

```text
normalized mean_i (%) = 100 x mean(E_i) / mean(E_Full-MIMA)
normalized SD_i (percentage points) = 100 x SD(E_i) / mean(E_Full-MIMA)
```

The reported dispersion is the sample standard deviation (`ddof=1`). Full-MIMA's
normalized standard deviation is not zero: normalization fixes its mean at 100%, while
its row-level energy estimates still vary around that mean.

### Command-ready latency

Command-ready latency is measured immediately before requirement-vector inference and
ends when the command sequence is ready. It includes:

- requirement-vector inference or the corresponding API request;
- cVAE/MLP configuration generation;
- candidate selection and energy ranking, when enabled; and
- command-sequence generation.

It excludes simulation replay, final geometry measurement, post-hoc energy audit, live
sensor acquisition, and physical robot execution. The reported timing uses one fixed
set of 100 samples, one configuration-generation seed, one worker, and model warm-up
before the timed set. The released timing records do not identify the hardware, so the
values must not be presented as measurements on a specific onboard device.

## Reproduction levels

Two reproduction levels are intentionally separate.

### Reconstruct the paper table

Table reconstruction reads the released row-level records. It does not call the teacher
or distilled services and does not rerun the generator, energy model, or simulation.
Extract `data/system_level_ablation_assets.zip` and set
`SYSTEM_ABLATION_BUNDLE` to the extracted directory containing
`reproduce_table.py`. From the repository root, run:

```bash
python3 experiments/system_level_ablation/scripts/reproduce_table.py \
  --bundle-dir "$SYSTEM_ABLATION_BUNDLE" \
  --output-dir "$TABLE_OUTPUT"
```

The reconstruction validates input hashes, sample IDs, seeds, timing coverage,
finite-energy counts, the derived tolerance, and all reported values after paper-level
rounding. Archive contents and checksums are documented in the
[data README](data/README.md).

### Rerun from raw sensor inputs

A full rerun starts from RGB, depth, point-cloud, and task inputs. It invokes a supplied
requirement-vector source, runs configuration generation and selection, and records
row-level success and energy. This path requires all model assets and a backend that
implements the documented full-chain contract; missing assets are rejected rather than
silently replaced.

```bash
python3 experiments/system_level_ablation/scripts/run_success_energy.py \
  --run-config "$RUN_CONFIG" \
  --dataset-dir "$SENSOR_DATASET" \
  --sample-ids-file "$EVALUATION_IDS" \
  --output-dir "$FULL_CHAIN_OUTPUT" \
  --backend PACKAGE.MODULE:FUNCTION \
  --methods full_mima,mllm_distilled,mllm_to_rf,mllm_to_dt,mllm_to_gbt,cvae_to_mlp,without_energy_optimizer \
  --seeds 1-10 \
  --workers 1
```

Command-ready timing is rerun separately with the fixed timing IDs and one worker:

```bash
python3 experiments/system_level_ablation/scripts/run_execution_time.py \
  --run-config "$RUN_CONFIG" \
  --dataset-dir "$SENSOR_DATASET" \
  --sample-ids-file "$TIMING_IDS" \
  --output-dir "$TIMING_OUTPUT" \
  --backend PACKAGE.MODULE:FUNCTION \
  --seed 7
```

The backend interface and asset requirements are defined in
[`BACKEND_API.md`](experiments/system_level_ablation/BACKEND_API.md). Detailed method
definitions and protocol checks are in the
[system-level ablation README](experiments/system_level_ablation/README.md).

## Energy calibration

The MuJoCo energy path supports legacy models for audit compatibility and the final
`dynamic_calibrated` mode used by the system-level evaluation. The final mode evaluates
calibrated aggregate power and adds an interval-specific offset for all 16 command
intervals, including the final return interval. It uses 50 Hz voltage/current traces;
transformation and sustainment are separated by joint-position convergence using robot
joint positions for real traces and MuJoCo `qpos` for simulation.

This calibration is dataset-specific and must not be described as a universal actuator
energy model. The implementation, validation procedure, retained legacy modes, and
focused tests are documented in the
[MuJoCo energy README](src/ros_mujoco/README.md).

## Third-party software

| Component | Upstream | Location |
| --- | --- | --- |
| Point-LIO | [HKU-MARS/Point-LIO](https://github.com/hku-mars/Point-LIO) | `src/point_lio_crimson/` |
| libb64 | [libb64/libb64](https://github.com/libb64/libb64) | `src/third_party/libb64/` |
| WebSocket++ | [zaphoyd/websocketpp](https://github.com/zaphoyd/websocketpp) | `src/third_party/websocketpp/` |

Third-party and upstream components retain their own licenses.

## License

Original work in this repository is licensed under the GNU General Public License v3.0.
See [`LICENSE.md`](LICENSE.md). Vendored and modified third-party components retain the
terms stated in their corresponding license files.
