# Reproducing MIMA results

[🏠 MIMA](../README.md) · [📚 Documentation](README.md)

Run commands from the repository root unless a section specifies another directory.

## 📋 Table of Contents

- [🧭 Choose a workflow](#choose-a-workflow)
- [📊 Reconstruct the module-level table](#reconstruct-the-module-level-table)
- [🧪 Reevaluate and retrain generator models](#reevaluate-and-retrain-generator-models)
- [🗂️ Prepare the system-level bundle](#prepare-the-system-level-bundle)
- [📐 System-level protocol](#system-level-protocol)
- [🧪 Reproduction](#reproduction)
- [⚡ Energy calibration](#energy-calibration)
- [✅ Verification](#verification)

<a id="choose-a-workflow"></a>

## 🧭 Choose a workflow

| Workflow | Inputs | Model inference |
| --- | --- | --- |
| Reconstruct reported tables | Archived metric JSON or row-level experiment records | None |
| Reevaluate module-level checkpoints | Released selected weights and original validation partition | Local cVAE/MLP inference |
| Retrain generator ablations | Released training and validation partitions | Local training |
| Run new system-level experiments | Sensor dataset, configured requirement service and pipeline backend | Full configured pipeline |

> [!IMPORTANT]
> Success Rate in Supplementary Table 5 and Achievement Rate in Supplementary
> Table 6 use different evaluation protocols. Their percentages must not be
> compared as measurements of the same event.

<a id="reconstruct-the-module-level-table"></a>

## 📊 Reconstruct the module-level table

This operation needs only Python's standard library:

```bash
python experiments/module_level_ablation/scripts/eval/print_validation_table.py \
  experiments/module_level_ablation/outputs/validation_metrics.json \
  experiments/module_level_ablation/outputs/cvae_mlp_ablation_metrics.json \
  experiments/module_level_ablation/outputs/mlp_gnn_ablation_metrics.json \
  experiments/module_level_ablation/outputs/mlp_ablation_metrics.json \
  --csv-output experiments/module_level_ablation/reproduced/validation_metrics_table.csv
```

The generated CSV must match the archived comparison table after the same
formatting and rounding. Achievement Rate is read from `success_at_64`, not
`valid_sample_rate`. The four reported rates are 88.46%, 38.46%, 74.04%, and 55.77%.

<a id="reevaluate-and-retrain-generator-models"></a>

## 🧪 Reevaluate and retrain generator models

```bash
python scripts/prepare_data.py module
cd experiments/module_level_ablation
python reevaluate.py --device auto
```

Install the generator environment in [INSTALLATION.md](INSTALLATION.md) first.
The original validation partition is `split_seed7`: 59,200 pairs, 3,549 source
structures, and 104 unique requirement/subspace queries. Training and validation
share no source structures. The query definitions occur in both partitions,
and the validation partition was used for checkpoint selection.

> [!NOTE]
> The cVAE samples a learned conditional diagonal Gaussian prior. Deterministic
> MLP variants repeat their single output through the common 64-output interface.
> The archived evaluation records identify PyTorch 2.8.0+cu129, CUDA 12.9 and an
> NVIDIA GeForce RTX 3090, with seed 7, batch size 512, query batch size 16,
> sampling temperature 1.0 and 64 outputs per query. CPU and CUDA random-number
> streams can produce different sampled candidates even with the same seed.
> Fresh prior sampling and retraining can therefore change Achievement Rate;
> reconstructing archived records is the deterministic path. Training launchers and hyperparameters are
> listed in the [module README](../experiments/module_level_ablation/README.md).

<a id="prepare-the-system-level-bundle"></a>

## 🗂️ Prepare the system-level bundle

```bash
python scripts/prepare_data.py system
python experiments/system_level_ablation/scripts/reproduce_table.py \
  --bundle-dir experiments/system_level_ablation/data_bundle/system_level_ablation_google_drive_assets \
  --output-dir experiments/system_level_ablation/reconstructed
```

<a id="system-level-protocol"></a>

## 📐 System-level protocol

The released protocol covers the following seven methods:

| Method | Requirement source | Generator | Candidate selection |
| --- | --- | --- | --- |
| Full-MIMA | Frozen Full-MIMA teacher | cVAE + Scale GNN | Energy-ranked feasible fallback |
| MLLM-distilled | Teacher-only ET32 | cVAE + Scale GNN | Energy-ranked feasible fallback |
| MLLM -> RF | Random forest | cVAE + Scale GNN | Energy-ranked feasible fallback |
| MLLM -> DT | Decision tree | cVAE + Scale GNN | Energy-ranked feasible fallback |
| MLLM -> GBT | Gradient-boosted tree | cVAE + Scale GNN | Energy-ranked feasible fallback |
| RTC generator -> MLP | Frozen Full-MIMA teacher | Deterministic MLP | Energy-ranked feasible fallback |
| w/o Energy optimizer | Frozen Full-MIMA teacher | cVAE + Scale GNN | First generated candidate; energy evaluated after selection |

The paper-table reconstruction produces:

| Method | Success rate (%) | Normalized energy (%) | Command-ready latency (s) |
| --- | ---: | ---: | ---: |
| Full-MIMA | 95.23 | 100.00 +/- 1.79 | 3.76 +/- 0.21 |
| MLLM-distilled | 87.38 | 100.04 +/- 1.82 | 0.74 +/- 0.21 |
| MLLM -> RF | 62.06 | 99.93 +/- 1.70 | 0.76 +/- 0.10 |
| MLLM -> DT | 15.98 | 99.88 +/- 1.69 | 0.73 +/- 0.09 |
| MLLM -> GBT | 28.32 | 99.80 +/- 1.61 | 0.75 +/- 0.10 |
| RTC generator -> MLP | 34.58 | 103.17 +/- 7.57 | 4.32 +/- 0.54 |
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
before the timed set.

<a id="reproduction"></a>

## 🧪 Reproduction

<a id="reconstruct-the-paper-table"></a>

The [bundle preparation command](#prepare-the-system-level-bundle) above
reconstructs the system-level table with explicit default paths. Reconstruction
validates input hashes, sample IDs, seeds, timing coverage, finite-energy
counts, the derived tolerance and reported values after paper-level rounding.

### Rerun from raw sensor inputs

A full rerun starts from RGB, depth, point-cloud, and task inputs. It invokes a supplied
requirement-vector source, runs configuration generation and selection, and records
row-level success and energy. This path requires configured model assets and a backend implementing the
[experiment interface](../experiments/system_level_ablation/BACKEND_API.md).

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
[`BACKEND_API.md`](../experiments/system_level_ablation/BACKEND_API.md). Detailed method
definitions and protocol checks are in the
[system-level ablation README](../experiments/system_level_ablation/README.md).

<a id="energy-calibration"></a>

## ⚡ Energy calibration

The ROS optimizer and system-level ablation candidate evaluation use the shared
`dynamic_calibrated` aggregate power model.
Legacy models remain available for historical replay and audit.

Transformation and sustainment are separated by joint-position
convergence using robot joint positions for real traces and MuJoCo `qpos` for
simulation. Real-robot voltage and current measurements were acquired at
approximately 3 Hz; the 50 Hz logging rate applies to the MuJoCo simulation.

The calibration was fitted to the repeated transformation-loop dataset. The
implementation, validation procedure and retained legacy modes are documented in the
[MuJoCo energy README](../src/ros_mujoco/README.md).


<a id="verification"></a>

## ✅ Verification

Run `python scripts/check_repository.py` for source, documentation, ROS package
configuration and artifact consistency. Run `ruff check .` and
`ruff format --check .` for maintained source quality.

Archive SHA-256 values are listed in `data/SHA256SUMS`. The dataset preparation
commands verify each archive before extraction. Full data verification is
described in [DOWNLOADS.md](DOWNLOADS.md).

## Teacher and student source

The [teacher/student programs](../MLLM/teacher_student/README.md#data-preparation-training-and-evaluation)
include sensor feature extraction, teacher/student training and evaluation,
inner-split selection and leakage auditing. Teacher and student weights are
not distributed because of third-party rights restrictions; frozen predictions
can still be used for archived table reconstruction.
