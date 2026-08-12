# MIMA System-Level Ablation

This module contains the formal experiment definitions, batch entry points,
and deterministic reporting code for the seven-row system-level ablation.
It replaces machine-bound research scripts with explicit configuration and a
documented backend interface.

## Supported workflows

Two workflows are intentionally separated:

1. **Table reconstruction** reads archived row-level records and reproduces the
   reported table without calling models or rerunning simulation.
2. **Full-chain execution** expands the configured methods over explicit sample
   IDs and seeds, invokes a supplied full-chain backend, and writes records in
   the same schema consumed by the reconstruction code.

The reconstruction path is complete when used with the system-level ablation
data bundle. Full-chain execution additionally requires the cVAE/MLP generator
weights and implementations, requirement-vector sources, robot model, and
energy model. Those assets are not silently inferred from repository layout.

## Separately distributed data bundle

The source repository does not contain the 587 MB experiment bundle. It is
distributed separately and contains:

- the fixed 107-sample sensor evaluation set;
- row-level success, energy, and command-ready timing records;
- frozen Full-MIMA and MLLM-distilled requirement-vector caches;
- the reconstructed paper table and its full-precision audit report;
- the original experiment code snapshot; and
- a complete SHA-256 manifest.

The bundle is sufficient for deterministic table reconstruction. It does not
contain the hosted model services or every model weight needed to rerun the
full chain from raw sensor inputs.

## Reported table

The current protocol in `configs/paper_protocol.json` reconstructs:

| Method | Success Rate (%) | Normalized Energy (%) | Execution Time (s) |
|---|---:|---:|---:|
| Full-MIMA | 95.23 | 100.00 +/- 1.79 | 3.76 +/- 0.21 |
| MLLM-distilled | 87.38 | 100.04 +/- 1.82 | 0.74 +/- 0.21 |
| MLLM -> RF | 62.06 | 99.93 +/- 1.70 | 0.76 +/- 0.10 |
| MLLM -> DT | 15.98 | 99.88 +/- 1.69 | 0.73 +/- 0.09 |
| MLLM -> GBT | 28.32 | 99.80 +/- 1.61 | 0.75 +/- 0.10 |
| cVAE -> MLP | 34.58 | 103.17 +/- 7.57 | 4.32 +/- 0.54 |
| w/o Energy optimizer | 94.95 | 104.87 +/- 14.77 | 3.13 +/- 0.03 |

## Method definitions

| Method | Requirement source | Generator | Candidate selection |
|---|---|---|---|
| Full-MIMA | frozen Full-MIMA teacher | cVAE | energy-ranked feasible fallback |
| MLLM-distilled | Teacher-only ET32 | cVAE | energy-ranked feasible fallback |
| MLLM -> RF | random forest | cVAE | energy-ranked feasible fallback |
| MLLM -> DT | decision tree | cVAE | energy-ranked feasible fallback |
| MLLM -> GBT | gradient-boosted tree | cVAE | energy-ranked feasible fallback |
| cVAE -> MLP | frozen Full-MIMA teacher | deterministic MLP | energy-ranked feasible fallback |
| w/o Energy optimizer | frozen Full-MIMA teacher | cVAE | first generated candidate; post-hoc energy audit only |

Only the named component changes in each ablation. The requirement-vector
clients and conventional baseline models are documented separately under
`MLLM/mima_requirement_vector/`.

## Success definition

An execution is successful only when all three conditions hold:

1. the full-chain backend reports successful command generation;
2. the generated width and height do not exceed the ground-truth passage; and
3. predicted width and height are each within the configured tolerance of the
   ground truth.

The reported tolerance is 3% of the midpoint of the evaluated height range:

```text
height range = [0.27, 0.43] m
midpoint = 0.35 m
tolerance = 0.03 x 0.35 m = 0.0105 m
```

The tolerance is applied only during table reconstruction. It does not modify
prediction, generation, candidate selection, motion planning, replay, geometry
measurement, or energy estimation.

## Energy definition

Energy is summarized over finite row-level estimates from the 107 samples and
10 configuration-generation seeds. Full-MIMA's mean finite energy is the 100%
reference:

```text
normalized mean_i (%) = 100 x mean(E_i) / mean(E_Full-MIMA)
normalized SD_i (percentage points) = 100 x SD(E_i) / mean(E_Full-MIMA)
```

Both standard deviations use the sample statistic (`ddof=1`). Full-MIMA's
normalized standard deviation is not zero because it describes row-level
dispersion around the reference mean.

## Execution-time definition

Timing uses an explicit fixed list of 100 samples, one configuration-generation
seed, and one worker. The backend must return command-ready latency measured
from immediately before requirement-vector inference through:

1. requirement-vector inference or the corresponding API request;
2. cVAE/MLP structure generation;
3. candidate selection and energy ranking, when enabled; and
4. command-sequence generation.

It excludes simulation replay, final geometry measurement, post-hoc energy
audit, live sensor acquisition, and physical robot execution. API-backed rows
include the request made by the backend. Model warm-up must occur before the
timed sample set. Hardware identity is not present in the archived timing rows,
so these values must not be described as measurements on a specific onboard
device.

## Reconstruct the table

The data bundle and output directory are mandatory arguments. No local or
server path is embedded in the script.

```bash
python scripts/reproduce_table.py \
  --bundle-dir /path/to/data_bundle \
  --output-dir /path/to/reconstructed_table
```

The script validates:

- all six input SHA-256 hashes;
- 107 unique evaluation IDs;
- 10 seeds and 1,070 rows per method;
- the common 100-sample timing set;
- the derived 0.0105 m tolerance;
- all finite-energy counts; and
- every value after paper-level rounding.

Outputs are `system_level_ablation_table.md`, a full-precision CSV, and
`audit_report.json`.

The bundle-root `reproduce_table.py` and
`reproduction/reproduce_system_level_ablation.py` invoke the same current
0.0105 m reconstruction. Historical experiment defaults are retained only
inside the explicitly marked code snapshot and do not define reported values.

## Rerun success and energy

Create a run configuration from `configs/run_config.example.json` and set every
asset explicitly. Empty fields are rejected before execution.

```bash
python scripts/run_success_energy.py \
  --run-config /path/to/run_config.json \
  --dataset-dir /path/to/evaluation_dataset \
  --sample-ids-file /path/to/sample_ids.txt \
  --output-dir /path/to/success_energy_output \
  --backend your_package.full_chain:run_request \
  --methods full_mima,mllm_distilled,mllm_to_rf,mllm_to_dt,mllm_to_gbt,cvae_to_mlp,without_energy_optimizer \
  --seeds 1-10 \
  --workers 1
```

Parallel workers may be enabled only when the supplied backend and hardware
support independent processes. Parallel success/energy execution is not a
timing measurement.

## Rerun command-ready timing

Use the explicit 100-ID timing list and one worker:

```bash
python scripts/run_execution_time.py \
  --run-config /path/to/run_config.json \
  --dataset-dir /path/to/evaluation_dataset \
  --sample-ids-file /path/to/timing_sample_ids.txt \
  --output-dir /path/to/timing_output \
  --backend your_package.full_chain:run_request \
  --seed 7
```

The backend API is specified in [`BACKEND_API.md`](BACKEND_API.md).
The mapping from the archived research snapshot to maintained public modules is
recorded in [`IMPLEMENTATION_MAP.md`](IMPLEMENTATION_MAP.md).

## Why the archived snapshot is not copied verbatim

The original research snapshot remains in the raw-data bundle for provenance.
Several scripts contained private network defaults, historical output paths,
dated checkpoint locations, and assumptions about an `external/KT_GVAE`
checkout. Copying those files into the public source tree would create a
misleading, non-portable entry point. This module preserves their experiment
semantics while replacing those assumptions with explicit assets and a stable
backend contract.
