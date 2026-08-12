# Archived-to-Public Implementation Map

The system-level archive under [`../../data/`](../../data/README.md) retains the
original research snapshot for exact provenance. The public repository
organizes the maintained implementation by responsibility instead of
reproducing that snapshot's directory assumptions.

| Archived responsibility | Maintained location | Status |
|---|---|---|
| Full-MIMA and MLLM-distilled service client | `MLLM/mima_requirement_vector/mima_vr/service_client.py` | Included; service URL is mandatory at invocation |
| RF/DT/GBT sensor features and inference | `MLLM/mima_requirement_vector/mima_vr/` | Included with fitted weights and hashes |
| Seven-method definitions | `experiments/system_level_ablation/mima_ablation/methods.py` | Included |
| Success and physical-fit rules | `experiments/system_level_ablation/mima_ablation/records.py` | Included |
| Success/energy and timing batch expansion | `experiments/system_level_ablation/mima_ablation/batch.py` | Included through the documented backend API |
| Table aggregation and integrity audit | `experiments/system_level_ablation/mima_ablation/reporting.py` | Included and verified against archived row-level records |
| Dynamic 50 Hz energy model | `src/ros_mujoco/scripts/ros_mujoco_utils/energy_calibration.py` | Included; supersedes historical energy defaults |
| MuJoCo energy replay | `src/ros_mujoco/scripts/mujoco_experiment_energy.py` | Included |
| Motion planning and command generation | `src/trans_planner/` | Included |
| cVAE and deterministic MLP structure generators | full-chain backend supplied by the user | Interface defined; generator implementation and weights are not in the available release assets |
| Full-MIMA teacher and Teacher-only ET32 model services | compatible service supplied by the user | Client contract included; local service implementations and weights are not in the available release assets |

## Deliberately retired defaults

The maintained experiment entry points do not provide defaults for:

- private network addresses;
- dataset or output directories;
- service cache directories;
- model checkpoints;
- structure-generator checkout locations; or
- robot and energy-model files.

These values are experiment inputs and must be supplied in a run configuration
or on the command line. `configs/run_config.example.json` intentionally leaves
asset fields empty, and execution fails before model loading until all assets
required by the selected methods are set.

## Reproducibility levels

- **Table-level reproduction:** complete from the repository's Git LFS data
  archive. It verifies input hashes, coverage, formulas, and final
  rounded values.
- **Baseline requirement inference:** complete for RF, DT, and GBT using the
  released weights under `MLLM/mima_requirement_vector/`.
- **Full-chain rerun:** the orchestration and record contracts are included,
  but execution requires the unavailable generator and teacher/student service
  assets identified above. No placeholder backend is presented as the paper's
  implementation.
