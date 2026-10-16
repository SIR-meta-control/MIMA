# System-Level Implementation Map

[🏠 MIMA](../../README.md) · [📚 Documentation](../../docs/README.md)

The system-level archive under [`../../data/`](../../data/README.md) retains the
original experiment scripts. Their corresponding public modules are listed below.

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
| cVAE and Scale GNN RTC generator | `src/generator/` | HTTP service, ROS adapter, implementation, and selected checkpoint included |
| Module-level cVAE/MLP variants | `experiments/module_level_ablation/` | Four model implementations and selected checkpoints included; system-level integration requires a configured backend |
| Full-MIMA teacher and Teacher-only ET32 inference | `MLLM/mima_requirement_vector/mima_vr/service_client.py` | `/predict` endpoint configured in the run settings |

<a id="run-configuration"></a>

## ⚙️ Run configuration

Supply the following experiment inputs in the run configuration:

- model-service addresses;
- dataset or output directories;
- service cache directories;
- model checkpoints;
- structure-generator checkout locations; or
- robot and energy-model files.

Set the service addresses and asset paths for the selected methods in
`configs/run_config.example.json` before execution.

<a id="reproducibility-levels"></a>

## 🧪 Experiment workflows

- **Table reconstruction:** read the Git LFS data archive and verify input
  hashes, coverage, formulas and final rounded values.
- **Baseline requirement inference:** run RF, DT and GBT with the fitted
  weights under `MLLM/mima_requirement_vector/`.
- **System-level experiments:** configure the pipeline backend and model
  endpoints, then run the batch scripts with the selected samples and seeds.
