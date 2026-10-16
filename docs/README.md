# MIMA Documentation

[🏠 MIMA](../README.md)

Choose a guide for installation, model and data preparation, or reproduction.
Package READMEs document the individual ROS and Python interfaces.

## 📋 Table of Contents

- [📚 Installation and downloads](#installation-and-downloads)
- [🔌 Runtime interfaces](#runtime-interfaces)
- [🧪 Experiments](#experiments)
- [⚡ Energy analysis](#energy-analysis)
- [🤝 Contributing and licenses](#contributing-and-licenses)

<a id="installation-and-downloads"></a>

## 📚 Installation and downloads

| Guide | Contents |
| --- | --- |
| [Installation](INSTALLATION.md) | Python, PyTorch and ROS 1 environments |
| [Downloads](DOWNLOADS.md) | Released models, Git LFS data, checksums and extraction |
| [Model card](MODEL_CARD.md) | Model identities, inference interfaces and evaluation settings |
| [Reproducibility](REPRODUCIBILITY.md) | Table reconstruction, checkpoint evaluation and experiment protocols |

<a id="runtime-interfaces"></a>

## 🔌 Runtime interfaces

| Component | Guide |
| --- | --- |
| Multimodal inference | [MLLM](../MLLM/README.md), [InternVL installation](../MLLM/INSTALLATION.md) |
| Requirement-vector models | [Interfaces and baselines](../MLLM/mima_requirement_vector/README.md), [model card](../MLLM/mima_requirement_vector/MODEL_CARD.md) |
| RTC generation | [Generator service and ROS adapter](../src/generator/README.md) |
| Geometry capture | [Graph modeler](../src/graph_modeler/README.md) |
| Energy selection | [Optimizer](../src/optimizer/README.md) |
| Kinematics and planning | [Kinematics interpreter](../src/kinematics_interpreter/README.md), [transformation planner](../src/trans_planner/README.md) |
| Robot control | [Control stack](../src/crimson_control/README.md), [motor interfaces](../src/motor_interface/README.md) |
| Shared ROS types | [Messages and services](../src/meta_msgs/README.md) |
| Robot assets | [Models](../src/models/README.md) |
| Interface testing | [Fixed-position test service](../src/test_dyn/README.md) |

<a id="experiments"></a>

## 🧪 Experiments

- [Experiment overview](../experiments/README.md)
- [Module-level ablations](../experiments/module_level_ablation/README.md)
- [System-level ablations](../experiments/system_level_ablation/README.md)
- [Full-chain backend API](../experiments/system_level_ablation/BACKEND_API.md)
- [System-level implementation map](../experiments/system_level_ablation/IMPLEMENTATION_MAP.md)
- [Research-data inventory](../data/README.md)

<a id="energy-analysis"></a>

## ⚡ Energy analysis

- [MuJoCo bridge and energy replay](../src/ros_mujoco/README.md)
- [Calibration records](../runs/real_result/trans_loop_real/README.md)
- [Calibration report](../runs/real_result/trans_loop_real/calibration_report.md)
- [Historical real-log analysis](../runs/real_result/trans_loop_real/analysis_energy/README.md)
- [Historical JSON-log analysis](../runs/real_result/trans_loop_real/analysis_energy_json_no_startup/README.md)

<a id="contributing-and-licenses"></a>

## 🤝 Contributing and licenses

- [Contributing](../CONTRIBUTING.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Citation metadata](../CITATION.cff)
- [Repository license](../LICENSE.md)
