# Installation

[🏠 MIMA](../README.md) · [📚 Documentation](README.md)

> [!NOTE]
> Run standalone commands from the repository root. ROS builds run from the
> Catkin workspace root. GPU model inference and the ROS host can use separate
> environments, connected through the generator HTTP service.

## 📋 Table of Contents

- [🐍 Environment overview](#environment-overview)
- [🐍 Standalone Python environment](#standalone-python-environment)
- [🧠 Generator](#generator)
- [🤖 ROS workspace](#ros-workspace)
- [🧠 Requirement-vector baselines](#requirement-vector-baselines)
- [✅ Development checks](#development-checks)

<a id="environment-overview"></a>

## 🐍 Environment overview

| Component | Environment | Dependencies |
| --- | --- | --- |
| Archived table reconstruction | Python 3.10 or newer | Standard library |
| RF/DT/GBT and protocol checks | Python 3.10 or newer | Requirement-vector development requirements |
| Generator inference and training | Python 3.10 or newer | PyTorch, NumPy, tqdm |
| Energy analysis and optimizer simulation | Python 3.10 or newer | MuJoCo, Gymnasium, SciPy and energy requirements |
| Robot interfaces | Linux, ROS 1 and Catkin | ROS dependencies, vendor drivers and corresponding Python runtime |
| Generic InternVL inference | Separate CUDA environment | [MLLM installation](../MLLM/INSTALLATION.md) |

The simulator, generator inference and generator training support CPU execution.
The generic InternVL implementation uses a separate CUDA environment. Select
the PyTorch build for the host's CUDA driver or use the CPU build below.

<a id="standalone-python-environment"></a>

## 🐍 Standalone Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-energy.txt
```

Windows users can activate the environment with `.venv\Scripts\Activate.ps1`.
ROS hardware interfaces target Linux.

If Ubuntu reports that `ensurepip` is unavailable, install the distribution's
`python3-venv` package before creating the environment. An existing Conda
environment with the corresponding dependencies can also be used.

<a id="generator"></a>

## 🧠 Generator

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/module_level_ablation/requirements.txt
```

For GPU inference, install a matching PyTorch CUDA build instead of the CPU
command. The [PyTorch installer](https://pytorch.org/get-started/locally/)
provides the supported builds. Selected checkpoints are ordinary repository
files under `experiments/module_level_ablation/checkpoints/`.

Verify imports before launching the service:

```bash
python -c "import torch, numpy; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
```

<a id="ros-workspace"></a>

## 🤖 ROS workspace

> [!IMPORTANT]
> Install and source the ROS 1 distribution used by your robot host. Keep its
> Python interpreter and installed ROS Python modules compatible. A standalone
> venv does not automatically supply `rospy`, generated messages or vendor drivers.

For a fresh workspace:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/SIR-meta-control/MIMA.git
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

The packages also define install rules for nodes, helper modules, configuration,
launch files and simulation meshes. To use an install space, run
`catkin_make install` and source `install/setup.bash` instead of `devel/setup.bash`.

For an existing checkout, link it into the workspace rather than cloning again:

```bash
mkdir -p ~/catkin_ws/src
ln -s "$(pwd)" ~/catkin_ws/src/MIMA
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

Run the link command from the MIMA repository root. The workspace finds the
packages recursively under `MIMA/src/`.

The C++ packages require Eigen, PCL, OpenCV, libcurl, ZBar, yaml-cpp,
nlohmann/json and Python development headers. The current Point-LIO source
includes both `livox_ros_driver` and `livox_ros_driver2` message headers;
install and source the matching vendor packages before building that package.
Hardware topics, serial devices and calibration parameters must match the robot.
Package-specific launch instructions are linked from the repository layout.
The [`graph_modeler` capture service](../src/graph_modeler/README.md#dependencies)
includes its geometry accessors and uses MuJoCo and Gymnasium from
`requirements-energy.txt`; no external `commons` package is required.

The optimizer's ROS Python interpreter also needs `requirements-energy.txt`.
The generator model service can run in the separate PyTorch environment; the
ROS adapter itself uses standard-library HTTP and ROS modules.

<a id="requirement-vector-baselines"></a>

## 🧠 Requirement-vector baselines

```bash
python -m pip install -r MLLM/mima_requirement_vector/requirements.txt
```

The fitted RF, DT and GBT weights run locally. For teacher and ET32 inference,
configure the `/predict` endpoint as described in the
[requirement-vector guide](../MLLM/mima_requirement_vector/README.md#full-mima-teacher-and-mllm-distilled-services).

<a id="development-checks"></a>

## ✅ Development checks

```bash
python -m pip install -r requirements-dev.txt
python scripts/check_repository.py
ruff check .
ruff format --check .
```

GitHub Actions runs the repository, lint and formatting checks on pushes and
pull requests. Validate runtime changes in the corresponding model, simulation
or ROS environment described above.

## Teacher and student source

The [teacher/student source guide](../MLLM/teacher_student/README.md#installation)
provides separate CPU/student and teacher dependency lists, plus commands for
launching the paper-specific services with authorized local weights.
