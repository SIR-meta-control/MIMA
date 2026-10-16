# optimizer

[🏠 MIMA](../../README.md) · [📚 Documentation](../../docs/README.md)

## 📋 Table of Contents

- [🏠 Overview](#overview)
- [🧩 Dependencies](#dependencies)
- [📚 Installation](#installation)
- [🚀 Usage](#usage)
- [🤖 Nodes](#nodes)
- [⚙️ Candidate simulation](#candidate-simulation)

<a id="overview"></a>

## 🏠 Overview

`optimizer` is a ROS1 package that evaluates a batch of candidate robot topologies and selects the minimum-energy solution.
It receives candidate RTCs, converts each to joint targets, simulates motion
and hold phases in MuJoCo, and selects the candidate with the smallest sum of
transformation and sustainment energy.

<a id="dependencies"></a>

## 🧩 Dependencies

- **ROS1 / Catkin**
  - `catkin`
  - `roscpp`
  - `rospy`
  - `std_msgs`
  - `meta_msgs`
- **ROS message/service packages used at runtime**
  - `dynamixel_msgs` (for `dynamixel_msgs/GetPos`)
- **Python libraries**
  - `numpy`
  - `mujoco`
  - `gymnasium` (MuJoCo environment wrapper)
  - `scipy` (for `scipy.spatial.transform`)
- **External assets**
  - MuJoCo model file: `../models/crimson/mjcf/crimson_scene.xml`

The node also imports `ros_mujoco_utils.energy_calibration` from `ros_mujoco`.

<a id="installation"></a>

## 📚 Installation

Build inside a standard catkin workspace:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/SIR-meta-control/MIMA.git
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

If Python dependencies are missing:

```bash
pip install numpy mujoco gymnasium scipy
```

<a id="usage"></a>

## 🚀 Usage

Launch the optimizer node:

```bash
roslaunch optimizer optimizer.launch
```

Or run directly:

```bash
rosrun optimizer pareto_optimizer_node.py
```

The node expects the following upstream services/topics to be available:

- Service `interpret_kinematics` (`meta_msgs/Topo2Angles`)
- Service `/dynamixel_control/pos` (`dynamixel_msgs/GetPos`)
- Topic `generated_topolist` (`meta_msgs/TopoList`)

Result topic:

- `optimal_topology` (`meta_msgs/TopologicalGraph`)

<a id="nodes"></a>

## 🤖 Nodes

### `pareto_topology_optimizer_node` (`scripts/pareto_optimizer_node.py`)

- **Subscribed Topics**
  - `generated_topolist` (`meta_msgs/TopoList`)
- **Published Topics**
  - `optimal_topology` (`meta_msgs/TopologicalGraph`)
- **Services (called)**
  - `interpret_kinematics` (`meta_msgs/Topo2Angles`)
  - `/dynamixel_control/pos` (`dynamixel_msgs/GetPos`)
- **Services (provided)**
  - None
- **Parameters**
  - `~max_workers` (default: `4`) - thread pool size for parallel topology evaluation
  - `~transition_steps` (default: `400`) - interpolation steps for main trajectory
  - `~constraint_steps` (default: `100`) - frames in each required start/end axis-alignment segment
  - `~sustain_steps` (default: `500`) - hold-phase simulation steps at final pose
  - `~fps` (default: `50`) - shared simulation control and energy integration rate
  - `~render` (default: `False`) - whether to render MuJoCo during simulation

<a id="candidate-simulation"></a>

## ⚙️ Candidate simulation

> [!IMPORTANT]
> Candidate evaluation starts from the measured current joint positions while
> the robot is stationary. The 17 actuator angles are mapped to their MuJoCo
> joints, the two passive frame joints are fitted to the loop constraints, and
> all initial velocities are zero. The model's nominal base orientation is
> retained and its height is set by the horizontal scene floor. Initialization
> requires a maximum loop-closure residual of 1 mm; inconsistent input poses
> raise an error before candidate simulation begins.

The motion path follows the execution planner's axis-alignment waypoints:
initial alignment when required, motion between aligned poses, and final
motion to an unaligned target when required. Adjacent segments share the same
boundary pose. The main segment uses `~transition_steps`; each required
alignment segment uses `~constraint_steps`.

At the default 50 Hz, each control frame advances 40 MJCF substeps of 0.0005 s,
so both simulated time and energy integration use 0.02 s per frame. The default
500-frame sustainment phase therefore lasts 10 s. Other configured rates must
also correspond to an integer number of MJCF substeps.
