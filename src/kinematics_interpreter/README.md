# kinematics_interpreter

[🏠 MIMA](../../README.md) · [📚 Documentation](../../docs/README.md)

## 📋 Table of Contents

- [🏠 Overview](#overview)
- [🧩 Dependencies](#dependencies)
- [📚 Installation](#installation)
- [🚀 Usage](#usage)
- [🤖 Nodes](#nodes)

<a id="overview"></a>

## 🏠 Overview

`kinematics_interpreter` is a ROS1 package that interprets topological graph data into joint control angles for the Crimson robot.

It provides two interfaces backed by the same core calculation:

- A **ROS service** (`interpret_kinematics`) that accepts a single `TopologicalGraph` and synchronously returns a 17-element joint angle vector. This is consumed by the `optimizer` package during parallel energy evaluation.
- A **topic subscriber** (`/optimal_topology`) that receives the final selected topology, computes the joint control vector, and publishes it to `/motor_ctrl` to drive the physical robot.

> [!NOTE]
> The kinematics computation uses MuJoCo pose math (`mju_mulPose`, `mju_mulQuat`) and a static YAML file of kinematic-chain reference poses to derive per-joint angle deviations.

<a id="dependencies"></a>

## 🧩 Dependencies

- **ROS1 / Catkin**
  - `catkin`
  - `rospy`
  - `std_msgs`
  - `message_generation` / `message_runtime`
  - `meta_msgs`
- **Python libraries**
  - `numpy`
  - `mujoco`
  - `scipy` (for `scipy.spatial.transform`)
  - `PyYAML` (provides the `yaml` module)
- **Static data**
  - `scripts/theoretically_relative_poses.yaml` — kinematic chain reference poses loaded at node startup

<a id="installation"></a>

## 📚 Installation

Build in a standard catkin workspace:

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
python3 -m pip install numpy mujoco scipy PyYAML
```

<a id="usage"></a>

## 🚀 Usage

Launch the kinematics interpreter service node:

```bash
roslaunch kinematics_interpreter main.launch
```

Or run directly:

```bash
rosrun kinematics_interpreter kin_ite_srv.py
```

Launch the test topology publisher (for development/debugging):

```bash
roslaunch kinematics_interpreter pub.launch
```

<a id="nodes"></a>

## 🤖 Nodes

### `kinematics_interpreter_node` (`scripts/kin_ite_srv.py`)

- **Subscribed Topics**
  - `/optimal_topology` (`meta_msgs/TopologicalGraph`) — receives the final selected topology and triggers motor command publishing
- **Published Topics**
  - `/motor_ctrl` (`std_msgs/Float32MultiArray`) — 17-element joint control vector sent to the motor driver
- **Services (provided)**
  - `interpret_kinematics` (`meta_msgs/Topo2Angles`) — accepts a `TopologicalGraph` and returns a 17-element joint angle array
- **Services (called)**
  - None
- **Parameters**
  - `flag_test` (default: `true`) — selects data layout of incoming edge data (`true`: edges include a leading joint-angle column; `false`: edges start directly from position data)
