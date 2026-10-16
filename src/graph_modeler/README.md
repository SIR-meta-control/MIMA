# graph_modeler

[🏠 MIMA](../../README.md) · [📚 Documentation](../../docs/README.md)

## 📋 Table of Contents

- [🏠 Overview](#overview)
- [🧩 Dependencies](#dependencies)
- [📚 Installation](#installation)
- [🚀 Usage](#usage)
- [🤖 Nodes](#nodes)

<a id="overview"></a>

## 🏠 Overview

`graph_modeler` is a ROS1 package that builds a structured topological graph representation of the Crimson robot from a given joint angle configuration.

It provides a ROS service (`model_graph`) that accepts a flat joint angle vector, drives the MuJoCo simulation to that pose, and captures the resulting robot state (joint positions/orientations, node poses, body scale, leg base transforms). This state is then packed into a `TopologicalGraph` message and returned to the caller.

The returned graph uses the RTC message format shared by the generator,
optimizer, and kinematics interpreter.

<a id="dependencies"></a>

## 🧩 Dependencies

The capture environment is included in `scripts/sim_env.py`. It uses
Gymnasium's MuJoCo runtime and restores the geometry accessors from the
original Crimson simulation: frame joints 2–8, node sites 1–8, and leg-base
sites in the order 16, 15, 17, 14. The capture definitions follow
[crimson_sim at `76c9131`](https://github.com/SIR-meta-control/crimson_sim/blob/76c9131f50bd4a34e9fbc564b6727054603b188f/crimson_mujoco_env.py).
No external `commons` package is needed.
Joint angles use MuJoCo's `jnt_qposadr` mapping so the floating root does not
shift the captured hinge angles. Poses use world coordinates and wxyz
quaternions; dimensions are the world-axis extents of all site centers.

- **ROS1 / Catkin**
  - `catkin`
  - `roscpp`
  - `rospy`
  - `std_msgs`
  - `meta_msgs`
- **Python libraries**
  - `numpy`
  - `mujoco`
  - `gymnasium`
  - `scipy`
- **External assets**
  - MuJoCo model file: `src/models/crimson/mjcf/crimson_scene.xml`
    (resolved from the checkout or the installed model package,
    independently of the working directory)

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

Install Python dependencies from the MIMA repository root using the ROS node's
Python interpreter:

```bash
pip install -r requirements-energy.txt
```

<a id="usage"></a>

## 🚀 Usage

Launch the graph modeler service node:

```bash
roslaunch graph_modeler graph_mod.launch
```

Or run directly:

```bash
rosrun graph_modeler graph_mod_srv.py
```

The node waits for calls to the `model_graph` service. Pass a 17-element joint angle vector as `req.angles` to receive the corresponding `TopologicalGraph` response.

<a id="nodes"></a>

## 🤖 Nodes

### `graph_modeler_node` (`scripts/graph_mod_srv.py`)

- **Subscribed Topics**
  - None
- **Published Topics**
  - None
- **Services (provided)**
  - `model_graph` (`meta_msgs/Angles2Topo`) — accepts a joint angle array (`req.angles`), simulates the robot in MuJoCo, and returns a `TopologicalGraph` containing:
    - `edges` — per-joint data: angle, 3D position, quaternion (8 values per joint, row-major `Float32MultiArray`)
    - `nodes` — per-body pose: 3D position, quaternion (7 values per node)
    - `adjacency` — identity matrix in the current capture service
    - `feature.scale` — bounding-box dimensions `[length, width, height]`
    - `feature.leg_base` — leg mounting site poses (4 legs × 7 floats)
    - `feature.leg_angles` — three copies of joint index 6 in the current capture service
- **Services (called)**
  - None
- **Parameters**
  - None
