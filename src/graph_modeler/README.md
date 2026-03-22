# graph_modeler

## Overview

`graph_modeler` is a ROS1 package that builds a structured topological graph representation of the Crimson robot from a given joint angle configuration.

It provides a ROS service (`model_graph`) that accepts a flat joint angle vector, drives the MuJoCo simulation to that pose, and captures the resulting robot state (joint positions/orientations, node poses, body scale, leg base transforms). This state is then packed into a `TopologicalGraph` message and returned to the caller.

The output graph is the primary input format consumed by the `generator` and `optimizer` pipeline stages for topology generation and energy evaluation.

## Dependencies

- **ROS1 / Catkin**
  - `catkin`
  - `roscpp`
  - `rospy`
  - `std_msgs`
  - `meta_msgs`
- **Python libraries**
  - `numpy`
  - `mujoco`
  - `scipy` (for `scipy.spatial.transform`)
- **External assets**
  - MuJoCo model file: `models/crimson/mjcf/crimson_scene.xml`

## Installation

Build in a standard catkin workspace:

```bash
cd ~/catkin_ws/src
git clone <your_repo_url>
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

If Python dependencies are missing:

```bash
pip install numpy mujoco scipy
```

## Usage

Launch the graph modeler service node:

```bash
roslaunch graph_modeler graph_mod.launch
```

Or run directly:

```bash
rosrun graph_modeler graph_mod_srv.py
```

The node waits for calls to the `model_graph` service. Pass a 17-element joint angle vector as `req.angles` to receive the corresponding `TopologicalGraph` response.

## Nodes

### `graph_modeler_node` (`scripts/graph_mod_srv.py`)

- **Subscribed Topics**
  - None
- **Published Topics**
  - None
- **Services (provided)**
  - `model_graph` (`meta_msgs/Angles2Topo`) — accepts a joint angle array (`req.angles`), simulates the robot in MuJoCo, and returns a fully populated `TopologicalGraph` containing:
    - `edges` — per-joint data: angle, 3D position, quaternion (8 values per joint, row-major `Float32MultiArray`)
    - `nodes` — per-body pose: 3D position, quaternion (7 values per node)
    - `adjacency` — square adjacency matrix of the robot graph
    - `feature.scale` — bounding-box dimensions `[length, width, height]`
    - `feature.leg_base` — leg mounting site poses (4 legs × 7 floats)
    - `feature.leg_angles` — representative leg joint angles (3-element list)
- **Services (called)**
  - None
- **Parameters**
  - None
