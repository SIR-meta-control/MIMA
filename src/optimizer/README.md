# optimizer

## Overview

`optimizer` is a ROS1 package that evaluates a batch of candidate robot topologies and selects the minimum-energy solution.
It receives generated topology candidates, converts each topology to joint targets through a kinematics service, simulates motion and hold phases in MuJoCo, and publishes the best topology.

## Dependencies

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

> Note: `dynamixel_msgs`, `numpy`, `mujoco`, `gymnasium`, and `scipy` are required by the Python code, even if some are not declared in `package.xml`.

## Installation

Build inside a standard catkin workspace:

```bash
cd ~/catkin_ws/src
git clone <your_repo_url>
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

If Python dependencies are missing:

```bash
pip install numpy mujoco gymnasium scipy
```

## Usage

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

## Nodes

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
  - `~constraint_steps` (default: `100`) - pre-trajectory constraint correction steps
  - `~sustain_steps` (default: `500`) - hold-phase simulation steps at final pose
  - `~fps` (default: `50`) - effective control frame rate for energy integration
  - `~render` (default: `False`) - whether to render MuJoCo during simulation
