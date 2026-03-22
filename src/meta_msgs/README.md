# meta_msgs

## Overview

`meta_msgs` is a ROS1 message and service definition package that provides the shared data types used across the Crimson robot pipeline — from topology generation to kinematics interpretation, graph modelling, and motion planning.

It defines no executable nodes. All other packages in this workspace (`optimizer`, `kinematics_interpreter`, `graph_modeler`, `generator`, etc.) declare `meta_msgs` as a dependency to access these common interfaces.

## Dependencies

- **ROS1 / Catkin**
  - `catkin`
  - `roscpp`
  - `rospy`
  - `std_msgs`
  - `message_generation`
  - `message_runtime`

## Installation

Build in a standard catkin workspace:

```bash
cd ~/catkin_ws/src
git clone <your_repo_url>
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Nodes

None. This is a message-only package.

---

## Message Definitions

### `meta_msgs/Global`

Global feature attributes of a robot topology.

| Field | Type | Description |
|---|---|---|
| `scale` | `float32[]` | Bounding-box dimensions `[length, width, height]` |
| `leg_base` | `std_msgs/Float32MultiArray` | Leg mounting site poses (4 legs × 7 floats: `[x, y, z, qw, qx, qy, qz]`) |
| `leg_angles` | `float32[]` | Representative leg joint angles |
| `locomotion_mode` | `uint8` | Active locomotion mode identifier |

---

### `meta_msgs/TopologicalGraph`

A complete graph representation of the robot's current physical configuration.

| Field | Type | Description |
|---|---|---|
| `nodes` | `std_msgs/Float32MultiArray` | Per-body poses, each row: `[x, y, z, qw, qx, qy, qz]` (7 floats) |
| `edges` | `std_msgs/Float32MultiArray` | Per-joint data, each row: `[angle, x, y, z, qw, qx, qy, qz]` (8 floats) |
| `adjacency` | `std_msgs/Float32MultiArray` | Square adjacency matrix (N×N, row-major) |
| `feature` | `meta_msgs/Global` | Global features of the configuration |

---

### `meta_msgs/TopoList`

A batch of topological graphs, used to pass multiple candidate topologies at once (e.g. from generator to optimizer).

| Field | Type | Description |
|---|---|---|
| `graphs` | `meta_msgs/TopologicalGraph[]` | Array of topological graph candidates |

---

## Service Definitions

### `meta_msgs/Topo2Angles`

Convert a topological graph to a joint angle vector.

| Direction | Field | Type | Description |
|---|---|---|---|
| Request | `graph_data` | `meta_msgs/TopologicalGraph` | Input topology |
| Response | `angles` | `float32[]` | 17-element joint angle vector (radians) |

**Used by:** `optimizer` (calls), `kinematics_interpreter` (provides)

---

### `meta_msgs/Angles2Topo`

Convert a joint angle vector to a topological graph.

| Direction | Field | Type | Description |
|---|---|---|---|
| Request | `angles` | `float32[]` | Joint angle vector |
| Response | `graph_data` | `meta_msgs/TopologicalGraph` | Resulting topology with full state capture |

**Used by:** `graph_modeler` (provides)

---

### `meta_msgs/TransPlanner`

Request a transition trajectory between configurations.

| Direction | Field | Type | Description |
|---|---|---|---|
| Request | `target` | `float32[]` | Target configuration |
| Request | `steps` | `uint8` | Number of interpolation steps |
| Response | `trajectory` | `std_msgs/Float32MultiArray` | Interpolated trajectory |

---

### `meta_msgs/GetStructure`

Query the robot's current structural description string.

| Direction | Field | Type | Description |
|---|---|---|---|
| Request | *(empty)* | — | — |
| Response | `structure` | `string` | Structural description |

---

### `meta_msgs/IsSim`

Query whether the system is running in simulation mode.

| Direction | Field | Type | Description |
|---|---|---|---|
| Request | *(empty)* | — | — |
| Response | `flag_sim` | `bool` | `true` if simulation, `false` if real hardware |

---

### `meta_msgs/GetID`

Query the robot's hardware ID.

| Direction | Field | Type | Description |
|---|---|---|---|
| Request | *(empty)* | — | — |
| Response | `id` | `uint8` | Robot hardware identifier |

---

### `meta_msgs/GetCurrTopo`

Query the robot's current topological graph.

| Direction | Field | Type | Description |
|---|---|---|---|
| Request | *(empty)* | — | — |
| Response | `curr_topo` | `meta_msgs/TopologicalGraph` | Current robot topology |

---

### `meta_msgs/GetCommon`

Query common robot metadata in a single call.

| Direction | Field | Type | Description |
|---|---|---|---|
| Request | *(empty)* | — | — |
| Response | `id` | `uint8` | Robot hardware identifier |
| Response | `is_sim` | `bool` | Simulation flag |
| Response | `structure` | `string` | Structural description |
