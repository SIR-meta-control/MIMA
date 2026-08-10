# trans_planner

## Overview

`trans_planner` is a ROS1 package that plans smooth joint-space trajectories between the robot’s current configuration and a target 17-DOF pose. It exposes:

- A **`trans_planner` service** (`meta_msgs/TransPlanner`) that returns an interpolated trajectory from the current state to a requested target.
- A **`motor_ctrl` subscriber** that reacts to optimal joint commands (e.g. from `kinematics_interpreter`) and executes the planned trajectory step-by-step on either **simulation** or **hardware**.

In **simulation** mode it queries `get_angles` (`crimson_sim`) for the current pose and publishes actuator commands on `planned_sim_actuator`. In **real** mode it reads joint positions via `/dynamixel_control/pos` (`dynamixel_msgs/GetPos`), converts with `real2sim`, and sends goal positions via `/dynamixel_control/sync_write` (`dynamixel_msgs/SetParam`).

Trajectory generation (`interpolation.py`) uses linear interpolation and optional **axis-alignment** segments for the torso joints so that constraints \(\phi_1 = \phi_2\) and \(\phi + \phi_1 + \phi_2 = 0\) are respected when needed.

## Dependencies

- **ROS1 / Catkin** (from `package.xml` / `CMakeLists.txt`)
  - `catkin`
  - `roscpp`, `rospy`
  - `meta_msgs`
- **Used at runtime by scripts** (ensure these packages are in the workspace / installed)
  - `std_msgs` (`Float32MultiArray`)
  - `dynamixel_msgs` (`GetPos`, `SetParam`)
  - `crimson_sim` (service `get_angles` — simulation only)
- **Python**
  - `numpy`

> **Note:** `std_msgs`, `dynamixel_msgs`, and `crimson_sim` are required to run `planner.py` but are not listed in `package.xml`; add `exec_depend` entries if you want the dependency graph to be complete.

## Installation

Build in a standard catkin workspace:

```bash
cd ~/catkin_ws/src
git clone <your_repo_url>
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Usage

**Simulation** (sets `flag_sim:=true`):

```bash
roslaunch trans_planner sim.launch
```

**Real hardware** (sets `flag_sim:=false`, `step_time:=0.05`):

```bash
roslaunch trans_planner real.launch
```

Or run the node directly (defaults: `flag_sim=true`, `step_time=0.05`):

```bash
rosrun trans_planner planner.py
```

**Prerequisites:**

- Simulation: `crimson_sim` node providing service `get_angles`.
- Real: `dynamixel_control` running with `/dynamixel_control/pos` available.

## Nodes

### `trans_planner` (`scripts/planner.py`)

- **Subscribed Topics**
  - `motor_ctrl` (`std_msgs/Float32MultiArray`) — target joint vector; triggers planning with a fixed `steps=100` and sequential command execution
- **Published Topics**
  - **If `~flag_sim` is `true`:** `planned_sim_actuator` (`std_msgs/Float32MultiArray`) — one row of the planned trajectory per message (sim actuator space; length may differ from 17 when simulation strips dependent joints — see `Interpolation` in `interpolation.py`)
  - **If `~flag_sim` is `false`:** `/dynamixel_control/sync_write` (`dynamixel_msgs/SetParam`) — `paramType=1` (goal position), motor IDs `1..17`, `params` from `sim2real(command)`
- **Services (provided)**
  - `trans_planner` (`meta_msgs/TransPlanner`)
    - **Request:** `target` (`float32[]`) — goal 17-DOF joint vector; `steps` (`uint8`) — number of interpolation samples
    - **Response:** `trajectory` (`std_msgs/Float32MultiArray`) — stacked trajectory (layout: flattened sequence of per-timestep joint vectors; same row layout as `Interpolation.gen_trajectory()` output)
- **Services (called)**
  - **If `~flag_sim` is `true`:** `get_angles` (`crimson_sim/get_angles`) — returns current `angles` for planning start state
  - **If `~flag_sim` is `false`:** `/dynamixel_control/pos` (`dynamixel_msgs/GetPos`) — reads `pos`, converted with `real2sim` for planning start state
- **Parameters**
  - `flag_sim` (default: `true`) — `true`: simulation publishers + `get_angles`; `false`: Dynamixel sync_write + `GetPos`
  - `step_time` (default: `0.05`) — sleep duration (seconds) between consecutive commands when replaying a trajectory from the `motor_ctrl` callback

## Repeated transformation experiment (`experiment.py`)

The experiment runner repeatedly executes the configured transformation
sequence. The default sequence traverses `c1` through `c9` and then returns
through the intermediate configurations to `c1`. Each repetition is recorded
in a separate rosbag. Interactive execution can pause for confirmation between
repetitions; unattended execution disables this pause.

### Configuration representation

The `c1` through `c9` entries in `config/experiment.yaml` contain 17 raw
Dynamixel encoder counts, ordered by motor ID `1..17` and bounded by
`0..4095`. The runner converts each configuration to the simulation joint-angle
space with `real2sim`, applies the torso-axis-constrained interpolation from
`interpolation.py`, converts each frame back with `sim2real`, and publishes the
result through `/dynamixel_control/sync_write` with `paramType=1`
(`GoalPosition`). This is the same hardware command path used by `planner.py`.

### Experiment markers

The runner publishes two `crimson_msgs/Trans` marker topics so that command
intervals can be recovered from the rosbag:

- `/crimson_control/transform`: published immediately before a transformation;
  `cfg` stores the zero-based sequence step and `mode` stores the configuration
  index.
- `/crimson_control/transformed`: published after the interpolated command
  frames have been sent.

### Recorded data

Each repetition produces
`<bag_prefix>_<repetition>_<timestamp>.bag` containing
`/crimson_control/transform`, `/crimson_control/transformed`, and
`/dynamixel_control/log`.

The `/dynamixel_control/log` message (`dynamixel_msgs/LogData`) records current
`I` in amperes, input voltage `U` in volts, and joint position `P` as raw
Dynamixel encoder counts. The position field is read from the Dynamixel
`PresentPosition` register at address 132.

### Interactive execution

```bash
rosrun trans_planner run_experiment.sh
# Use an already running Dynamixel driver:
START_DYNAMIXEL=0 rosrun trans_planner run_experiment.sh
```

### Unattended execution

```bash
roslaunch trans_planner experiment.launch
# Use an already running Dynamixel driver:
roslaunch trans_planner experiment.launch start_dynamixel:=false
```

### Experiment parameters (`config/experiment.yaml`)

- `configs`: measured 17-dimensional encoder-count vectors for `c1..c9`.
- `sequence`: ordered configuration traversal.
- `steps` and `step_time`: interpolation-frame count and frame period in
  seconds.
- `settle_time`: hold duration after each transformation, in seconds.
- `num_experiments`: number of repeated experiments; the default is 10.
- `bag_dir`, `bag_prefix`, and `bag_topics`: rosbag output settings.
