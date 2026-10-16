# test_dyn

[🏠 MIMA](../../README.md) · [📚 Documentation](../../docs/README.md)

## 📋 Table of Contents

- [🏠 Overview](#overview)
- [🧩 Dependencies](#dependencies)
- [📚 Installation](#installation)
- [🚀 Usage](#usage)
- [🤖 Nodes](#nodes)
- [📝 Integration notes](#integration-notes)

<a id="overview"></a>

## 🏠 Overview

`test_dyn` is a small **test / stub** ROS1 package that mimics the Dynamixel **encoder read** interface used by the real robot stack. It advertises the same service name as `dynamixel_control` (`/dynamixel_control/pos`) and implements `dynamixel_msgs/GetPos`, returning a **fixed 17-element list** of raw encoder ticks (Dynamixel-style `uint32` values).

Use it when you want to exercise nodes that call `GetPos` (e.g. `optimizer`, `trans_planner` in hardware mode, or other tooling) **without** physical motors or the full `dynamixel_control` node.

> [!IMPORTANT]
> Only one node can provide `/dynamixel_control/pos` in a single ROS master. Do **not** run this stub at the same time as the real `dynamixel_control` node, or you will get a service name conflict.

<a id="dependencies"></a>

## 🧩 Dependencies

- **ROS1 / Catkin**
  - `catkin`, `rospy`
- **ROS messages**
  - `dynamixel_msgs` (for `dynamixel_msgs/GetPos` — defined in `motor_interface` / `dynamixel_msgs`)

> [!NOTE]
> `package.xml` lists `meta_msgs` / `std_msgs` / `roscpp` but the script only imports `rospy` and `dynamixel_msgs`. Ensure `dynamixel_msgs` is built and sourced before running.

<a id="installation"></a>

## 📚 Installation

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/SIR-meta-control/MIMA.git
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

<a id="usage"></a>

## 🚀 Usage

`message_saver.py` records the generator, optimizer and motor-control topics.
Its `~output_dir` defaults to `$ROS_HOME/mima/runs` (or `~/.ros/mima/runs` when
`ROS_HOME` is unset), so installed nodes do not write into their installation
prefix. Set this private parameter to choose another recording directory.

Launch the stub node:

```bash
roslaunch test_dyn pos.launch
```

Or run directly:

```bash
rosrun test_dyn test_dyn_srv.py
```

Verify the service:

```bash
rosservice call /dynamixel_control/pos
```

You should receive 17 encoder values (see **Fixed mock values** below).

<a id="nodes"></a>

## 🤖 Nodes

### `test_dyn_node` (`scripts/test_dyn_srv.py`)

- **Subscribed Topics**
  - None
- **Published Topics**
  - None
- **Services (provided)**
  - `/dynamixel_control/pos` (`dynamixel_msgs/GetPos`)
    - **Request:** empty
    - **Response:** `pos` (`uint32[]`) — length **17**, fixed mock encoder readings (same order as the rest of the Crimson stack: one value per controlled joint)
- **Services (called)**
  - None
- **Parameters**
  - None (values are hard-coded in the script)

### Fixed mock values

The service always returns:

```text
[1024, 1024, 1536, 1024, 1536, 2560, 3123, 4000, 1536, 3123, 4000, 1536, 3123, 4000, 2560, 3123, 4000]
```

To simulate different poses, edit `test_positions` in `scripts/test_dyn_srv.py`.

<a id="integration-notes"></a>

## 📝 Integration notes

- Downstream code often converts these ticks to radians via `real2sim` (see `optimizer` / `trans_planner` utilities). The numeric list is only meaningful relative to that mapping and your motor calibration.
- For full hardware tests, replace this package with `dynamixel_control` and real servos.
