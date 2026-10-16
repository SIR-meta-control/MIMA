# motor_interface

[🏠 MIMA](../../README.md) · [📚 Documentation](../../docs/README.md)

## 📋 Table of Contents

- [🏠 Overview](#overview)
- [🧩 Dependencies](#dependencies)
- [📚 Installation](#installation)
- [🚀 Usage](#usage)
- [🤖 Nodes](#nodes)

<a id="overview"></a>

## 🏠 Overview

`motor_interface` provides low-level ROS1 motor communication and control for two actuator stacks:

- **Dynamixel stack** (`dynamixel_control`, `dynamixel_msgs`, `dynamixel_sdk`)
- **LK stack** (`lk_control`, `lk_msgs`)

It exposes ROS services/topics for commanding motors, reading state/position feedback, and publishing runtime telemetry over serial links.

<a id="dependencies"></a>

## 🧩 Dependencies

- **ROS1 / Catkin**
  - `catkin`
  - `roscpp`
  - `rospy`
  - `std_msgs`
  - `geometry_msgs`
  - `message_generation`
  - `serial`
- **ROS packages in this module**
  - `dynamixel_control`
  - `dynamixel_msgs`
  - `dynamixel_sdk`
  - `lk_control`
  - `lk_msgs`
- **System libraries**
  - `yaml-cpp` (used by `dynamixel_control` and `lk_control`)
- **Runtime requirements**
  - Access to configured serial devices (for example `/dev/ttyTHS0`, `/dev/ttyCH341USB0`)

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

> [!NOTE]
> If serial support is missing on your system, install ROS serial dependencies (distribution-specific package names).

<a id="usage"></a>

## 🚀 Usage

Launch Dynamixel interface:

```bash
roslaunch dynamixel_control dynamixel_control.launch
```

Launch LK interface:

```bash
roslaunch lk_control lk.launch
```

You may also run nodes directly:

```bash
rosrun dynamixel_control dynamixel_control_node
rosrun lk_control lk_node
```

<a id="nodes"></a>

## 🤖 Nodes

### `dynamixel_control_node` (`dynamixel_control/src/dynamixel_control_node.cpp`)

- **Subscribed Topics**
  - `/dynamixel_control/sync_write` (`dynamixel_msgs/SetParam`)
  - `/dynamixel_control/torque_enable` (`std_msgs/Bool`)
- **Published Topics**
  - `/dynamixel_control/state` (`dynamixel_msgs/State`)
  - `/dynamixel_control/log` (`dynamixel_msgs/LogData`)
- **Services (provided)**
  - `/dynamixel_control/sync_read` (`dynamixel_msgs/GetParam`)
  - `/dynamixel_control/ping` (`dynamixel_msgs/Ping`)
  - `/dynamixel_control/reboot` (`dynamixel_msgs/Reboot`)
  - `/dynamixel_control/pos` (`dynamixel_msgs/GetPos`)
- **Services (called)**
  - None
- **Parameters**
  - `/dynamixel_control/dyn_yaml_path` (default in launch: `$(find dynamixel_control)/config/dynamixel.yaml`)
  - YAML config defaults (`dynamixel.yaml`):
    - `port`: `/dev/ttyTHS0`
    - `baudrate`: `115200`
    - `timeout`: `100`
    - `looprate`: `3`
    - `log`: `false`
    - `feedback`: `true`
    - `torque`: `true`
    - `address`: `[64, 116, 126, 128, 132, 144, 146]`

### `lk_node` (`lk_control/src/node.cpp`)

- **Subscribed Topics**
  - None
- **Published Topics**
  - `/lk/feedback` (`lk_msgs/LogUI`)
- **Services (provided)**
  - `/lk/command` (`lk_msgs/Command`)
  - `/lk/cmd_vel` (`lk_msgs/CmdVel`)
  - `/lk/brdcst_vel` (`lk_msgs/BrdcstVel`)
  - `/lk/read_state1` (`lk_msgs/State1`)
  - `/lk/read_state2` (`lk_msgs/State2`)
  - `/lk/b_read1` (`lk_msgs/BrdcstState1`)
  - `/lk/b_read2` (`lk_msgs/BrdcstState2`)
- **Services (called)**
  - None
- **Parameters**
  - `/lk_control/lk_yaml_path` (default in launch: `$(find lk_control)/config/lk.yaml`)
  - YAML config defaults (`lk.yaml`):
    - `port`: `/dev/ttyCH341USB0`
    - `baudrate`: `57600`
    - `timeout`: `100`
    - `looprate`: `10`
