# crimson_control

## Overview

`crimson_control` is the top-level ROS1 control stack for the Crimson reconfigurable quadruped robot. It brings together morphology transformation, legged/omnidirectional motion control, keyboard teleoperation, STM32 serial communication, and sensor integration into a unified system.

The repository is organised as a meta-folder containing four catkin packages:

| Package | Role |
|---|---|
| `crimson_control` | Main controller — morphology transformation, gait/motion execution |
| `crimson_msgs` | Custom message definitions (`Motion`, `Trans`) |
| `joy_stick` | Keyboard teleoperation node |
| `stserial` | Serial bridge to the STM32 relay board |

> **Note:** Motor drivers (`dynamixel_control`, `lk_control`) and their message packages live in the separate `motor_interface` module but are required at runtime.

## Dependencies

- **ROS1 / Catkin**
  - `catkin`, `roscpp`, `rospy`
  - `std_msgs`, `geometry_msgs`, `sensor_msgs`
  - `message_generation` / `message_runtime`
  - `message_filters`
  - `serial`
  - `teb_local_planner`
- **ROS packages (in this repo or motor_interface)**
  - `crimson_msgs`, `dynamixel_msgs`, `dynamixel_control`
  - `lk_msgs`, `lk_control`
- **System libraries**
  - `yaml-cpp`
- **Optional sensor drivers (used in `all.launch` / `crimson.launch`)**
  - `realsense2_camera` (Intel RealSense D-series)
  - `livox_ros_driver` (Livox LiDAR)
  - `point_lio_crimson` (LiDAR-inertial odometry)
  - `thermal_ros_driver` (thermal camera)

## Installation

```bash
cd ~/catkin_ws/src
git clone <your_repo_url>
sudo apt update
sudo apt-get install ros-${ROS_DISTRO}-serial
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Usage

**Launch everything at once (including sensors and SLAM):**

```bash
roslaunch crimson_control crimson.launch
```

**Or launch components in order:**

```bash
roslaunch stserial st.launch
roslaunch dynamixel_control dynamixel_control.launch
roslaunch lk_control lk.launch
roslaunch crimson_control crimson_control.launch
roslaunch joy_stick joy.launch
```

## Nodes

### `crimson_control_node` (`crimson_control/src/node.cpp`)

The central controller. Handles morphology transformation requests and motion commands, drives Dynamixel and LK motors via their respective ROS interfaces, and supports an optional automated execution mode.

- **Subscribed Topics**
  - `/crimson/transform` (`crimson_msgs/Trans`) — triggers a morphology change
  - `/crimson/motion` (`crimson_msgs/Motion`) — sets velocity/gait commands
  - `/crimson/autorun` (`std_msgs/Bool`) — enables/disables automated sequence playback
- **Published Topics**
  - `/crimson/transformed` (`crimson_msgs/Trans`) — confirms the completed transformation state
- **Services (provided)**
  - None
- **Services (called)**
  - `/dynamixel_control/pos` (`dynamixel_msgs/GetPos`) — reads current joint positions
  - `/dynamixel_control/sync_read` (`dynamixel_msgs/GetParam`) — reads Dynamixel registers
- **Parameters**
  - `/crimson/group_yaml_path` (default: `$(find crimson_control)/config/group.yaml`)
  - `/crimson/omni_yaml_path` (default: `$(find crimson_control)/config/motion/omni.yaml`)
  - `/crimson/quad_yaml_path` (default: `$(find crimson_control)/config/motion/quad.yaml`)
  - `/crimson/trans_yaml_path` (default: `$(find crimson_control)/config/trans/trans.yaml`)

---

### `ckey_node` (`joy_stick/src/node.cpp`)

Keyboard teleoperation node. Reads raw keyboard input and publishes transformation and motion commands. Supports entering/exiting morphology-change mode via `Tab`.

- **Subscribed Topics**
  - `/move_base/reach_goal` (`std_msgs/UInt8`) — navigation planner goal reached callback
- **Published Topics**
  - `/crimson/motion` (`crimson_msgs/Motion`) — velocity commands
  - `/crimson/transform` (`crimson_msgs/Trans`) — morphology change commands
  - `/dynamixel_control/sync_write` (`dynamixel_msgs/SetParam`) — direct motor write (e.g. head pitch)
  - `/dynamixel_control/torque_enable` (`std_msgs/Bool`) — motor torque on/off
  - `/omni/disable` (`std_msgs/Bool`) — disable omnidirectional motion controller
  - `/quad/disable` (`std_msgs/Bool`) — disable quadruped motion controller
  - `/crimson/autorun` (`std_msgs/Bool`) — enable/disable auto-run sequence
  - `/track_enable` (`std_msgs/Bool`) — enable tracking mode
- **Services (called)**
  - `/dynamixel_control/pos` (`dynamixel_msgs/GetPos`) — read current joint positions
- **Parameters**
  - `/joy_stick/joy_yaml_path` (default: `$(find joy_stick)/config/joy.yaml`)
  - YAML config defaults (`joy.yaml`):
    - `mode`: `0`
    - `vx`: `1`
    - `stride`: `300`
    - `theta`: `30`

---

### `st_node` (`stserial/src/node.cpp`)

Serial bridge node for communication with the STM32 microcontroller (relay board).

- **Subscribed Topics**
  - `/st/set_relay` (`std_msgs/Bool`) — sends relay on/off command over serial (`"1\r\n"` / `"0\r\n"`)
- **Published Topics**
  - None
- **Services**
  - None
- **Parameters**
  - `/stserial/st_yaml_path` (default: `$(find stserial)/config/st.yaml`)
  - YAML config defaults (`st.yaml`):
    - `port`: `/dev/ttyCH341USB1`
    - `baudrate`: `115200`
    - `timeout`: `100`
    - `looprate`: `50`

---

## Notes

- **Do not use multi-threading in `dynamixel_control`.**
- The `all.launch` and `crimson.launch` files additionally start external sensor nodes (`realsense2_camera`, `livox_ros_driver`, `point_lio_crimson`, `thermal_ros_driver`). These packages must be installed separately and are not part of this repository.
