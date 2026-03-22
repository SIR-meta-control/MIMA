This directory contains ROS 1 (Catkin) packages and MLLM used in the work titled **Metamorphous adaptability in robotic systems through intelligent structural evolvement**.

<div align="center">
  <div>
    <img src="figure/MIMA.png" width=75%>
  </div>
</div>
<font color=#a0a0a0 size=2>The Multimodal Intelligent Metamorphosis Architecture (MIMA) for guiding robot metamorphosis to adapt to the environment. a, MIMA processes both cognition of the environment and tasks (perception space) and of a robot structure (metamorphosis space) and establishes the mapping between cognitive spaces through multiple neural-network models. It consists of b, multimodal data inputs block, c, MLLM inference block, and d, geometrical inference block for metamorphosis. The MIMA generates the topologies and optimizes the metamorphosis configuration states.
  </font>

### Components

| Component                    | Description                                      |
| ---------------------------- | ------------------------------------------------ |
| `README.md`                  | The README file of the project                   |
| `bags`                       | The rosbag files of the project                  |
| `runs`                       | The running results of the project               |
| `src`                        | The source code of the project                   |
| `src/meta_msgs`              | Shared message definitions                       |
| `src/generator`              | Topological configuration generation             |
| `src/optimizer`              | Energy evaluation and optimal topology selection |
| `src/kinematics_interpreter` | Graph to joint angle conversion                  |
| `src/graph_modeler`          | Joint angle to graph conversion                  |
| `src/trans_planner`          | Joint angle trajectory interpolation             |
| `src/test_dyn`               | Joint motor interface for testing                |
| `src/motor_interface`        | Joint motor interface for real hardware          |
| `src/crimson_control`        | Implementation layer for the robot               |
| `src/models/crimson/urdf`    | Robot description and simulation assets          |
| `MLLM`                       | The MLLM used in the project                     |

### Third-party software included or relied upon

| Component            | Upstream                                                                         | Location in this tree           |
| -------------------- | -------------------------------------------------------------------------------- | ------------------------------- |
| Point-LIO (HKU-MARS) | [https://github.com/hku-mars/Point-LIO](https://github.com/hku-mars/Point-LIO)   | `point_lio_crimson/` (modified) |
| libb64               | [https://github.com/libb64/libb64](https://github.com/libb64/libb64)             | `third_party/libb64/`           |
| WebSocket++ 0.8.x    | [https://github.com/zaphoyd/websocketpp](https://github.com/zaphoyd/websocketpp) | `third_party/websocketpp/`      |

The `point_lio_crimson` package links against **libb64** headers from `third_party/libb64/`. **WebSocket++** is vendored alongside other code; it is not required to build the `point_lio_crimson` targets as configured in this repository.

---

## 1. System requirements

### Operating system and middleware

- **OS:** Ubuntu **20.04** (64-bit, `x86_64`).
- **ROS:** **ROS Noetic** with **Catkin**.
- **Build tools:** `build-essential`, **CMake** ≥ 2.8.3, **GCC** (as provided for Noetic), **C++14**.

### Software dependencies (version notes)

**ROS packages (representative list — also install transitive dependencies):**

- Core: `ros-noetic-catkin`, `ros-noetic-roscpp`, `ros-noetic-rospy`, `ros-noetic-std-msgs`, `ros-noetic-sensor-msgs`, `ros-noetic-geometry-msgs`, `ros-noetic-nav-msgs`, `ros-noetic-tf`
- Messages / filters: `ros-noetic-message-generation`, `ros-noetic-message-runtime`, `ros-noetic-message-filters`
- visualization: `ros-noetic-pcl-ros`, `ros-noetic-pcl-conversions`, `ros-noetic-cv-bridge`, `ros-noetic-eigen-conversions`, `ros-noetic-rviz`, `ros-noetic-robot-state-publisher`, `ros-noetic-joint-state-publisher-gui`, `ros-noetic-roslaunch`
- planning: `ros-noetic-teb-local-planner`, `ros-noetic-serial`
- Simulation: `ros-noetic-gazebo-ros` and related Gazebo packages as needed
- **LiDAR driver (separate workspace):** [livox_ros_driver](https://github.com/Livox-SDK/livox_ros_driver) — required for Livox-focused workflows; build and `source` its workspace before building or running `point_lio_crimson` (see the driver documentation and [`point_lio_crimson/README.md`](point_lio_crimson/README.md))

**System libraries (many are pulled in via `rosdep` or ROS metapackages):**

- **Eigen3** — `libeigen3-dev`
- **PCL** — `libpcl-dev` (≥ 1.8)
- **OpenCV** — `libopencv-dev`
- **libcurl** — `libcurl4-openssl-dev`
- **ZBar** — `libzbar-dev`
- **yaml-cpp** — `libyaml-cpp-dev` (for `crimson_control`)
- **Python development headers** — `python3-dev` (and `libpython3-dev` on Ubuntu)
- **nlohmann/json** — `nlohmann-json3-dev`
- **OpenMP** — optional; improves parallelism when available (`libomp-dev` / compiler OpenMP support)

**Header-only dependency (matplotlib-cpp):**  
CMake calls `find_path(... matplotlibcpp.h)`. Install the header, for example:

```bash
sudo wget -O /usr/local/include/matplotlibcpp.h https://raw.githubusercontent.com/lava/matplotlib-cpp/master/matplotlibcpp.h
```

(Or clone [matplotlib-cpp](https://github.com/lava/matplotlib-cpp) and add its directory to `CMAKE_INCLUDE_PATH` / your environment.)

**Metapackage recommendation:** Installing **`ros-noetic-desktop-full`** covers most ROS libraries above and Python bindings used by tutorials; a minimal install is possible using `ros-noetic-ros-base` plus the packages listed explicitly.

### Python packages (names only — install commands are in §2)

`rospy` (via ROS deb packages), `numpy`, `matplotlib`, `PyYAML`, `scipy`, `mujoco`

Use the **same Python 3 interpreter** as ROS Noetic (`/usr/bin/python3`). Avoid activating a Conda env that shadows `/opt/ros/noetic` when running `rosrun` / `roslaunch`.

### Versions the software has been tested on

Development and builds are expected on **Ubuntu 20.04** with **ROS Noetic**. **Other Ubuntu versions or ROS 2 have not been validated** for this tree.

### Non-standard hardware

- **LiDAR + IMU:** Required for live operation with a physical sensor suite; message types and timing must match your driver and launch configuration.
- **Dynamixel servos / custom serial hardware:** Referenced by `motor_interface` and `crimson_control` packages when driving the physical robot.
- **NVIDIA GPU with appropriate drivers:** required for the `generator` package.

For **software-only** builds, a normal `x86_64` PC with NVIDIA GPU is sufficient; some launch files will not be meaningful without the corresponding sensors or actuators.

---

## 2. Installation guide

### Layout

Let `$WS` be your Catkin workspace root (e.g. `~/catkin_ws`). Place this repository’s `src` tree at **`$WS/src`** (so packages appear as `$WS/src/generator`, `$WS/src/meta_msgs`, …).

### Steps

1. **Install ROS Noetic** following [http://wiki.ros.org/noetic/Installation/Ubuntu](http://wiki.ros.org/noetic/Installation/Ubuntu). Suggested: `ros-noetic-desktop-full`.

2. **Install system and ROS dependencies** (adjust if you use a minimal ROS install):

   ```bash
   sudo apt update
   sudo apt install -y \
     build-essential cmake git \
     libeigen3-dev libpcl-dev libopencv-dev \
     libcurl4-openssl-dev libzbar-dev \
     libyaml-cpp-dev nlohmann-json3-dev \
     python3-dev libpython3-dev \
     ros-noetic-pcl-ros ros-noetic-pcl-conversions \
     ros-noetic-cv-bridge ros-noetic-eigen-conversions \
     ros-noetic-teb-local-planner ros-noetic-serial \
     ros-noetic-rviz ros-noetic-robot-state-publisher \
     ros-noetic-joint-state-publisher-gui ros-noetic-roslaunch \
     ros-noetic-gazebo-ros
   ```

3. **matplotlibcpp.h**:

   ```bash
   sudo wget -O /usr/local/include/matplotlibcpp.h \
     https://raw.githubusercontent.com/lava/matplotlib-cpp/master/matplotlibcpp.h
   ```

4. **Initialize rosdep** (once per machine):

   ```bash
   sudo rosdep init || true
   rosdep update
   cd $WS
   rosdep install --from-paths src --ignore-src -r -y
   ```

5. **Livox ROS driver** (if you use Livox hardware or corresponding bags): clone and build in a separate workspace, then add to `~/.bashrc`:

   ```bash
   source /path/to/livox_ws/devel/setup.bash
   ```

6. **Build the Catkin workspace:**

   ```bash
   cd $WS
   catkin_make -DCMAKE_BUILD_TYPE=Release
   # or: catkin build
   source $WS/devel/setup.bash
   ```

   Add `source $WS/devel/setup.bash` to `~/.bashrc` if desired.

### Python dependencies

Prefer **APT** for packages available as Ubuntu debs (matches ROS Noetic’s `python3`):

```bash
sudo apt install -y python3-pip python3-numpy python3-matplotlib python3-yaml python3-scipy
```

**MuJoCo Python binding** is not always packaged; install for the **user** so it does not require mixing Conda with ROS:

```bash
pip3 install --user mujoco
```

If `pip3` is missing:

```bash
sudo apt install -y python3-pip
```

**Virtual environments:** ROS Noetic expects system Python and `/opt/ros/noetic` on `PYTHONPATH`. Using a `venv` for ROS nodes is unsupported here; use system/site-packages or `pip3 install --user` as above. Do not run `rosrun`/`roslaunch` under a Conda base env unless you know how to merge `PYTHONPATH` with Noetic.

### Typical install time

On a normal desktop (broadband, SSD), **approximately 90-120 minutes** from a clean Ubuntu 20.04 install through ROS setup, NVIDIA driver installation, dependency installation, and the first full `catkin_make`

---

## 3. Demo

There are two demos available:

- `generation` demo: When a requirement vector is input into the generator, it will generate a list of topological configurations and send it to the optimizer.
  The optimizer will then select the minimum-energy solution and send it to the kinematics interpreter.
  The kinematics interpreter will then convert the topology to a joint angle vector and send it to the motor interface.
  A example result is shown below: `$WS/runs/example_results.json`.
- `visualization` demo: We provide a 30s rosbag file for visualization.
  The rosbag file is uploaded to the Google drive: https://drive.google.com/file/d/1ue5oxmCAtbVviLTJvWY8hgjgFmvbxKsd/view?usp=sharing
  It contains the sensor data, the navigation data and the topological graph signal. You can visualize the data in RViz with the configurationfile `$WS/src/point_lio_crimson/rviz_cfg/indoor.rviz`.

---

## 4. Instructions for use

To start the generation demo, run the following command:

```bash
roslaunch generator main.launch
```

The result will be saved to `$WS/runs/result_{time}.json`.

To start the visualization demo, run the following command:

```bash
rviz -d $WS/src/point_lio_crimson/rviz_cfg/indoor.rviz
rosbag play $WS/bags/test.bag
```

---

## 5. License

**Original work** in this repository (the metamorphous-robotics research code and assets authored for this project, excluding vendored third-party trees listed below) is licensed under the **GNU General Public License v3.0**. The full license text is in [`LICENSE.md`](LICENSE.md).

**Third-party and upstream components** keep their own terms:

| Location                       | License / reference                                                                                                                                                                                   |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/point_lio_crimson/`       | GNU GPL v2 — [`src/point_lio_crimson/LICENSE`](src/point_lio_crimson/LICENSE); nested `include/IKFoM/` — [`src/point_lio_crimson/include/IKFoM/LICENSE`](src/point_lio_crimson/include/IKFoM/LICENSE) |
| `src/third_party/libb64/`      | [`src/third_party/libb64/LICENSE.md`](src/third_party/libb64/LICENSE.md)                                                                                                                              |
| `src/third_party/websocketpp/` | Vendored [WebSocket++](https://github.com/zaphoyd/websocketpp) (BSD 3-Clause; see upstream `COPYING` in that repository)                                                                              |
| `MLLM/`                        | Per-subtree — e.g. [`MLLM/lmdeploy/LICENSE`](MLLM/lmdeploy/LICENSE), [`MLLM/internvl_chat_llava/LICENSE`](MLLM/internvl_chat_llava/LICENSE)                                                           |

When you redistribute or combine binaries, you must comply with **all** applicable licenses (including GPL obligations for GPL-covered parts).

---
