# generator

ROS 1 package that runs **learned robot-topology generation** in the loop: it subscribes to a compact **detection / requirement vector**, runs a PyTorch model (`RobotConfigurationNet`), and publishes a list of candidate **topological graphs** (nodes, edges, adjacency, and global leg/scale features) as `meta_msgs` messages for downstream planning or visualization.

---

## Overview

- **Role**: Bridge between a perception or task-specification pipeline (8-D `Float32MultiArray`) and the rest of the stack that consumes `meta_msgs/TopoList`.
- **Model**: Variational encoder–decoder with graph convolutional readout, bar-linkage conditioning (`4-bar` / `6-bar` / `8-bar`), and optional confidence scores. At runtime the node calls `forward(..., train=False)` to draw multiple diverse configurations per input message.
- **Artifacts**: Weights are loaded from a PyTorch checkpoint; graph kinematic priors are loaded from `graph_imputation.npy` (path comes from checkpoint `args` or defaults).

This package does **not** implement offline dataset loading, train/validation splitting, or training loops in the ROS nodes—those belong to your external training codebase. The sections below describe **runtime** conditioning and tensor-to-message assembly only.

---

## Dependencies

### ROS

- **ROS 1** with **catkin** (e.g. Melodic or Noetic).
- **Catkin packages** (from `package.xml` / Python imports):
  - `rospy`
  - `std_msgs`
  - `sensor_msgs` (declared in `package.xml`; not required by the Python nodes reviewed here)
  - **`meta_msgs`** — message types `TopologicalGraph`, `TopoList`, `Global` (used by the generator; ensure this package is in your workspace and that `package.xml` lists `meta_msgs` if you rely on `rosdep`).

> **Note:** `CMakeLists.txt` lists `meta_msgs` in `find_package(catkin ...)`, but `package.xml` does not declare it yet. Add `<depend>meta_msgs</depend>` (or split build/exec depends) for a consistent workspace build.

### Python (runtime)

- **PyTorch** (`torch`)
- **NumPy** (`numpy`)
- **PyTorch Geometric** (`torch_geometric`) — `GCNConv`, `Data`, `Batch`, `global_mean_pool`

### Other launch dependencies (optional demo)

`launch/main.launch` **includes** additional packages (`test_dyn`, `optimizer`, `kinematics_interpreter`, `trans_planner`). Those are only required if you run that full launch file as-is.

---

## Installation

1. Place the package under your Catkin workspace source tree, for example:

   `catkin_ws/src/generator`

2. Ensure `meta_msgs` (and any packages from your demo launch) are present in the same workspace.

3. Build:

   ```bash
   cd ~/catkin_ws
   catkin_make   # or catkin build
   source devel/setup.bash
   ```

4. **Python environment**: The repository scripts use machine-specific shebangs (e.g. a fixed Conda path). For portability, run nodes with the interpreter that has PyTorch and PyTorch Geometric installed, or adjust shebangs / use `rosrun` after `chmod +x` on the scripts.

5. **Checkpoint and graph file**: Place your trained `*.pt` checkpoint where your parameters point (see `launch/main.launch` for an example). The checkpoint may embed `args` including `graph_imputation_path`; if that path is relative, it is resolved under `generator/generation/`.

---

## Usage

### Launch file (full stack example)

```bash
roslaunch generator main.launch
```

This launch file starts `generator_node`, a delayed `manual_publisher`, and several **external** nodes from other packages. Inspect `launch/main.launch` and trim includes if you only need the generator.

The `<rosparam param="detection_vector">...</rosparam>` entry in that file is **not** read by `generator_node`; the demo supplies input by publishing on `detection_vector_topic` (e.g. via `manual_publisher`).

### Run the generator node directly

```bash
rosrun generator main.py
```

Set private parameters (see **Nodes** → `generator_node`) for `model_path`, topics, and `num_configs`.

### Manual test publisher

```bash
rosrun generator manual_publisher.py
```

Publishes a single hard-coded `Float32MultiArray` on `/detection/vector/manually` (see script for default values).

### Alternate entry point

`scripts/robot_config_generator.py` can also be run as a standalone node (`robot_config_generator`) with the same `RobotConfigGenerator` logic as `node/main.py`.

---

## Data conditioning, sampling, and message assembly

### Input vector (`Float32MultiArray`)

- The subscriber receives a **flat list of floats** copied into `vreq` and wrapped as a batch of shape `(1, 8)`.
- In **training code** (`generation/model.py`, `determine_bar_from_vreq`), indices **5–7** are interpreted as task flags (inspect / load / pack). The **ROS inference path** uses `train=False`, where each candidate configuration samples a **random** bar type (`4-bar`, `6-bar`, `8-bar`) per draw; the full vector still conditions the VAE-style encoders.

### “Preprocessing” inside the node

1. **Load checkpoint** → restore `model_state_dict` and optional `args` (`batch_size`, `graph_imputation_path`, etc.).
2. **Load graph imputation** (`.npy`) → tensors `T_node_leg`, `T_node_edge`, `S_edge_spacing` used for edge/leg frame propagation inside the network.
3. **Forward** with `train=False` → `num_configs` stochastic samples (manual seed from time + small Gaussian jitter on latents).
4. **Post-process tensors** → each sample is converted to `meta_msgs/TopologicalGraph`:
   - `nodes` / `edges`: `Float32MultiArray` with layout **8×7** (position + quaternion per row).
   - `adjacency`: float list encoding linkage pattern (depends on inferred `bar` string).
   - `feature` (`meta_msgs/Global`): `scale`, `leg_angles`, `leg_base` (4×7 layout), `locomotion_mode` set to `0` in the current code.

### Train/validation split

Not implemented in this ROS package. Training uses `forward(..., train=True)` with ground-truth `nodes`, `leg_angle`, and `bar_list` in your external training scripts; dataset splits belong there.

---

## Nodes

### `generator_node` (`node/main.py`)

Main inference node wrapping `RobotConfigGenerator`.

| Direction | Name | Type |
|-----------|------|------|
| **Subscribes** | `~detection_vector_topic` (default `/detection/vector`) | `std_msgs/Float32MultiArray` |
| **Publishes** | `~generated_configs_topic` (default `/generated_topolist`) | `meta_msgs/TopoList` |

**Parameters** (private, `~` namespace):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | `""` | Path to PyTorch checkpoint (**required**; node logs error and aborts setup if empty or missing). |
| `num_configs` | `10` | Number of candidate graphs sampled per input message. |
| `detection_vector_topic` | `/detection/vector` | Input topic name. |
| `generated_configs_topic` | `/generated_topolist` | Output topic name. |

**Services:** none.

---

### `manual_detection_publisher` (`scripts/manual_publisher.py`)

One-shot helper for testing.

| Direction | Name | Type |
|-----------|------|------|
| **Publishes** | `/detection/vector/manually` | `std_msgs/Float32MultiArray` |

**Parameters:** none (topic and payload are hard-coded in the script).

**Services:** none.

---

### `robot_config_generator` (optional; `scripts/robot_config_generator.py` `__main__`)

Same behavior and interface as `generator_node` (same class and parameters). Useful if you install/run the script directly instead of `node/main.py`.

---

## Message summary

- **`meta_msgs/TopoList`**: `TopologicalGraph[] graphs`
- **`meta_msgs/TopologicalGraph`**: `nodes`, `edges`, `adjacency` (`std_msgs/Float32MultiArray`), `feature` (`meta_msgs/Global`)
- **`meta_msgs/Global`**: `float32[] scale`, `leg_base` (`Float32MultiArray`), `float32[] leg_angles`, `uint8 locomotion_mode`

---

## License

See `package.xml` (`license` field is currently a placeholder).
