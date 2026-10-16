# GVAE Generator Service

[🏠 MIMA](../../README.md) · [📚 Documentation](../../docs/README.md)

This ROS 1 package integrates the current constraint-aware GVAE structure
generator into MIMA. Model inference runs in a standalone HTTP service, while a
lightweight ROS adapter translates requirement-vector topics into service calls
and publishes valid candidates as `meta_msgs/TopoList`.

PyTorch inference, graph imputation, constraint checks, and candidate ranking
run in the model service. The ROS adapter uses `rospy`, `std_msgs`, `meta_msgs`,
and Python's standard HTTP library.

## 📋 Table of Contents

- [🏗️ Runtime Architecture](#runtime-architecture)
- [📦 Included Model](#included-model)
- [📐 Requirement-Vector Mapping](#requirement-vector-mapping)
- [🧩 Dependencies](#dependencies)
- [🚀 Start the Model Service](#start-the-model-service)
- [🔌 HTTP API](#http-api)
- [🤖 Start the ROS Adapter](#start-the-ros-adapter)
- [🔌 ROS Interface](#ros-interface)
- [🚀 Example request](#focused-validation)

<a id="runtime-architecture"></a>

## 🏗️ Runtime Architecture

```text
MIMA requirement vector topic
  [wp, hp, dp, hs, fl, fi, fp]
                 |
                 |
                 v
GVAE request [dp, wp, hp, load, inspect, pack]
                 |
                 | POST /generate
                 v
         GVAE HTTP service
  CVAE -> Scale GNN -> graph imputation
  -> hard constraints -> score -> diversity
                 |
                 | JSON Top-K candidates
                 v
          ROS adapter node
                 |
                 v
       /generated_topolist
```

The service also accepts the native six-dimensional GVAE vector directly.

<a id="included-model"></a>

## 📦 Included Model

The package contains:

```text
generation/
  gvae/                         current GVAE Python package
  graph_imputation.yaml         deterministic graph transforms
  runtime.py                    persistent Top-K inference engine
  service_contract.py           request/response validation and vector mapping
  service_client.py             standard-library HTTP client
scripts/
  generator_service.py          HTTP model server
  robot_config_generator.py     ROS-to-HTTP adapter
```

Selected model states are provided under
`experiments/module_level_ablation/checkpoints/`: use
`selected/cvae_gnn/best_model.pt` for the generator and
`bar_classifier/bar_classifier.pt` for bar-type prediction. The commands below
pass these paths explicitly.

The full generator predicts eight node poses and three leg angles. Scale is
estimated by the topology-aware GNN. Edges and leg bases are recovered through
the fixed YAML transforms, and only candidates passing all quaternion, spacing,
bar geometry, angle, size, and task checks are returned.

<a id="requirement-vector-mapping"></a>

## 📐 Requirement-Vector Mapping

MIMA supplies:

```text
[wp_m, hp_m, dp_m, hs_m, fl, fi, fp]
```

GVAE consumes:

```text
[x, y, z, load, inspect, pack]
```

The fixed adapter is:

```text
x       = dp_m
y       = wp_m
z       = hp_m
load    = fl
inspect = fi
pack    = fp
```

<a id="dependencies"></a>

## 🧩 Dependencies

Model-service environment:

- Python 3.10 or newer
- PyTorch
- NumPy

ROS adapter environment:

- ROS 1 and catkin
- `rospy`
- `std_msgs`
- `meta_msgs`

Build the ROS workspace normally:

```bash
rosdep install --from-paths src --ignore-src -r -y
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

<a id="start-the-model-service"></a>

## 🚀 Start the Model Service

Activate the Python environment containing PyTorch and run:

```bash
conda activate gvae
python3 src/generator/scripts/generator_service.py \
  --host 127.0.0.1 \
  --port 8091 \
  --model experiments/module_level_ablation/checkpoints/selected/cvae_gnn/best_model.pt \
  --bar-classifier experiments/module_level_ablation/checkpoints/bar_classifier/bar_classifier.pt \
  --device auto
```

To specify the checkpoint and graph-imputation file explicitly:

```bash
python3 src/generator/scripts/generator_service.py \
  --model experiments/module_level_ablation/checkpoints/selected/cvae_gnn/best_model.pt \
  --bar-classifier experiments/module_level_ablation/checkpoints/bar_classifier/bar_classifier.pt \
  --graph-imputation src/generator/generation/graph_imputation.yaml \
  --device cuda
```

Check readiness:

```bash
curl http://127.0.0.1:8091/health
```

The model is loaded once at startup. Requests are serialized around inference
to preserve deterministic seed behavior and avoid concurrent mutation of the
PyTorch random-number state.

`--model` and `--bar-classifier` are required. The commands above point to the
checkpoints included in the checkout; an installed ROS workspace can use
explicit paths to the same artifacts on the model-service host.

<a id="http-api"></a>

## 🔌 HTTP API

### `GET /health`

Returns service status, model family, scale mode, artifact names, device, and
uptime.

### `POST /generate`

Native GVAE request:

```json
{
  "vreq": [0.8, 0.6, 0.5, 0, 0, 0],
  "bar_types": "auto",
  "samples_per_bar": 64,
  "top_k": 10,
  "temperature": 1.0,
  "diversity_threshold": 0.02,
  "min_per_bar": 1,
  "seed": 7
}
```

Named MIMA request is also accepted:

```json
{
  "v_r": {
    "wp_m": 0.6,
    "hp_m": 0.5,
    "dp_m": 0.8,
    "hs_m": 0.2,
    "fl": 0,
    "fi": 0,
    "fp": 0
  },
  "samples_per_bar": 64,
  "top_k": 10,
  "seed": 7
}
```

The response contains bar probabilities, sampling and rejection summaries,
hard-constraint thresholds, inference time, and ranked candidates. Each
candidate includes full `nodes`, 8-column `edges` (angle plus pose), `scale`,
`leg_base`, `leg_angle`, scores, and constraint residuals.

> [!NOTE]
> Invalid input returns HTTP 400. Model failures return HTTP 500. A valid request
> may return an empty candidate list when no sample passes every hard constraint.

<a id="start-the-ros-adapter"></a>

## 🤖 Start the ROS Adapter

Start the service first, then source the ROS workspace and launch only the
adapter:

```bash
roslaunch generator generator_client.launch \
  generator_service_url:=http://127.0.0.1:8091 \
  requirement_vector_topic:=/requirement/vector \
  generated_configs_topic:=/generated_topolist
```

For a configured robot, start the Dynamixel driver first, then launch the
generation-to-control nodes:

```bash
roslaunch generator main.launch \
  generator_service_url:=http://127.0.0.1:8091
```

> [!IMPORTANT]
> The driver must provide `/dynamixel_control/pos` and consume
> `/dynamixel_control/sync_write`. The optimizer and planner read the measured
> joint positions from this service. `main.launch` does not start the fixed-pose
> test service. For isolated interface tests, that service is available through
> `roslaunch test_dyn pos.launch`.

Publish a 6-D or 7-D `std_msgs/Float32MultiArray` on the configured input topic.

<a id="ros-interface"></a>

## 🔌 ROS Interface

| Direction | Name | Type |
| --- | --- | --- |
| Subscribe | `~requirement_vector_topic` | `std_msgs/Float32MultiArray` |
| Publish | `~generated_configs_topic` | `meta_msgs/TopoList` |

Private parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `service_url` | `http://127.0.0.1:8091` | GVAE HTTP service base URL |
| `service_timeout_s` | `120.0` | Per-request HTTP timeout |
| `samples_per_bar` | `64` | Latent samples generated for each selected bar type |
| `top_k` | `10` | Maximum valid candidates published |
| `bar_types` | `auto` | `auto`, `all`, or comma-separated bar names |
| `temperature` | `1.0` | Conditional-prior sampling temperature |
| `diversity_threshold` | `0.02` | Minimum normalized distance during Top-K selection |
| `min_per_bar` | `1` | Minimum retained candidate per sampled bar type when possible |
| `seed` | `7` | Reproducible service sampling seed |
| `requirement_vector_topic` | `/requirement/vector` | Input topic |
| `generated_configs_topic` | `/generated_topolist` | Output topic |

The HTTP response preserves each edge as
`[angle, x, y, z, qw, qx, qy, qz]`. The ROS adapter publishes
`[x, y, z, qw, qx, qy, qz]` because the current kinematics interpreter runs
with `flag_test=false` and expects seven edge-pose columns.

<a id="focused-validation"></a>

## 🚀 Example request

Start the model service and submit a request:

```bash
curl -X POST http://127.0.0.1:8091/generate \
  -H 'Content-Type: application/json' \
  -d '{"vreq":[0.8,0.6,0.5,0,0,0],"samples_per_bar":2,"top_k":1,"seed":7}'
```
