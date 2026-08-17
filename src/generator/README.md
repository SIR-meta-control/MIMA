# GVAE Generator Service

This ROS 1 package integrates the current constraint-aware GVAE structure
generator into MIMA. Model inference runs in a standalone HTTP service, while a
lightweight ROS adapter translates requirement-vector topics into service calls
and publishes valid candidates as `meta_msgs/TopoList`.

The process boundary is intentional: the model service owns PyTorch, CUDA, the
generator weights, graph imputation, hard constraints, ranking, and diversity;
the ROS process requires only `rospy`, `std_msgs`, `meta_msgs`, and Python's
standard HTTP library.

## Runtime Architecture

```text
MIMA requirement vector topic
  [wp, hp, dp, hs, fl, fi, fp]
                 |
                 | drop hs; reorder dimensions
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

## Included Model

The old `RobotConfigurationNet` implementation and
`graph_imputation.npy` have been retired. The package now contains:

```text
generation/
  gvae/                         current GVAE Python package
  graph_imputation.yaml         deterministic graph transforms
  checkpoints/
    full_generator.pt           CVAE + Scale GNN checkpoint
    bar_classifier.pt           p(bar_type | vreq) checkpoint
  runtime.py                    persistent Top-K inference engine
  service_contract.py           request/response validation and vector mapping
  service_client.py             standard-library HTTP client
scripts/
  generator_service.py          HTTP model server
  robot_config_generator.py     ROS-to-HTTP adapter
```

The full generator predicts eight node poses and three leg angles. Scale is
estimated by the topology-aware GNN. Edges and leg bases are recovered through
the fixed YAML transforms, and only candidates passing all quaternion, spacing,
bar geometry, angle, size, and task checks are returned.

## Requirement-Vector Mapping

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

`hs_m` is deliberately ignored. It is not added to another dimension and does
not modify any model input, threshold, or task flag.

## Dependencies

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

## Start the Model Service

Activate the Python environment containing PyTorch and run:

```bash
conda activate gvae
python3 src/generator/scripts/generator_service.py \
  --host 127.0.0.1 \
  --port 8091 \
  --device auto
```

The default artifacts are resolved relative to this package. Override them only
when testing another checkpoint:

```bash
python3 src/generator/scripts/generator_service.py \
  --model src/generator/generation/checkpoints/full_generator.pt \
  --bar-classifier src/generator/generation/checkpoints/bar_classifier.pt \
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

## HTTP API

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

Named MIMA request is also accepted and uses the same `hs_m`-dropping mapping:

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

Invalid input returns HTTP 400. Model failures return HTTP 500. A valid request
may return an empty candidate list when no sample passes every hard constraint.

## Start the ROS Adapter

Start the service first, then source the ROS workspace and launch only the
adapter:

```bash
roslaunch generator generator_client.launch \
  generator_service_url:=http://127.0.0.1:8091 \
  requirement_vector_topic:=/requirement/vector \
  generated_configs_topic:=/generated_topolist
```

The complete legacy stack can be launched with:

```bash
roslaunch generator main.launch \
  generator_service_url:=http://127.0.0.1:8091
```

No manual publisher is started or installed. A real requirement-vector
producer must publish a 6-D or 7-D `std_msgs/Float32MultiArray` on the configured
input topic.

## ROS Interface

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
`[angle, x, y, z, qw, qx, qy, qz]`. The ROS adapter intentionally publishes
`[x, y, z, qw, qx, qy, qz]` because the current kinematics interpreter runs
with `flag_test=false` and expects seven edge-pose columns.

## Focused Validation

Run the service-contract tests without ROS:

```bash
python3 -m unittest discover -s src/generator/test -v
```

For an end-to-end check, start the model service and submit a small request:

```bash
curl -X POST http://127.0.0.1:8091/generate \
  -H 'Content-Type: application/json' \
  -d '{"vreq":[0.8,0.6,0.5,0,0,0],"samples_per_bar":2,"top_k":1,"seed":7}'
```
