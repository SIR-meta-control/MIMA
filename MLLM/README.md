# MLLM Multimodal Inference

This repository provides multimodal inference scripts built on top of InternVL-style chat models. The main script supports joint reasoning with:

- RGB image
- Depth image
- LiDAR point cloud
- LiDAR map point cloud
- Odometry vector
- Natural language instruction

The default end-to-end entrypoint is infer.py.

## Installation

Please follow:

- [INSTALLATION.md](./INSTALLATION.md)

## Quick Start

### 1. Prepare model path

Place your model weights in a local directory and pass that directory via --model-path.


### 2. Prepare input files

You need the following inputs:

- One RGB image file, for example .jpg or .png
- One depth image file aligned with the RGB frame
- One LiDAR point cloud file in .ply format
- One map point cloud file in .ply format
- One odometry vector in comma-separated float format


### 3. Run inference

```bash
python infer.py \
  --model-path /path/to/your/model \
  --instruction "Describe the scene and infer the task type." \
  --image-path /path/to/rgb.png \
  --depth-path /path/to/depth.png \
  --point-cloud-path /path/to/current_cloud.ply \
  --map-point-cloud-path /path/to/map_cloud.ply \
  --odometry "0.1,0.2,0.3,0.0,0.0,1.57" \
  --max-num 6 \
  --max-new-tokens 2048
```

### 4. (Optional) Use audio2text for instruction

You can use any off-the-shelf audio2text (ASR) module to transcribe speech into text,
then pass the transcript to `--instruction`.

Typical flow:

1. Convert audio to text with an existing ASR module.
2. Use the transcribed text as the instruction input for `infer.py`.


## Command Line Arguments

infer.py supports the following options:

- --model-path: Local model directory path
- --instruction: Task instruction in natural language
- `--instruction` can also be text produced by an audio2text module
- --image-path: RGB image path
- --depth-path: Depth image path
- --point-cloud-path: Current LiDAR point cloud path (.ply)
- --map-point-cloud-path: Map LiDAR point cloud path (.ply)
- --odometry: Comma-separated odometry vector
- --max-num: Maximum number of dynamic image tiles
- --max-new-tokens: Maximum generated tokens

## Input Requirements

### RGB and depth images

- Recommended to be spatially aligned
- Any format supported by PIL can be used
- The script converts them to RGB and applies normalization

### Point cloud files

- File format must be .ply
- At least 3 columns are required for xyz
- The script samples NPOINTS = 2048 for each point cloud

### Odometry

- Use comma-separated float values
- Dimension should match model configuration, typically 6

## Runtime Notes

- CUDA is required by the current model chat implementation in this repository
- If the GPU supports bfloat16, bfloat16 is used; otherwise float16 is used
