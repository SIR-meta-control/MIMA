# MIMA Requirement-Vector Inference

[🏠 MIMA](../../README.md) · [📚 Documentation](../../docs/README.md)

This directory contains the paper-specific interfaces and conventional
baselines used to infer the seven-dimensional MIMA requirement vector

```text
v_r = [w_p, h_p, d_p, h_s, f_l, f_i, f_p].
```

It complements the generic InternVL implementation in the parent directory.
The code here records the inference contract used by Full-MIMA,
MLLM-distilled, and the RF/DT/GBT replacement ablations.

## 📋 Table of Contents

- [📦 Inference methods](#inference-methods)
- [📁 Layout](#layout)
- [📚 Installation](#installation)
- [🧠 Full-MIMA teacher and MLLM-distilled services](#full-mima-teacher-and-mllm-distilled-services)
- [🚀 RF, DT, and GBT inference](#rf-dt-and-gbt-inference)
- [🚆 Baseline training](#baseline-training)
- [✅ Verification](#verification)

<a id="release-scope"></a>
<a id="inference-methods"></a>

## 📦 Inference methods

| Method | Interface and assets |
| --- | --- |
| Full-MIMA teacher | `/predict` client, response parser and batch inference cache |
| MLLM-distilled | ET32 service client, deployment metadata and model checksums |
| MLLM -> RF/DT/GBT | Deterministic 16-feature adapter, fitted weights, training and inference code |

RF, DT, and GBT run locally using a 16-feature sensor adapter. MLLM-distilled
uses an 85-dimensional sensor summary. Its implementation, teacher preprocessing
and service programs are provided in [teacher_student](../teacher_student/README.md).
The teacher and student weights are excluded because of third-party rights
restrictions; service deployment requires authorized local weights.

<a id="layout"></a>

## 📁 Layout

```text
mima_requirement_vector/
  mima_vr/                 shared schema, service client, and baseline code
  scripts/                 service cache, local inference, and baseline training
  models/baselines/        released RF, DT, and GBT weights
  models/teacher_only_et32 ET32 model metadata and checksums
  MODEL_CARD.md            method definitions and available assets
```

<a id="installation"></a>

## 📚 Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

<a id="full-mima-teacher-and-mllm-distilled-services"></a>

## 🧠 Full-MIMA teacher and MLLM-distilled services

Both services use the same JSON request contract. `backend=internvl` selects a
Full-MIMA-compatible teacher; `backend=student` selects the distilled backend.
Sensor file paths must be absolute paths visible to the service host.

```bash
python scripts/predict_service.py \
  --service-url http://SERVICE_HOST:PORT \
  --backend student \
  --expected-model strict_student_teacher_only_et32_s32768 \
  --rgb /data/sample/rgb.png \
  --depth /data/sample/depth.npy \
  --point-cloud /data/sample/point_cloud.npy \
  --sample-id sample_001 \
  --scenario tunnel
```

Use `--backend internvl` for the Full-MIMA teacher. The parser requires all
seven output fields to be finite JSON numbers. Missing fields, numeric strings,
NaN, and infinity are rejected.

For a manifest-indexed dataset:

```bash
python scripts/cache_service_predictions.py \
  --dataset-dir /data/evaluation_dataset \
  --output-dir /data/output_cache \
  --service-url http://SERVICE_HOST:PORT \
  --backend student \
  --expected-model strict_student_teacher_only_et32_s32768 \
  --sample-ids-file /data/evaluation_dataset/sample_ids.txt \
  --no-resume
```

The dataset manifest is JSON Lines and must provide `sample_id`, `scenario`,
and sensor paths. Relative sensor paths are resolved against `--dataset-dir`.

<a id="rf-dt-and-gbt-inference"></a>

## 🚀 RF, DT, and GBT inference

The released conventional baselines accept an RGB image, a 2-D depth array in
meters, and an `N x 3` point-cloud array whose columns are lateral `x`, forward
`y`, and vertical `z`.

```bash
python scripts/predict_baseline.py \
  --model rf \
  --rgb /data/sample/rgb.png \
  --depth /data/sample/depth.npy \
  --point-cloud /data/sample/point_cloud.npy \
  --scenario tunnel
```

The 16 input features are five RGB statistics, four depth statistics, four
point-cloud statistics, and three neutral command fields.

<a id="baseline-training"></a>

## 🚆 Baseline training

`train_baselines.py` accepts a CSV with `split`, the 16 feature columns listed
in `mima_vr/schema.py`, and all seven requirement-vector target columns. Only
rows marked `split=train` are fitted.

```bash
python scripts/train_baselines.py \
  --feature-table /data/training_features.csv \
  --output-dir /data/trained_baselines \
  --random-state 42
```

This entry point exposes the exact estimator families and fixed
hyperparameters used to produce the released baseline weights.

<a id="verification"></a>

## ✅ Verification

Fitted baseline weights are listed in `models/baselines/checksums.json`.
The ET32 model card is based on deployment metadata and cached predictions.
System-level success, energy, and command-ready timing are evaluated under
[`experiments/system_level_ablation/`](../../experiments/system_level_ablation/README.md).
