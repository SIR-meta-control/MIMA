# MIMA Requirement-Vector Inference

This directory contains the paper-specific interfaces and conventional
baselines used to infer the seven-dimensional MIMA requirement vector

```text
v_r = [w_p, h_p, d_p, h_s, f_l, f_i, f_p].
```

It complements the generic InternVL implementation in the parent directory.
The code here records the inference contract used by Full-MIMA,
MLLM-distilled, and the RF/DT/GBT replacement ablations.

## Release scope

| Method | Public capability in this directory |
| --- | --- |
| Full-MIMA teacher | Strict `/predict` client, response parser, and batch cache entry point |
| MLLM-distilled | The same client and cache entry point, plus the adopted Teacher-only ET32 model card and expected hashes |
| w/o MLLM -> RF/DT/GBT | Deterministic 16-feature adapter, training and inference code, metadata, and fitted weights |

The hosted Full-MIMA teacher server, its paper-specific prompt, and its
checkpoint are not distributed here. The MLLM-distilled `student.joblib`, its
85-D feature extractor, and its training/service implementation are also not
present in the available release assets. Their expected identities and hashes
are recorded so that a future release can be verified without silently
substituting another model. Consequently, this directory can call a compatible
teacher or student service, but it cannot start those two services or retrain
the adopted student locally.

The RF/DT/GBT path is fully local. These baselines use a separate 16-feature
sensor adapter and must not be described as using the student's 85-D feature
extractor.

## Layout

```text
mima_requirement_vector/
  mima_vr/                 shared schema, service client, and baseline code
  scripts/                 service cache, local inference, and baseline training
  models/baselines/        released RF, DT, and GBT weights
  models/teacher_only_et32 expected ET32 artifact identity and hashes
  tests/                   focused contract and feature tests
  MODEL_CARD.md            method definitions and evidence boundaries
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the focused tests with:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

## Full-MIMA teacher and MLLM-distilled services

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

## RF, DT, and GBT inference

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
point-cloud statistics, and three neutral command fields. The reported MuJoCo
ablation used sensor-only inputs, so all three command fields are zero.

## Baseline training

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
hyperparameters used to produce the released baseline weights. The original
training sensor records are not duplicated in this source tree.

## Reproducibility boundary

- The service client and parser reproduce the recorded API contract, not the
  unpublished server internals.
- The ET32 facts in `MODEL_CARD.md` are backed by deployment metadata and
  cached predictions; they are not a substitute for the missing weight and
  extractor files.
- The fitted baseline weights are checked by SHA-256 in
  `models/baselines/checksums.json`.
- System-level success, energy, and full-chain timing are downstream metrics;
  they are not recomputed by this inference-only directory.
