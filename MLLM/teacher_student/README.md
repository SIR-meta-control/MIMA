# Teacher and student source code

[MIMA](../../README.md) · [Requirement-vector interfaces](../mima_requirement_vector/README.md)

This directory provides the paper-specific Full-MIMA teacher and Teacher-only
ET32 student programs: sensor preprocessing, the 85-dimensional student feature
extractor, inference services, training, distillation and evaluation utilities.
The code was imported from the implementation used for the recorded deployment,
source revision `39647f66c5b9bacb308023344f95939e95133308`.

The teacher and student weights are not distributed because of third-party
rights restrictions. Supply locally held, authorized weights to run these
models. Archived results remain available through the
[reproduction guide](../../docs/REPRODUCIBILITY.md).

## Installation

Run the following commands from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The student, feature extraction and tabular evaluation use CPU libraries.
For the teacher, first install matching PyTorch and torchvision packages for
your CUDA runtime, then install `requirements-teacher.txt`. The bundled
teacher source uses Transformers 4.37.2; keep it in a separate environment from
other model stacks. For teacher fine-tuning, also install
`requirements-training.txt`; the bundled distributed training utilities import
DeepSpeed even when a separate DeepSpeed configuration is not supplied.

## Run a student service

An authorized student bundle must contain `student.joblib` and its matching
`metadata.json`. The feature column order, feature version and sampling limit
come from that metadata. The recorded ET32 uses 32 trees and a sensor sampling
limit of 32,768; the input dimension is 85.

```bash
python scripts/serve.py --backend student --host 127.0.0.1 --port 2610 \
  --student-model-dir /path/to/authorized/student_bundle
```

The service exposes `GET /ping` and `POST /predict`. It returns
`v_r = [wp_m, hp_m, dp_m, hs_m, fl, fi, fp]` in the existing MIMA response schema.
Use the [existing client](../mima_requirement_vector/README.md#full-mima-teacher-and-mllm-distilled-services)
to submit RGB, depth and point-cloud paths. These paths must be visible to the
service host. Set `--host` explicitly when serving another machine.

## Run a teacher service

An authorized teacher checkpoint must include its model shards, tokenizer and
configuration. The paper-specific sensor model and pure-PyTorch point-cloud
operator fallbacks are included under `vendor/internvl_chat`.

```bash
python scripts/serve.py --backend internvl --host 127.0.0.1 --port 2611 \
  --internvl-checkpoint /path/to/authorized/checkpoint \
  --internvl-prompt-profile grid --device cuda:0
```

For single-sample inference, use `scripts/predict_internvl_vr.py --help`.
`INTERNVL_VR_CHECKPOINT` can also set the default checkpoint directory.
The `grid` profile preserves the teacher's sensor-summary prompt and native
centimeter-grid output contract. The other profiles are retained as research
options.

## Data preparation, training and evaluation

All entry points below accept explicit input/output paths; consult `--help`
for their arguments. A sensor manifest is JSONL with a unique `sample_id`,
`scenario`, RGB/depth/point-cloud paths and the seven requirement-vector labels.
Paths can be absolute or relative to the manifest directory. The scripts do not
download private datasets or connect to the authors' server.

| Task | Entry points |
| --- | --- |
| Create deterministic dataset splits and audit overlap | `split_mujoco_manifest.py`, `audit_split_leakage.py` |
| Convert sensor manifests to InternVL training inputs | `build_internvl_vr_dataset.py` |
| Initialize and fine-tune a sensor teacher from authorized inputs | `build_clean_internvl_sensor_init.py`, `train_internvl_vr_4gpu.sh`, `train_strict_internvl_teacher.sh` |
| Train and select compact students | `train_distilled_student.py`, `select_distilled_student_config.py`, `select_distillation_supervision.py` |
| Evaluate predictions and select a teacher checkpoint | `evaluate_internvl_vr.py`, `evaluate_distilled_student.py`, `evaluate_vr_tolerance.py`, `select_internvl_checkpoint.py` |
| Export prediction caches and summarize results | `export_internvl_vr_cache.py`, `export_student_vr_cache.py`, `build_strict_distillation_result_table.py` |
| Measure service latency | `benchmark_student_service.py` |

For example, fit a teacher-only ET32 using explicit training/evaluation
manifests and teacher predictions for the training samples:

```bash
python scripts/train_distilled_student.py \
  --train-manifest /path/to/train/manifest.jsonl \
  --test-manifest /path/to/evaluation/manifest.jsonl \
  --teacher-predictions /path/to/teacher_train_predictions.csv \
  --teacher-weight 1.0 --model-type extratrees \
  --n-estimators 32 --summary-max-samples 32768 --n-jobs 1 \
  --model-name teacher_only_et32 \
  --model-dir /path/to/new_student --output-dir /path/to/results
```

The teacher prediction CSV contains `sample_id` and the seven `*_pred` columns.
Final teacher-only fitting uses those targets; original labels are used for
evaluation. Select hyperparameters on inner validation data before reporting
held-out results. Passing `--teacher-weight 0` fits a GT-only model, and a value
between 0 and 1 mixes the two supervision sources.

The teacher launcher requires `MODEL_NAME_OR_PATH` and `META_PATH` to point to
authorized initialization weights and prepared training metadata. It exposes
GPU count, output path and training settings through environment variables.
Use the recorded experiment configuration when running the launcher.

## Layout and release boundaries

```text
teacher_student/
  refactored_mllm/      sensor features, models, inference and service code
  scripts/             data preparation, training, selection and evaluation
  vendor/internvl_chat/ teacher architecture, training code and operator fallbacks
  requirements.txt     student and CPU tooling dependencies
  requirements-teacher.txt
```

Local `models/`, `data/`, `outputs/` and common weight extensions are ignored
by Git. The generic multimodal scripts elsewhere in `MLLM/` are preserved.
The RF/DT/GBT baseline service resolves the existing weights under
`MLLM/mima_requirement_vector/models/baselines/`.

Original MIMA code uses the repository's [GPL v3.0 license](../../LICENSE.md).
Bundled model sources retain their original notices; see
[vendor details](vendor/README.md) and [third-party notices](../../docs/THIRD_PARTY_NOTICES.md).
