# MIMA model card

[🏠 MIMA](../README.md) · [📚 Documentation](README.md)

## 📋 Table of Contents

- [🏗️ Architecture and interfaces](#architecture-and-interfaces)
- [📦 Assets and execution](#assets-and-execution)
- [📊 Evaluation](#evaluation)
- [🎯 Intended use](#intended-use)
- [📄 Licenses](#licenses)

<a id="architecture-and-interfaces"></a>

## 🏗️ Architecture and interfaces

Full-MIMA maps multimodal observations and task inputs to the seven-dimensional
requirement vector `[wp, hp, dp, hs, fl, fi, fp]`. The generator adapter maps it to
`[dp, wp, hp, load, inspect, pack]`.

Within a chosen configuration subspace, the cVAE samples a learned conditional
diagonal Gaussian prior whose mean and variance depend on the encoded
requirement vector and subspace index. Its decoder generates body-link poses
and limb angles. Fixed graph imputation recovers the remaining pose fields,
and the Scale GNN predicts external dimensions. Hard constraints and diversity
selection produce RTC candidates. Code and interfaces are documented in the
[generator README](../src/generator/README.md).

The energy optimizer evaluates transformation and sustainment energy from the
stationary current configuration. Its implementation uses fixed equal weights
and directly minimizes the sum. The control period and energy-integration period
are both 0.02 s at 50 Hz. See the [optimizer README](../src/optimizer/README.md).

<a id="assets-and-execution"></a>

## 📦 Assets and execution

| Model | Assets in this repository | Execution |
| --- | --- | --- |
| Full generator and three module ablations | Four selected `best_model.pt` files, graph parameters, model source and evaluation JSONs | Local inference and reevaluation |
| Bar-type classifier | Selected classifier checkpoint and configuration | Local generator runtime |
| RF, DT and GBT | Fitted joblib weights, metadata, checksums and 16-feature adapter | Local requirement-vector prediction |
| Full-MIMA teacher | Paper-specific preprocessing, prompts, model source, training and service code | Local `/predict` service with authorized weights |
| Teacher-only ET32 | 85-dimensional feature extractor, training, inference and service code; recorded model identity/hashes | Local `/predict` service with authorized weights |

Model identifiers and evaluation settings are documented in the
[requirement-vector model card](../MLLM/mima_requirement_vector/MODEL_CARD.md).
Teacher and student weights are excluded because of third-party rights
restrictions. See the [source and deployment guide](../MLLM/teacher_student/README.md).
Configure teacher and ET32 endpoints using the
[inference guide](../MLLM/mima_requirement_vector/README.md#full-mima-teacher-and-mllm-distilled-services).

<a id="evaluation"></a>

## 📊 Evaluation

The module-level reevaluation uses 59,200 pairs from 3,549 held-out source
structures and 104 requirement/subspace queries. No evaluated source structure
belongs to the training partition. The validation partition was used for
checkpoint selection and the query definitions are shared with training.

> [!NOTE]
> Achievement Rate counts queries with at least one feasible output among 64
> outputs; deterministic MLP variants repeat one output. System-level Success
> Rate additionally assesses command generation and passage geometry under its
> own protocol. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the metric definitions.

<a id="intended-use"></a>

## 🎯 Intended use

The released models support research on the I-MR geometry and the evaluated
passage/task domain. Outputs require the implemented hard-constraint checks
and a robot-specific control configuration. Reuse on other robots, geometries
or sensor distributions requires new validation.

<a id="licenses"></a>

## 📄 Licenses

Original MIMA source uses [GPL v3.0](../LICENSE.md). Third-party source, pretrained
backbones and datasets retain the terms of their providers. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
