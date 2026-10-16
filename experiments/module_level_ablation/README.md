# MIMA Module-Level Ablation

[🏠 MIMA](../../README.md) · [📚 Documentation](../../docs/README.md)

This module contains the source code, selected checkpoints, and archived metric
records for the configuration-generator ablation. It compares four
architectures while retaining the same request representation, graph
imputation, physical constraints, sampling budget, and evaluation code.

## 📋 Table of Contents

- [🧠 Compared architectures](#compared-architectures)
- [📁 Contents](#contents)
- [📊 Metrics](#metrics)
- [🧪 Reproduction levels](#reproduction-levels)
- [🗂️ Evaluation sets](#evaluation-sets)
- [✅ Integrity](#integrity)

<a id="compared-architectures"></a>

## 🧠 Compared architectures

| Method | Configuration generator | Scale estimator |
| --- | --- | --- |
| Full generator | conditional variational autoencoder (cVAE) | Scale GNN |
| Scale GNN -> MLP | cVAE | topology-free MLP |
| cVAE -> MLP | deterministic request MLP | Scale GNN |
| cVAE + Scale GNN -> MLP | deterministic request MLP | direct MLP head |

Only the named component is replaced in each ablation. The fixed graph
imputation and hard-constraint implementation are shared by all four rows.

<a id="contents"></a>

## 📁 Contents

```text
checkpoints/  selected model states and their run metadata
configs/      fixed graph-imputation parameters
models/       cVAE, MLP, Scale GNN, geometry, and constraint code
outputs/      archived metric JSON files and the exported comparison table
scripts/      data preparation, training, evaluation, and inference entry points
```

The complete datasets and checkpoint histories are provided in
`data/module_level_ablation_assets.zip`, tracked with Git LFS. Extract the
archive to a temporary location and copy its `datasets/` directory into this
module before model reevaluation or training. The four selected `best_model.pt`
states required for reevaluation are already included under
`checkpoints/selected/`.

<a id="metrics"></a>

## 📊 Metrics

Reconstruction metrics are evaluated against target structures:

- **Orientation (rad):** mean geodesic quaternion error of reconstructed node
  orientations.
- **Location (mm):** mean absolute error over reconstructed node coordinates.
- **Ql (rad):** mean absolute error of the three leg angles.
- **w x/y/z (mm):** axis-wise mean absolute error of assembled scale.

Set-level metrics are evaluated over unique requirement-vector and
configuration-subspace queries, with 64 outputs per query. cVAE variants sample
from the prior; deterministic MLP variants repeat the same prediction 64 times.
For the latter, Achievement Rate equals single-output feasibility.

- **Achievement Rate:** fraction of requests with at least one candidate that
  passes every hard constraint.
- **Valid Rate:** fraction of all generated candidates that pass every hard
  constraint.
- **Coverage@64:** fraction of compatible validation structures matched within
  the configured normalized feature-distance threshold.
- **Diversity@64:** mean pairwise normalized feature distance among valid
  candidates.

Achievement Rate and Valid Rate are distinct. The former is computed from
`success_at_64`; the latter is computed from `valid_sample_rate`.

<a id="reproduction-levels"></a>

## 🧪 Reproduction levels

### Reconstruct the archived output table

This operation reads the four archived JSON files and performs no model
inference:

```bash
python scripts/eval/print_validation_table.py \
  outputs/validation_metrics.json \
  outputs/cvae_mlp_ablation_metrics.json \
  outputs/mlp_gnn_ablation_metrics.json \
  outputs/mlp_ablation_metrics.json \
  --csv-output reproduced/validation_metrics_table.csv
```

The generated table must match `outputs/validation_metrics_table.csv`.

### Reevaluate the selected checkpoints

Install the dependencies, place `datasets/` from the repository data archive at
the module root, and run from this directory:

```bash
python reevaluate.py --device auto
```

> [!NOTE]
> This is a fresh stochastic prior-generation evaluation. Device-level numerical
> behavior can affect sampled set metrics even with a fixed seed.

### Retrain the architectures

Training requires the raw and processed datasets in the repository data
archive. The training entry points are under `scripts/train/`, and the shell
wrappers are under `scripts/runners/`. Retraining creates new checkpoints and
is not a deterministic reconstruction of the archived outputs.

Activate the generator environment from the [installation guide](../../docs/INSTALLATION.md#generator)
before running a launcher. Launchers use the active `python`; set `PYTHON` to
an absolute interpreter path to choose another environment. Additional arguments
are forwarded to the training script, for example:

```bash
# From this module directory, after preparing datasets/:
bash scripts/runners/run_structure_generator_training.sh --device cpu --epochs 1
```

This example changes the device and epoch budget for a local run. With no
overrides, the launchers retain the archived training hyperparameters. New
checkpoints go under `ckpts/` and logs under `logs/`.

<a id="evaluation-sets"></a>

## 🗂️ Evaluation sets

Supplementary Table 6 reports reevaluation of all four selected checkpoints on
their original validation partition,
`datasets/processed_dataset/split_seed7/val_pairs.jsonl`. Reconstruction errors
are computed over 59,200 requirement-configuration pairs representing 3,549
source structures. The validation and training partitions share no source
structure IDs.

Set-level metrics use 104 unique requirement-vector and configuration-subspace
queries from the same validation partition, with 64 outputs per query. The
revised Achievement Rates are 88.46% (Full generator), 38.46% (Scale GNN -> MLP),
74.04% (cVAE -> MLP), and 55.77% (cVAE + Scale GNN -> MLP).

This partition was also used for checkpoint selection during training, so the
results describe validation performance. All 104 query definitions also occur
in the training partition. `reevaluate.py` uses this validation partition by
default.

The archived metric JSON files and comparison table in `outputs/` contain
this reevaluation. [`outputs/evaluation_audit.json`](outputs/evaluation_audit.json)
records the partition sizes, source-structure overlap, checkpoint and partition
hashes, and evaluation settings.

<a id="integrity"></a>

## ✅ Integrity

`MANIFEST.sha256` records every source-tree research artifact. The Git LFS data
archive includes its own manifest. Paths in both
manifests are relative to their respective roots.
