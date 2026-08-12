# MIMA Module-Level Ablation

This module contains the source code, selected checkpoints, and archived metric
records for the configuration-generator ablation. It compares four
architectures while retaining the same request representation, graph
imputation, physical constraints, sampling budget, and evaluation code.

## Compared architectures

| Method | Configuration generator | Scale estimator |
| --- | --- | --- |
| Full generator | conditional variational autoencoder (cVAE) | Scale GNN |
| Scale GNN -> MLP | cVAE | topology-free MLP |
| cVAE -> MLP | deterministic request MLP | Scale GNN |
| cVAE + Scale GNN -> MLP | deterministic request MLP | direct MLP head |

Only the named component is replaced in each ablation. The fixed graph
imputation and hard-constraint implementation are shared by all four rows.

## Contents

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

## Metrics

Reconstruction metrics are evaluated against target structures:

- **Orientation (rad):** mean geodesic quaternion error of reconstructed node
  orientations.
- **Location (mm):** mean absolute error over reconstructed node coordinates.
- **Ql (rad):** mean absolute error of the three leg angles.
- **w x/y/z (mm):** axis-wise mean absolute error of assembled scale.

Set-level metrics use 64 prior samples for each request:

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

## Reproduction levels

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

This is a fresh stochastic prior-generation evaluation. Device-level numerical
behavior can affect sampled set metrics even with a fixed seed.

### Retrain the architectures

Training requires the raw and processed datasets in the repository data
archive. The training entry points are under `scripts/train/`, and the shell
wrappers are under `scripts/runners/`. Retraining creates new checkpoints and
is not a deterministic reconstruction of the archived outputs.

## Evidence boundary

The archived files in `outputs/` were evaluated on the 20% split containing
117,580 pairs and 7,099 structures. The selected checkpoints were trained with
the separate 10% split whose validation partition contains 59,200 pairs and
3,549 structures. Exactly 3,550 structures in the archived 20% evaluation
partition belong to the selected checkpoints' training partition.

Consequently, the archived output table is retained as original experimental
evidence, but it must not be described as a leakage-free evaluation on 59,200
held-out pairs. A manuscript table defined as evaluation on the 59,200-pair
held-out partition must be populated from a reevaluation using
`split_seed7/val_pairs.jsonl`. In addition, a value derived from
`valid_sample_rate` must be labelled Valid Rate rather than Achievement Rate.

## Integrity

`MANIFEST.sha256` records every source-tree research artifact. The Git LFS data
archive includes its own manifest. Paths in both
manifests are relative to their respective roots.
