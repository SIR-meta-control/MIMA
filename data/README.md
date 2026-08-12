# Research Data

This directory contains the archived data used by the released experiments.
The ZIP archives are tracked with Git LFS. After cloning the repository, fetch
their contents with:

```bash
git lfs pull
```

## Archives

| Archive | Size (bytes) | Contents | SHA-256 |
| --- | ---: | --- | --- |
| `experiment_fig_4.zip` | 11,058,934 | Raw ROS bags and exported CSV records supporting the Figure 4 robot experiments and speed trials | `474be9d9357fccdbbda9b36154109dcb75a9819d8d12e211a21c48975c1ed1a4` |
| `module_level_ablation_assets.zip` | 210,109,010 | Raw and processed configuration-generator datasets, training splits, selected runs, checkpoint histories, and integrity metadata | `06d833abda324f3d8bbac558f565e9d779c99866ecba3c0032ba6f9ba7c79cfc` |
| `system_level_ablation_assets.zip` | 208,664,678 | Fixed sensor evaluation set, row-level success, energy and timing records, requirement-vector caches, provenance snapshot, reconstruction scripts, and integrity metadata | `9ad74ee05c2423c101b5ff872d34b6e59f7c35ec773c8c8140c2cff794ea339d` |

The hashes are also provided in `SHA256SUMS`. Verify the archives from this
directory with:

```bash
sha256sum -c SHA256SUMS
```

## Use

The system-level archive is self-contained for deterministic reconstruction of
the reported seven-row ablation table. Extract it and use the archive directory
that contains `reproduce_table.py` as the bundle root. Reconstruction uses the
archived row-level records and does not invoke model services or rerun the
generator, energy model, or simulation.

The module-level archive contains the larger datasets and checkpoint histories
used by `experiments/module_level_ablation/`. Extract it to a temporary location,
then place its `datasets/` directory at the module root before checkpoint
reevaluation or training. The source tree already contains the selected model
states required by the standard reevaluation workflow.

The Figure 4 archive preserves both the original ROS bags and their exported CSV
records. The two representations are retained so that analyses can use the
source logs while individual traces can be inspected without ROS bag tooling.

Each ablation archive includes its own manifest and README describing its
internal evidence boundary. Extracted working directories and regenerated
outputs are not tracked by the repository.
