# Research Data

This directory contains the source data and archived records used by the
released experiments. Large ZIP archives and split ROS bag parts are tracked
with Git LFS. After
cloning the repository, fetch their contents with:

```bash
git lfs pull
```

## Data files

| File | Size (bytes) | Contents | SHA-256 |
| --- | ---: | --- | --- |
| `Figure 2.zip` | 23,340 | MATLAB-derived Figure 2c and 2d spline samples and keypoints, with both curves uniformly sampled at 300 points, record manifests, source-script hashes, and integrity metadata | `c8ff0a326f69f377656698eead1f9c6a808cad0b57d6eb45339b35f0c88f8778` |
| `Extended_Data_Figure_6c_Source_Data.xlsx` | 9,792 | Transformation and sustainment energy source values for the seven feasible and Pareto-optimal configurations shown in Extended Data Figure 6c | `9efe91d13b87ae54b3c824e3a474c66d065d7e3ab604cf5fcee84389cd6da05d` |
| `test.bag.part001` | 1,900,000,000 | Byte-range part 1 of 3 of the supplied ROS bag `test.bag` | `24b3d3fe964056909887dd985fae3a4abd5e73652fd40eb32f2fef98ea9a7df6` |
| `test.bag.part002` | 1,900,000,000 | Byte-range part 2 of 3 of the supplied ROS bag `test.bag` | `e0ebbf9d212a690485c415ea246db56218a43523f2dbafc3060f1054f59d45b5` |
| `test.bag.part003` | 384,949,833 | Byte-range part 3 of 3 of the supplied ROS bag `test.bag` | `21ba390b9108aee368a8d917d061bb0cd982e3d0a81203bb365928fb609db2e5` |
| `Extended_Data_Figure_6c_Source_Data.xlsx` | 9,792 | Transformation and sustainment energy source values for the seven feasible and Pareto-optimal configurations shown in Extended Data Figure 6c | `9efe91d13b87ae54b3c824e3a474c66d065d7e3ab604cf5fcee84389cd6da05d` |
| `experiment_fig_4.zip` | 11,058,934 | Raw ROS bags and exported CSV records supporting the Figure 4 robot experiments and speed trials | `474be9d9357fccdbbda9b36154109dcb75a9819d8d12e211a21c48975c1ed1a4` |
| `module_level_ablation_assets.zip` | 210,109,010 | Raw and processed configuration-generator datasets, training splits, selected runs, checkpoint histories, and integrity metadata | `06d833abda324f3d8bbac558f565e9d779c99866ecba3c0032ba6f9ba7c79cfc` |
| `system_level_ablation_assets.zip` | 208,664,678 | Fixed sensor evaluation set, row-level success, energy and timing records, requirement-vector caches, provenance snapshot, reconstruction scripts, and integrity metadata | `9ad74ee05c2423c101b5ff872d34b6e59f7c35ec773c8c8140c2cff794ea339d` |
| `Supplementary Method 9.2.zip` | 263,958 | Record-level latency, resource-monitoring, stability, communication-loss, and onboard-validation data for Supplementary Method 9.2 | `fa5d82007f2e31d4942d9e36367be26505e13c64bcbd6b04134ef18f3b5ba283` |
| `Supplementary Method 6.3.zip` | 14,365,363 | Real-robot and calibrated MuJoCo transformation commands, motor measurements, and energy-verification records for Supplementary Method 6.3 | `0d0e6a0916a34a96bd41ca49d9a6c14c0f69336c0e1613dcce66b20fa70ccece` |

## Split ROS Bag

The supplied `test.bag` is a ROS bag V2.0 with an original size of
4,184,949,833 bytes and SHA-256
`654f02bc2379f09480e2b3a6a68d46cf326a4d0bd5815a249aff50e510d1e35d`.
It is split byte-for-byte into three parts because the complete file exceeds
the per-file limit of the GitHub LFS remote. Fetch all LFS objects, then run
the following in the directory containing the parts.

PowerShell:

```powershell
$parts = Get-ChildItem .\test.bag.part* | Sort-Object Name
$output = [IO.File]::Create('.\test.bag')
try {
    foreach ($part in $parts) {
        $inputStream = [IO.File]::OpenRead($part.FullName)
        try { $inputStream.CopyTo($output) } finally { $inputStream.Dispose() }
    }
} finally { $output.Dispose() }
```

Linux/macOS:

```bash
cat test.bag.part001 test.bag.part002 test.bag.part003 > test.bag
```

After reassembly, verify the original SHA-256 shown above.

The hashes are also provided in `SHA256SUMS`. Verify the data files from this
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

The Extended Data Figure 6c workbook contains the plotted transformation and
sustainment energy values together with the classification of each configuration
as feasible or Pareto optimal.

Each ablation archive includes its own manifest and README describing its
internal evidence boundary. Extracted working directories and regenerated
outputs are not tracked by the repository.
