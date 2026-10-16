# Downloads and data integrity

[🏠 MIMA](../README.md) · [📚 Documentation](README.md)

## 📋 Table of Contents

- [📦 Repository assets](#repository-assets)
- [✅ Fetch and verify](#fetch-and-verify)
- [🗂️ Prepare working datasets](#prepare-working-datasets)
- [📼 ROS bag](#ros-bag)
- [🧪 Reproduction](#reproduction)

<a id="repository-assets"></a>

## 📦 Repository assets

| Asset | Location | Retrieval |
| --- | --- | --- |
| Four selected generator checkpoints | `experiments/module_level_ablation/checkpoints/selected/` | Included by Git |
| Bar-type classifier | `experiments/module_level_ablation/checkpoints/bar_classifier/` | Included by Git |
| Fitted RF, DT and GBT weights | `MLLM/mima_requirement_vector/models/baselines/` | Included by Git |
| Training and validation datasets | `data/module_level_ablation_assets.zip` | Git LFS |
| System-level records and fixed sensor evaluation set | `data/system_level_ablation_assets.zip` | Git LFS |
| Figure 2/4, calibration, onboard tests and split ROS bag | `data/` | See [data inventory](../data/README.md) |

Model identifiers and inference interfaces are documented in
[MODEL_CARD.md](MODEL_CARD.md).

<a id="fetch-and-verify"></a>

## ✅ Fetch and verify

Selected checkpoints and RF/DT/GBT weights are ordinary Git files. Local
model inference does not require any of the large data archives.

Fetch the archive needed for your workflow:

| Workflow | Git LFS archive |
| --- | --- |
| Generator reevaluation or training | `data/module_level_ablation_assets.zip` |
| Archived system-level table reconstruction | `data/system_level_ablation_assets.zip` |
| Figure data, calibration or bag replay | Corresponding files in the [data inventory](../data/README.md) |

```bash
git lfs install
git lfs pull --include="data/module_level_ablation_assets.zip"
# For system-level reconstruction instead:
git lfs pull --include="data/system_level_ablation_assets.zip"
```

The preparation commands below verify each archive before extraction. To
retrieve and verify the entire released data collection, including all bag
segments, run:

```bash
git lfs pull
python scripts/check_repository.py --verify-data --verify-archives
```

> [!NOTE]
> `data/SHA256SUMS` records the exact released file contents. A Git LFS pointer
> is not the data file. Full collection verification requires every file in
> that manifest; fetching one archive is sufficient for its own preparation.

<a id="prepare-working-datasets"></a>

## 🗂️ Prepare working datasets

```bash
python scripts/prepare_data.py module
python scripts/prepare_data.py system
```

The preparation script verifies the archive hash before extraction and rejects
archive paths outside the destination and symbolic links. It extracts module
datasets under `experiments/module_level_ablation/datasets/` and the system
bundle under `experiments/system_level_ablation/data_bundle/`. Existing files
with different contents are not overwritten.

The datasets and reconstructed outputs remain outside version control. The
selected model checkpoints in the source tree remain unchanged.

<a id="ros-bag"></a>

## 📼 ROS bag

`test.bag` is distributed as three byte-range segments. Reassembly commands and
the complete file's SHA-256 are provided in [data/README.md](../data/README.md#split-ros-bag).
Run reassembly from the data directory after fetching all three LFS objects.

<a id="reproduction"></a>

## 🧪 Reproduction

Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) to reconstruct the archived tables,
reevaluate selected models or retrain the ablations. These are separate operations
with different requirements and numerical reproducibility guarantees.
