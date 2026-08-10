import csv
import json
import shutil
from pathlib import Path

from mima_ablation.batch import run_batch
from mima_ablation.methods import METHOD_SPECS


def test_batch_writes_success_energy_and_timing_contracts():
    root = Path.cwd() / ".test_runtime"
    shutil.rmtree(root, ignore_errors=True)
    try:
        dataset = root / "dataset"
        sample_dir = dataset / "samples" / "sample_w0.50_h0.35"
        sample_dir.mkdir(parents=True)
        for filename in ("rgb.png", "depth.npy", "point_cloud.npy"):
            (sample_dir / filename).write_bytes(b"fixture")
        record = {
            "sample_id": "sample_w0.50_h0.35",
            "scenario": "tunnel",
            "rgb_path": "samples/sample_w0.50_h0.35/rgb.png",
            "depth_path": "samples/sample_w0.50_h0.35/depth.npy",
            "point_cloud_path": "samples/sample_w0.50_h0.35/point_cloud.npy",
        }
        (dataset / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        ids_path = root / "sample_ids.txt"
        ids_path.write_text("sample_w0.50_h0.35\n", encoding="utf-8")
        assets = {
            "requirement_sources": {"full_mima_teacher": "fixture"},
            "structure_models": {"cvae": "fixture"},
            "energy_model": "fixture",
            "robot_model": "fixture",
        }

        success_summary = run_batch(
            mode="success_energy",
            backend_path="tests.fake_backend:run_request",
            dataset_dir=dataset,
            sample_ids_file=ids_path,
            output_dir=root / "success",
            method_specs=(METHOD_SPECS["full_mima"],),
            seeds=(1, 2),
            assets=assets,
            protocol={},
            workers=1,
            progress=False,
        )
        assert success_summary["row_count"] == 2
        with (root / "success" / "details.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["energy_j"] for row in rows] == ["101.0", "102.0"]
        assert all(row["pred_wp_m"] == "0.5" for row in rows)

        timing_summary = run_batch(
            mode="timing",
            backend_path="tests.fake_backend:run_request",
            dataset_dir=dataset,
            sample_ids_file=ids_path,
            output_dir=root / "timing",
            method_specs=(METHOD_SPECS["full_mima"],),
            seeds=(7,),
            assets=assets,
            protocol={},
            workers=1,
            progress=False,
        )
        assert timing_summary["row_count"] == 1
        with (root / "timing" / "execution_time_details.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            row = next(csv.DictReader(handle))
        assert row["execution_time_s"] == "0.25"
    finally:
        shutil.rmtree(root, ignore_errors=True)
