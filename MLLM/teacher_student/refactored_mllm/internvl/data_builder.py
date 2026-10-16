from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from refactored_mllm.internvl.prompts import VR_REQUIREMENT_PROMPT, format_vr_answer
from refactored_mllm.internvl.sensor_scale import (
    format_sensor_scale,
    sensor_scale_metadata,
)


REQUIREMENT_COLUMNS = ("wp_m", "hp_m", "dp_m", "hs_m", "fl", "fi", "fp")


def build_internvl_vr_dataset(
    manifest_paths: Iterable[str | Path],
    output_root: str | Path,
    *,
    dataset_name: str = "mujoco_vr",
    copy_assets: bool = True,
    limit: int | None = None,
    decimals: int = 4,
) -> dict[str, Path | int]:
    output_root = Path(output_root)
    dataset_root = output_root / dataset_name
    annotation_path = output_root / f"{dataset_name}.jsonl"
    meta_path = output_root / f"{dataset_name}_meta.json"
    dataset_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    written = 0
    with annotation_path.open("w", encoding="utf-8") as out:
        for manifest_path in manifest_paths:
            manifest = Path(manifest_path)
            run_dir = manifest.parent
            with manifest.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if limit is not None and written >= limit:
                        break
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    item = convert_record(
                        record,
                        run_dir=run_dir,
                        dataset_root=dataset_root,
                        copy_assets=copy_assets,
                        decimals=decimals,
                    )
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    written += 1
            if limit is not None and written >= limit:
                break

    meta = {
        dataset_name: {
            "root": str(dataset_root.resolve()),
            "annotation": str(annotation_path.resolve()),
            "data_augment": False,
            "repeat_time": 1,
            "length": written,
        }
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "dataset_root": dataset_root,
        "annotation": annotation_path,
        "meta": meta_path,
        "length": written,
    }


def convert_record(
    record: dict[str, object],
    *,
    run_dir: Path,
    dataset_root: Path,
    copy_assets: bool,
    decimals: int,
) -> dict[str, object]:
    missing = [column for column in REQUIREMENT_COLUMNS if column not in record]
    if missing:
        raise ValueError(
            f"Missing v_r columns for {record.get('sample_id')}: {missing}"
        )

    sample_id = _safe_name(str(record["sample_id"]))
    sample_dir = dataset_root / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    rgb_source = _resolve(run_dir, record, "rgb_path")
    depth_source = _resolve(run_dir, record, "depth_path")
    point_cloud_source = _resolve(run_dir, record, "point_cloud_path")

    rgb_rel = Path("samples") / sample_id / "rgb.png"
    depth_rel = Path("samples") / sample_id / "depth.png"
    pc_rel = Path("samples") / sample_id / "pointcloud.ply"

    if copy_assets:
        _copy_rgb(rgb_source, dataset_root / rgb_rel)
        _write_depth_png(depth_source, dataset_root / depth_rel)
        _write_point_cloud_ply(point_cloud_source, dataset_root / pc_rel)
    else:
        rgb_rel = rgb_source.resolve()
        depth_rel = _ensure_depth_png(depth_source)
        pc_rel = _ensure_point_cloud_ply(point_cloud_source)

    scale_metadata = sensor_scale_metadata(
        depth_source,
        point_cloud_source,
        rgb_path=rgb_source,
    )

    return {
        "id": str(record["sample_id"]),
        "image": _as_posix(rgb_rel),
        "depth": _as_posix(depth_rel),
        "pointcloud": _as_posix(pc_rel),
        "scenario": record.get("scenario") or record.get("target_type"),
        "source_manifest": str(run_dir),
        "sensor_scale": scale_metadata,
        "conversations": [
            {
                "from": "human",
                "value": (
                    f"{VR_REQUIREMENT_PROMPT}\n{format_sensor_scale(scale_metadata)}"
                ),
            },
            {"from": "gpt", "value": format_vr_answer(record, decimals=decimals)},
        ],
    }


def _resolve(run_dir: Path, record: dict[str, object], key: str) -> Path:
    raw = record.get(key)
    if raw is None:
        raise KeyError(key)
    path = Path(str(raw))
    if not path.is_absolute():
        path = run_dir / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _copy_rgb(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(target)


def _write_depth_png(source: Path, target: Path) -> None:
    depth = np.asarray(np.load(source), dtype=float)
    if depth.ndim != 2:
        raise ValueError(f"Depth array must be 2D: {source}")
    finite = depth[np.isfinite(depth)]
    finite = finite[finite > 0]
    if finite.size == 0:
        normalized = np.zeros(depth.shape, dtype=np.uint8)
    else:
        low = float(np.percentile(finite, 1))
        high = float(np.percentile(finite, 99))
        if high <= low:
            high = low + 1.0
        clipped = np.clip(depth, low, high)
        clipped = np.nan_to_num(clipped, nan=high, posinf=high, neginf=low)
        normalized = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalized, mode="L").save(target)


def _write_point_cloud_ply(source: Path, target: Path) -> None:
    points = np.asarray(np.load(source), dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Point cloud must be Nx3: {source}")
    points = points[:, :3]
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def _ensure_depth_png(source: Path) -> Path:
    if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        return source.resolve()
    raise ValueError("copy_assets=False requires depth images, not .npy arrays")


def _ensure_point_cloud_ply(source: Path) -> Path:
    if source.suffix.lower() == ".ply":
        return source.resolve()
    raise ValueError("copy_assets=False requires .ply point clouds, not .npy arrays")


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _as_posix(path: Path) -> str:
    return path.as_posix()
