from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import requests
from PIL import Image


def encode_image_base64(path: str | Path, image_format: str = "JPEG") -> str:
    with Image.open(path) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def send_sensor_request(
    *,
    server_url: str,
    image_path: str | Path,
    pointcloud_data: list[dict[str, float]] | list[list[float]],
    depth_path: str | Path | None = None,
    depth_data: list[list[float]] | None = None,
    text: str = "",
    model: str = "dt",
    backend: str = "internvl",
    sample_id: str = "sensor_sample",
    scenario: str = "open_ground",
    timeout: float = 60.0,
) -> dict[str, Any]:
    payload = {
        "model_index": 0,
        "backend": backend,
        "model": model,
        "image": encode_image_base64(image_path),
        "pointcloud_data": pointcloud_data,
        "text": text,
        "sample_id": sample_id,
        "scenario": scenario,
    }
    if depth_path is not None:
        payload["depth_path"] = str(depth_path)
    if depth_data is not None:
        payload["depth_data"] = depth_data
    response = requests.post(
        server_url.rstrip("/") + "/predict",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()
