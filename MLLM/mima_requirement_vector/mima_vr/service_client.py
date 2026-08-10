"""Client and strict response parser for Full-MIMA and MLLM-distilled services."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib import request as urllib_request

from .schema import REQUIREMENT_KEYS


PostCallable = Callable[..., Any]


class _JSONResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = int(status)
        self.body = body

    def raise_for_status(self) -> None:
        if not 200 <= self.status < 300:
            raise RuntimeError(f"requirement-vector service returned HTTP {self.status}")

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


def post_json(url: str, *, json_body: Mapping[str, Any], timeout: float) -> _JSONResponse:
    body = json.dumps(json_body).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=timeout) as response:
        return _JSONResponse(status=int(response.status), body=response.read())


@dataclass(frozen=True)
class RequirementVectorServiceClient:
    """Call a service implementing the paper's common ``/predict`` contract."""

    base_url: str
    backend: str
    model_index: int = 0
    timeout_s: float = 180.0
    request_post: PostCallable = post_json

    @property
    def predict_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/predict"

    def predict_from_sensor_paths(
        self,
        *,
        rgb_path: str | Path,
        depth_path: str | Path,
        point_cloud_path: str | Path,
        sample_id: str = "sensor_sample",
        scenario: str = "open_ground",
        task_command: str = "",
        model_key: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_index": int(self.model_index),
            "backend": self.backend,
            "sample_id": str(sample_id),
            "scenario": str(scenario),
            "rgb_path": str(Path(rgb_path).resolve()),
            "depth_path": str(Path(depth_path).resolve()),
            "point_cloud_path": str(Path(point_cloud_path).resolve()),
        }
        if task_command:
            payload["text"] = str(task_command)

        response = self.request_post(
            self.predict_url,
            json_body=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, Mapping):
            raise ValueError("requirement-vector service response must be a JSON object")
        return normalize_service_prediction(
            result,
            sample_id=str(sample_id),
            scenario=str(scenario),
            model_key=model_key or self.backend,
        )


def normalize_service_prediction(
    result: Mapping[str, Any],
    *,
    sample_id: str,
    scenario: str,
    model_key: str,
) -> dict[str, Any]:
    """Validate all seven fields without silently coercing malformed API values."""

    raw_vr = result.get("v_r") or result.get("text")
    if not isinstance(raw_vr, Mapping):
        raise ValueError("service response must contain object field 'v_r' or 'text'")

    missing = [key for key in REQUIREMENT_KEYS if key not in raw_vr]
    if missing:
        raise ValueError(f"service response v_r is missing keys: {missing}")

    values: dict[str, float] = {}
    for key in REQUIREMENT_KEYS:
        value = raw_vr[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"service response v_r.{key} must be a JSON number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"service response v_r.{key} must be finite")
        values[key] = numeric

    return {
        "sample_id": str(result.get("sample_id", sample_id)),
        "scenario": str(result.get("scenario", scenario)),
        "model": str(result.get("model", model_key)),
        "model_key": str(model_key),
        "feature_version": result.get("feature_version"),
        "v_r": values,
        "used_pointcloud": result.get("used_pointcloud"),
        "timing": result.get("timing", {}),
        "checkpoint": result.get("checkpoint"),
        "service_response": dict(result),
    }
