from __future__ import annotations

import base64
import io
import tempfile
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
import numpy as np
from PIL import Image

from refactored_mllm.features import load_or_decode_point_cloud
from refactored_mllm.distillation import (
    DEFAULT_STUDENT_MODEL_DIR,
    build_distill_feature_frame_from_payload,
    format_student_prediction,
    load_student_bundle,
    predict_student_frame,
    student_summary_max_samples,
)
from refactored_mllm.internvl.inference import (
    DEFAULT_FINETUNED_CHECKPOINT,
    DEFAULT_INTERNVL_SOURCE,
    InternVLVRPredictor,
)
from refactored_mllm.models import DEFAULT_MODEL_DIR
from refactored_mllm.pipeline import predict_vr_from_payload


class VRModelService:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 2610,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        student_model_dir: str | Path = DEFAULT_STUDENT_MODEL_DIR,
        default_model: str = "dt",
        backend: str = "internvl",
        internvl_checkpoint: str | Path = DEFAULT_FINETUNED_CHECKPOINT,
        internvl_source: str | Path = DEFAULT_INTERNVL_SOURCE,
        device: str = "cuda:0",
        internvl_prompt_profile: str = "grid",
    ) -> None:
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.model_dir = Path(model_dir)
        self.student_model_dir = Path(student_model_dir)
        self.default_model = default_model
        self.backend = backend
        self.internvl_checkpoint = Path(internvl_checkpoint)
        self.internvl_source = Path(internvl_source)
        self.device = device
        self.internvl_prompt_profile = internvl_prompt_profile
        self._internvl_predictor: InternVLVRPredictor | None = None
        self._student_bundle: tuple[Any, dict[str, Any]] | None = None
        self.setup_routes()

    def setup_routes(self) -> None:
        @self.app.route("/ping", methods=["GET"])
        def ping() -> Any:
            return jsonify({"status": "ok", "server_time": time.time()})

        @self.app.route("/predict", methods=["POST"])
        def predict() -> Any:
            return self.handle_predict()

    def handle_predict(self) -> Any:
        server_recv = time.time()
        try:
            payload = request.get_json(force=True)
            if not isinstance(payload, dict):
                return jsonify({"error": "request JSON must be an object"}), 400

            model_index = payload.get("model_index", 0)
            if model_index not in (0, "0", None):
                return jsonify(
                    {
                        "error": (
                            "Only model_index=0 is supported after refactor; "
                            "the old OpenVocabulary/Whisper/SD branches were removed."
                        )
                    }
                ), 400

            start = time.time()
            backend = str(payload.get("backend") or self.backend).lower()
            if backend == "internvl":
                prediction = self._predict_with_internvl(payload)
            elif backend == "baseline":
                prediction = predict_vr_from_payload(
                    payload,
                    model_dir=self.model_dir,
                    default_model_key=self.default_model,
                )
            elif backend == "student":
                prediction = self._predict_with_student(payload)
            else:
                return jsonify(
                    {"error": "backend must be 'internvl', 'baseline', or 'student'"}
                ), 400
            end = time.time()

            response = {
                **prediction,
                "text": prediction["v_r"],
                "backend": backend,
                "success_metric_version": "vr_abs_error_all_7_dims_le_0.01m",
                "used_pointcloud": bool(
                    payload.get("pointcloud_data")
                    or payload.get("point_cloud_data")
                    or payload.get("point_cloud")
                    or payload.get("points")
                    or payload.get("point_cloud_path")
                    or payload.get("pc_path")
                ),
                "timing": {
                    "server_recv_time": server_recv,
                    "inference_ms": (end - start) * 1000.0,
                    "server_total_ms": (end - server_recv) * 1000.0,
                },
            }
            return jsonify(response)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    def run(self) -> None:
        self.app.run(host=self.host, port=self.port)

    def _predict_with_internvl(self, payload: dict[str, Any]) -> dict[str, object]:
        predictor = self._get_internvl_predictor()
        with tempfile.TemporaryDirectory(prefix="refactored_mllm_") as temp_dir:
            temp_root = Path(temp_dir)
            rgb_path = _materialize_rgb(payload, temp_root)
            depth_path = _materialize_depth(payload, temp_root)
            point_cloud_path = _materialize_point_cloud(payload, temp_root)
            prediction = predictor.predict(
                rgb_path=rgb_path,
                depth_path=depth_path,
                point_cloud_path=point_cloud_path,
            )
        return {
            "sample_id": str(payload.get("sample_id", "sensor_sample")),
            "scenario": str(payload.get("scenario", "open_ground")),
            "model": "internvl",
            "checkpoint": str(self.internvl_checkpoint),
            **prediction,
        }

    def _get_internvl_predictor(self) -> InternVLVRPredictor:
        if self._internvl_predictor is None:
            self._internvl_predictor = InternVLVRPredictor(
                checkpoint=self.internvl_checkpoint,
                internvl_source=self.internvl_source,
                device=self.device,
                prompt_profile=self.internvl_prompt_profile,
            )
        return self._internvl_predictor

    def _predict_with_student(self, payload: dict[str, Any]) -> dict[str, object]:
        model, metadata = self._get_student_bundle()
        frame = build_distill_feature_frame_from_payload(
            payload,
            summary_max_samples=student_summary_max_samples(metadata),
        )
        prediction_frame = predict_student_frame(model, frame, metadata)
        return format_student_prediction(
            frame,
            prediction_frame,
            model_dir=self.student_model_dir,
            metadata=metadata,
        )

    def _get_student_bundle(self) -> tuple[Any, dict[str, Any]]:
        if self._student_bundle is None:
            self._student_bundle = load_student_bundle(self.student_model_dir)
        return self._student_bundle


def _materialize_rgb(payload: dict[str, Any], temp_root: Path) -> Path:
    rgb_path = payload.get("rgb_path") or payload.get("image_path")
    if rgb_path:
        return Path(str(rgb_path))

    image_payload = payload.get("image")
    if image_payload is None:
        raise ValueError("InternVL backend requires image, rgb_path, or image_path")

    raw = _decode_payload_bytes(image_payload)
    target = temp_root / "rgb.png"
    with Image.open(io.BytesIO(raw)) as image:
        image.convert("RGB").save(target)
    return target


def _materialize_depth(payload: dict[str, Any], temp_root: Path) -> Path:
    depth_path = payload.get("depth_path")
    if depth_path:
        return Path(str(depth_path))

    if "depth" in payload:
        target = temp_root / "depth.npy"
        np.save(target, np.asarray(payload["depth"], dtype=np.float32))
        return target
    if "depth_data" in payload:
        target = temp_root / "depth.npy"
        np.save(target, np.asarray(payload["depth_data"], dtype=np.float32))
        return target

    raise ValueError("InternVL backend requires depth_path, depth, or depth_data")


def _materialize_point_cloud(payload: dict[str, Any], temp_root: Path) -> Path:
    point_cloud_path = (
        payload.get("point_cloud_path")
        or payload.get("pointcloud_path")
        or payload.get("pc_path")
    )
    if point_cloud_path:
        return Path(str(point_cloud_path))

    point_cloud_data = (
        payload.get("pointcloud_data")
        or payload.get("point_cloud_data")
        or payload.get("point_cloud")
        or payload.get("points")
    )
    if point_cloud_data is None:
        raise ValueError("InternVL backend requires point-cloud data or path")

    target = temp_root / "point_cloud.npy"
    np.save(target, load_or_decode_point_cloud(data=point_cloud_data))
    return target


def _decode_payload_bytes(payload: str | bytes) -> bytes:
    if isinstance(payload, bytes):
        return payload

    stripped = payload.strip()
    if "," in stripped and stripped.lower().startswith("data:"):
        stripped = stripped.split(",", 1)[1]

    try:
        return base64.b64decode(stripped, validate=True)
    except Exception:
        return payload.encode("latin1")
