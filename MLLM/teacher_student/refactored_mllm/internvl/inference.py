from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

from refactored_mllm.internvl.prompts import (
    internvl_prompt_for_profile,
    prompt_profile_uses_sensor_scale,
)
from refactored_mllm.internvl.sensor_scale import (
    format_sensor_scale,
    sensor_scale_metadata,
)
from refactored_mllm.point_cloud import load_raw_point_cloud
from refactored_mllm.schema import REQUIREMENT_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERNVL_SOURCE = PROJECT_ROOT / "vendor" / "internvl_chat"
DEFAULT_FINETUNED_CHECKPOINT = Path(
    os.environ.get("INTERNVL_VR_CHECKPOINT", PROJECT_ROOT / "models" / "teacher")
)


class InternVLVRPredictor:
    def __init__(
        self,
        *,
        checkpoint: str | Path = DEFAULT_FINETUNED_CHECKPOINT,
        internvl_source: str | Path = DEFAULT_INTERNVL_SOURCE,
        device: str = "cuda:0",
        max_num: int = 6,
        prompt_profile: str = "grid",
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.internvl_source = Path(internvl_source)
        self.device = device
        self.max_num = max_num
        self.prompt_profile = prompt_profile
        self.prompt = internvl_prompt_for_profile(prompt_profile)
        self.include_sensor_scale = prompt_profile_uses_sensor_scale(prompt_profile)
        self._load_runtime()

    def _load_runtime(self) -> None:
        if not (self.checkpoint / "config.json").is_file():
            raise FileNotFoundError(
                "Teacher weights are not distributed with MIMA. Supply an authorized "
                "checkpoint directory via --internvl-checkpoint, --checkpoint, or "
                "INTERNVL_VR_CHECKPOINT."
            )
        if str(self.internvl_source) not in sys.path:
            sys.path.insert(0, str(self.internvl_source))

        import torch
        from transformers import AutoTokenizer

        from refactored_mllm.internvl.preprocessing import load_image, load_depth
        from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel

        self.torch = torch
        self.load_image = load_image
        self.load_depth = load_depth
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
        )
        config = InternVLChatConfig.from_pretrained(self.checkpoint)
        attn_implementation = "eager"
        if config.llm_config.model_type == "internlm2":
            config.llm_config.attn_implementation = attn_implementation
        else:
            config.llm_config._attn_implementation = attn_implementation
        self.model = (
            InternVLChatModel.from_pretrained(
                self.checkpoint,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            .eval()
            .to(torch.device(self.device))
        )
        self.generation_config = {
            "num_beams": 1,
            "max_new_tokens": 128,
            "do_sample": False,
        }

    def predict(
        self,
        *,
        rgb_path: str | Path,
        depth_path: str | Path,
        point_cloud_path: str | Path,
        prompt: str | None = None,
    ) -> dict[str, object]:
        torch = self.torch
        effective_prompt = prompt or self.prompt
        if self.include_sensor_scale:
            effective_prompt = (
                f"{effective_prompt}\n"
                f"{format_sensor_scale(sensor_scale_metadata(depth_path, point_cloud_path, rgb_path=rgb_path))}"
            )
        pixel_values = self.load_image(rgb_path, max_num=self.max_num)
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        depth_values = self.load_depth(depth_path, max_num=self.max_num)
        if depth_values.ndim == 5 and depth_values.size(0) == 1:
            depth_values = depth_values.squeeze(0)
        if depth_values.ndim == 4 and depth_values.size(1) == 1:
            depth_values = depth_values.repeat(1, 3, 1, 1)
        depth_values = depth_values.to(torch.bfloat16).to(self.device)
        point_cloud = load_raw_point_cloud(
            _load_point_cloud_any(point_cloud_path),
            normalize=True,
        )
        point_cloud_values = (
            torch.from_numpy(point_cloud)
            .expand(pixel_values.size(0), -1, -1)
            .unsqueeze(0)
            .to(torch.bfloat16)
            .to(self.device)
        )

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            effective_prompt,
            self.generation_config,
            depth_values=depth_values,
            point_cloud=point_cloud_values,
        )
        parsed = parse_vr_response(response)
        return build_precision_audit_result(
            raw_model_text=response,
            parsed_v_r=parsed,
            prompt_profile=self.prompt_profile,
        )


def parse_vr_response(response: str) -> dict[str, float]:
    text = response.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    text = text.replace("'", '"')
    payload = json.loads(text)
    missing = [column for column in REQUIREMENT_COLUMNS if column not in payload]
    if missing:
        raise ValueError(
            f"InternVL response missing v_r keys: {missing}; raw={response!r}"
        )
    non_numeric = [
        column
        for column in REQUIREMENT_COLUMNS
        if isinstance(payload[column], bool)
        or not isinstance(payload[column], (int, float))
    ]
    if non_numeric:
        raise ValueError(
            f"InternVL response contains non-numeric v_r values for {non_numeric}; "
            f"raw={response!r}"
        )
    parsed = {column: float(payload[column]) for column in REQUIREMENT_COLUMNS}
    invalid = [column for column, value in parsed.items() if not math.isfinite(value)]
    if invalid:
        raise ValueError(
            f"InternVL response contains non-finite v_r values for {invalid}; raw={response!r}"
        )
    return parsed


def build_precision_audit_result(
    *,
    raw_model_text: str,
    parsed_v_r: dict[str, float],
    prompt_profile: str,
) -> dict[str, object]:
    parsed_before_postprocess = {
        column: float(parsed_v_r[column]) for column in REQUIREMENT_COLUMNS
    }
    final_v_r = dict(parsed_before_postprocess)
    native_subcentimeter = any(
        not _is_centimeter_grid_value(final_v_r[column]) for column in ("wp_m", "hp_m")
    )
    return {
        "raw_response": raw_model_text,
        "raw_model_text": raw_model_text,
        "parsed_v_r_before_postprocess": parsed_before_postprocess,
        "v_r": final_v_r,
        "precision": {
            "rounding_applied": False,
            "wp_hp_unit": "m",
            "source": "model_native",
            "prompt_profile": prompt_profile,
            "native_subcentimeter_output_observed": native_subcentimeter,
            "training_label_resolution_m": 0.01,
            "subcentimeter_accuracy_validated": False,
        },
    }


def _is_centimeter_grid_value(value: float, *, tolerance: float = 1e-9) -> bool:
    scaled = float(value) * 100.0
    return abs(scaled - round(scaled)) <= tolerance


def _load_point_cloud_any(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=float)
    from refactored_mllm.point_cloud import load_point_cloud

    return load_point_cloud(path)
