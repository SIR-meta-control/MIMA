from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "sensor_clean_init"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a clean InternVL sensor checkpoint from the open "
            "Mini-InternVL base without inheriting legacy 4-code fine-tuning."
        )
    )
    parser.add_argument("--open-base", type=Path, required=True)
    parser.add_argument(
        "--legacy-sensor-config",
        type=Path,
        required=True,
        help="Checkpoint/config that contains the legacy depth/point-cloud branch config.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--internvl-source",
        type=Path,
        default=PROJECT_ROOT / "vendor" / "internvl_chat",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if str(args.internvl_source) not in sys.path:
        sys.path.insert(0, str(args.internvl_source))

    import torch
    from transformers import AutoTokenizer

    from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel

    open_config = json.loads(
        (args.open_base / "config.json").read_text(encoding="utf-8")
    )
    legacy_config = json.loads(
        (args.legacy_sensor_config / "config.json").read_text(encoding="utf-8")
    )
    open_config["point_cloud_config"] = legacy_config["point_cloud_config"]
    open_config["template"] = "phi3-chat"
    open_config["dynamic_image_size"] = True
    open_config["use_thumbnail"] = True
    open_config["max_dynamic_patch"] = 6
    open_config["min_dynamic_patch"] = 1
    open_config["force_image_size"] = 448
    open_config["downsample_ratio"] = 0.5
    open_config["select_layer"] = -1
    open_config["_name_or_path"] = str(args.open_base)

    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(open_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = InternVLChatConfig.from_pretrained(output_dir)
    if config.llm_config.model_type == "internlm2":
        config.llm_config.attn_implementation = "eager"
    else:
        config.llm_config._attn_implementation = "eager"

    model = InternVLChatModel.from_pretrained(
        args.open_base,
        config=config,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )

    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(
        args.open_base,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.save_pretrained(output_dir)
    for name in ("generation_config.json", "preprocessor_config.json"):
        source = args.open_base / name
        if source.exists():
            shutil.copy2(source, output_dir / name)

    metadata = {
        "source_open_base": str(args.open_base),
        "legacy_sensor_config": str(args.legacy_sensor_config),
        "inherits_legacy_4code_weights": False,
        "notes": (
            "Vision/language weights are loaded from the open Mini-InternVL base. "
            "Depth/point-cloud branch config is injected so the vendored legacy "
            "sensor model can consume RGBD/point-cloud inputs; missing branch "
            "weights are initialized by the model constructor/from_pretrained."
        ),
    }
    (output_dir / "clean_sensor_init_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
