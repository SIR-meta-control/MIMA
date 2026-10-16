from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refactored_mllm.internvl.inference import (
    DEFAULT_FINETUNED_CHECKPOINT,
    DEFAULT_INTERNVL_SOURCE,
)
from refactored_mllm.internvl.prompts import PROMPT_PROFILES
from refactored_mllm.distillation import DEFAULT_STUDENT_MODEL_DIR
from refactored_mllm.models import DEFAULT_MODEL_DIR, allowed_model_keys
from refactored_mllm.service import VRModelService


ABS_DEFAULT_MODEL_DIR = PROJECT_ROOT / DEFAULT_MODEL_DIR
ABS_DEFAULT_STUDENT_MODEL_DIR = PROJECT_ROOT / DEFAULT_STUDENT_MODEL_DIR


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the refactored MLLM VR service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2610)
    parser.add_argument(
        "--backend", choices=("internvl", "baseline", "student"), default="internvl"
    )
    parser.add_argument("--model", choices=allowed_model_keys(), default="dt")
    parser.add_argument("--model-dir", type=Path, default=ABS_DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--student-model-dir", type=Path, default=ABS_DEFAULT_STUDENT_MODEL_DIR
    )
    parser.add_argument(
        "--internvl-checkpoint", type=Path, default=DEFAULT_FINETUNED_CHECKPOINT
    )
    parser.add_argument("--internvl-source", type=Path, default=DEFAULT_INTERNVL_SOURCE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--internvl-prompt-profile",
        choices=tuple(PROMPT_PROFILES),
        default="grid",
        help=(
            "InternVL prompt contract. Use checkpoint_native to reproduce the June 30 "
            "checkpoint and continuous_audit only for native-precision auditing."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    service = VRModelService(
        host=args.host,
        port=args.port,
        model_dir=args.model_dir,
        student_model_dir=args.student_model_dir,
        default_model=args.model,
        backend=args.backend,
        internvl_checkpoint=args.internvl_checkpoint,
        internvl_source=args.internvl_source,
        device=args.device,
        internvl_prompt_profile=args.internvl_prompt_profile,
    )
    service.run()


if __name__ == "__main__":
    main()
