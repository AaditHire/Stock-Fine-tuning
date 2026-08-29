"""Run the pinned Stage 6 local QLoRA experiment."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from finpulse_llm.training.config import load_training_config
from finpulse_llm.training.runner import run_training, verify_training_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/qwen3_4b_stage6.toml"),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify configuration and locked dataset hashes without loading the model.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one optimizer step on two examples before the full experiment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config = load_training_config(args.config)
    if args.preflight_only:
        result = {
            "status": "preflight_passed",
            "run_name": config.run_name,
            "effective_batch_size": config.trainer.effective_batch_size,
            "input_hashes": verify_training_inputs(config),
        }
    else:
        result = run_training(config, smoke_test=args.smoke_test)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
