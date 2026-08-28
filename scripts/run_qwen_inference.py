"""Run memory-safe Qwen3-4B inference and save a machine-readable report."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from finpulse_llm.inference.config import load_model_config
from finpulse_llm.inference.prompts import FINANCE_SMOKE_PROMPTS
from finpulse_llm.inference.runner import run_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "models" / "qwen3_4b.toml"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "benchmarks" / "stage2_qwen3_4b.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use only files already present in the local Hugging Face cache",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="custom prompt; repeat for multiple prompts (defaults to the Stage 2 smoke suite)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "huggingface"))
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    config = load_model_config(args.config)
    prompts = tuple(args.prompt) if args.prompt else FINANCE_SMOKE_PROMPTS
    metrics = run_inference(config, prompts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(metrics.to_dict(), indent=2))
    print(f"Saved report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
