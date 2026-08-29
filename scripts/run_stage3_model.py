"""Run and objectively score one model on the Stage 3 benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from finpulse_llm.evaluation.stage3 import load_benchmark, score_responses
from finpulse_llm.inference.config import load_model_config
from finpulse_llm.inference.runner import run_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = PROJECT_ROOT / "benchmarks" / "stage3_base_models.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "huggingface"))
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    cases = load_benchmark(args.benchmark)
    config = load_model_config(args.model_config)
    metrics = run_inference(config, [case.prompt for case in cases])
    metrics_dict = metrics.to_dict()
    scoring = score_responses(cases, metrics_dict.pop("results"))
    report = {
        "benchmark": str(args.benchmark),
        "benchmark_case_count": len(cases),
        "model": metrics_dict,
        "scoring": scoring,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": metrics.model_id, **scoring}, indent=2))
    print(f"Saved report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
