"""Run a small deterministic general-reasoning regression sentinel on base and adapter."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from run_stage7_adapter import inspect_adapter

from finpulse_llm.evaluation.stage3 import load_benchmark, score_responses
from finpulse_llm.inference.config import load_model_config
from finpulse_llm.inference.runner import run_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "benchmarks" / "stage3_base_models.json"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "models" / "qwen3_4b_eval.toml"
ADAPTER_DIR = PROJECT_ROOT / "models" / "adapters" / "finpulse-qwen3-4b-seed-v1"
OUTPUT = PROJECT_ROOT / "results" / "benchmarks" / "stage7_general_regression.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--reuse-base-report",
        type=Path,
        help="Reuse the validated base section from an existing sentinel report.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    adapter_dir = args.adapter_dir.resolve()
    output = args.output.resolve()
    reuse_base_report = (
        args.reuse_base_report.resolve() if args.reuse_base_report else None
    )
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cases = tuple(
        case for case in load_benchmark(BENCHMARK) if case.category == "general_reasoning"
    )
    config = load_model_config(MODEL_CONFIG)
    adapter = inspect_adapter(adapter_dir, config.model_id)
    if adapter["base_model_revision"] != config.revision:
        raise ValueError("Adapter and base configuration revisions differ")

    if reuse_base_report:
        prior = json.loads(reuse_base_report.read_text(encoding="utf-8"))
        if prior.get("case_ids") != [case.id for case in cases]:
            raise ValueError("Reusable base report has different sentinel cases")
        if prior.get("base_model_id") != config.model_id:
            raise ValueError("Reusable base report has a different model ID")
        if prior.get("base_model_revision") != config.revision:
            raise ValueError("Reusable base report has a different model revision")
        base_report = prior["base"]
    else:
        base_metrics = run_inference(config, [case.prompt for case in cases]).to_dict()
        base_responses = base_metrics.pop("results")
        base_report = {
            "model": base_metrics,
            "scoring": score_responses(cases, base_responses),
        }
    adapter_config = replace(config, model_id=str(adapter_dir), revision=None)
    adapter_metrics = run_inference(
        adapter_config, [case.prompt for case in cases]
    ).to_dict()
    adapter_responses = adapter_metrics.pop("results")
    report = {
        "schema_version": 1,
        "status": "complete",
        "source_benchmark": "benchmarks/stage3_base_models.json",
        "case_ids": [case.id for case in cases],
        "case_count": len(cases),
        "decoding": "greedy",
        "base_model_id": config.model_id,
        "base_model_revision": config.revision,
        "adapter": adapter,
        "base": base_report,
        "fine_tuned": {
            "model": adapter_metrics,
            "scoring": score_responses(cases, adapter_responses),
        },
        "limitation": (
            "This three-case development sentinel is not the frozen financial benchmark and "
            "is too small for a broad general-capability claim."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "base": report["base"]["scoring"]["overall_score"],
                "fine_tuned": report["fine_tuned"]["scoring"]["overall_score"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
