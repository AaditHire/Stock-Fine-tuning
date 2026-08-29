"""Run the frozen Stage 4 benchmark locally with resumable per-case checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from finpulse_llm.evaluation.stage4 import (
    load_frozen_benchmark,
    score_stage4_responses,
    verify_manifest,
)
from finpulse_llm.inference.config import load_model_config
from finpulse_llm.inference.runner import GenerationResult, run_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval" / "finpulse_eval_v1.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "eval" / "finpulse_eval_v1.manifest.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "models" / "qwen3_4b_eval.toml"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "benchmarks" / "stage4_qwen3_4b_baseline.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "results" / "benchmarks" / "stage4_qwen3_4b_baseline.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--limit", type=int, help="Run only the first N cases for a smoke test")
    parser.add_argument("--category", help="Run only one benchmark category")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _checkpoint(
    benchmark_hash: str,
    model_id: str,
    case_count: int,
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "in_progress",
        "benchmark_id": "finpulse_eval_v1",
        "benchmark_sha256": benchmark_hash,
        "model_id": model_id,
        "case_count": case_count,
        "completed_case_count": len(responses),
        "responses": responses,
    }


def _load_resume(
    path: Path, benchmark_hash: str, model_id: str, case_ids: list[str]
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    stored_model_id = raw.get("model_id", raw.get("model", {}).get("model_id"))
    if raw.get("benchmark_sha256") != benchmark_hash or stored_model_id != model_id:
        raise ValueError("Checkpoint does not match this benchmark and model")
    responses = list(raw.get("responses", raw.get("scoring", {}).get("cases", [])))
    completed_ids = [item["case_id"] for item in responses]
    if completed_ids != case_ids[: len(completed_ids)]:
        raise ValueError("Checkpoint responses are not the expected ordered prefix")
    return responses


def render_markdown(report: dict[str, Any]) -> str:
    scoring = report["scoring"]
    model = report["model"]
    cases = scoring["cases"]
    total_tokens = sum(item["output_tokens"] for item in cases)
    total_seconds = sum(item["generation_seconds"] for item in cases)
    weakest = sorted(scoring["category_scores"].items(), key=lambda item: item[1])[:3]
    lines = [
        "# Stage 4 Qwen3-4B frozen baseline",
        "",
        f"- Benchmark: `{report['benchmark_id']}` ({report['case_count']} cases)",
        f"- Overall: **{scoring['overall_score']:.1%}** "
        f"({scoring['checks_passed']}/{scoring['checks_total']} checks)",
        f"- Aggregate generation speed: **{total_tokens / total_seconds:.2f} tokens/second**",
        f"- Peak total device VRAM: **{model['peak_gpu_device_used_mib']:.1f} MiB**",
        f"- Peak process RAM: **{model['peak_process_ram_mib']:.1f} MiB**",
        "",
        "## Category scores",
        "",
        "| Category | Score |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {category} | {score:.1%} |"
        for category, score in scoring["category_scores"].items()
    )
    lines.extend(
        [
            "",
            "## Weakest categories",
            "",
            *(f"- `{category}`: {score:.1%}" for category, score in weakest),
            "",
            "This is the pre-training baseline. The benchmark is frozen and excluded from all "
            "future training data.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "huggingface"))
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    manifest = verify_manifest(args.dataset, args.manifest)
    all_cases = load_frozen_benchmark(args.dataset)
    selected_cases = (
        tuple(case for case in all_cases if case.category == args.category)
        if args.category
        else all_cases
    )
    if args.category and not selected_cases:
        raise ValueError(f"Unknown or empty category: {args.category}")
    cases = selected_cases[: args.limit] if args.limit else selected_cases
    config = load_model_config(args.model_config)
    case_ids = [case.id for case in cases]
    responses = (
        _load_resume(args.output, manifest["dataset_sha256"], config.model_id, case_ids)
        if args.resume
        else []
    )
    if len(responses) == len(cases):
        print("All requested cases are already complete; no model load needed.")
        return 0

    remaining = cases[len(responses) :]

    def save_result(result: GenerationResult) -> None:
        case = remaining[len(responses) - initial_response_count]
        item = {"case_id": case.id, **asdict(result)}
        responses.append(item)
        checkpoint = _checkpoint(
            manifest["dataset_sha256"], config.model_id, len(cases), responses
        )
        _write_json_atomic(args.output, checkpoint)

    initial_response_count = len(responses)
    metrics = run_inference(config, [case.prompt for case in remaining], on_result=save_result)
    metrics_dict = metrics.to_dict()
    metrics_dict.pop("results")
    scoring = score_stage4_responses(cases, responses)
    report = {
        "schema_version": 1,
        "status": "complete",
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_sha256": manifest["dataset_sha256"],
        "case_count": len(cases),
        "model": metrics_dict,
        "generation_config": str(args.model_config),
        "scoring": scoring,
    }
    _write_json_atomic(args.output, report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "model": metrics.model_id,
                "cases": len(cases),
                "overall_score": scoring["overall_score"],
                "category_scores": scoring["category_scores"],
            },
            indent=2,
        )
    )
    print(f"Saved baseline to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
