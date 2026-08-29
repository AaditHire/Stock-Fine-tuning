"""Run base Qwen or the Stage 6B adapter on the Stage 5B development screen."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from run_stage4_baseline import _load_resume, _write_json_atomic
from run_stage7_adapter import inspect_adapter

from finpulse_llm.data.pipeline import file_sha256, load_jsonl
from finpulse_llm.evaluation.stage7b import score_development_responses
from finpulse_llm.inference.config import load_model_config
from finpulse_llm.inference.runner import GenerationResult, run_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data/development/finpulse_stage5b_v1.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/processed/finpulse_stage5b_v1.manifest.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/models/qwen3_4b_stage7b_dev.toml"
DEFAULT_ADAPTER = PROJECT_ROOT / "models/adapters/finpulse-qwen3-4b-stage5b-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("base", "adapter"), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _checkpoint(
    dataset_hash: str,
    model_id: str,
    case_count: int,
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "in_progress",
        "benchmark_id": "finpulse_stage5b_development",
        "benchmark_sha256": dataset_hash,
        "dataset_sha256": dataset_hash,
        "model_id": model_id,
        "case_count": case_count,
        "completed_case_count": len(responses),
        "responses": responses,
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset_hash = file_sha256(args.dataset)
    if dataset_hash != manifest.get("development_sha256"):
        raise ValueError("Development split SHA-256 does not match the locked manifest")
    examples = load_jsonl([args.dataset])
    config = load_model_config(args.model_config)
    adapter = None
    inference_config = config
    model_identity = config.model_id
    if args.model == "adapter":
        adapter = inspect_adapter(args.adapter_dir.resolve(), config.model_id)
        if adapter["base_model_revision"] != config.revision:
            raise ValueError("Adapter and development config pin different base revisions")
        model_identity = f"{config.model_id}+lora@{adapter['weights_sha256']}"
        inference_config = replace(config, model_id=str(args.adapter_dir.resolve()), revision=None)
    output = args.output or (
        PROJECT_ROOT / f"results/benchmarks/stage7b_dev_{args.model}.json"
    )
    case_ids = [item["id"] for item in examples]
    responses = (
        _load_resume(output, dataset_hash, model_identity, case_ids) if args.resume else []
    )
    if len(responses) == len(examples):
        print("All development cases are already complete; no model load needed.")
        return 0
    remaining = examples[len(responses) :]
    initial_count = len(responses)

    def save_result(result: GenerationResult) -> None:
        example = remaining[len(responses) - initial_count]
        responses.append({"case_id": example["id"], **asdict(result)})
        _write_json_atomic(
            output,
            _checkpoint(dataset_hash, model_identity, len(examples), responses),
        )

    metrics = run_inference(
        inference_config,
        [item["messages"][1]["content"] for item in remaining],
        on_result=save_result,
    )
    metrics_dict = metrics.to_dict()
    metrics_dict.pop("results")
    metrics_dict["model_id"] = config.model_id
    metrics_dict["model_revision"] = config.revision
    scoring = score_development_responses(examples, responses)
    report = {
        "schema_version": 1,
        "status": "complete",
        "benchmark_id": "finpulse_stage5b_development",
        "dataset_sha256": dataset_hash,
        "case_count": len(examples),
        "model_id": model_identity,
        "model": metrics_dict,
        "adapter": adapter,
        "generation_config": str(args.model_config.relative_to(PROJECT_ROOT)),
        "scoring": scoring,
    }
    _write_json_atomic(output, report)
    print(
        json.dumps(
            {
                "model": args.model,
                "overall_score": scoring["overall_score"],
                "checks_passed": scoring["checks_passed"],
                "checks_total": scoring["checks_total"],
                "task_type_scores": scoring["task_type_scores"],
            },
            indent=2,
        )
    )
    print(f"Saved development report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
