"""Run one Stage 7C candidate on the locked Stage 5C development split."""

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
from finpulse_llm.evaluation.stage7c import score_development_responses
from finpulse_llm.inference.config import load_model_config
from finpulse_llm.inference.runner import GenerationResult, run_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data/development/finpulse_stage5c_v1.jsonl"
MANIFEST = PROJECT_ROOT / "data/processed/finpulse_stage5c_v1.manifest.json"
CONFIG = PROJECT_ROOT / "configs/models/qwen3_4b_stage7c_dev.toml"
CANDIDATES = {
    "checkpoint-15": PROJECT_ROOT / "models/checkpoints/stage6c-qwen3-4b-stage5d-v1/checkpoint-15",
    "checkpoint-30": PROJECT_ROOT / "models/checkpoints/stage6c-qwen3-4b-stage5d-v1/checkpoint-30",
    "checkpoint-45": PROJECT_ROOT / "models/checkpoints/stage6c-qwen3-4b-stage5d-v1/checkpoint-45",
    "final": PROJECT_ROOT / "models/adapters/finpulse-qwen3-4b-stage5d-v1",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("base", *CANDIDATES), required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--model-config", type=Path, default=CONFIG)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    # Canonical Stage 7C reports record batch 8, but the longest final-adapter batch
    # peaked at 5,939.6 MiB. Default future invocations to the safer measured fallback.
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def _checkpoint(
    dataset_hash: str,
    model_id: str,
    case_count: int,
    batch_size: int,
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "in_progress",
        "benchmark_id": "finpulse_stage5c_development",
        "benchmark_sha256": dataset_hash,
        "dataset_sha256": dataset_hash,
        "model_id": model_id,
        "case_count": case_count,
        "inference_batch_size": batch_size,
        "completed_case_count": len(responses),
        "responses": responses,
    }


def main() -> int:
    args = _parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset_hash = file_sha256(args.dataset)
    if dataset_hash != manifest.get("development_sha256"):
        raise ValueError("Development split SHA-256 does not match the locked manifest")
    all_examples = load_jsonl([args.dataset])
    examples = all_examples[: args.limit] if args.limit else all_examples
    config = load_model_config(args.model_config)
    system_prompts = {item["messages"][0]["content"] for item in examples}
    if system_prompts != {config.system_prompt}:
        raise ValueError("Generation system prompt differs from the locked development data")

    adapter = None
    inference_config = config
    model_identity = config.model_id
    if args.candidate != "base":
        adapter_dir = (args.adapter_dir or CANDIDATES[args.candidate]).resolve()
        adapter = inspect_adapter(adapter_dir, config.model_id)
        revision = adapter["base_model_revision"]
        if revision is not None and revision != config.revision:
            raise ValueError("Adapter and development config pin different base revisions")
        model_identity = f"{config.model_id}+lora@{adapter['weights_sha256']}"
        inference_config = replace(config, model_id=str(adapter_dir), revision=None)
    output = args.output or PROJECT_ROOT / f"results/benchmarks/stage7c_dev_{args.candidate}.json"
    case_ids = [item["id"] for item in examples]
    if args.resume and output.is_file():
        checkpoint = json.loads(output.read_text(encoding="utf-8"))
        if checkpoint.get("inference_batch_size") != args.batch_size:
            raise ValueError("Checkpoint uses a different inference batch size")
    responses = _load_resume(output, dataset_hash, model_identity, case_ids) if args.resume else []
    if len(responses) == len(examples):
        print("All requested development cases are already complete; no model load needed.")
        return 0

    remaining = examples[len(responses) :]
    initial_count = len(responses)

    def save_result(result: GenerationResult) -> None:
        example = remaining[len(responses) - initial_count]
        responses.append({"case_id": example["id"], **asdict(result)})
        _write_json_atomic(
            output,
            _checkpoint(
                dataset_hash,
                model_identity,
                len(examples),
                args.batch_size,
                responses,
            ),
        )

    metrics = run_inference(
        inference_config,
        [item["messages"][1]["content"] for item in remaining],
        on_result=save_result,
        batch_size=args.batch_size,
    )
    metrics_dict = metrics.to_dict()
    metrics_dict.pop("results")
    metrics_dict["model_id"] = config.model_id
    metrics_dict["model_revision"] = config.revision
    scoring = score_development_responses(examples, responses)
    report = {
        "schema_version": 1,
        "status": "complete",
        "benchmark_id": "finpulse_stage5c_development",
        "dataset_sha256": dataset_hash,
        "case_count": len(examples),
        "candidate": args.candidate,
        "model_id": model_identity,
        "model": metrics_dict,
        "adapter": adapter,
        "generation_config": str(args.model_config.relative_to(PROJECT_ROOT)),
        "inference_batch_size": args.batch_size,
        "scoring": scoring,
    }
    _write_json_atomic(output, report)
    print(
        json.dumps(
            {
                "candidate": args.candidate,
                "answer_accuracy": scoring["answer_accuracy"],
                "format_accuracy": scoring["format_accuracy"],
                "task_type_scores": scoring["task_type_scores"],
                "source_scores": scoring["source_scores"],
            },
            indent=2,
        )
    )
    print(f"Saved development report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
