"""Evaluate the Stage 6 adapter on the untouched frozen Stage 4 benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from run_stage4_baseline import _load_resume, _write_json_atomic

from finpulse_llm.evaluation.stage4 import (
    load_frozen_benchmark,
    score_stage4_responses,
    sha256_file,
    verify_manifest,
)
from finpulse_llm.inference.config import load_model_config
from finpulse_llm.inference.runner import GenerationResult, run_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval" / "finpulse_eval_v1.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "eval" / "finpulse_eval_v1.manifest.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "models" / "qwen3_4b_eval.toml"
DEFAULT_ADAPTER = PROJECT_ROOT / "models" / "adapters" / "finpulse-qwen3-4b-seed-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "benchmarks" / "stage7_qwen3_4b_adapter.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--expected-adapter-sha256",
        help="Required identity lock when a trainer checkpoint omits base revision metadata",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Run only the first N cases for a smoke test")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def inspect_adapter(adapter_dir: Path, expected_base_model: str) -> dict[str, Any]:
    """Verify the adapter identity without loading the 4B base model."""

    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Incomplete LoRA adapter directory: {adapter_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != expected_base_model:
        raise ValueError("Adapter base model does not match the frozen evaluation model")
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Adapter is not a causal-language-model LoRA")
    return {
        "directory": str(adapter_dir.relative_to(PROJECT_ROOT)),
        "weights_sha256": sha256_file(weights_path),
        "weights_size_bytes": weights_path.stat().st_size,
        "base_model_id": config["base_model_name_or_path"],
        "base_model_revision": config.get("revision"),
        "rank": config["r"],
        "alpha": config["lora_alpha"],
        "target_modules": sorted(config["target_modules"]),
    }


def _checkpoint(
    benchmark_hash: str,
    adapter_identity: str,
    case_count: int,
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "in_progress",
        "benchmark_id": "finpulse_eval_v1",
        "benchmark_sha256": benchmark_hash,
        "model_id": adapter_identity,
        "case_count": case_count,
        "completed_case_count": len(responses),
        "responses": responses,
    }


def main() -> int:
    args = _parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    manifest = verify_manifest(args.dataset, args.manifest)
    all_cases = load_frozen_benchmark(args.dataset)
    cases = all_cases[: args.limit] if args.limit else all_cases
    base_config = load_model_config(args.model_config)
    adapter = inspect_adapter(args.adapter_dir.resolve(), base_config.model_id)
    if (
        args.expected_adapter_sha256 is not None
        and adapter["weights_sha256"] != args.expected_adapter_sha256.casefold()
    ):
        raise ValueError("Adapter SHA-256 does not match the explicitly selected candidate")
    if adapter["base_model_revision"] is None and args.expected_adapter_sha256 is None:
        raise ValueError(
            "Adapter omits base revision metadata; provide --expected-adapter-sha256"
        )
    if (
        adapter["base_model_revision"] is not None
        and adapter["base_model_revision"] != base_config.revision
    ):
        raise ValueError("Adapter and evaluation configuration pin different base revisions")
    adapter_identity = f"{base_config.model_id}+lora@{adapter['weights_sha256']}"
    case_ids = [case.id for case in cases]
    responses = (
        _load_resume(args.output, manifest["dataset_sha256"], adapter_identity, case_ids)
        if args.resume
        else []
    )
    if len(responses) == len(cases):
        print("All requested adapter cases are already complete; no model load needed.")
        return 0

    remaining = cases[len(responses) :]
    initial_response_count = len(responses)

    def save_result(result: GenerationResult) -> None:
        case = remaining[len(responses) - initial_response_count]
        responses.append({"case_id": case.id, **asdict(result)})
        _write_json_atomic(
            args.output,
            _checkpoint(manifest["dataset_sha256"], adapter_identity, len(cases), responses),
        )

    # Unsloth resolves a local PEFT directory, loads its pinned base, then attaches the adapter.
    inference_config = replace(
        base_config,
        model_id=str(args.adapter_dir.resolve()),
        revision=None,
    )
    metrics = run_inference(
        inference_config,
        [case.prompt for case in remaining],
        on_result=save_result,
    )
    metrics_dict = metrics.to_dict()
    metrics_dict.pop("results")
    metrics_dict["model_id"] = base_config.model_id
    metrics_dict["model_revision"] = base_config.revision
    scoring = score_stage4_responses(cases, responses)
    report = {
        "schema_version": 1,
        "status": "complete",
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_sha256": manifest["dataset_sha256"],
        "case_count": len(cases),
        "model_id": adapter_identity,
        "model": metrics_dict,
        "adapter": adapter,
        "generation_config": str(args.model_config.relative_to(PROJECT_ROOT)),
        "scoring": scoring,
    }
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "overall_score": scoring["overall_score"],
                "checks_passed": scoring["checks_passed"],
                "checks_total": scoring["checks_total"],
                "category_scores": scoring["category_scores"],
            },
            indent=2,
        )
    )
    print(f"Saved adapter evaluation to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
