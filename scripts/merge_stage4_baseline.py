"""Merge a corrected category rerun into the complete Stage 4 baseline and re-score it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_stage4_baseline import _write_json_atomic, render_markdown

from finpulse_llm.evaluation.stage4 import (
    load_frozen_benchmark,
    score_stage4_responses,
    verify_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/eval/finpulse_eval_v1.jsonl"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/eval/finpulse_eval_v1.manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def _responses(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(report["scoring"]["cases"])


def main() -> int:
    args = _args()
    manifest = verify_manifest(args.dataset, args.manifest)
    cases = load_frozen_benchmark(args.dataset)
    original = json.loads(args.original.read_text(encoding="utf-8"))
    replacement = json.loads(args.replacement.read_text(encoding="utf-8"))
    merged = {item["case_id"]: item for item in _responses(original)}
    merged.update({item["case_id"]: item for item in _responses(replacement)})
    ordered: list[dict[str, Any]] = []
    for case in cases:
        response = merged.get(case.id)
        if response is None:
            raise ValueError(f"Missing response for {case.id}")
        if response["prompt"] != case.prompt:
            raise ValueError(f"Response prompt does not match final benchmark for {case.id}")
        response = {
            key: value
            for key, value in response.items()
            if key not in {"category", "score", "checks"}
        }
        ordered.append(response)

    scoring = score_stage4_responses(cases, ordered)
    run_models = [original["model"], replacement["model"]]
    model = dict(original["model"])
    for field in (
        "peak_process_ram_mib",
        "peak_system_ram_used_mib",
        "peak_gpu_device_used_mib",
        "peak_gpu_allocated_mib",
        "peak_gpu_reserved_mib",
    ):
        model[field] = max(run[field] for run in run_models)
    report = {
        "schema_version": 1,
        "status": "complete",
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_sha256": manifest["dataset_sha256"],
        "case_count": len(cases),
        "model": model,
        "generation_config": original["generation_config"],
        "run_count": 2,
        "merge_note": "144 unchanged responses plus a 16-case corrected calculation rerun",
        "scoring": scoring,
    }
    _write_json_atomic(args.output, report)
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    summary = {
        "overall_score": scoring["overall_score"],
        "category_scores": scoring["category_scores"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
