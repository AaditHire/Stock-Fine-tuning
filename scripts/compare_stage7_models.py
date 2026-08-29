"""Create machine-readable and Markdown Stage 7 comparison reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finpulse_llm.evaluation.stage4 import load_frozen_benchmark, verify_manifest
from finpulse_llm.evaluation.stage7 import compare_reports, render_comparison_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=PROJECT_ROOT / "results/benchmarks/stage4_qwen3_4b_baseline.json",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=PROJECT_ROOT / "results/benchmarks/stage7_qwen3_4b_adapter.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/eval/finpulse_eval_v1.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/eval/finpulse_eval_v1.manifest.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "results/benchmarks/stage7_comparison.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "results/benchmarks/stage7_comparison.md",
    )
    parser.add_argument(
        "--general-regression",
        type=Path,
        default=PROJECT_ROOT / "results/benchmarks/stage7_general_regression.json",
    )
    parser.add_argument("--training-examples", type=int, default=33)
    parser.add_argument("--validation-examples", type=int, default=7)
    parser.add_argument("--stage-label", default="Stage 7")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    verify_manifest(args.dataset, args.manifest)
    frozen_cases = load_frozen_benchmark(args.dataset)
    base = json.loads(args.base.read_text(encoding="utf-8"))
    adapter = json.loads(args.adapter.read_text(encoding="utf-8"))
    general_regression = (
        json.loads(args.general_regression.read_text(encoding="utf-8"))
        if args.general_regression.is_file()
        else None
    )
    report = compare_reports(
        base,
        adapter,
        frozen_cases,
        general_regression,
        training_examples=args.training_examples,
        validation_examples=args.validation_examples,
        stage_label=args.stage_label,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_comparison_markdown(report), encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "conclusion": report["conclusion"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
