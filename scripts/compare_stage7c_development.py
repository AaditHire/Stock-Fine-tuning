"""Compare all Stage 7C development reports and select at most one candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_stage4_baseline import _write_json_atomic

from finpulse_llm.evaluation.stage7c import compare_candidates, render_comparison_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=PROJECT_ROOT / "results/benchmarks/stage7c_dev_base.json",
    )
    parser.add_argument("--candidate", type=Path, action="append")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "results/benchmarks/stage7c_dev_comparison.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "results/benchmarks/stage7c_dev_comparison.md",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidate_paths = args.candidate or [
        PROJECT_ROOT / f"results/benchmarks/stage7c_dev_{name}.json"
        for name in ("checkpoint-15", "checkpoint-30", "checkpoint-45", "final")
    ]
    base = json.loads(args.base.read_text(encoding="utf-8"))
    candidates = [json.loads(path.read_text(encoding="utf-8")) for path in candidate_paths]
    report = compare_candidates(base, candidates)
    _write_json_atomic(args.json_output, report)
    args.markdown_output.write_text(render_comparison_markdown(report), encoding="utf-8")
    summary = {
        "selected_candidate": report["selected_candidate"],
        "candidates": [
            {
                "candidate": item["candidate"],
                "answer_accuracy": item["answer_accuracy"],
                "gate_passed": item["gate_passed"],
            }
            for item in report["candidates"]
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
