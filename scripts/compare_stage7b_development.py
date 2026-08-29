"""Compare Stage 7B development reports and apply the frozen-run gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finpulse_llm.evaluation.stage7b import (
    compare_development_reports,
    render_development_comparison,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=PROJECT_ROOT / "results/benchmarks/stage7b_dev_base.json",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=PROJECT_ROOT / "results/benchmarks/stage7b_dev_adapter.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "results/benchmarks/stage7b_dev_comparison.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "results/benchmarks/stage7b_dev_comparison.md",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    adapter = json.loads(args.adapter.read_text(encoding="utf-8"))
    report = compare_development_reports(base, adapter)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_development_comparison(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
