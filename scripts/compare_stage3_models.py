"""Compare two scored Stage 3 reports and select the stronger base model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs=2, type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_tokens_per_second(report: dict[str, Any]) -> float:
    cases = report["scoring"]["cases"]
    tokens = sum(case["output_tokens"] for case in cases)
    seconds = sum(case["generation_seconds"] for case in cases)
    return tokens / seconds


def _selection_key(report: dict[str, Any]) -> tuple[float, float, float, float]:
    scores = report["scoring"]
    categories = scores["category_scores"]
    return (
        float(scores["overall_score"]),
        float(categories.get("hallucination_resistance", 0)),
        float(categories.get("financial_calculation", 0)),
        _aggregate_tokens_per_second(report),
    )


def _model_summary(report: dict[str, Any]) -> dict[str, Any]:
    model = report["model"]
    scoring = report["scoring"]
    return {
        "model_id": model["model_id"],
        "overall_score": scoring["overall_score"],
        "checks_passed": scoring["checks_passed"],
        "checks_total": scoring["checks_total"],
        "category_scores": scoring["category_scores"],
        "aggregate_tokens_per_second": round(_aggregate_tokens_per_second(report), 3),
        "load_seconds": model["load_seconds"],
        "peak_process_ram_mib": model["peak_process_ram_mib"],
        "peak_gpu_device_used_mib": model["peak_gpu_device_used_mib"],
        "peak_gpu_allocated_mib": model["peak_gpu_allocated_mib"],
    }


def _markdown(comparison: dict[str, Any]) -> str:
    summaries = comparison["models"]
    categories = sorted(
        set().union(*(summary["category_scores"].keys() for summary in summaries))
    )
    lines = [
        "# Stage 3 base-model comparison",
        "",
        "This is a small development benchmark, not the frozen Stage 4 evaluation set.",
        "",
        "## Decision",
        "",
        f"Selected base model: `{comparison['selected_model']}`.",
        "",
        comparison["selection_reason"],
        "",
        "## Overall results",
        "",
        "| Model | Score | Checks | tok/s | Load (s) | Device VRAM (MiB) | Process RAM (MiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| `{summary['model_id']}` | {summary['overall_score']:.1%} | "
            f"{summary['checks_passed']}/{summary['checks_total']} | "
            f"{summary['aggregate_tokens_per_second']:.2f} | {summary['load_seconds']:.2f} | "
            f"{summary['peak_gpu_device_used_mib']:.1f} | "
            f"{summary['peak_process_ram_mib']:.1f} |"
        )
    lines.extend(["", "## Category scores", ""])
    lines.append("| Category | " + " | ".join(f"`{s['model_id']}`" for s in summaries) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in summaries) + " |")
    for category in categories:
        values = [f"{summary['category_scores'].get(category, 0):.1%}" for summary in summaries]
        lines.append(f"| {category} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Selection rule",
            "",
            "Models are ranked by overall automated score, then hallucination resistance, "
            "financial calculation score, and finally generation speed. Resource measurements "
            "remain visible and can override the result only if a model is impractical on "
            "6 GB VRAM.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    reports = [_load(path) for path in args.reports]
    ranked = sorted(reports, key=_selection_key, reverse=True)
    selected = ranked[0]
    runner_up = ranked[1]
    selected_id = selected["model"]["model_id"]
    selected_score = selected["scoring"]["overall_score"]
    runner_up_score = runner_up["scoring"]["overall_score"]
    reason = (
        f"The selected model scored {selected_score:.1%} versus {runner_up_score:.1%}. "
        "Tie-breakers prioritize hallucination resistance and financial calculations before speed."
    )
    comparison = {
        "selected_model": selected_id,
        "selection_reason": reason,
        "selection_rule": [
            "overall_score",
            "hallucination_resistance",
            "financial_calculation",
            "aggregate_tokens_per_second",
        ],
        "models": [_model_summary(report) for report in reports],
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(_markdown(comparison), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
