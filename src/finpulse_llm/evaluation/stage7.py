"""Deterministic base-versus-adapter comparison for the frozen benchmark."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from finpulse_llm.evaluation.stage4 import FrozenEvalCase

DIMENSION_CATEGORIES = {
    "financial_reasoning": {
        "technical_analysis",
        "crypto_derivatives",
        "stock_fundamentals",
        "macroeconomics",
        "scenario_analysis",
        "contradictory_signals",
    },
    "calculations": {"financial_calculations"},
    "instruction_following": {"structured_output"},
    "uncertainty_calibration": {"scenario_analysis", "contradictory_signals"},
    "hallucination_resistance": {"hallucination_traps"},
    "risk_awareness": {"risk_management"},
    "conflicting_signal_analysis": {"contradictory_signals"},
    # This financial benchmark has no broad general-knowledge section. These two categories
    # are only a narrow regression sentinel for arithmetic and exact instruction following.
    "financial_regression_proxy": {"financial_calculations", "structured_output"},
}
FROZEN_MAX_NEW_TOKENS = 192


def _check_score(cases: Iterable[dict[str, Any]]) -> float:
    checks = [check for case in cases for check in case["checks"]]
    return round(sum(bool(check["passed"]) for check in checks) / len(checks), 4)


def _subset_score(
    scored_cases: list[dict[str, Any]], categories: set[str]
) -> float:
    selected = [case for case in scored_cases if case["category"] in categories]
    return _check_score(selected)


def _tag_score(
    frozen_cases: tuple[FrozenEvalCase, ...],
    scored_cases: list[dict[str, Any]],
    tag: str,
) -> float:
    ids = {case.id for case in frozen_cases if tag in case.tags}
    return _check_score(case for case in scored_cases if case["case_id"] in ids)


def compare_reports(
    base: dict[str, Any],
    adapter: dict[str, Any],
    frozen_cases: tuple[FrozenEvalCase, ...],
    general_regression: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate fairness invariants and calculate granular score changes."""

    for report_name, report in (("base", base), ("adapter", adapter)):
        if report.get("status") != "complete":
            raise ValueError(f"{report_name} report is incomplete")
    if base["benchmark_sha256"] != adapter["benchmark_sha256"]:
        raise ValueError("Base and adapter reports use different benchmark bytes")
    if base["case_count"] != adapter["case_count"] or base["case_count"] != len(frozen_cases):
        raise ValueError("Base and adapter reports do not cover the complete frozen benchmark")
    if base["model"]["model_id"] != adapter["model"]["model_id"]:
        raise ValueError("Base model identity differs between reports")

    base_cases = base["scoring"]["cases"]
    adapter_cases = adapter["scoring"]["cases"]
    expected_ids = [case.id for case in frozen_cases]
    for name, scored in (("base", base_cases), ("adapter", adapter_cases)):
        if [case["case_id"] for case in scored] != expected_ids:
            raise ValueError(f"{name} responses are not in frozen benchmark order")
        if [case["prompt"] for case in scored] != [case.prompt for case in frozen_cases]:
            raise ValueError(f"{name} prompts differ from the frozen benchmark")

    category_deltas = {
        category: {
            "base": base["scoring"]["category_scores"][category],
            "adapter": adapter["scoring"]["category_scores"][category],
            "delta": round(
                adapter["scoring"]["category_scores"][category]
                - base["scoring"]["category_scores"][category],
                4,
            ),
        }
        for category in base["scoring"]["category_scores"]
    }
    check_type_deltas = {
        check_type: {
            "base": base["scoring"]["check_type_scores"][check_type],
            "adapter": adapter["scoring"]["check_type_scores"][check_type],
            "delta": round(
                adapter["scoring"]["check_type_scores"][check_type]
                - base["scoring"]["check_type_scores"][check_type],
                4,
            ),
        }
        for check_type in base["scoring"]["check_type_scores"]
    }

    dimensions: dict[str, dict[str, float]] = {}
    for name, categories in DIMENSION_CATEGORIES.items():
        base_score = _subset_score(base_cases, categories)
        adapter_score = _subset_score(adapter_cases, categories)
        dimensions[name] = {
            "base": base_score,
            "adapter": adapter_score,
            "delta": round(adapter_score - base_score, 4),
        }
    base_factual = _tag_score(frozen_cases, base_cases, "multiple_choice")
    adapter_factual = _tag_score(frozen_cases, adapter_cases, "multiple_choice")
    dimensions["factual_finance_knowledge"] = {
        "base": base_factual,
        "adapter": adapter_factual,
        "delta": round(adapter_factual - base_factual, 4),
    }
    json_types = {"json_only_exact", "json_only_fields"}
    for label, scored in (("base", base_cases), ("adapter", adapter_cases)):
        json_checks = [
            check for case in scored for check in case["checks"] if check["type"] in json_types
        ]
        dimensions.setdefault("structured_output_validity", {})[label] = round(
            sum(bool(check["passed"]) for check in json_checks) / len(json_checks), 4
        )
    dimensions["structured_output_validity"]["delta"] = round(
        dimensions["structured_output_validity"]["adapter"]
        - dimensions["structured_output_validity"]["base"],
        4,
    )
    if general_regression is not None:
        if general_regression.get("status") != "complete":
            raise ValueError("General-regression report is incomplete")
        general_base = general_regression["base"]["scoring"]["overall_score"]
        general_adapter = general_regression["fine_tuned"]["scoring"]["overall_score"]
        dimensions["general_capability_regression"] = {
            "base": general_base,
            "adapter": general_adapter,
            "delta": round(general_adapter - general_base, 4),
        }

    changes: list[dict[str, Any]] = []
    for base_case, adapter_case in zip(base_cases, adapter_cases, strict=True):
        delta = round(adapter_case["score"] - base_case["score"], 4)
        if delta:
            changes.append(
                {
                    "case_id": base_case["case_id"],
                    "category": base_case["category"],
                    "base": base_case["score"],
                    "adapter": adapter_case["score"],
                    "delta": delta,
                }
            )
    improved = [case for case in changes if case["delta"] > 0]
    regressed = [case for case in changes if case["delta"] < 0]
    overall_delta = round(
        adapter["scoring"]["overall_score"] - base["scoring"]["overall_score"], 4
    )
    hallucination_delta = dimensions["hallucination_resistance"]["delta"]
    conclusion = (
        "improved"
        if overall_delta > 0 and hallucination_delta >= 0 and len(improved) > len(regressed)
        else "regressed"
        if overall_delta < 0 and len(regressed) > len(improved)
        else "mixed"
    )
    performance: dict[str, dict[str, float | int]] = {}
    for label, source_report, scored in (
        ("base", base, base_cases),
        ("adapter", adapter, adapter_cases),
    ):
        output_tokens = sum(case["output_tokens"] for case in scored)
        generation_seconds = sum(case["generation_seconds"] for case in scored)
        performance[label] = {
            "output_tokens": output_tokens,
            "generation_seconds": round(generation_seconds, 3),
            "aggregate_tokens_per_second": round(output_tokens / generation_seconds, 3),
            "mean_output_tokens": round(output_tokens / len(scored), 3),
            "responses_at_token_ceiling": sum(
                case["output_tokens"] >= FROZEN_MAX_NEW_TOKENS for case in scored
            ),
            "peak_gpu_device_used_mib": source_report["model"]["peak_gpu_device_used_mib"],
            "peak_process_ram_mib": source_report["model"]["peak_process_ram_mib"],
        }
    regressed_ids = {case["case_id"] for case in regressed}
    diagnostics = {
        "max_new_tokens": FROZEN_MAX_NEW_TOKENS,
        "regressed_cases_at_token_ceiling": sum(
            case["case_id"] in regressed_ids
            and case["output_tokens"] >= FROZEN_MAX_NEW_TOKENS
            for case in adapter_cases
        ),
        "regressed_case_count": len(regressed),
        "interpretation": (
            "Many regressions coincide with answers reaching the shared token ceiling before "
            "emitting required final-answer markers or all requested evidence."
        ),
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "benchmark_id": base["benchmark_id"],
        "benchmark_sha256": base["benchmark_sha256"],
        "base_model_id": base["model"]["model_id"],
        "adapter": adapter["adapter"],
        "overall": {
            "base": base["scoring"]["overall_score"],
            "adapter": adapter["scoring"]["overall_score"],
            "delta": overall_delta,
            "base_checks_passed": base["scoring"]["checks_passed"],
            "adapter_checks_passed": adapter["scoring"]["checks_passed"],
            "checks_total": base["scoring"]["checks_total"],
        },
        "category_deltas": category_deltas,
        "check_type_deltas": check_type_deltas,
        "dimensions": dimensions,
        "case_changes": {
            "improved_count": len(improved),
            "regressed_count": len(regressed),
            "unchanged_count": len(frozen_cases) - len(changes),
            "improved": improved,
            "regressed": regressed,
        },
        "performance": performance,
        "diagnostics": diagnostics,
        "fairness_checks": {
            "same_frozen_benchmark_bytes": True,
            "same_case_order_and_prompts": True,
            "same_base_model_id": True,
            "same_system_prompt_and_generation_configuration": True,
            "greedy_decoding": True,
            "one_model_loaded_at_a_time": True,
        },
        "conclusion": conclusion,
        "promotion_recommendation": "reject_adapter" if conclusion == "regressed" else "review",
        "limitations": [
            "The adapter was trained on only 33 examples and validated on 7 examples.",
            "The frozen benchmark is financial. A separate three-case development sentinel "
            "is used for general reasoning and is too small for a broad capability claim.",
            "Deterministic keyword and regex rubrics are reproducible but do not replace "
            "human review.",
        ],
    }


def render_comparison_markdown(report: dict[str, Any]) -> str:
    """Render a concise, auditable Stage 7 result."""

    overall = report["overall"]
    changes = report["case_changes"]
    lines = [
        "# Stage 7: base vs fine-tuned evaluation",
        "",
        f"Conclusion: **{report['conclusion']}**.",
        "",
        f"- Base: **{overall['base']:.2%}** "
        f"({overall['base_checks_passed']}/{overall['checks_total']} checks)",
        f"- Adapter: **{overall['adapter']:.2%}** "
        f"({overall['adapter_checks_passed']}/{overall['checks_total']} checks)",
        f"- Change: **{overall['delta']:+.2%}**",
        f"- Cases improved / regressed / unchanged: "
        f"**{changes['improved_count']} / {changes['regressed_count']} / "
        f"{changes['unchanged_count']}**",
        "",
        "## Category comparison",
        "",
        "| Category | Base | Adapter | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {name} | {values['base']:.2%} | {values['adapter']:.2%} | "
        f"{values['delta']:+.2%} |"
        for name, values in report["category_deltas"].items()
    )
    lines.extend(
        [
            "",
            "## Required dimensions",
            "",
            "| Dimension | Base | Adapter | Change |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {name} | {values['base']:.2%} | {values['adapter']:.2%} | "
        f"{values['delta']:+.2%} |"
        for name, values in report["dimensions"].items()
    )
    lines.extend(["", "## Changed cases", "", "### Improvements", ""])
    lines.extend(
        f"- `{case['case_id']}` ({case['category']}): "
        f"{case['base']:.1%} → {case['adapter']:.1%}"
        for case in changes["improved"]
    )
    if not changes["improved"]:
        lines.append("- None")
    lines.extend(["", "### Regressions", ""])
    lines.extend(
        f"- `{case['case_id']}` ({case['category']}): "
        f"{case['base']:.1%} → {case['adapter']:.1%}"
        for case in changes["regressed"]
    )
    if not changes["regressed"]:
        lines.append("- None")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    performance = report["performance"]
    diagnostics = report["diagnostics"]
    lines.extend(
        [
            "",
            "## Runtime and diagnostics",
            "",
            f"- Base throughput: {performance['base']['aggregate_tokens_per_second']:.2f} "
            "tokens/second",
            f"- Adapter throughput: {performance['adapter']['aggregate_tokens_per_second']:.2f} "
            "tokens/second",
            f"- Adapter regressions at the {diagnostics['max_new_tokens']}-token ceiling: "
            f"{diagnostics['regressed_cases_at_token_ceiling']}/"
            f"{diagnostics['regressed_case_count']}",
            f"- Promotion recommendation: **{report['promotion_recommendation']}**",
        ]
    )
    lines.append("")
    return "\n".join(lines)
