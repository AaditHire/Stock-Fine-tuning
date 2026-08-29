"""Task-aware scoring and gating for the Stage 7B development screen."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from finpulse_llm.data.text import comparison_text
from finpulse_llm.data.validation import REFUSAL_MARKERS, UNSUPPORTED_LIVE_VALUE

FINAL_PATTERN = re.compile(r"FINAL:\s*([^\n]+)\s*$")
UNCERTAINTY_MARKERS = ("conditional", "uncertain", "may", "could", "depends", "not certain")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
}


def _tokens(value: str) -> set[str]:
    return {token for token in comparison_text(value).split() if token not in STOPWORDS}


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def _score_final(expected: str, response: str) -> list[dict[str, Any]]:
    expected_match = FINAL_PATTERN.search(expected)
    actual_match = FINAL_PATTERN.search(response)
    expected_value = comparison_text(expected_match.group(1)) if expected_match else ""
    actual_value = comparison_text(actual_match.group(1)) if actual_match else ""
    exact_format = len(FINAL_PATTERN.findall(response)) == 1 and actual_match is not None
    return [
        _check("final_marker_format", exact_format, "exactly one terminal FINAL marker"),
        _check(
            "final_value",
            bool(actual_value) and actual_value == expected_value,
            f"expected normalized final value: {expected_value}",
        ),
    ]


def _score_json(expected: str, response: str) -> list[dict[str, Any]]:
    parsed: Any = None
    valid = False
    try:
        parsed = json.loads(response)
        valid = response.strip().startswith("{") and response.strip().endswith("}")
    except json.JSONDecodeError:
        pass
    expected_object = json.loads(expected)
    return [
        _check("json_only", valid, "response must be one unfenced JSON object"),
        _check("json_values", valid and parsed == expected_object, "JSON values must match inputs"),
    ]


def _score_refusal(response: str) -> list[dict[str, Any]]:
    lowered = response.casefold()
    refusal = any(marker in lowered for marker in REFUSAL_MARKERS)
    unsupported_value = UNSUPPORTED_LIVE_VALUE.search(response) is not None
    return [
        _check("access_limitation", refusal, "must disclose lack of live-data access"),
        _check("no_fabricated_value", not unsupported_value, "must not supply an exact live value"),
    ]


def _score_factual(expected: str, response: str) -> list[dict[str, Any]]:
    expected_tokens = _tokens(expected)
    response_tokens = _tokens(response)
    coverage = (
        len(expected_tokens & response_tokens) / len(expected_tokens)
        if expected_tokens
        else 0
    )
    words = len(response.split())
    return [
        _check("reference_token_coverage", coverage >= 0.45, f"coverage={coverage:.3f}"),
        _check("concise", words <= 45, f"word_count={words}"),
    ]


def _score_analysis(example: dict[str, Any], response: str) -> list[dict[str, Any]]:
    prompt = example["messages"][1]["content"]
    scenario = prompt.split(":", maxsplit=1)[-1].split(". Explain", maxsplit=1)[0]
    clauses = scenario.split(", but ", maxsplit=1)
    response_tokens = _tokens(response)
    signal_checks = []
    for clause in clauses:
        clause_tokens = _tokens(clause)
        overlap = len(clause_tokens & response_tokens)
        signal_checks.append(overlap >= min(2, len(clause_tokens)))
    words = len(response.split())
    length_label = example["metadata"]["response_length"]
    minimum, maximum = (65, 145) if length_label == "long" else (30, 90)
    lowered = response.casefold()
    return [
        _check("both_signals", len(signal_checks) == 2 and all(signal_checks), str(signal_checks)),
        _check("invalidation", "invalidat" in lowered, "must state invalidation"),
        _check(
            "uncertainty",
            any(marker in lowered for marker in UNCERTAINTY_MARKERS),
            "must use calibrated uncertainty language",
        ),
        _check("requested_length", minimum <= words <= maximum, f"word_count={words}"),
    ]


def score_development_responses(
    examples: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(examples) != len(responses):
        raise ValueError("Development response count does not match examples")
    scored_cases: list[dict[str, Any]] = []
    task_checks: dict[str, list[bool]] = defaultdict(list)
    category_checks: dict[str, list[bool]] = defaultdict(list)
    for example, generated in zip(examples, responses, strict=True):
        if generated["case_id"] != example["id"]:
            raise ValueError("Development response order mismatch")
        task = example["metadata"]["task_type"]
        expected = example["messages"][2]["content"]
        response = generated["response"]
        if example["metadata"]["response_format"] == "final_marker":
            checks = _score_final(expected, response)
        elif example["metadata"]["response_format"] == "json_only":
            checks = _score_json(expected, response)
        elif task == "refusal":
            checks = _score_refusal(response)
        elif task == "factual":
            checks = _score_factual(expected, response)
        elif task == "analysis":
            checks = _score_analysis(example, response)
        else:
            raise ValueError(f"No development scorer for task type: {task}")
        passed = sum(bool(check["passed"]) for check in checks)
        for check in checks:
            task_checks[task].append(bool(check["passed"]))
            category_checks[example["metadata"]["category"]].append(bool(check["passed"]))
        scored_cases.append(
            {
                "case_id": example["id"],
                "category": example["metadata"]["category"],
                "task_type": task,
                "response_format": example["metadata"]["response_format"],
                "prompt": example["messages"][1]["content"],
                "expected": expected,
                **generated,
                "checks": checks,
                "score": round(passed / len(checks), 4),
            }
        )
    checks_passed = sum(sum(values) for values in task_checks.values())
    checks_total = sum(len(values) for values in task_checks.values())
    return {
        "overall_score": round(checks_passed / checks_total, 4),
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "task_type_scores": {
            key: round(sum(values) / len(values), 4) for key, values in sorted(task_checks.items())
        },
        "category_scores": {
            key: round(sum(values) / len(values), 4)
            for key, values in sorted(category_checks.items())
        },
        "cases": scored_cases,
    }


def compare_development_reports(base: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    if base["dataset_sha256"] != adapter["dataset_sha256"]:
        raise ValueError("Development reports use different dataset bytes")
    if base["generation_config"] != adapter["generation_config"]:
        raise ValueError("Development reports use different generation configuration")
    base_scores = base["scoring"]["task_type_scores"]
    adapter_scores = adapter["scoring"]["task_type_scores"]
    task_deltas = {
        task: {
            "base": base_scores[task],
            "adapter": adapter_scores[task],
            "delta": round(adapter_scores[task] - base_scores[task], 4),
        }
        for task in base_scores
    }
    overall_base = base["scoring"]["overall_score"]
    overall_adapter = adapter["scoring"]["overall_score"]
    gate_checks = {
        "overall_improves_by_5_points": overall_adapter - overall_base >= 0.05,
        "calculation_at_least_90_percent": adapter_scores["calculation"] >= 0.90,
        "calculation_not_worse": adapter_scores["calculation"] >= base_scores["calculation"],
        "instruction_following_at_least_90_percent": (
            adapter_scores["instruction_following"] >= 0.90
        ),
        "refusal_not_worse": adapter_scores["refusal"] >= base_scores["refusal"],
        "analysis_within_5_points": adapter_scores["analysis"] >= base_scores["analysis"] - 0.05,
        "factual_within_5_points": adapter_scores["factual"] >= base_scores["factual"] - 0.05,
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "dataset_sha256": base["dataset_sha256"],
        "overall": {
            "base": overall_base,
            "adapter": overall_adapter,
            "delta": round(overall_adapter - overall_base, 4),
        },
        "task_type_deltas": task_deltas,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "next_action": (
            "run_frozen_benchmark" if all(gate_checks.values()) else "reject_before_frozen"
        ),
        "limitation": (
            "The development split is row-disjoint but shares template families with training; "
            "this gate is a screen, not a release-quality result."
        ),
    }


def render_development_comparison(report: dict[str, Any]) -> str:
    lines = [
        "# Stage 7B development gate",
        "",
        f"Gate passed: **{report['gate_passed']}**",
        "",
        f"- Base: **{report['overall']['base']:.2%}**",
        f"- Adapter: **{report['overall']['adapter']:.2%}**",
        f"- Change: **{report['overall']['delta']:+.2%}**",
        "",
        "| Task | Base | Adapter | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {task} | {values['base']:.2%} | {values['adapter']:.2%} | "
        f"{values['delta']:+.2%} |"
        for task, values in report["task_type_deltas"].items()
    )
    lines.extend(["", "## Gate checks", ""])
    lines.extend(f"- {name}: **{passed}**" for name, passed in report["gate_checks"].items())
    lines.extend(["", report["limitation"], ""])
    return "\n".join(lines)
