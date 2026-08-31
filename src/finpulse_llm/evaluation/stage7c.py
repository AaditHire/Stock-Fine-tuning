"""Deterministic scoring and candidate selection for the Stage 7C development gate."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any

FINAL_PATTERN = re.compile(r"(?im)^FINAL:\s*([^\r\n]+)\s*$")
NUMBER_PATTERN = re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", re.I)
SOURCE_NAMES = {
    "btech-software/cosimo-cfa-frm-71k": "cosimo",
    "bevaya/FinQA": "finqa",
}


def _source_name(example: dict[str, Any]) -> str:
    reference = str(example["metadata"]["source"]["reference"])
    for prefix, name in SOURCE_NAMES.items():
        if reference.startswith(prefix):
            return name
    raise ValueError(f"Unsupported Stage 5C development source: {reference}")


def _canonical_final(value: str) -> str:
    """Normalize harmless numeric formatting while retaining labels, units, and signs."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("−", "-").replace("–", "-")
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)

    def normalize_number(match: re.Match[str]) -> str:
        try:
            number = Decimal(match.group(0))
        except InvalidOperation:
            return match.group(0)
        if number == 0:
            return "0"
        rendered = format(number.normalize(), "f")
        return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered

    normalized = NUMBER_PATTERN.sub(normalize_number, normalized)
    normalized = re.sub(r"[^a-z0-9%+\-/]+", " ", normalized)
    return " ".join(normalized.split())


def _final_value(text: str) -> tuple[str | None, int]:
    matches = FINAL_PATTERN.findall(text)
    if not matches:
        return None, 0
    return matches[-1].strip(), len(matches)


def score_development_responses(
    examples: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score terminal format and exact normalized final values for ordered responses."""

    if len(examples) != len(responses):
        raise ValueError("Development response count does not match examples")
    aggregates: dict[str, dict[str, list[bool]]] = {
        "task_type": defaultdict(list),
        "source": defaultdict(list),
        "category": defaultdict(list),
    }
    format_results: list[bool] = []
    value_results: list[bool] = []
    scored_cases: list[dict[str, Any]] = []

    for example, generated in zip(examples, responses, strict=True):
        if generated.get("case_id") != example["id"]:
            raise ValueError("Development response order mismatch")
        if example["metadata"]["response_format"] != "final_marker":
            raise ValueError(
                f"Case {example['id']} does not use the Stage 7C final-marker contract"
            )
        expected, expected_count = _final_value(example["messages"][2]["content"])
        actual, actual_count = _final_value(str(generated["response"]))
        if expected is None or expected_count != 1:
            raise ValueError(f"Case {example['id']} has an invalid reference FINAL marker")
        format_passed = actual is not None and actual_count == 1
        value_passed = format_passed and _canonical_final(actual) == _canonical_final(expected)
        task = str(example["metadata"]["task_type"])
        source = _source_name(example)
        category = str(example["metadata"]["category"])
        format_results.append(format_passed)
        value_results.append(value_passed)
        for dimension, key in (("task_type", task), ("source", source), ("category", category)):
            aggregates[dimension][key].append(value_passed)
        scored_cases.append(
            {
                **generated,
                "case_id": example["id"],
                "task_type": task,
                "source": source,
                "category": category,
                "expected_final": expected,
                "actual_final": actual,
                "format_passed": format_passed,
                "value_passed": value_passed,
            }
        )

    def rates(values: dict[str, list[bool]]) -> dict[str, float]:
        return {key: round(sum(items) / len(items), 4) for key, items in sorted(values.items())}

    return {
        "answer_accuracy": round(sum(value_results) / len(value_results), 4),
        "answers_correct": sum(value_results),
        "answers_total": len(value_results),
        "format_accuracy": round(sum(format_results) / len(format_results), 4),
        "format_correct": sum(format_results),
        "task_type_scores": rates(aggregates["task_type"]),
        "source_scores": rates(aggregates["source"]),
        "category_scores": rates(aggregates["category"]),
        "cases": scored_cases,
    }


def compare_candidates(
    base: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply the predeclared no-regression gate and select at most one candidate."""

    if not candidates:
        raise ValueError("At least one adapter candidate report is required")
    reports = [base, *candidates]
    for report in reports:
        if report.get("status") != "complete":
            raise ValueError("All Stage 7C reports must be complete")
        if report.get("case_count") != 450:
            raise ValueError("Stage 7C requires all 450 development cases")
        if report.get("dataset_sha256") != base.get("dataset_sha256"):
            raise ValueError("Development reports use different dataset bytes")
        if report.get("generation_config") != base.get("generation_config"):
            raise ValueError("Development reports use different generation configuration")
        if report.get("inference_batch_size") != base.get("inference_batch_size"):
            raise ValueError("Development reports use different inference batch sizes")
        if [case["case_id"] for case in report["scoring"]["cases"]] != [
            case["case_id"] for case in base["scoring"]["cases"]
        ]:
            raise ValueError("Development reports use different case order")

    base_scores = base["scoring"]
    comparisons: list[dict[str, Any]] = []
    for candidate in candidates:
        scores = candidate["scoring"]
        gate_checks = {
            "strictly_improves_answer_accuracy": (
                scores["answer_accuracy"] > base_scores["answer_accuracy"]
            ),
            "format_not_worse": scores["format_accuracy"] >= base_scores["format_accuracy"],
            "calculation_not_worse": (
                scores["task_type_scores"]["calculation"]
                >= base_scores["task_type_scores"]["calculation"]
            ),
            "multiple_choice_not_worse": (
                scores["task_type_scores"]["multiple_choice"]
                >= base_scores["task_type_scores"]["multiple_choice"]
            ),
            "cosimo_not_worse": (
                scores["source_scores"]["cosimo"] >= base_scores["source_scores"]["cosimo"]
            ),
            "finqa_not_worse": (
                scores["source_scores"]["finqa"] >= base_scores["source_scores"]["finqa"]
            ),
        }
        comparisons.append(
            {
                "candidate": candidate["candidate"],
                "adapter": candidate["adapter"],
                "answer_accuracy": scores["answer_accuracy"],
                "answer_delta": round(
                    scores["answer_accuracy"] - base_scores["answer_accuracy"], 4
                ),
                "format_accuracy": scores["format_accuracy"],
                "task_type_scores": scores["task_type_scores"],
                "source_scores": scores["source_scores"],
                "gate_checks": gate_checks,
                "gate_passed": all(gate_checks.values()),
            }
        )

    eligible = [item for item in comparisons if item["gate_passed"]]
    selected = max(
        eligible,
        key=lambda item: (
            item["answer_accuracy"],
            item["format_accuracy"],
            -int(str(item["candidate"]).removeprefix("checkpoint-").replace("final", "999")),
        ),
        default=None,
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "stage": "7C",
        "dataset_sha256": base["dataset_sha256"],
        "case_count": base["case_count"],
        "selection_policy": {
            "primary_metric": "answer_accuracy",
            "tie_breakers": ["format_accuracy", "earlier_checkpoint"],
            "eligibility": (
                "Strictly improve overall answer accuracy and regress no task type, source "
                "family, or format compliance versus base."
            ),
        },
        "base": {
            "answer_accuracy": base_scores["answer_accuracy"],
            "format_accuracy": base_scores["format_accuracy"],
            "task_type_scores": base_scores["task_type_scores"],
            "source_scores": base_scores["source_scores"],
        },
        "candidates": comparisons,
        "selected_candidate": selected["candidate"] if selected else None,
        "selected_adapter": selected["adapter"] if selected else None,
        "next_action": "stop_before_frozen_benchmark",
    }


def render_comparison_markdown(report: dict[str, Any]) -> str:
    """Render the Stage 7C selection result."""

    lines = [
        "# Stage 7C development evaluation",
        "",
        f"Selected candidate: **{report['selected_candidate'] or 'none'}**",
        "",
        f"- Development split: **{report['case_count']} cases**",
        f"- Base answer accuracy: **{report['base']['answer_accuracy']:.2%}**",
        f"- Base format accuracy: **{report['base']['format_accuracy']:.2%}**",
        "",
        "| Candidate | Answer | Change | Format | Gate |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    lines.extend(
        f"| {item['candidate']} | {item['answer_accuracy']:.2%} | "
        f"{item['answer_delta']:+.2%} | {item['format_accuracy']:.2%} | "
        f"{'pass' if item['gate_passed'] else 'fail'} |"
        for item in report["candidates"]
    )
    lines.extend(
        [
            "",
            "## Task and source scores",
            "",
            "| Model | Calculation | Multiple choice | Cosimo | FinQA |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| base | {report['base']['task_type_scores']['calculation']:.2%} | "
            f"{report['base']['task_type_scores']['multiple_choice']:.2%} | "
            f"{report['base']['source_scores']['cosimo']:.2%} | "
            f"{report['base']['source_scores']['finqa']:.2%} |",
        ]
    )
    lines.extend(
        f"| {item['candidate']} | {item['task_type_scores']['calculation']:.2%} | "
        f"{item['task_type_scores']['multiple_choice']:.2%} | "
        f"{item['source_scores']['cosimo']:.2%} | "
        f"{item['source_scores']['finqa']:.2%} |"
        for item in report["candidates"]
    )
    lines.extend(
        [
            "",
            "The frozen 160-case benchmark was not run. Stage 7C uses only the locked, "
            "family-disjoint Stage 5C development split and stops before the next stage.",
            "",
        ]
    )
    return "\n".join(lines)
