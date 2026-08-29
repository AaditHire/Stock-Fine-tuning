"""Objective scoring helpers for the small Stage 3 base-model benchmark."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    category: str
    prompt: str
    checks: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CheckResult:
    type: str
    passed: bool
    detail: str


def load_benchmark(path: str | Path) -> tuple[BenchmarkCase, ...]:
    """Load the Stage 3 JSON benchmark and reject duplicate IDs."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = tuple(
        BenchmarkCase(
            id=str(item["id"]),
            category=str(item["category"]),
            prompt=str(item["prompt"]),
            checks=tuple(item["checks"]),
        )
        for item in raw["cases"]
    )
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark case IDs must be unique")
    if not cases:
        raise ValueError("Benchmark must contain at least one case")
    return cases


def _extract_json(response: str) -> Any:
    stripped = response.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(response):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(response[index:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("response does not contain valid JSON")


def _json_fields(value: Any, check: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(value, dict):
        return False, "JSON value is not an object"
    expected_keys = set(check["exact_keys"])
    if set(value) != expected_keys:
        return False, f"keys={sorted(value)} expected={sorted(expected_keys)}"

    validators = {
        "string": lambda item: isinstance(item, str),
        "array": lambda item: isinstance(item, list),
        "number_0_1": lambda item: (
            isinstance(item, (int, float)) and not isinstance(item, bool) and 0 <= item <= 1
        ),
    }
    for field, type_name in check["field_types"].items():
        validator = validators[type_name]
        if not validator(value.get(field)):
            return False, f"field {field!r} is not {type_name}"
    return True, "valid JSON object and field types"


def score_check(response: str, check: dict[str, Any]) -> CheckResult:
    """Evaluate one declarative check against a model response."""

    check_type = str(check["type"])
    lowered = response.casefold()
    passed = False
    detail = ""

    if check_type == "regex":
        passed = re.search(str(check["pattern"]), response, re.IGNORECASE | re.DOTALL) is not None
        detail = f"required regex: {check['pattern']}"
    elif check_type == "regex_not":
        passed = re.search(str(check["pattern"]), response, re.IGNORECASE | re.DOTALL) is None
        detail = f"forbidden regex: {check['pattern']}"
    elif check_type == "contains_any":
        values = [str(value).casefold() for value in check["values"]]
        matched = [value for value in values if value in lowered]
        passed = bool(matched)
        detail = f"matched={matched}"
    elif check_type == "contains_all":
        values = [str(value).casefold() for value in check["values"]]
        missing = [value for value in values if value not in lowered]
        passed = not missing
        detail = f"missing={missing}"
    elif check_type == "keyword_count":
        values = [str(value).casefold() for value in check["values"]]
        matched = [value for value in values if value in lowered]
        passed = len(matched) >= int(check["minimum"])
        detail = f"matched={matched}, minimum={check['minimum']}"
    elif check_type == "json_exact":
        try:
            actual = _extract_json(response)
            expected = check["value"]
            passed = actual == expected
            detail = f"actual={actual!r}, expected={expected!r}"
        except ValueError as exc:
            detail = str(exc)
    elif check_type == "json_only_exact":
        try:
            actual = json.loads(response.strip())
            expected = check["value"]
            passed = actual == expected
            detail = f"actual={actual!r}, expected={expected!r}"
        except json.JSONDecodeError:
            detail = "response is not only one valid JSON value"
    elif check_type == "json_fields":
        try:
            passed, detail = _json_fields(_extract_json(response), check)
        except ValueError as exc:
            detail = str(exc)
    elif check_type == "json_only_fields":
        try:
            passed, detail = _json_fields(json.loads(response.strip()), check)
        except json.JSONDecodeError:
            detail = "response is not only one valid JSON value"
    else:
        raise ValueError(f"Unknown benchmark check type: {check_type}")

    return CheckResult(type=check_type, passed=passed, detail=detail)


def score_responses(
    cases: tuple[BenchmarkCase, ...], responses: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score ordered inference results and calculate category aggregates."""

    if len(cases) != len(responses):
        raise ValueError(f"Expected {len(cases)} responses, received {len(responses)}")

    scored_cases: list[dict[str, Any]] = []
    category_points: dict[str, list[float]] = defaultdict(list)
    total_passed = 0
    total_checks = 0

    for case, inference in zip(cases, responses, strict=True):
        results = tuple(score_check(str(inference["response"]), check) for check in case.checks)
        passed_count = sum(result.passed for result in results)
        score = passed_count / len(results)
        total_passed += passed_count
        total_checks += len(results)
        category_points[case.category].append(score)
        scored_cases.append(
            {
                "id": case.id,
                "category": case.category,
                "prompt": case.prompt,
                "response": inference["response"],
                "input_tokens": inference["input_tokens"],
                "output_tokens": inference["output_tokens"],
                "generation_seconds": inference["generation_seconds"],
                "tokens_per_second": inference["tokens_per_second"],
                "score": round(score, 4),
                "checks": [asdict(result) for result in results],
            }
        )

    return {
        "overall_score": round(total_passed / total_checks, 4),
        "checks_passed": total_passed,
        "checks_total": total_checks,
        "category_scores": {
            category: round(sum(scores) / len(scores), 4)
            for category, scores in sorted(category_points.items())
        },
        "cases": scored_cases,
    }
