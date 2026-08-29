import json
from pathlib import Path

import pytest

from finpulse_llm.evaluation.stage3 import load_benchmark, score_check, score_responses

BENCHMARK_PATH = Path(__file__).parents[1] / "benchmarks" / "stage3_base_models.json"


def test_stage3_benchmark_has_required_coverage() -> None:
    cases = load_benchmark(BENCHMARK_PATH)
    categories = {case.category for case in cases}

    assert len(cases) == 15
    assert {
        "financial_calculation",
        "finance_reasoning",
        "finance_knowledge",
        "hallucination_resistance",
        "structured_output",
        "instruction_following",
        "general_reasoning",
    } <= categories


@pytest.mark.parametrize(
    ("response", "check", "expected"),
    [
        ("FINAL: 20 shares", {"type": "regex", "pattern": r"FINAL:\s*20\s+shares$"}, True),
        ("It will rise", {"type": "regex_not", "pattern": r"\bwill rise\b"}, False),
        (
            "Volume and support",
            {"type": "keyword_count", "values": ["volume", "support"], "minimum": 2},
            True,
        ),
        (
            '{"status":"insufficient_data"}',
            {"type": "json_exact", "value": {"status": "insufficient_data"}},
            True,
        ),
        (
            '```json\n{"status":"insufficient_data"}\n```',
            {"type": "json_only_exact", "value": {"status": "insufficient_data"}},
            False,
        ),
    ],
)
def test_score_check(response: str, check: dict, expected: bool) -> None:
    assert score_check(response, check).passed is expected


def test_score_responses_rejects_wrong_count() -> None:
    cases = load_benchmark(BENCHMARK_PATH)
    with pytest.raises(ValueError, match="Expected 15"):
        score_responses(cases, [])


def test_benchmark_is_valid_json() -> None:
    raw = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    assert raw["version"] == 1


def test_positive_funding_check_rejects_reversed_payment_direction() -> None:
    cases = load_benchmark(BENCHMARK_PATH)
    case = next(item for item in cases if item.id == "funding_mechanics")

    correct = score_check("Positive funding means buyers pay sellers.", case.checks[0])
    reversed_direction = score_check("Positive funding means shorts pay longs.", case.checks[0])

    assert correct.passed is True
    assert reversed_direction.passed is False


def test_json_only_fields_rejects_fenced_json_and_explanation() -> None:
    check = {
        "type": "json_only_fields",
        "exact_keys": ["trend", "confidence", "risks"],
        "field_types": {"trend": "string", "confidence": "number_0_1", "risks": "array"},
    }
    plain = '{"trend":"mixed","confidence":0.5,"risks":[]}'
    fenced = f"```json\n{plain}\n```\nExplanation follows."

    assert score_check(plain, check).passed is True
    assert score_check(fenced, check).passed is False
