from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from finpulse_llm.evaluation.stage4 import load_frozen_benchmark
from finpulse_llm.evaluation.stage7 import compare_reports, render_comparison_markdown

ROOT = Path(__file__).parents[1]
CASES = load_frozen_benchmark(ROOT / "data/eval/finpulse_eval_v1.jsonl")
COMPARISON = ROOT / "results/benchmarks/stage7_comparison.json"


def _report() -> dict:
    scored = []
    category_scores = {}
    check_types: dict[str, list[bool]] = {}
    for case in CASES:
        checks = [
            {"type": check["type"], "passed": True, "detail": "test"}
            for check in case.checks
        ]
        for check in checks:
            check_types.setdefault(check["type"], []).append(True)
        scored.append(
            {
                "case_id": case.id,
                "category": case.category,
                "prompt": case.prompt,
                "score": 1.0,
                "checks": checks,
                "output_tokens": 10,
                "generation_seconds": 1.0,
            }
        )
        category_scores[case.category] = 1.0
    return {
        "status": "complete",
        "benchmark_id": "finpulse_eval_v1",
        "benchmark_sha256": "frozen-hash",
        "case_count": len(CASES),
        "model": {
            "model_id": "base",
            "peak_gpu_device_used_mib": 100,
            "peak_process_ram_mib": 100,
        },
        "adapter": {"weights_sha256": "adapter"},
        "scoring": {
            "overall_score": 1.0,
            "checks_passed": sum(len(case.checks) for case in CASES),
            "checks_total": sum(len(case.checks) for case in CASES),
            "category_scores": category_scores,
            "check_type_scores": {key: 1.0 for key in check_types},
            "cases": scored,
        },
    }


def test_comparison_detects_one_regression() -> None:
    base = _report()
    adapter = deepcopy(base)
    changed = adapter["scoring"]["cases"][0]
    changed["checks"][0]["passed"] = False
    changed["score"] = 0.0
    adapter["scoring"]["overall_score"] = 0.9974
    adapter["scoring"]["checks_passed"] -= 1
    adapter["scoring"]["category_scores"][changed["category"]] = 0.9375
    adapter["scoring"]["check_type_scores"][changed["checks"][0]["type"]] = 0.98

    result = compare_reports(base, adapter, CASES)

    assert result["case_changes"]["regressed_count"] == 1
    assert result["case_changes"]["improved_count"] == 0
    assert result["conclusion"] == "regressed"
    assert result["promotion_recommendation"] == "reject_adapter"
    assert "Stage 7" in render_comparison_markdown(result)


def test_comparison_rejects_different_benchmark_hash() -> None:
    base = _report()
    adapter = _report()
    adapter["benchmark_sha256"] = "different"

    with pytest.raises(ValueError, match="different benchmark"):
        compare_reports(base, adapter, CASES)


def test_comparison_accepts_general_regression_sentinel() -> None:
    general = {
        "status": "complete",
        "base": {"scoring": {"overall_score": 1.0}},
        "fine_tuned": {"scoring": {"overall_score": 2 / 3}},
    }

    result = compare_reports(_report(), _report(), CASES, general)

    assert result["dimensions"]["general_capability_regression"]["delta"] == -0.3333


def test_completed_stage7_result_rejects_adapter() -> None:
    import json

    report = json.loads(COMPARISON.read_text(encoding="utf-8"))

    assert report["benchmark_sha256"] == (
        "bfd1b847d2042f6a59f8a8a5f0dfe0826729dc68a0d390a356c2f1fd3b1781fa"
    )
    assert report["overall"]["base"] == 0.9056
    assert report["overall"]["adapter"] == 0.8444
    assert report["dimensions"]["hallucination_resistance"]["adapter"] == 1.0
    assert report["conclusion"] == "regressed"
    assert report["promotion_recommendation"] == "reject_adapter"
