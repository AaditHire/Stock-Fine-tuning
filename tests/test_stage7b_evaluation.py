from __future__ import annotations

import copy
from pathlib import Path

from finpulse_llm.data.pipeline import load_jsonl
from finpulse_llm.evaluation.stage7b import (
    compare_development_reports,
    score_development_responses,
)
from finpulse_llm.inference.config import load_model_config

ROOT = Path(__file__).parents[1]
DEVELOPMENT = ROOT / "data/development/finpulse_stage5b_v1.jsonl"
CONFIG = ROOT / "configs/models/qwen3_4b_stage7b_dev.toml"
DEV_COMPARISON = ROOT / "results/benchmarks/stage7b_dev_comparison.json"
FROZEN_COMPARISON = ROOT / "results/benchmarks/stage7b_comparison.json"


def _perfect_report() -> dict:
    examples = load_jsonl([DEVELOPMENT])
    responses = [
        {
            "case_id": item["id"],
            "prompt": item["messages"][1]["content"],
            "response": item["messages"][2]["content"],
            "input_tokens": 1,
            "output_tokens": 1,
            "generation_seconds": 1.0,
            "tokens_per_second": 1.0,
        }
        for item in examples
    ]
    return {
        "dataset_sha256": "locked-dev",
        "generation_config": "configs/models/qwen3_4b_stage7b_dev.toml",
        "scoring": score_development_responses(examples, responses),
    }


def test_reference_development_answers_pass_task_aware_scoring() -> None:
    report = _perfect_report()

    assert report["scoring"]["overall_score"] == 1.0
    assert all(score == 1.0 for score in report["scoring"]["task_type_scores"].values())


def test_development_scorer_rejects_case_order_mismatch() -> None:
    examples = load_jsonl([DEVELOPMENT])[:1]
    response = {
        "case_id": "wrong",
        "response": examples[0]["messages"][2]["content"],
    }

    try:
        score_development_responses(examples, [response])
    except ValueError as exc:
        assert "order mismatch" in str(exc)
    else:
        raise AssertionError("Expected order mismatch to fail")


def test_development_gate_requires_material_overall_improvement() -> None:
    adapter = _perfect_report()
    base = copy.deepcopy(adapter)
    base["scoring"]["overall_score"] = 0.96

    comparison = compare_development_reports(base, adapter)

    assert comparison["gate_passed"] is False
    assert comparison["gate_checks"]["overall_improves_by_5_points"] is False


def test_development_config_matches_stage5b_system_prompt() -> None:
    import tomllib

    model_config = load_model_config(CONFIG)
    with (ROOT / "configs/data/training_pipeline_stage5b.toml").open("rb") as handle:
        data_config = tomllib.load(handle)

    assert model_config.system_prompt == data_config["system_prompt"].strip()
    assert model_config.generation.do_sample is False
    assert model_config.generation.max_new_tokens == 192


def test_completed_stage7b_gate_and_frozen_result() -> None:
    import json

    development = json.loads(DEV_COMPARISON.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_COMPARISON.read_text(encoding="utf-8"))

    assert development["dataset_sha256"] == (
        "46936e4b8063c685202fc39a3fd43c4f2507cdf0e667250f4fc38921f1adacf3"
    )
    assert development["gate_passed"] is True
    assert development["overall"] == {
        "base": 0.75,
        "adapter": 0.9113,
        "delta": 0.1613,
    }
    assert frozen["benchmark_sha256"] == (
        "bfd1b847d2042f6a59f8a8a5f0dfe0826729dc68a0d390a356c2f1fd3b1781fa"
    )
    assert frozen["stage_label"] == "Stage 7B"
    assert frozen["overall"]["adapter"] == 0.7092
    assert frozen["dimensions"]["hallucination_resistance"]["adapter"] == 1.0
    assert frozen["promotion_recommendation"] == "reject_adapter"
