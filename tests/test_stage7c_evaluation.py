from __future__ import annotations

import copy
from pathlib import Path

import pytest

from finpulse_llm.data.pipeline import load_jsonl
from finpulse_llm.evaluation.stage7c import compare_candidates, score_development_responses
from finpulse_llm.inference.config import load_model_config

ROOT = Path(__file__).parents[1]
DEVELOPMENT = ROOT / "data/development/finpulse_stage5c_v1.jsonl"
CONFIG = ROOT / "configs/models/qwen3_4b_stage7c_dev.toml"


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
        "status": "complete",
        "dataset_sha256": "locked-development",
        "case_count": 450,
        "candidate": "base",
        "generation_config": "configs/models/qwen3_4b_stage7c_dev.toml",
        "inference_batch_size": 8,
        "scoring": score_development_responses(examples, responses),
    }


def test_reference_answers_pass_stage7c_scoring() -> None:
    scoring = _perfect_report()["scoring"]

    assert scoring["answer_accuracy"] == 1.0
    assert scoring["format_accuracy"] == 1.0
    assert scoring["answers_total"] == 450
    assert scoring["source_scores"] == {"cosimo": 1.0, "finqa": 1.0}


def test_numeric_formatting_is_normalized_but_terminal_marker_is_required() -> None:
    examples = load_jsonl([DEVELOPMENT])
    numeric = next(item for item in examples if "FINAL: 4,769.81" in item["messages"][2]["content"])
    response = {"case_id": numeric["id"], "response": "Calculation.\nFINAL: 4769.810"}

    scoring = score_development_responses([numeric], [response])
    assert scoring["answer_accuracy"] == 1.0

    response["response"] = "The answer is 4769.81"
    scoring = score_development_responses([numeric], [response])
    assert scoring["answer_accuracy"] == 0.0
    assert scoring["format_accuracy"] == 0.0


def test_selection_requires_improvement_without_subgroup_regression() -> None:
    base = _perfect_report()
    base["scoring"]["answer_accuracy"] = 0.8
    base["scoring"]["format_accuracy"] = 0.9
    base["scoring"]["task_type_scores"] = {"calculation": 0.8, "multiple_choice": 0.8}
    base["scoring"]["source_scores"] = {"cosimo": 0.8, "finqa": 0.8}
    candidate = copy.deepcopy(base)
    candidate["candidate"] = "checkpoint-15"
    candidate["adapter"] = {"weights_sha256": "candidate"}
    candidate["scoring"]["answer_accuracy"] = 0.81

    comparison = compare_candidates(base, [candidate])
    assert comparison["selected_candidate"] == "checkpoint-15"

    candidate["scoring"]["source_scores"]["finqa"] = 0.79
    comparison = compare_candidates(base, [candidate])
    assert comparison["selected_candidate"] is None


def test_stage7c_config_matches_locked_system_prompt() -> None:
    examples = load_jsonl([DEVELOPMENT])
    config = load_model_config(CONFIG)

    assert {item["messages"][0]["content"] for item in examples} == {config.system_prompt}
    assert config.generation.do_sample is False
    assert config.generation.enable_thinking is False
    assert config.generation.max_new_tokens == 192


def test_stage7c_rejects_partial_reports() -> None:
    base = _perfect_report()
    candidate = copy.deepcopy(base)
    candidate["candidate"] = "checkpoint-15"
    candidate["adapter"] = {"weights_sha256": "candidate"}
    candidate["case_count"] = 449

    with pytest.raises(ValueError, match="450"):
        compare_candidates(base, [candidate])
