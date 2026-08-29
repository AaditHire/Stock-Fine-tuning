import json
from collections import Counter
from pathlib import Path

import pytest

from finpulse_llm.evaluation.stage4 import (
    EXPECTED_CATEGORIES,
    load_frozen_benchmark,
    prompt_fingerprint,
    score_stage4_responses,
    verify_manifest,
)

ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data" / "eval" / "finpulse_eval_v1.jsonl"
MANIFEST = ROOT / "data" / "eval" / "finpulse_eval_v1.manifest.json"


def test_frozen_benchmark_structure_and_balance() -> None:
    cases = load_frozen_benchmark(DATASET)

    assert len(cases) == 160
    assert Counter(case.category for case in cases) == {
        category: 16 for category in EXPECTED_CATEGORIES
    }
    assert all(case.exclude_from_training and case.split == "eval" for case in cases)


def test_manifest_matches_frozen_dataset() -> None:
    manifest = verify_manifest(DATASET, MANIFEST)

    assert manifest["status"] == "frozen"
    assert manifest["case_count"] == 160


def test_stage4_prompts_do_not_duplicate_stage3_prompts() -> None:
    stage3 = json.loads((ROOT / "benchmarks" / "stage3_base_models.json").read_text())
    stage3_fingerprints = {prompt_fingerprint(item["prompt"]) for item in stage3["cases"]}
    stage4_fingerprints = {
        prompt_fingerprint(case.prompt) for case in load_frozen_benchmark(DATASET)
    }

    assert stage3_fingerprints.isdisjoint(stage4_fingerprints)


def test_scoring_rejects_case_order_mismatch() -> None:
    cases = load_frozen_benchmark(DATASET)
    responses = [
        {
            "case_id": case.id,
            "response": "FINAL: A",
            "input_tokens": 1,
            "output_tokens": 1,
            "generation_seconds": 1.0,
            "tokens_per_second": 1.0,
        }
        for case in cases
    ]
    responses[0]["case_id"] = "wrong"

    with pytest.raises(ValueError, match="order mismatch"):
        score_stage4_responses(cases, responses)
