from __future__ import annotations

import json
from pathlib import Path

from finpulse_llm.data.config import load_data_config
from finpulse_llm.data.pipeline import file_sha256, load_jsonl
from finpulse_llm.data.validation import validate_example

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/data/stage5d_sampling.toml"
VALIDATION_CONFIG = ROOT / "configs/data/training_pipeline_stage5b.toml"
PARENT_TRAIN = ROOT / "data/train/finpulse_stage5c_v1.jsonl"
TRAIN = ROOT / "data/train/finpulse_stage5d_v1.jsonl"
VALIDATION = ROOT / "data/validation/finpulse_stage5c_v1.jsonl"
DEVELOPMENT = ROOT / "data/development/finpulse_stage5c_v1.jsonl"
MANIFEST = ROOT / "data/processed/finpulse_stage5d_v1.manifest.json"
QUALITY = ROOT / "data/processed/finpulse_stage5d_v1.quality.json"
FROZEN_SHA256 = "bfd1b847d2042f6a59f8a8a5f0dfe0826729dc68a0d390a356c2f1fd3b1781fa"


def test_stage5d_manifest_locks_balanced_train_and_unchanged_holdouts() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["sampling_config_sha256"] == file_sha256(CONFIG)
    assert manifest["train_sha256"] == file_sha256(TRAIN)
    assert manifest["validation_sha256"] == file_sha256(VALIDATION)
    assert manifest["development_sha256"] == file_sha256(DEVELOPMENT)
    assert manifest["protected_stage4_sha256"] == FROZEN_SHA256
    assert manifest["training_approval"] == "not_granted"


def test_stage5d_is_a_unique_valid_subset_of_stage5c_training() -> None:
    validation_config = load_data_config(VALIDATION_CONFIG)
    parent = load_jsonl([PARENT_TRAIN])
    selected = load_jsonl([TRAIN])
    parent_ids = {row["id"] for row in parent}
    selected_ids = {row["id"] for row in selected}

    assert len(selected) == 900
    assert len(selected_ids) == 900
    assert selected_ids <= parent_ids
    assert all(not validate_example(row, validation_config).errors for row in selected)


def test_stage5d_balance_contract() -> None:
    quality = json.loads(QUALITY.read_text(encoding="utf-8"))

    assert quality["source_counts"] == {
        "cosimo": 400,
        "finqa": 200,
        "project_behavior": 300,
    }
    assert quality["task_type_counts"] == {
        "analysis": 100,
        "calculation": 525,
        "factual": 75,
        "instruction_following": 50,
        "multiple_choice": 125,
        "refusal": 25,
    }
    assert quality["response_format_counts"]["json_only"] == 75
    assert quality["calculation_share"] == 0.5833
    assert quality["holdouts"]["unchanged"] is True
