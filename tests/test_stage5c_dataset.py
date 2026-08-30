from __future__ import annotations

import json
from pathlib import Path

from finpulse_llm.data.config import load_data_config
from finpulse_llm.data.pipeline import file_sha256, load_jsonl
from finpulse_llm.data.validation import validate_example

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/data/training_pipeline_stage5b.toml"
SOURCE_CONFIG = ROOT / "configs/data/stage5c_sources.toml"
RAW = ROOT / "data/raw/finpulse_stage5c_v1.jsonl"
TRAIN = ROOT / "data/train/finpulse_stage5c_v1.jsonl"
VALIDATION = ROOT / "data/validation/finpulse_stage5c_v1.jsonl"
DEVELOPMENT = ROOT / "data/development/finpulse_stage5c_v1.jsonl"
MANIFEST = ROOT / "data/processed/finpulse_stage5c_v1.manifest.json"
QUALITY = ROOT / "data/processed/finpulse_stage5c_v1.quality.json"
FROZEN_SHA256 = "bfd1b847d2042f6a59f8a8a5f0dfe0826729dc68a0d390a356c2f1fd3b1781fa"


def test_stage5c_manifest_locks_sources_outputs_and_frozen_benchmark() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["source_config_sha256"] == file_sha256(SOURCE_CONFIG)
    assert manifest["raw_sha256"] == file_sha256(RAW)
    assert manifest["train_sha256"] == file_sha256(TRAIN)
    assert manifest["validation_sha256"] == file_sha256(VALIDATION)
    assert manifest["development_sha256"] == file_sha256(DEVELOPMENT)
    assert manifest["protected_stage4_sha256"] == FROZEN_SHA256
    assert manifest["sources"]["cosimo"]["revision"] == (
        "42244d29c6b9912683213a08d1a9c5b0373b381b"
    )
    assert manifest["sources"]["finqa"]["revision"] == (
        "3d6a736bc67e06bc15fbf3618d88204a57c5b25e"
    )


def test_stage5c_counts_uniqueness_and_schema() -> None:
    config = load_data_config(CONFIG)
    raw = load_jsonl([RAW])
    splits = {
        "train": load_jsonl([TRAIN]),
        "validation": load_jsonl([VALIDATION]),
        "development": load_jsonl([DEVELOPMENT]),
    }

    assert len(raw) == 4800
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 3900,
        "validation": 450,
        "development": 450,
    }
    assert raw == splits["train"] + splits["validation"] + splits["development"]
    ids = [{row["id"] for row in rows} for rows in splits.values()]
    assert len(set().union(*ids)) == 4800
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert all(not validate_example(row, config).errors for row in raw)


def test_stage5c_quality_gate_records_source_and_family_isolation() -> None:
    quality = json.loads(QUALITY.read_text(encoding="utf-8"))

    assert quality["status"] == "complete"
    assert quality["source_counts"] == {
        "cosimo": 3000,
        "finqa": 1500,
        "project_behavior": 300,
    }
    assert quality["source_family_overlaps"] == {
        "train_validation": [],
        "train_development": [],
        "validation_development": [],
    }
    assert max(
        statistics["maximum"] for statistics in quality["token_counts"].values()
    ) <= 512
    assert quality["candidate_rejections_before_quota_fill"][
        "cosimo_duplicate_conversation"
    ] > 0
    assert "finance_alpaca" in quality["audited_exclusions"]
