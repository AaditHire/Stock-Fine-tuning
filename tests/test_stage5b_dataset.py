from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

from finpulse_llm.data.config import load_data_config
from finpulse_llm.data.leakage import EvaluationLeakageIndex
from finpulse_llm.data.pipeline import build_quality_report, file_sha256, load_jsonl, run_pipeline
from finpulse_llm.data.validation import validate_example

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/data/training_pipeline_stage5b.toml"
RAW = ROOT / "data/raw/finpulse_stage5b_v1.jsonl"
TRAIN = ROOT / "data/train/finpulse_stage5b_v1.jsonl"
VALIDATION = ROOT / "data/validation/finpulse_stage5b_v1.jsonl"
DEVELOPMENT = ROOT / "data/development/finpulse_stage5b_v1.jsonl"
MANIFEST = ROOT / "data/processed/finpulse_stage5b_v1.manifest.json"
QUALITY = ROOT / "data/processed/finpulse_stage5b_v1.quality.json"


def _load_builder():
    path = ROOT / "scripts/build_stage5b_corpus.py"
    spec = importlib.util.spec_from_file_location("build_stage5b_corpus", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage5b_builder_is_reproducible() -> None:
    module = _load_builder()
    expected = "".join(
        json.dumps(item, ensure_ascii=False) + "\n" for item in module.build_records(CONFIG)
    )

    assert len(module.build_records(CONFIG)) == 500
    assert RAW.read_text(encoding="utf-8") == expected


def test_stage5b_pipeline_accepts_every_record_and_creates_three_disjoint_splits() -> None:
    config = load_data_config(CONFIG)
    leakage = EvaluationLeakageIndex.from_files(
        ROOT / "benchmarks/stage3_base_models.json",
        ROOT / "data/eval/finpulse_eval_v1.jsonl",
    )
    result = run_pipeline(load_jsonl([RAW]), config, leakage)
    quality = build_quality_report(result, config)

    assert len(result.accepted) == 500
    assert (len(result.train), len(result.validation), len(result.development)) == (398, 51, 51)
    assert not result.rejections
    split_ids = [
        {item["id"] for item in split}
        for split in (result.train, result.validation, result.development)
    ]
    assert split_ids[0].isdisjoint(split_ids[1])
    assert split_ids[0].isdisjoint(split_ids[2])
    assert split_ids[1].isdisjoint(split_ids[2])
    assert quality["all_distributions_within_tolerance"] is True
    assert quality["task_type_counts"]["calculation"] == 150
    assert quality["response_format_counts"]["final_marker"] == 250
    assert quality["response_format_counts"]["json_only"] == 75


def test_stage5b_manifest_locks_all_splits_and_frozen_benchmark() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    quality = json.loads(QUALITY.read_text(encoding="utf-8"))

    assert manifest["train_sha256"] == file_sha256(TRAIN)
    assert manifest["validation_sha256"] == file_sha256(VALIDATION)
    assert manifest["development_sha256"] == file_sha256(DEVELOPMENT)
    assert manifest["protected_stage4_sha256"] == (
        "bfd1b847d2042f6a59f8a8a5f0dfe0826729dc68a0d390a356c2f1fd3b1781fa"
    )
    assert quality["accepted_records"] == 500
    assert quality["rejected_records"] == 0


def test_stage5b_format_contracts_reject_common_regressions() -> None:
    config = load_data_config(CONFIG)
    records = load_jsonl([RAW])
    json_example = copy.deepcopy(
        next(item for item in records if item["metadata"]["response_format"] == "json_only")
    )
    json_example["messages"][2]["content"] = (
        f"```json\n{json_example['messages'][2]['content']}\n```"
    )
    marker_example = copy.deepcopy(
        next(item for item in records if item["metadata"]["response_format"] == "final_marker")
    )
    marker_example["messages"][2]["content"] = "The calculation is complete."

    assert "json_only response must be exactly valid JSON" in validate_example(
        json_example, config
    ).errors
    assert "final_marker response must end with exactly one FINAL marker" in validate_example(
        marker_example, config
    ).errors
