import copy
import json
from pathlib import Path

import pytest

from finpulse_llm.data.config import load_data_config
from finpulse_llm.data.leakage import EvaluationLeakageIndex
from finpulse_llm.data.pipeline import (
    build_quality_report,
    file_sha256,
    load_jsonl,
    run_pipeline,
)
from finpulse_llm.data.text import normalize_text
from finpulse_llm.data.validation import validate_example

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "data" / "training_pipeline.toml"
SEED_PATH = ROOT / "data" / "raw" / "finpulse_seed_v1.jsonl"
TRAIN_PATH = ROOT / "data" / "train" / "finpulse_seed_v1.jsonl"
VALIDATION_PATH = ROOT / "data" / "validation" / "finpulse_seed_v1.jsonl"
MANIFEST_PATH = ROOT / "data" / "processed" / "finpulse_seed_v1.manifest.json"


@pytest.fixture
def config():
    return load_data_config(CONFIG_PATH)


@pytest.fixture
def leakage():
    return EvaluationLeakageIndex.from_files(
        ROOT / "benchmarks" / "stage3_base_models.json",
        ROOT / "data" / "eval" / "finpulse_eval_v1.jsonl",
    )


def test_seed_pipeline_is_clean_balanced_and_reproducible(config, leakage) -> None:
    records = load_jsonl([SEED_PATH])
    first = run_pipeline(records, config, leakage)
    second = run_pipeline(records, config, leakage)
    quality = build_quality_report(first, config)

    assert len(first.accepted) == 40
    assert len(first.train) == 33
    assert len(first.validation) == 7
    assert not first.rejections
    assert first.train == second.train
    assert first.validation == second.validation
    assert quality["distribution_within_tolerance"] is True
    assert quality["actual_distribution"] == quality["expected_distribution"]
    assert {item["id"] for item in first.train}.isdisjoint(
        item["id"] for item in first.validation
    )


def test_exact_duplicate_is_rejected(config, leakage) -> None:
    original = load_jsonl([SEED_PATH])[0]
    duplicate = copy.deepcopy(original)
    duplicate["id"] = "fp_ta_9999"
    result = run_pipeline([original, duplicate], config, leakage)

    assert len(result.accepted) == 1
    assert "exact duplicate conversation" in result.rejections[0].reasons


def test_near_duplicate_prompt_is_rejected(config, leakage) -> None:
    original = load_jsonl([SEED_PATH])[0]
    duplicate = copy.deepcopy(original)
    duplicate["id"] = "fp_ta_9997"
    duplicate["messages"][1]["content"] += " Carefully."
    duplicate["messages"][2]["content"] += " The conclusion remains conditional."
    result = run_pipeline([original, duplicate], config, leakage)

    assert len(result.accepted) == 1
    assert any("near-duplicate user prompt" in reason for reason in result.rejections[0].reasons)


def test_exact_evaluation_prompt_is_rejected(config, leakage) -> None:
    example = copy.deepcopy(load_jsonl([SEED_PATH])[0])
    protected = json.loads(
        (ROOT / "data" / "eval" / "finpulse_eval_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    example["id"] = "fp_ta_9998"
    example["messages"][1]["content"] = protected["prompt"]
    result = run_pipeline([example], config, leakage)

    assert not result.accepted
    assert any("evaluation leakage: stage4/" in reason for reason in result.rejections[0].reasons)


def test_near_evaluation_paraphrase_is_rejected(config, leakage) -> None:
    example = copy.deepcopy(load_jsonl([SEED_PATH])[0])
    protected = json.loads(
        (ROOT / "data" / "eval" / "finpulse_eval_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    example["id"] = "fp_ta_9996"
    example["messages"][1]["content"] = protected["prompt"] + " Explain carefully."
    result = run_pipeline([example], config, leakage)

    assert any("evaluation leakage: stage4/" in reason for reason in result.rejections[0].reasons)


def test_live_request_without_access_limitation_is_rejected(config) -> None:
    example = copy.deepcopy(load_jsonl([SEED_PATH])[0])
    example["messages"][1]["content"] = "What is BTC's exact live funding rate right now?"
    validation = validate_example(example, config)

    assert "live-data request lacks an explicit access limitation" in validation.errors


def test_live_refusal_followed_by_fabricated_value_is_rejected(config) -> None:
    example = copy.deepcopy(load_jsonl([SEED_PATH])[-2])
    example["messages"][2]["content"] += " I cannot verify it, but the rate is 0.01%."
    validation = validate_example(example, config)

    assert "live-data response contains an unsupported exact value" in validation.errors


def test_unreviewed_or_extra_metadata_is_rejected(config) -> None:
    example = copy.deepcopy(load_jsonl([SEED_PATH])[0])
    example["metadata"]["review"]["status"] = "needs_review"
    example["metadata"]["unexpected"] = True
    validation = validate_example(example, config)

    assert "metadata keys are invalid" in validation.errors
    assert "only explicitly reviewed examples may enter a split" in validation.errors


def test_text_normalization_preserves_paragraphs() -> None:
    value = "  Full-width Ａ  \r\n\r\n\r\n  Next\tline "
    assert normalize_text(value) == "Full-width A\n\nNext line"


def test_generated_manifest_hashes_and_hugging_face_loading() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["train_sha256"] == file_sha256(TRAIN_PATH)
    assert manifest["validation_sha256"] == file_sha256(VALIDATION_PATH)

    datasets = pytest.importorskip("datasets")
    loaded = datasets.load_dataset(
        "json",
        data_files={"train": str(TRAIN_PATH), "validation": str(VALIDATION_PATH)},
    )
    assert loaded.num_rows == {"train": 33, "validation": 7}
    assert loaded["train"].features["messages"].feature.keys() == {"role", "content"}
