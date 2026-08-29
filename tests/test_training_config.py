from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from finpulse_llm.training.config import load_training_config
from finpulse_llm.training.runner import sha256_file, verify_training_inputs

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "training" / "qwen3_4b_stage6.toml"
STAGE6B_CONFIG = ROOT / "configs" / "training" / "qwen3_4b_stage6b.toml"


def test_stage6_configuration_is_memory_conservative() -> None:
    config = load_training_config(CONFIG, ROOT)

    assert config.model.load_in_4bit is True
    assert config.model.max_sequence_length == 512
    assert config.trainer.per_device_train_batch_size == 1
    assert config.trainer.effective_batch_size == 4
    assert config.lora.rank == 16
    assert config.lora.alpha == 32
    assert set(config.lora.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }


def test_training_inputs_match_locked_manifest() -> None:
    config = load_training_config(CONFIG, ROOT)
    result = verify_training_inputs(config)
    manifest = json.loads(config.data.manifest_file.read_text(encoding="utf-8"))

    assert result["train_sha256"] == manifest["train_sha256"]
    assert result["validation_sha256"] == manifest["validation_sha256"]
    assert result["protected_stage4_sha256"] == manifest["protected_stage4_sha256"]


def test_changed_training_input_is_rejected(tmp_path: Path) -> None:
    config = load_training_config(CONFIG, ROOT)
    train_copy = tmp_path / "train.jsonl"
    train_copy.write_text("{}\n", encoding="utf-8")
    changed = replace(config, data=replace(config.data, train_file=train_copy))

    assert sha256_file(train_copy) != config.data.train_sha256
    with pytest.raises(ValueError, match="Training split SHA-256"):
        verify_training_inputs(changed)


def test_non_4bit_configuration_is_rejected(tmp_path: Path) -> None:
    changed = CONFIG.read_text(encoding="utf-8").replace(
        "load_in_4bit = true", "load_in_4bit = false"
    )
    path = tmp_path / "unsafe.toml"
    path.write_text(changed, encoding="utf-8")

    with pytest.raises(ValueError, match="requires 4-bit"):
        load_training_config(path, ROOT)


def test_stage6b_configuration_uses_locked_corrective_data() -> None:
    config = load_training_config(STAGE6B_CONFIG, ROOT)
    result = verify_training_inputs(config)

    assert config.run_name == "stage6b-qwen3-4b-stage5b-v1"
    assert config.model.load_in_4bit is True
    assert config.model.max_sequence_length == 512
    assert config.trainer.per_device_train_batch_size == 1
    assert config.trainer.effective_batch_size == 4
    assert config.trainer.learning_rate == 0.0001
    assert config.trainer.num_train_epochs == 1.0
    assert result["train_sha256"] == config.data.train_sha256
    assert result["validation_sha256"] == config.data.validation_sha256
    assert result["protected_stage4_sha256"] == (
        "bfd1b847d2042f6a59f8a8a5f0dfe0826729dc68a0d390a356c2f1fd3b1781fa"
    )
