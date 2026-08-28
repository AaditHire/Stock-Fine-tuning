from pathlib import Path

import pytest

from finpulse_llm.inference.config import load_model_config
from finpulse_llm.inference.prompts import FINANCE_SMOKE_PROMPTS
from finpulse_llm.inference.runner import _MemorySampler

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "models" / "qwen3_4b.toml"


def test_qwen_config_is_memory_safe() -> None:
    config = load_model_config(CONFIG_PATH)

    assert config.model_id == "unsloth/Qwen3-4B-bnb-4bit"
    assert config.load_in_4bit is True
    assert config.max_sequence_length == 2048
    assert config.generation.max_new_tokens <= 256
    assert config.generation.enable_thinking is False


def test_missing_config_has_clear_error() -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_model_config(CONFIG_PATH.with_name("missing.toml"))


def test_smoke_prompts_cover_live_data_refusal() -> None:
    assert len(FINANCE_SMOKE_PROMPTS) >= 3
    assert any("current funding rate" in prompt for prompt in FINANCE_SMOKE_PROMPTS)


def test_memory_sampler_starts_and_stops_without_blocking() -> None:
    sampler = _MemorySampler(interval_seconds=0.01)
    sampler.start()
    sampler.stop()

    assert sampler.peak_process_rss > 0
    assert sampler.peak_system_used > 0
