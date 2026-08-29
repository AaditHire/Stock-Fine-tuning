from pathlib import Path

import pytest

from finpulse_llm.inference.config import load_model_config
from finpulse_llm.inference.prompts import FINANCE_SMOKE_PROMPTS
from finpulse_llm.inference.runner import _generation_arguments, _MemorySampler

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "models" / "qwen3_4b.toml"
PHI_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "models" / "phi4_mini.toml"
EVAL_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "models" / "qwen3_4b_eval.toml"


def test_qwen_config_is_memory_safe() -> None:
    config = load_model_config(CONFIG_PATH)

    assert config.model_id == "unsloth/Qwen3-4B-bnb-4bit"
    assert config.load_in_4bit is True
    assert config.max_sequence_length == 2048
    assert config.generation.max_new_tokens <= 256
    assert config.generation.enable_thinking is False


def test_stage3_model_configs_use_identical_benchmark_settings() -> None:
    qwen = load_model_config(CONFIG_PATH)
    phi = load_model_config(PHI_CONFIG_PATH)

    assert phi.model_id == "unsloth/Phi-4-mini-instruct-bnb-4bit"
    assert phi.load_in_4bit is True
    assert phi.trust_remote_code is False
    assert phi.revision == "cece1fd36f04ff79f55ec861f206ca4e16acea6e"
    assert phi.max_sequence_length == qwen.max_sequence_length
    assert phi.seed == qwen.seed
    assert phi.system_prompt == qwen.system_prompt
    assert phi.generation == qwen.generation


def test_stage4_eval_config_uses_deterministic_decoding() -> None:
    config = load_model_config(EVAL_CONFIG_PATH)

    assert config.model_id == "unsloth/Qwen3-4B-bnb-4bit"
    assert config.load_in_4bit is True
    assert config.generation.do_sample is False
    assert config.generation.max_new_tokens == 192
    arguments = _generation_arguments(config)
    assert arguments["do_sample"] is False
    assert "temperature" not in arguments


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
