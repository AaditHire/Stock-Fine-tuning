"""Configuration models for local LLM inference."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling settings passed to Hugging Face ``generate``."""

    max_new_tokens: int
    do_sample: bool
    enable_thinking: bool
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float


@dataclass(frozen=True)
class ModelConfig:
    """Model-loading and generation settings kept outside application code."""

    model_id: str
    revision: str | None
    max_sequence_length: int
    load_in_4bit: bool
    trust_remote_code: bool
    seed: int
    system_prompt: str
    generation: GenerationConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"Missing required configuration key: {key}") from exc


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and validate a model configuration from TOML."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Model configuration does not exist: {config_path}")

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    generation_raw = _require(raw, "generation")
    generation = GenerationConfig(
        max_new_tokens=int(_require(generation_raw, "max_new_tokens")),
        do_sample=bool(generation_raw.get("do_sample", True)),
        enable_thinking=bool(_require(generation_raw, "enable_thinking")),
        temperature=float(_require(generation_raw, "temperature")),
        top_p=float(_require(generation_raw, "top_p")),
        top_k=int(_require(generation_raw, "top_k")),
        min_p=float(_require(generation_raw, "min_p")),
        repetition_penalty=float(_require(generation_raw, "repetition_penalty")),
    )
    config = ModelConfig(
        model_id=str(_require(raw, "model_id")),
        revision=str(raw["revision"]) if raw.get("revision") else None,
        max_sequence_length=int(_require(raw, "max_sequence_length")),
        load_in_4bit=bool(_require(raw, "load_in_4bit")),
        trust_remote_code=bool(raw.get("trust_remote_code", False)),
        seed=int(_require(raw, "seed")),
        system_prompt=str(_require(raw, "system_prompt")).strip(),
        generation=generation,
    )
    _validate(config)
    return config


def _validate(config: ModelConfig) -> None:
    if not config.model_id:
        raise ValueError("model_id must not be empty")
    if config.max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if not config.load_in_4bit:
        raise ValueError("Local benchmark models must use 4-bit loading for the 6 GB GPU")
    if config.generation.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if config.generation.temperature <= 0:
        raise ValueError("temperature must be greater than zero for Qwen3 sampling")
    if not 0 < config.generation.top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")
