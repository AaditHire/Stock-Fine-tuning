"""Configuration loading for the Stage 5 dataset pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPipelineConfig:
    """Quality thresholds and reproducibility settings kept outside pipeline code."""

    dataset_id: str
    seed: int
    validation_ratio: float
    near_duplicate_threshold: float
    evaluation_leakage_threshold: float
    distribution_tolerance: float
    min_user_characters: int
    min_assistant_characters: int
    max_message_characters: int
    fail_on_rejection: bool
    system_prompt: str
    expected_distribution: dict[str, float]


def load_data_config(path: str | Path) -> DataPipelineConfig:
    """Load and validate the TOML data-pipeline configuration."""

    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    config = DataPipelineConfig(
        dataset_id=str(raw["dataset_id"]),
        seed=int(raw["seed"]),
        validation_ratio=float(raw["validation_ratio"]),
        near_duplicate_threshold=float(raw["near_duplicate_threshold"]),
        evaluation_leakage_threshold=float(raw["evaluation_leakage_threshold"]),
        distribution_tolerance=float(raw["distribution_tolerance"]),
        min_user_characters=int(raw["min_user_characters"]),
        min_assistant_characters=int(raw["min_assistant_characters"]),
        max_message_characters=int(raw["max_message_characters"]),
        fail_on_rejection=bool(raw["fail_on_rejection"]),
        system_prompt=str(raw["system_prompt"]).strip(),
        expected_distribution={
            str(key): float(value) for key, value in raw["expected_distribution"].items()
        },
    )
    if not 0 < config.validation_ratio < 0.5:
        raise ValueError("validation_ratio must be between 0 and 0.5")
    for name, value in (
        ("near_duplicate_threshold", config.near_duplicate_threshold),
        ("evaluation_leakage_threshold", config.evaluation_leakage_threshold),
        ("distribution_tolerance", config.distribution_tolerance),
    ):
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if abs(sum(config.expected_distribution.values()) - 1.0) > 1e-9:
        raise ValueError("expected_distribution must sum to 1.0")
    return config
