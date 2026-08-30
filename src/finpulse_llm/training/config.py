"""Validated configuration for local QLoRA training."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelTrainingConfig:
    model_id: str
    revision: str
    load_in_4bit: bool
    max_sequence_length: int
    trust_remote_code: bool


@dataclass(frozen=True)
class TrainingDataConfig:
    train_file: Path
    validation_file: Path
    manifest_file: Path
    train_sha256: str
    validation_sha256: str


@dataclass(frozen=True)
class LoraTrainingConfig:
    rank: int
    alpha: int
    dropout: float
    bias: str
    target_modules: tuple[str, ...]
    use_gradient_checkpointing: str
    use_rslora: bool


@dataclass(frozen=True)
class TrainerConfig:
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    lr_scheduler_type: str
    optimizer: str
    precision: str
    max_grad_norm: float
    logging_steps: int
    eval_strategy: str
    packing: bool
    train_on_responses_only: bool
    eval_steps: int | None = None
    save_strategy: str = "no"
    save_steps: int | None = None
    save_total_limit: int | None = None

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class TrainingOutputConfig:
    adapter_dir: Path
    working_dir: Path
    metrics_file: Path


@dataclass(frozen=True)
class QLoRATrainingConfig:
    run_name: str
    seed: int
    model: ModelTrainingConfig
    data: TrainingDataConfig
    lora: LoraTrainingConfig
    trainer: TrainerConfig
    output: TrainingOutputConfig


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def load_training_config(
    path: str | Path, repository_root: str | Path | None = None
) -> QLoRATrainingConfig:
    """Load Stage 6 TOML and reject settings unsafe for the 6 GB target GPU."""

    config_path = Path(path).resolve()
    root = Path(repository_root).resolve() if repository_root else config_path.parents[2]
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    model = ModelTrainingConfig(**raw["model"])
    data_raw = raw["data"]
    data = TrainingDataConfig(
        train_file=_resolve(root, data_raw["train_file"]),
        validation_file=_resolve(root, data_raw["validation_file"]),
        manifest_file=_resolve(root, data_raw["manifest_file"]),
        train_sha256=str(data_raw["train_sha256"]),
        validation_sha256=str(data_raw["validation_sha256"]),
    )
    lora = LoraTrainingConfig(
        rank=int(raw["lora"]["rank"]),
        alpha=int(raw["lora"]["alpha"]),
        dropout=float(raw["lora"]["dropout"]),
        bias=str(raw["lora"]["bias"]),
        target_modules=tuple(raw["lora"]["target_modules"]),
        use_gradient_checkpointing=str(raw["lora"]["use_gradient_checkpointing"]),
        use_rslora=bool(raw["lora"]["use_rslora"]),
    )
    trainer = TrainerConfig(
        num_train_epochs=float(raw["trainer"]["num_train_epochs"]),
        per_device_train_batch_size=int(raw["trainer"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(raw["trainer"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(raw["trainer"]["gradient_accumulation_steps"]),
        learning_rate=float(raw["trainer"]["learning_rate"]),
        weight_decay=float(raw["trainer"]["weight_decay"]),
        warmup_steps=int(raw["trainer"]["warmup_steps"]),
        lr_scheduler_type=str(raw["trainer"]["lr_scheduler_type"]),
        optimizer=str(raw["trainer"]["optimizer"]),
        precision=str(raw["trainer"]["precision"]),
        max_grad_norm=float(raw["trainer"]["max_grad_norm"]),
        logging_steps=int(raw["trainer"]["logging_steps"]),
        eval_strategy=str(raw["trainer"]["eval_strategy"]),
        packing=bool(raw["trainer"]["packing"]),
        train_on_responses_only=bool(raw["trainer"]["train_on_responses_only"]),
        eval_steps=(
            int(raw["trainer"]["eval_steps"])
            if "eval_steps" in raw["trainer"]
            else None
        ),
        save_strategy=str(raw["trainer"].get("save_strategy", "no")),
        save_steps=(
            int(raw["trainer"]["save_steps"])
            if "save_steps" in raw["trainer"]
            else None
        ),
        save_total_limit=(
            int(raw["trainer"]["save_total_limit"])
            if "save_total_limit" in raw["trainer"]
            else None
        ),
    )
    output_raw = raw["output"]
    output = TrainingOutputConfig(
        adapter_dir=_resolve(root, output_raw["adapter_dir"]),
        working_dir=_resolve(root, output_raw["working_dir"]),
        metrics_file=_resolve(root, output_raw["metrics_file"]),
    )
    config = QLoRATrainingConfig(
        run_name=str(raw["run_name"]),
        seed=int(raw["seed"]),
        model=model,
        data=data,
        lora=lora,
        trainer=trainer,
        output=output,
    )
    _validate(config)
    return config


def _validate(config: QLoRATrainingConfig) -> None:
    if not config.model.load_in_4bit:
        raise ValueError("Stage 6 requires 4-bit loading; full fine-tuning is out of scope")
    if not 128 <= config.model.max_sequence_length <= 2048:
        raise ValueError("max_sequence_length must be between 128 and 2048 for this 6 GB profile")
    if config.trainer.per_device_train_batch_size != 1:
        raise ValueError("The RTX 3060 6 GB profile requires micro-batch size 1")
    if config.lora.rank <= 0 or config.lora.alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not config.lora.target_modules:
        raise ValueError("At least one LoRA target module is required")
    if config.trainer.effective_batch_size < 1:
        raise ValueError("Effective batch size must be positive")
    if config.trainer.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be positive")
    if not 0 < config.trainer.learning_rate <= 0.001:
        raise ValueError("learning_rate must be in (0, 0.001]")
    if config.trainer.warmup_steps < 0:
        raise ValueError("warmup_steps must not be negative")
    if config.trainer.eval_strategy not in {"no", "epoch", "steps"}:
        raise ValueError("eval_strategy must be no, epoch, or steps")
    if config.trainer.eval_strategy == "steps" and (
        config.trainer.eval_steps is None or config.trainer.eval_steps <= 0
    ):
        raise ValueError("eval_steps must be positive when eval_strategy is steps")
    if config.trainer.save_strategy not in {"no", "epoch", "steps"}:
        raise ValueError("save_strategy must be no, epoch, or steps")
    if config.trainer.save_strategy == "steps" and (
        config.trainer.save_steps is None or config.trainer.save_steps <= 0
    ):
        raise ValueError("save_steps must be positive when save_strategy is steps")
    if config.trainer.save_total_limit is not None and config.trainer.save_total_limit <= 0:
        raise ValueError("save_total_limit must be positive when supplied")
    if config.trainer.precision not in {"bf16", "fp16"}:
        raise ValueError("precision must be bf16 or fp16")
