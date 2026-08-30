"""Local, memory-conscious QLoRA training with Unsloth and TRL."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import platform
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finpulse_llm.training.config import QLoRATrainingConfig

LOGGER = logging.getLogger(__name__)
MIB = 1024**2


class _ResourceSampler:
    """Capture host and device peaks that point-in-time trainer metrics miss."""

    def __init__(self, torch_module: Any, interval_seconds: float = 0.1) -> None:
        import psutil

        self._psutil = psutil
        self._process = psutil.Process()
        self._torch = torch_module
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_process_rss = 0
        self.peak_system_used = 0
        self.peak_gpu_device_used = 0

    def _record(self) -> None:
        self.peak_process_rss = max(self.peak_process_rss, self._process.memory_info().rss)
        memory = self._psutil.virtual_memory()
        self.peak_system_used = max(self.peak_system_used, memory.total - memory.available)
        free_bytes, total_bytes = self._torch.cuda.mem_get_info()
        self.peak_gpu_device_used = max(self.peak_gpu_device_used, total_bytes - free_bytes)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._record()
            self._stop.wait(self._interval_seconds)

    def start(self) -> None:
        self._record()
        self._thread.start()

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=2)
        self._record()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_training_inputs(config: QLoRATrainingConfig) -> dict[str, Any]:
    """Fail before loading the GPU if reviewed dataset files changed unexpectedly."""

    for path in (config.data.train_file, config.data.validation_file, config.data.manifest_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required training input does not exist: {path}")
    observed_train = sha256_file(config.data.train_file)
    observed_validation = sha256_file(config.data.validation_file)
    if observed_train != config.data.train_sha256:
        raise ValueError("Training split SHA-256 does not match the locked Stage 6 configuration")
    if observed_validation != config.data.validation_sha256:
        raise ValueError("Validation split SHA-256 does not match the locked Stage 6 configuration")
    manifest = json.loads(config.data.manifest_file.read_text(encoding="utf-8"))
    if manifest.get("train_sha256") != observed_train:
        raise ValueError("Training split does not match the Stage 5 manifest")
    if manifest.get("validation_sha256") != observed_validation:
        raise ValueError("Validation split does not match the Stage 5 manifest")
    train_examples = len(_read_jsonl(config.data.train_file))
    validation_examples = len(_read_jsonl(config.data.validation_file))
    steps_per_epoch = math.ceil(train_examples / config.trainer.effective_batch_size)
    return {
        "train_sha256": observed_train,
        "validation_sha256": observed_validation,
        "protected_stage4_sha256": manifest.get("protected_stage4_sha256"),
        "train_examples": train_examples,
        "validation_examples": validation_examples,
        "estimated_optimizer_steps": math.ceil(
            steps_per_epoch * config.trainer.num_train_epochs
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"Dataset split is empty: {path}")
    return rows


def _format_rows(
    rows: list[dict[str, Any]], tokenizer: Any
) -> tuple[list[dict[str, str]], list[int]]:
    texts: list[dict[str, str]] = []
    token_lengths: list[int] = []
    for row in rows:
        rendered = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        texts.append({"text": rendered})
        token_lengths.append(len(tokenizer(rendered, add_special_tokens=False)["input_ids"]))
    return texts, token_lengths


def _adapter_parameter_counts(model: Any) -> tuple[int, int]:
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
        return int(trainable), int(total)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def _latest_evaluation_metrics(
    log_history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the evaluation already run by an epoch/steps strategy, if present."""

    return next(
        (dict(entry) for entry in reversed(log_history) if "eval_loss" in entry),
        None,
    )


def run_training(config: QLoRATrainingConfig, *, smoke_test: bool = False) -> dict[str, Any]:
    """Train and save only a LoRA adapter; never merge or upload model weights."""

    input_hashes = verify_training_inputs(config)

    # Heavy ML imports stay out of config validation and unit tests.
    import torch
    # isort: off
    # Unsloth must patch Transformers/TRL before either package is imported.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    # isort: on

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Stage 6 refuses CPU training")
    total_vram = torch.cuda.get_device_properties(0).total_memory
    if total_vram < 5 * 1024**3:
        raise RuntimeError("This profile expects a CUDA GPU with at least 5 GiB VRAM")
    if config.trainer.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configuration requests bf16, but this GPU/PyTorch stack lacks support")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sampler = _ResourceSampler(torch)
    sampler.start()
    started = time.perf_counter()
    model = tokenizer = trainer = None
    try:
        LOGGER.info(
            "Loading pinned base model %s at %s",
            config.model.model_id,
            config.model.revision,
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model.model_id,
            revision=config.model.revision,
            max_seq_length=config.model.max_sequence_length,
            load_in_4bit=config.model.load_in_4bit,
            trust_remote_code=config.model.trust_remote_code,
            local_files_only=True,
        )

        available_modules = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        missing_targets = sorted(set(config.lora.target_modules) - available_modules)
        if missing_targets:
            raise ValueError(f"LoRA target modules absent from model: {missing_targets}")

        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            target_modules=list(config.lora.target_modules),
            use_gradient_checkpointing=config.lora.use_gradient_checkpointing,
            random_state=config.seed,
            use_rslora=config.lora.use_rslora,
            loftq_config=None,
        )
        trainable_parameters, total_parameters = _adapter_parameter_counts(model)

        train_rows = _read_jsonl(config.data.train_file)
        validation_rows = _read_jsonl(config.data.validation_file)
        if smoke_test:
            train_rows = train_rows[:2]
            validation_rows = validation_rows[:1]
        train_texts, train_lengths = _format_rows(train_rows, tokenizer)
        validation_texts, validation_lengths = _format_rows(validation_rows, tokenizer)
        longest = max(train_lengths + validation_lengths)
        if longest > config.model.max_sequence_length:
            raise ValueError(
                f"Longest example is {longest} tokens, above configured maximum "
                f"{config.model.max_sequence_length}; refusing silent truncation"
            )

        training_args = SFTConfig(
            output_dir=str(config.output.working_dir),
            run_name=config.run_name + ("-smoke" if smoke_test else ""),
            max_length=config.model.max_sequence_length,
            dataset_text_field="text",
            packing=config.trainer.packing,
            per_device_train_batch_size=config.trainer.per_device_train_batch_size,
            per_device_eval_batch_size=config.trainer.per_device_eval_batch_size,
            gradient_accumulation_steps=(
                1 if smoke_test else config.trainer.gradient_accumulation_steps
            ),
            num_train_epochs=config.trainer.num_train_epochs,
            max_steps=(1 if smoke_test else -1),
            learning_rate=config.trainer.learning_rate,
            weight_decay=config.trainer.weight_decay,
            warmup_steps=config.trainer.warmup_steps,
            lr_scheduler_type=config.trainer.lr_scheduler_type,
            optim=config.trainer.optimizer,
            max_grad_norm=config.trainer.max_grad_norm,
            logging_steps=1 if smoke_test else config.trainer.logging_steps,
            logging_first_step=True,
            eval_strategy="no" if smoke_test else config.trainer.eval_strategy,
            eval_steps=config.trainer.eval_steps or 500,
            save_strategy="no" if smoke_test else config.trainer.save_strategy,
            save_steps=config.trainer.save_steps or 500,
            save_total_limit=config.trainer.save_total_limit,
            report_to="none",
            seed=config.seed,
            data_seed=config.seed,
            fp16=config.trainer.precision == "fp16",
            bf16=config.trainer.precision == "bf16",
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=Dataset.from_list(train_texts),
            eval_dataset=Dataset.from_list(validation_texts),
            args=training_args,
        )
        if config.trainer.train_on_responses_only:
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )

        LOGGER.info("Starting %s training run", "one-step smoke" if smoke_test else "full")
        train_result = trainer.train()
        if smoke_test:
            evaluation = {}
        else:
            evaluation = _latest_evaluation_metrics(trainer.state.log_history)
            if evaluation is None:
                evaluation = trainer.evaluate()
        torch.cuda.synchronize()
        sampler.stop()

        adapter_dir = (
            config.output.adapter_dir.with_name(config.output.adapter_dir.name + "-smoke")
            if smoke_test
            else config.output.adapter_dir
        )
        model.peft_config["default"].revision = config.model.revision
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        adapter_file = adapter_dir / "adapter_model.safetensors"
        if not adapter_file.is_file() or adapter_file.stat().st_size == 0:
            raise RuntimeError("Training completed, but the LoRA adapter file was not saved")
        try:
            adapter_display_path = str(adapter_dir.relative_to(Path.cwd()))
        except ValueError:
            adapter_display_path = str(adapter_dir)
        elapsed = time.perf_counter() - started
        report = {
            "schema_version": 1,
            "stage": 6,
            "run_name": training_args.run_name,
            "smoke_test": smoke_test,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "status": "completed",
            "base_model": {
                "model_id": config.model.model_id,
                "revision": config.model.revision,
                "quantization": "bitsandbytes 4-bit",
            },
            "dataset": {
                **input_hashes,
                "train_examples": len(train_rows),
                "validation_examples": len(validation_rows),
                "train_token_min": min(train_lengths),
                "train_token_mean": round(sum(train_lengths) / len(train_lengths), 2),
                "train_token_max": max(train_lengths),
                "validation_token_max": max(validation_lengths),
            },
            "lora": {**asdict(config.lora), "target_modules": list(config.lora.target_modules)},
            "trainer": {
                **asdict(config.trainer),
                "effective_batch_size": 1 if smoke_test else config.trainer.effective_batch_size,
                "max_sequence_length": config.model.max_sequence_length,
            },
            "parameters": {
                "trainable": trainable_parameters,
                "total": total_parameters,
                "trainable_percent": round(100 * trainable_parameters / total_parameters, 6),
            },
            "metrics": {
                "train": train_result.metrics,
                "evaluation": evaluation,
                "log_history": trainer.state.log_history,
                "duration_seconds_wall": round(elapsed, 3),
                "peak_process_ram_mib": round(sampler.peak_process_rss / MIB, 1),
                "peak_system_ram_used_mib": round(sampler.peak_system_used / MIB, 1),
                "peak_gpu_device_used_mib": round(sampler.peak_gpu_device_used / MIB, 1),
                "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB, 1),
                "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
            },
            "environment": {
                "os": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_vram_mib": round(total_vram / MIB, 1),
            },
            "adapter": {
                "directory": adapter_display_path,
                "weights_file": adapter_file.name,
                "weights_size_bytes": adapter_file.stat().st_size,
                "weights_sha256": sha256_file(adapter_file),
            },
        }
        metrics_file = (
            config.output.metrics_file.with_name(config.output.metrics_file.stem + "_smoke.json")
            if smoke_test
            else config.output.metrics_file
        )
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        return report
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(
            "QLoRA exceeded available VRAM. Close GPU-heavy applications; do not increase "
            "batch size or sequence length on this 6 GB profile."
        ) from exc
    finally:
        sampler.stop()
        del trainer, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
