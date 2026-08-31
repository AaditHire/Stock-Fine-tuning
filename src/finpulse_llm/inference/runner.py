"""Memory-conscious Unsloth inference with reproducible performance measurements."""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from finpulse_llm.inference.config import ModelConfig

LOGGER = logging.getLogger(__name__)
MIB = 1024**2


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    generation_seconds: float
    tokens_per_second: float


@dataclass(frozen=True)
class RunMetrics:
    model_id: str
    quantization: str
    max_sequence_length: int
    load_seconds: float
    peak_process_ram_mib: float
    peak_system_ram_used_mib: float
    peak_gpu_device_used_mib: float
    peak_gpu_allocated_mib: float
    peak_gpu_reserved_mib: float
    torch_version: str
    torch_cuda_version: str | None
    gpu_name: str
    results: tuple[GenerationResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _MemorySampler:
    """Sample RAM in the background because a single final RSS value misses peaks."""

    def __init__(self, torch_module: Any | None = None, interval_seconds: float = 0.1) -> None:
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

    def _record_sample(self) -> None:
        process_rss = self._process.memory_info().rss
        memory = self._psutil.virtual_memory()
        system_used = memory.total - memory.available
        self.peak_process_rss = max(self.peak_process_rss, process_rss)
        self.peak_system_used = max(self.peak_system_used, system_used)
        if self._torch is not None and self._torch.cuda.is_available():
            free_bytes, total_bytes = self._torch.cuda.mem_get_info()
            self.peak_gpu_device_used = max(
                self.peak_gpu_device_used, total_bytes - free_bytes
            )

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._record_sample()
            self._stop.wait(self._interval_seconds)

    def start(self) -> None:
        self._record_sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._record_sample()


def _generation_arguments(config: ModelConfig) -> dict[str, Any]:
    generation = config.generation
    arguments: dict[str, Any] = {
        "max_new_tokens": generation.max_new_tokens,
        "do_sample": generation.do_sample,
        "repetition_penalty": generation.repetition_penalty,
        "use_cache": True,
    }
    if generation.do_sample:
        arguments.update(
            temperature=generation.temperature,
            top_p=generation.top_p,
            top_k=generation.top_k,
            min_p=generation.min_p,
        )
    return arguments


def run_inference(
    config: ModelConfig,
    prompts: Sequence[str],
    on_result: Callable[[GenerationResult], None] | None = None,
    batch_size: int = 1,
) -> RunMetrics:
    """Load one 4-bit model, answer every prompt, and return measured results."""

    if not prompts:
        raise ValueError("At least one prompt is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Heavy imports stay inside the execution path so config/tests remain lightweight.
    import torch
    from unsloth import FastModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to load Qwen3-4B on CPU")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    sampler = _MemorySampler(torch)
    sampler.start()
    model = None
    tokenizer = None
    generated: list[GenerationResult] = []

    try:
        LOGGER.info("Loading %s in 4-bit mode", config.model_id)
        load_started = time.perf_counter()
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.model_id,
            revision=config.revision,
            max_seq_length=config.max_sequence_length,
            load_in_4bit=config.load_in_4bit,
            trust_remote_code=config.trust_remote_code,
        )
        FastModel.for_inference(model)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            LOGGER.info(
                "Generating responses %d-%d/%d",
                start + 1,
                start + len(batch_prompts),
                len(prompts),
            )
            rendered = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": config.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=config.generation.enable_thinking,
                )
                for prompt in batch_prompts
            ]
            tokenizer.padding_side = "left"
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_sequence_length - config.generation.max_new_tokens,
            ).to(model.device)
            padded_input_tokens = int(inputs["input_ids"].shape[-1])
            input_token_counts = [int(value) for value in inputs["attention_mask"].sum(dim=1)]

            torch.cuda.synchronize()
            generation_started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, **_generation_arguments(config))
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - generation_started

            continuation_ids = output[:, padded_input_tokens:]
            responses = tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)
            seconds_per_response = generation_seconds / len(batch_prompts)
            pad_token_id = tokenizer.pad_token_id
            for prompt, response, input_tokens, output_ids in zip(
                batch_prompts,
                responses,
                input_token_counts,
                continuation_ids,
                strict=True,
            ):
                output_tokens = sum(
                    int(token_id) != pad_token_id for token_id in output_ids.tolist()
                )
                generated.append(
                    GenerationResult(
                        prompt=prompt,
                        response=response.strip(),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        generation_seconds=round(seconds_per_response, 3),
                        tokens_per_second=round(
                            output_tokens / seconds_per_response, 3
                        ),
                    )
                )
                if on_result is not None:
                    on_result(generated[-1])

        sampler.stop()
        return RunMetrics(
            model_id=config.model_id,
            quantization="bitsandbytes 4-bit",
            max_sequence_length=config.max_sequence_length,
            load_seconds=round(load_seconds, 3),
            peak_process_ram_mib=round(sampler.peak_process_rss / MIB, 1),
            peak_system_ram_used_mib=round(sampler.peak_system_used / MIB, 1),
            peak_gpu_device_used_mib=round(sampler.peak_gpu_device_used / MIB, 1),
            peak_gpu_allocated_mib=round(torch.cuda.max_memory_allocated() / MIB, 1),
            peak_gpu_reserved_mib=round(torch.cuda.max_memory_reserved() / MIB, 1),
            torch_version=torch.__version__,
            torch_cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(0),
            results=tuple(generated),
        )
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(
            "Qwen3-4B exceeded available VRAM. Close GPU-heavy desktop applications and retry."
        ) from exc
    finally:
        sampler.stop()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
