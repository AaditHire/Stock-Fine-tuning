# finpulse-llm

`finpulse-llm` is a standalone learning project for running, benchmarking, QLoRA fine-tuning, evaluating, and exporting a small financial language model on consumer hardware.

This repository is deliberately independent of the existing FinPulse project. It has no FinPulse imports, integrations, live market APIs, RAG, vector databases, embeddings, agents, or MCP components.

## Current status

Stage 2 is complete: Qwen3-4B runs locally through Unsloth in 4-bit mode, with measured RAM, VRAM, load time, and generation speed. No training or dataset creation has started.

## Hardware target

- Windows 11 laptop
- NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM)
- AMD Ryzen 7 6800HS
- 16 GB RAM

All later configurations must fit this machine. Training will use 4-bit QLoRA rather than full fine-tuning, and only one model will be loaded at a time.

## Recommended environment

Start with a dedicated **native Windows virtual environment**. As of August 2026, Unsloth officially supports direct Windows training, bitsandbytes supports its CUDA backend on Windows, and the current native PyTorch installation already detects this GPU. Native Windows also avoids reserving part of the limited 16 GB RAM for a WSL2 virtual machine.

WSL2 with Ubuntu remains the fallback if a later training stage exposes a reproducible native-Windows problem. Stage 2 validated Unsloth, Triton for Windows, bitsandbytes, and CUDA inference natively, so there is currently no reason to switch.

The global Python installation is for diagnostics only. The working ML stack is isolated in `.venv` and pinned in `requirements-ml.txt`. Do not install training packages globally.

## Observed environment (2026-08-28)

| Component | Observed value |
| --- | --- |
| OS | Windows 11 Home Single Language, 64-bit, build 26200 |
| Python | 3.13.0 |
| Git | 2.53.0.windows.3 |
| RAM | 15.24 GiB usable |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| VRAM | 6 GiB |
| NVIDIA driver | 610.74 |
| Driver-reported CUDA capability | 13.3 |
| PyTorch | 2.9.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| `torch.cuda.is_available()` | `True` |
| WSL2 | Not installed |

The isolated Stage 2 environment uses PyTorch 2.11.0+cu130, Unsloth 2026.8.22, Transformers 5.5.0, TRL 0.24.0, PEFT 0.20.0, and bitsandbytes 0.50.2.

The CUDA version shown by `nvidia-smi` is the newest CUDA runtime the driver can support; it is not the same as the CUDA runtime bundled with PyTorch. For this host those values are 13.3 and 13.0 respectively, which is normal.

## Repository layout

```text
finpulse-llm/
├── configs/
│   ├── models/
│   └── training/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── eval/
├── models/
├── notebooks/
├── results/
│   ├── benchmarks/
│   └── training/
├── scripts/
│   └── check_environment.py
├── src/finpulse_llm/
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   ├── training/
│   └── utils/
└── tests/
```

Generated data, model weights, adapters, checkpoints, and caches are ignored by Git. Small JSON benchmark summaries may be versioned; large generated outputs remain excluded.

## Stage 2 setup and inference

Create the validated native-Windows environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run the four-prompt finance smoke suite:

```powershell
.\.venv\Scripts\python.exe scripts\run_qwen_inference.py
```

Run a custom prompt:

```powershell
.\.venv\Scripts\python.exe scripts\run_qwen_inference.py --prompt "Explain RSI divergence."
```

After the model is cached, force fully offline loading with `--offline`.

Configuration lives in `configs/models/qwen3_4b.toml`. The model cache is stored under `models/huggingface/` and is ignored by Git. The measured report is `results/benchmarks/stage2_qwen3_4b.json`.

On this machine, the cached model loaded in 8.2 seconds, used a peak 3.73 GiB of total device VRAM (including Windows GPU use), and generated about 10.8 tokens/second across the smoke suite. See `docs/stage2.md` for the complete findings and explanations.

## Environment diagnostic

Run the human-readable report from the repository root:

```powershell
python scripts/check_environment.py
```

For machine-readable output:

```powershell
python scripts/check_environment.py --json
```

PyTorch is optional for the script. If it is absent, the report says so rather than failing. The script also queries `nvidia-smi` when available.

## Development checks

The package uses a `src` layout, so local checks can run without installing it:

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest-tmp
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
```

## Staged roadmap

Development stops after each stage and resumes only when explicitly requested.

1. **Repository and environment foundation** — inspect the machine, choose the platform, scaffold the repository, and add diagnostics.
2. **ML environment and Qwen local inference** — create a pinned environment, load only Qwen3-4B in 4-bit mode, and measure memory and speed. No training.
3. **Base-model benchmark** — compare Qwen3-4B and `microsoft/Phi-4-mini-instruct` with identical prompts and measured resource use.
4. **Frozen FinPulse evaluation benchmark** — create a separate 150–250 item evaluation set, evaluation runner, and base-model baseline.
5. **Financial instruction dataset pipeline** — define provenance-aware conversational data, validation, cleaning, deduplication, leakage checks, and splits.
6. **QLoRA fine-tuning with Unsloth** — run conservative, configurable adapter training within the 6 GB VRAM limit.
7. **Base vs. fine-tuned evaluation** — compare both models on the frozen benchmark and report improvements and regressions.
8. **Local export and inference** — export adapters/merged artifacts and investigate GGUF, llama.cpp, and Ollama use.

## Scope and safety principles

- Teach analysis patterns and uncertainty, not changing prices or current market facts.
- Never fabricate live market values when no data is supplied.
- Keep the frozen evaluation set out of training data.
- Prefer reproducible configuration files and measured results.
- Estimate VRAM use before expensive operations and stop on unsafe configurations.
- Never commit secrets, large datasets, model weights, or generated checkpoints.

## License

No license has been selected yet. Until one is added, the repository should be treated as all rights reserved.
