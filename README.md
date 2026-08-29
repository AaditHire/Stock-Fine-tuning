# finpulse-llm

`finpulse-llm` is a standalone learning project for running, benchmarking, QLoRA fine-tuning, evaluating, and exporting a small financial language model on consumer hardware.

This repository is deliberately independent of the existing FinPulse project. It has no FinPulse imports, integrations, live market APIs, RAG, vector databases, embeddings, agents, or MCP components.

## Current status

Stage 7 is complete: the Stage 6 adapter scored 84.44% on the frozen benchmark, below the base model's 90.56%, so it is rejected as a release candidate. Hallucination resistance improved to 100%, but calculations, factual finance checks, instruction following, and most domain categories regressed. The result confirms that the 40-example seed validated mechanics but was not sufficient for a successful specialist model.

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

## Stage 3 base-model benchmark

Run either model independently so only one occupies GPU memory:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage3_model.py --model-config configs\models\qwen3_4b.toml --output results\benchmarks\stage3_qwen3_4b.json --offline
.\.venv\Scripts\python.exe scripts\run_stage3_model.py --model-config configs\models\phi4_mini.toml --output results\benchmarks\stage3_phi4_mini.json --offline
```

Rebuild the comparison report from the saved results:

```powershell
.\.venv\Scripts\python.exe scripts\compare_stage3_models.py results\benchmarks\stage3_qwen3_4b.json results\benchmarks\stage3_phi4_mini.json --json-output results\benchmarks\stage3_comparison.json --markdown-output results\benchmarks\stage3_comparison.md
```

Qwen scored 82.8% (24/29 checks), compared with Phi's 75.9% (22/29). Qwen was stronger on the calculation cases, while Phi was faster and handled both live-data refusal traps. Both fit comfortably below 4 GiB of measured total device VRAM. This 15-prompt development benchmark guided model choice; it remains deliberately separate from the frozen Stage 4 evaluation set. See `docs/stage3.md` for methodology and limitations.

## Stage 4 frozen evaluation

Verify the frozen dataset and its SHA-256 manifest without modifying either file:

```powershell
.\.venv\Scripts\python.exe scripts\build_stage4_benchmark.py --check
```

Run or resume the complete Qwen baseline locally:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage4_baseline.py --offline --resume
```

The benchmark contains 160 project-authored questions, with 16 cases in each required category. Its final SHA-256 is `bfd1b847d2042f6a59f8a8a5f0dfe0826729dc68a0d390a356c2f1fd3b1781fa`. Qwen scored 90.56% (355/392 rubric checks), generated 20,785 tokens at an aggregate 10.69 tokens/second, and peaked at 3,809.5 MiB total device VRAM.

The evaluation JSONL, answers, checks, and paraphrases must never enter training data. See `docs/stage4.md` for the category results, audit notes, and leakage policy.

## Stage 5 instruction-data pipeline

Verify the reviewed seed source, then rebuild the deterministic splits:

```powershell
.\.venv\Scripts\python.exe scripts\build_stage5_seed.py
.\.venv\Scripts\python.exe scripts\build_training_dataset.py data\raw\finpulse_seed_v1.jsonl
```

The seed build contains 40 accepted and zero rejected examples: 33 train and 7 validation. Its category distribution exactly matches the requested 25% technical analysis, 20% crypto derivatives, 15% stock fundamentals, 15% macroeconomics, 10% risk management, 10% scenario analysis, and 5% terminology/miscellaneous.

The JSONL files use conversational `messages` and load directly through Hugging Face `datasets`. Every example records category, subtopics, difficulty, source type, license, reference, and review status. Exact and fuzzy checks reject training prompts that overlap Stage 3 or the frozen Stage 4 benchmark. See `docs/stage5.md` for the schema and quality rules.

## Stage 6 QLoRA fine-tuning

Check the pinned inputs without loading CUDA, then optionally run a one-step smoke test before the complete seed experiment:

```powershell
.\.venv\Scripts\python.exe scripts\train_qlora.py --preflight-only
.\.venv\Scripts\python.exe scripts\train_qlora.py --smoke-test
.\.venv\Scripts\python.exe scripts\train_qlora.py
```

The configuration is in `configs/training/qwen3_4b_stage6.toml`. It uses a 4-bit frozen base, rank-16 LoRA adapters across all major attention and MLP projections, 512-token context, micro-batch 1, gradient accumulation 4, Unsloth checkpointing, and one epoch. The measured run trained 33.0 million parameters (0.814%), peaked at 5,114.4 MiB total device VRAM, and saved a 66.1 MB adapter under the Git-ignored `models/adapters/` directory. See `docs/stage6.md` for the measurements, caveats, and concept explanations.

## Stage 7 base-versus-adapter evaluation

Run or resume the adapter on the unchanged 160-case benchmark, then rebuild the comparison:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage7_adapter.py --resume
.\.venv\Scripts\python.exe scripts\run_stage7_general_regression.py
.\.venv\Scripts\python.exe scripts\compare_stage7_models.py
```

The base passed 355/392 checks (90.56%); the adapter passed 331/392 (84.44%). Hallucination resistance rose from 87.5% to 100%, while calculations fell from 75% to 50% and technical analysis fell from 95.31% to 70.31%. The adapter remains useful as an experiment but should not be exported as the finished model. See `docs/stage7.md` and `results/benchmarks/stage7_comparison.md` for the complete audit.

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
