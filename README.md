# finpulse-llm

`finpulse-llm` is a standalone learning project for running, benchmarking, QLoRA fine-tuning, evaluating, and exporting a small financial language model on consumer hardware.

This repository is deliberately independent of the existing FinPulse project. It has no FinPulse imports, integrations, live market APIs, RAG, vector databases, embeddings, agents, or MCP components.

## Current status

Stage 6C training is complete after Stage 7B rejected the corrective adapter. The balanced 900-row Stage 5D view trained a rank-8 LoRA adapter for 57 updates at `5e-5`, reaching mean training loss 1.6623 and validation loss 0.8718. Peak total device VRAM was 5,273.9 MiB. The adapter remains an unevaluated candidate: Stage 7C and Stage 8 have not started.

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

## Stage 5B corrective dataset

After Stage 7 rejected the seed adapter, Stage 5B built a separate 500-example corrective corpus while preserving the original Stage 5 files. The corpus contains 150 verified calculation drills, 100 concise multiple-choice tasks, 75 factual tasks, 100 conditional analyses, 25 live-data refusals, and 50 strict instruction-following tasks. Half of all answers use exact `FINAL:` markers and 75 are JSON-only.

The deterministic split contains 398 training, 51 validation, and 51 development examples. The development holdout is reserved for candidate selection before returning to the unchanged frozen benchmark. Stage 6B trained one corrective adapter using only train and validation data; see `docs/stage5b.md` and `docs/stage6b.md` for the data and measured training run.

## Stage 7 base-versus-adapter evaluation

Run or resume the adapter on the unchanged 160-case benchmark, then rebuild the comparison:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage7_adapter.py --resume
.\.venv\Scripts\python.exe scripts\run_stage7_general_regression.py
.\.venv\Scripts\python.exe scripts\compare_stage7_models.py
```

The base passed 355/392 checks (90.56%); the adapter passed 331/392 (84.44%). Hallucination resistance rose from 87.5% to 100%, while calculations fell from 75% to 50% and technical analysis fell from 95.31% to 70.31%. The adapter remains useful as an experiment but should not be exported as the finished model. See `docs/stage7.md` and `results/benchmarks/stage7_comparison.md` for the complete audit.

## Stage 7B corrective evaluation

The Stage 6B adapter first passed all seven checks in the 51-case Stage 5B development gate,
improving from 75.00% to 91.13%. On the unchanged frozen benchmark it then fell to 70.92%
(278/392), a 19.64-point regression from the saved base result. Financial calculations fell to
18.75%, while hallucination resistance remained at 100%; the three-case general sentinel stayed
at 2/3. The development split's shared template families made it an unreliable predictor of
out-of-template transfer. See `docs/stage7b.md` and
`results/benchmarks/stage7b_comparison.md` for the full audit.

## Stage 5C pinned external-data collection

Stage 5C collected 3,000 code-verified Cosimo finance examples, 1,500 FinQA financial-report reasoning examples, and 300 project-authored behavior examples. The deterministic result contains 3,900 train, 450 validation, and 450 development records. Whole external template/document families are isolated between splits, every rendered record fits 512 tokens, and the frozen benchmark hash remains unchanged.

The collection rejected 2,997 duplicate Cosimo conversations, demonstrating why the external sources must be curated rather than ingested blindly. The resulting corpus is still 90.5% calculation tasks; Stage 5D therefore creates a smaller balanced training view instead of training on the pool directly. See `docs/stage5c.md` for source revisions, licenses, limitations, and reproduction.

## Stage 5D balanced training view

Stage 5D deterministically selects 900 unique Stage 5C training rows: all 300 project-behavior examples, 400 Cosimo examples, and 200 FinQA examples. The resulting view contains 525 calculations, 125 multiple-choice tasks, 100 analyses, 75 factual tasks, 50 instruction-following tasks, and 25 live-data refusals. It retains all 75 JSON-only examples and uses the unchanged Stage 5C validation/development holdouts. See `docs/stage5d.md` for the sampling contract and measured distribution.

## Stage 6C gentle configuration preflight

Stage 6C halves LoRA rank from 16 to 8, lowers learning rate from `1e-4` to `5e-5`, and raises gradient accumulation from 4 to 16. All 900 training rows are exposed once in approximately 57 optimizer updates. The CPU-only preflight verified dataset and frozen-benchmark hashes without loading model weights or CUDA. Checkpoints are configured every 15 steps with a four-checkpoint limit. See `docs/stage6c.md` for the rationale and remaining GPU-memory risk.

The subsequent one-step GPU smoke test completed in the native Windows environment, trained 16.5 million adapter parameters, peaked at 4,085.5 MiB total device VRAM, and saved a structurally valid 504-tensor rank-8 adapter. Its zero warmup learning rate means it was a mechanics and memory test, not a quality result.

The authorized full run then completed all 57 updates, saved four checkpoints, and produced a 33.1 MB bfloat16 adapter. Logged loss declined from 3.275 to 1.103 near the end; validation loss was 0.8718. These are training-mechanics results only. The adapter must pass a Stage 7C development comparison before at most one candidate returns to the frozen benchmark. See `docs/stage6c.md` and `results/training/stage6c_qwen3_4b_stage5d_v1.json`.

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

Corrective iteration after Stage 7:

- **Stage 5B — complete:** expanded behavior-balanced data and an independent development holdout.
- **Stage 6B — complete:** one conservative QLoRA candidate trained using only the locked Stage 5B train/validation splits.
- **Stage 7B — complete, adapter rejected:** the development gate passed, but the frozen score regressed from 90.56% to 70.92%; stop before Stage 8.
- **Stage 5C — complete:** collected and locked a pinned 4,800-example external/project candidate corpus; no training was run.
- **Stage 5D — complete:** produced a locked 900-row balanced training view and preserved the Stage 5C holdouts; no training was run.
- **Stage 6C — complete:** the rank-8 adapter trained for 57 updates within 5,273.9 MiB VRAM; it remains unevaluated and must pass Stage 7C before promotion.

## Scope and safety principles

- Teach analysis patterns and uncertainty, not changing prices or current market facts.
- Never fabricate live market values when no data is supplied.
- Keep the frozen evaluation set out of training data.
- Prefer reproducible configuration files and measured results.
- Estimate VRAM use before expensive operations and stop on unsafe configurations.
- Never commit secrets, large datasets, model weights, or generated checkpoints.

## License

No license has been selected yet. Until one is added, the repository should be treated as all rights reserved.
