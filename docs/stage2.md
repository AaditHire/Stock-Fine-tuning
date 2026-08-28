# Stage 2: ML environment and Qwen3-4B local inference

Stage 2 validates that the selected software stack and Qwen3-4B can run on the RTX 3060 Laptop GPU without exceeding 6 GB VRAM. It does not train a model and is not the formal Stage 3 benchmark.

## Validated stack

| Component | Version |
| --- | --- |
| Python | 3.13.0 |
| PyTorch | 2.11.0+cu130 |
| CUDA runtime bundled with PyTorch | 13.0 |
| Unsloth | 2026.8.22 |
| Unsloth Zoo | 2026.8.16 |
| Transformers | 5.5.0 |
| TRL | 0.24.0 |
| PEFT | 0.20.0 |
| bitsandbytes | 0.50.2 |
| Accelerate | 1.14.0 |
| Datasets | 4.3.0 |
| Triton for Windows | 3.7.1.post27 |

The environment is native Windows. CUDA matrix multiplication, all required library imports, 4-bit model loading, and generation succeeded.

Current Unsloth constrains PyTorch to versions below 2.12. The default PyPI Windows build selected by pip was CPU-only, so `requirements-ml.txt` explicitly selects `torch==2.11.0+cu130` and `torchvision==0.26.0+cu130` from the official PyTorch CUDA 13.0 index. This is why reproducing the tested environment from the requirements file matters.

## Model and configuration

- Model: `unsloth/Qwen3-4B-bnb-4bit`
- Downloaded checkpoint size: approximately 2.67 GB
- Quantization: bitsandbytes 4-bit
- Context limit used by this project: 2,048 tokens
- Batch size: 1 prompt at a time
- Thinking mode: disabled for faster smoke tests
- Maximum generated tokens: 256
- Sampling: temperature 0.7, top-p 0.8, top-k 20
- Seed: 3407

The project uses the Unsloth-provided 4-bit checkpoint rather than downloading the much larger full-precision model and quantizing it locally. Only Qwen was downloaded; Phi was not downloaded.

After the first download, `scripts/run_qwen_inference.py --offline` prevents Hugging Face Hub requests and loads only from the ignored project cache.

## Measured result

The final cached run produced the following measurements:

| Measurement | Result |
| --- | ---: |
| Cached load time | 8.218 seconds |
| First load including download | about 391.5 seconds |
| Peak process RAM | 3,720.1 MiB |
| Peak total system RAM used | 12,628.9 MiB |
| Peak total device VRAM used | 3,817.5 MiB |
| Peak PyTorch allocated VRAM | 2,691.1 MiB |
| Peak PyTorch reserved VRAM | 2,752.0 MiB |
| Aggregate generation throughput | about 10.8 tokens/second |

`peak total device VRAM used` includes memory already used by Windows and desktop applications. PyTorch allocated/reserved figures isolate memory managed by this inference process more closely.

The machine retained more than 2 GB of VRAM headroom during this smoke test. Training will require additional memory for adapters, gradients, optimizer state, and activations, so this does not by itself prove that every Stage 6 training configuration will fit.

## What the smoke prompts revealed

The model handled the qualitative RSI and company-margin prompts reasonably and included caveats. However, it showed two serious baseline failures:

1. It fabricated a current BTC funding rate of `0.0003` despite a system instruction explicitly stating that no live data was available.
2. It wrote the correct position-sizing formula but evaluated `$100 / $5` as `200` instead of `20` units.

These failures are preserved in `results/benchmarks/stage2_qwen3_4b.json`. They demonstrate why a repeatable benchmark is necessary and why fluent wording must not be mistaken for correct financial reasoning. Stage 3 will compare models more systematically; Stage 4 will later create the frozen evaluation set.

## Concepts encountered

### Quantization and 4-bit loading

A normal model stores each weight with many bits, often 16 or 32. Four-bit quantization stores most weights using only four bits, reducing the memory needed for a 4-billion-parameter model enough to fit this GPU. Some auxiliary values remain at higher precision, so actual memory use is greater than the simple parameter-count calculation.

Quantization saves memory but can slightly change model quality. It also does not make every part of inference four-bit: activations, the KV cache, and some computations use higher precision.

### Unsloth

Unsloth patches model loading and computation paths to reduce memory use and improve speed. Here it loads the pre-quantized Qwen checkpoint and prepares it for efficient inference. Later, it will also help make QLoRA training practical on limited VRAM.

### Tokenization

The model does not read text directly. The tokenizer turns the system message and user prompt into integer token IDs. The smoke prompts used roughly 107–138 input tokens, and generated answers used 88–172 output tokens.

### Context length

Context length is the maximum token budget shared by the prompt, conversation history, and generated response. Qwen supports much longer contexts, but this project deliberately starts at 2,048 tokens because attention activations and the KV cache consume more memory as context grows.

### Sampling and seed

Sampling selects among plausible next tokens rather than always choosing the single most likely token. Temperature, top-p, and top-k control how broad that choice is. The seed makes repeated tests more reproducible, although GPU kernels and library changes can still introduce small differences.

QLoRA, LoRA adapters, gradient accumulation, epochs, learning rate, and training/validation loss have not been used yet. They will be explained when the project reaches training.
