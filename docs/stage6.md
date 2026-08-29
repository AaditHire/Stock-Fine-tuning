# Stage 6: local QLoRA fine-tuning

Stage 6 completed the first local adapter fine-tune of the selected Qwen3-4B base model. The run used Unsloth, TRL, PEFT, bitsandbytes 4-bit loading, and the reviewed Stage 5 seed data. It did not merge, export, upload, or evaluate the adapter on the frozen Stage 4 benchmark.

## Why this configuration fits 6 GB VRAM

The longest reviewed example is 236 tokens, so a 512-token training context preserves every example without paying the activation-memory cost of 2,048 tokens. Micro-batch size 1 is the main VRAM safeguard. Four micro-batches are accumulated before each optimizer update, producing an effective batch size of 4 and nine updates across the 33-example epoch.

LoRA rank 16 and alpha 32 follow the common `alpha = 2 × rank` heuristic. Adapters are attached to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`, covering the major attention and MLP linear layers. Unsloth gradient checkpointing trades extra recomputation for lower activation memory. The GPU reports bfloat16 support, so adapter computations use bfloat16 while the frozen base remains in 4-bit storage.

## Reproduce the run

Verify configuration and immutable input hashes without loading the model:

```powershell
.\.venv\Scripts\python.exe scripts\train_qlora.py --preflight-only
```

Run a one-step GPU smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\train_qlora.py --smoke-test
```

Run the configured seed experiment:

```powershell
.\.venv\Scripts\python.exe scripts\train_qlora.py
```

The script operates offline against the Stage 2 model cache. It verifies the Stage 5 split hashes before loading CUDA, refuses silent sequence truncation, checks every configured target module against the loaded architecture, trains only on assistant-response tokens, measures RAM/VRAM peaks, and saves only the adapter plus tokenizer metadata.

## Measured result

| Measurement | Result |
| --- | ---: |
| Train / validation examples | 33 / 7 |
| Epochs / optimizer steps | 1 / 9 |
| Trainable parameters | 33,030,144 (0.814% of logical parameters) |
| Logged loss, first → final step | 5.313 → 2.581 |
| Mean training loss | 3.674 |
| Final validation loss | 2.888 |
| Trainer runtime | 25.93 seconds |
| Peak total device VRAM used | 5,114.4 MiB of 6,144 MiB |
| Peak PyTorch allocated / reserved | 3,121.2 / 3,602.0 MiB |
| Peak process RAM | 3,680.7 MiB |
| Adapter weights | 66,127,776 bytes |

The falling training loss shows that optimizer updates changed the adapter in the expected direction. Validation loss is lower than the mean training loss partly because the training mean includes the much higher early steps; it is not proof of generalization. With only 40 total examples, this run validates the mechanics and hardware profile, not meaningful financial specialization. Stage 7 must use the untouched frozen benchmark before any quality claim is made.

## Concepts encountered

- **4-bit quantization** stores the frozen base weights in a compact form. This is what makes a 4B model practical on a 6 GB GPU.
- **LoRA** adds small trainable low-rank matrices to selected model layers instead of changing all base weights.
- **QLoRA** combines a quantized frozen base with trainable LoRA matrices. Only 0.814% of the logical parameters were trained here.
- **Unsloth** patches model and trainer operations to reduce memory and improve training speed. Its gradient checkpointing recomputes some activations rather than storing all of them.
- **An adapter** is the learned LoRA delta. It is about 66 MB and still requires the exact pinned base model when loaded.
- **Tokenization** converts each rendered chat into token IDs. Auditing lengths allowed the project to choose 512 safely without truncation.
- **Context length** is the maximum tokens accepted per training example. Larger values consume more activation memory even when most examples are short.
- **Gradient accumulation** processes several micro-batches before an optimizer update. It provides a larger effective batch without loading those examples simultaneously.
- **An epoch** is one pass through the training split. One epoch limits memorization risk on this tiny seed.
- **Learning rate** controls update size. The run warmed up for one step to `2e-4`, then decayed linearly.
- **Training loss** measures error on examples used for optimization; **validation loss** measures error on held-out examples not used for updates.
- **Overfitting** occurs when training examples are memorized without generalization. The tiny seed makes this a serious risk, which is why no extra epochs were run.
- **Dataset leakage** would make evaluation misleading. The pipeline re-verifies the locked training hashes and records the protected Stage 4 benchmark hash before training.

## Artifacts and version control

The machine-readable report is `results/training/stage6_qwen3_4b_seed_v1.json`. Adapter weights are under `models/adapters/finpulse-qwen3-4b-seed-v1/` and remain ignored by Git. The adapter safetensors file contains 504 LoRA tensors, records rank 16/alpha 32 and all seven targets, and has SHA-256 `b89cb6a4aab35fb66309462fee5359333df9ab601fb67eadae4020350bf9abaa`.

Current guidance consulted for the configuration:

- Unsloth LoRA hyperparameters: <https://docs.unsloth.ai/basics/lora-parameters-encyclopedia>
- TRL SFTTrainer: <https://huggingface.co/docs/trl/en/sft_trainer>
- PEFT LoRA: <https://huggingface.co/docs/peft/package_reference/lora>

