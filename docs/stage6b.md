# Stage 6B: corrective QLoRA fine-tuning

## Outcome

Stage 6B completed one local QLoRA epoch using the locked Stage 5B train and validation splits. It created a new adapter and left the rejected Stage 6 seed adapter untouched. It did not train on the 51-example development holdout or frozen benchmark, merge weights, export a runtime model, or begin Stage 7B evaluation.

## Configuration

The run retained the proven 6 GB memory envelope: 4-bit Qwen3-4B, rank-16 LoRA with alpha 32, all seven attention and MLP projection targets, 512-token context, micro-batch 1, gradient accumulation 4, response-only loss, bfloat16 adapter computation, and Unsloth gradient checkpointing.

The learning rate was reduced from `2e-4` to `1e-4` because Stage 6B performs 100 optimizer steps rather than the seed run's nine. Five steps warm up the learning rate, followed by linear decay. Training remains limited to one epoch.

The longest training conversation is 258 tokens and the longest validation conversation is 254 tokens, so no sequence is truncated at the 512-token context.

## Measured run

| Measurement | Result |
| --- | ---: |
| Train / validation examples | 398 / 51 |
| Epochs / optimizer steps | 1 / 100 |
| Trainable parameters | 33,030,144 (0.814%) |
| Logged loss, first → final step | 4.430 → 0.132 |
| Mean training loss | 0.754 |
| Final validation loss | 0.191 |
| Trainer runtime | 348.84 seconds |
| Wall-clock measured section | 377.99 seconds |
| Peak total device VRAM used | 5,117.4 MiB of 6,144 MiB |
| Peak PyTorch allocated / reserved | 3,147.6 / 3,822.0 MiB |
| Peak process RAM | 3,799.6 MiB |
| Adapter weights | 66,127,776 bytes |

Loss fell quickly because many Stage 5B examples are structured exact-format and numeric drills. The low validation loss confirms that the adapter learned similar held-out rows, but it is not evidence of broader financial improvement because validation shares template families with training. Stage 7B must test behavior on the development screen and unchanged frozen benchmark before any promotion decision.

## Artifacts

- Configuration: `configs/training/qwen3_4b_stage6b.toml`
- Report: `results/training/stage6b_qwen3_4b_stage5b_v1.json`
- Git-ignored adapter: `models/adapters/finpulse-qwen3-4b-stage5b-v1/`
- Adapter SHA-256: `4dbfab3baa3fe052b95f8334e7b3657fcf253d2a3947af9f83e18da48e289a56`

Reproduce the locked preflight and run:

```powershell
.\.venv\Scripts\python.exe scripts\train_qlora.py --config configs\training\qwen3_4b_stage6b.toml --preflight-only
.\.venv\Scripts\python.exe scripts\train_qlora.py --config configs\training\qwen3_4b_stage6b.toml
```

The redundant one-step smoke run was intentionally skipped because the same model, LoRA targets, trainer path, context, and memory controls had already completed successfully in Stage 6. Stage 6B's token audit showed a smaller maximum than the configured context.

## Next boundary

Stage 7B should first build deterministic scoring for the Stage 5B development holdout, then compare the new adapter and base model fairly. Only a candidate that passes that screen should consume another complete run of the frozen 160-case benchmark. Stage 7B has not started.
