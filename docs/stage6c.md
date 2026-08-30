# Stage 6C: gentle QLoRA configuration and CPU preflight

## Outcome

Stage 6C configuration, CPU-only preflight, one-step GPU smoke test, and the explicitly authorized full training run are complete. The run trained one rank-8 QLoRA adapter on the locked 900-row Stage 5D view and evaluated loss on the unchanged 450-row Stage 5C validation split.

No Stage 5C development evaluation, frozen benchmark evaluation, merge, or export was performed. The adapter remains an unevaluated candidate rather than a release model.

## Corrective configuration

| Setting | Stage 6B | Stage 6C preflight | Reason |
| --- | ---: | ---: | --- |
| Training examples | 398 | 900 | Broader balanced data |
| Validation examples | 51 | 450 | External family-disjoint holdout |
| LoRA rank / alpha | 16 / 32 | 8 / 16 | Halve adapter capacity |
| Learning rate | `1e-4` | `5e-5` | Reduce update magnitude |
| Micro-batch | 1 | 1 | Preserve the proven 6 GB envelope |
| Gradient accumulation | 4 | 16 | Stabilize gradients and reduce update count |
| Effective batch | 4 | 16 | 57 updates over all 900 examples |
| Epochs | 1 | 1 | Avoid repeated exposure |
| Context | 512 | 512 | Required by the locked data; no configured truncation |

The Stage 6B adapter trained approximately 33.0 million LoRA parameters. Rank 8 should reduce that to roughly 16.5 million, but the exact count remains unmeasured until a model is loaded during an explicitly authorized GPU smoke test.

All seven attention and MLP projection targets remain enabled. This retains coverage while rank reduction constrains capacity more evenly than removing entire module families.

## Checkpoint and evaluation policy

The expected 57 optimizer updates save checkpoints every 15 steps, retaining at most four. This should preserve candidates near steps 15, 30, and 45 plus trainer-managed final state without unbounded disk growth. Intermediate checkpoints can later be tested on the locked development screen rather than choosing solely from final training loss.

Validation runs once at the end of the epoch. Running the 450-example validation set every 15 steps would add substantial compute without improving the preflight decision.

Neither validation nor checkpoint selection may use the frozen Stage 4 benchmark. That benchmark remains a final comparison only after a candidate passes the development gate.

## Preflight result

- Status: `preflight_passed`
- Train / validation examples: 900 / 450
- Estimated optimizer updates: 57
- Effective batch size: 16
- Training SHA-256: `b7fc3917f53e230bf7ec704b12afb7a9683703d88380b5e9c789568afa89a6d4`
- Validation SHA-256: `31d17917a44f6ed7f9f321e639d8c492468735c913f0b4f3bc1639a883b913ea`
- Frozen benchmark SHA-256: `bfd1b847d2042f6a59f8a8a5f0dfe0826729dc68a0d390a356c2f1fd3b1781fa`

## GPU smoke result

The smoke test loaded the pinned 4-bit Qwen base, attached all seven LoRA target families, tokenized two training examples and one validation example, applied response-only masking, completed one forward/backward/optimizer cycle, and saved a separate smoke adapter.

| Measurement | Result |
| --- | ---: |
| Trainable parameters | 16,515,072 (0.4089%) |
| Smoke train / validation examples | 2 / 1 |
| Train token range | 200–320 |
| One-step trainer runtime | 20.81 seconds |
| Measured wall time | 34.49 seconds |
| Peak total device VRAM used | 4,085.5 MiB of 6,143.5 MiB |
| Peak PyTorch allocated / reserved | 2,936.2 / 3,020.0 MiB |
| Peak process RAM | 3,798.5 MiB |
| Adapter tensors | 504 float32 rank-8 matrices |
| Adapter size | 66,126,768 bytes |
| Adapter SHA-256 | `4e5f714e204a1502de2aba4096188ac34ef32ded4e059f1dfb4b4cb242070c27` |

Peak total device use is approximately 1,032 MiB below the measured Stage 6B peak. This provides comfortable smoke-test headroom, although the full run can still encounter a longer 511-token example and must retain the same micro-batch and context safeguards.

The single smoke step logged learning rate zero because the configured five-step warmup begins at zero. The smoke therefore validates memory allocation, masking, backward propagation, optimizer plumbing, and serialization—not learning quality or useful parameter movement. The full configured run reaches non-zero learning rates during warmup.

## Full training result

The full run completed all 57 expected optimizer updates, saved checkpoints at steps 15, 30, 45, and 57, evaluated validation loss, and serialized the final adapter.

| Measurement | Result |
| --- | ---: |
| Train / validation examples | 900 / 450 |
| Train token min / mean / max | 107 / 248.05 / 494 |
| Validation token maximum | 496 |
| Optimizer updates | 57 |
| Trainable parameters | 16,515,072 (0.4089%) |
| Logged loss, first → near-final | 3.275 → 1.103 |
| Mean training loss | 1.6623 |
| Final validation loss | 0.8718 |
| Trainer runtime, including epoch validation | 1,519.74 seconds |
| Measured wall time | 1,803.22 seconds |
| Peak total device VRAM used | 5,273.9 MiB of 6,143.5 MiB |
| Peak PyTorch allocated / reserved | 3,340.5 / 3,920.0 MiB |
| Peak process RAM | 3,791.7 MiB |
| Final adapter size | 33,096,960 bytes |
| Final adapter SHA-256 | `8ada68682d359273a1090f47c14acd4138156cf6b8116ab3251956b43ee6f97f` |

The declining loss and stable gradient norms show that the adapter learned the training objective without numerical instability. The lower validation loss is encouraging for the held-out external calculation/template families, but it is not evidence of general financial improvement. Both earlier rejected adapters also achieved low loss while regressing behaviorally.

The measured wall time includes an unnecessary second traversal of the validation split: `eval_strategy = "epoch"` evaluated at epoch end, then the runner called `evaluate()` again for its report. Both passes produced the same `0.8718` loss. The runner now reuses epoch-end metrics, preventing this duplicate GPU work in future experiments; the saved Stage 6C report preserves the measurements from the run as executed.

## Files

- `configs/training/qwen3_4b_stage6c.toml`: proposed GPU experiment
- `results/training/stage6c_preflight.json`: saved CPU-only result
- `results/training/stage6c_qwen3_4b_stage5d_v1_smoke.json`: local Git-ignored smoke metrics
- `models/adapters/finpulse-qwen3-4b-stage5d-v1-smoke/`: local Git-ignored smoke adapter
- `results/training/stage6c_qwen3_4b_stage5d_v1.json`: full measured training report
- `models/adapters/finpulse-qwen3-4b-stage5d-v1/`: local Git-ignored full adapter
- `models/checkpoints/stage6c-qwen3-4b-stage5d-v1/`: four local Git-ignored recovery checkpoints
- `scripts/train_qlora.py`: optional preflight-report output
- `src/finpulse_llm/training/config.py`: checkpoint/evaluation configuration validation
- `src/finpulse_llm/training/runner.py`: input counts, update estimate, and bounded checkpoints

Reproduce the CPU-only preflight:

```powershell
.\.venv\Scripts\python.exe scripts\train_qlora.py --config configs\training\qwen3_4b_stage6c.toml --preflight-only --preflight-output results\training\stage6c_preflight.json
```

## Remaining risk and next boundary

The full Stage 6C run passed its memory and training-mechanics gates. The next explicit stage is Stage 7C development evaluation. It should compare the base model, intermediate checkpoints, and final adapter on the untouched Stage 5C development split before selecting at most one candidate for the expensive frozen benchmark. No candidate should be exported or promoted based on training/validation loss alone.
