# Stage 5D: balanced training view

## Outcome

Stage 5D created a deterministic 900-example training view from the locked Stage 5C training pool. It reduces the effective calculation share from 90.3% in the Stage 5C training split to 58.3% without duplicating or synthesizing additional rows. No model weights were loaded, and no training or evaluation was run.

The Stage 5C validation and development files remain unchanged. They preserve external Cosimo generator-family and FinQA report-document isolation and are locked by the Stage 5D manifest.

## Sampling policy

| Source/task | Selected | Policy |
| --- | ---: | --- |
| Project behavior | 300 | Retain every scarce behavior row |
| Cosimo calculations | 300 | 50 from each of six available financial categories |
| Cosimo multiple choice | 100 | Deterministic sample from 102 available rows |
| FinQA calculations | 200 | Retain all 60 non-stock rows plus 140 stock-fundamental rows |

The resulting task distribution is:

| Task type | Count | Share |
| --- | ---: | ---: |
| Calculation | 525 | 58.3% |
| Multiple choice | 125 | 13.9% |
| Analysis | 100 | 11.1% |
| Factual | 75 | 8.3% |
| Instruction following | 50 | 5.6% |
| Refusal | 25 | 2.8% |

All 75 JSON-only examples, all 50 long-response examples, and all 25 refusal examples from the Stage 5C training pool are retained. The view contains 650 final-marker, 175 plain, and 75 JSON-only outputs.

## Why this is safer

Training directly on all 3,900 Stage 5C training rows would make calculations 90.3% of updates and reduce the scarce project behaviors to 7.7%. The 900-row view makes project behavior one third of updates while still adding 600 diverse external examples. It also avoids literal oversampling, which would repeatedly expose the model to identical behavior answers and increase memorization risk.

This balance does not itself authorize training. Stage 6C still needs a conservative QLoRA configuration and CPU-only preflight. Given both earlier adapter regressions, the next configuration should use fewer trainable parameters and gentler optimization than Stage 6B, save frequent checkpoints, and select candidates using only the locked development data—not the frozen benchmark.

## Files and reproduction

- `configs/data/stage5d_sampling.toml`: parent lock, deterministic seed, and selection quotas
- `scripts/build_stage5d_training_view.py`: parent verification and balanced selection builder
- `data/train/finpulse_stage5d_v1.jsonl`: 900-example training view
- `data/processed/finpulse_stage5d_v1.quality.json`: measured distribution
- `data/processed/finpulse_stage5d_v1.manifest.json`: training/holdout/frozen hashes

Rebuild without loading a model:

```powershell
.\.venv\Scripts\python.exe scripts\build_stage5d_training_view.py
```

## Next boundary

Stage 5D stopped before training. Stage 6C subsequently completed a gentler rank-8 configuration and CPU-only preflight. A one-step GPU smoke test and the full training run remain separate explicit decisions; see `docs/stage6c.md`.
