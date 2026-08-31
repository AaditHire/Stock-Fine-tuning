# Stage 7C: development checkpoint selection

## Outcome

Stage 7C evaluated base Qwen3-4B, checkpoints 15/30/45, and the final Stage 6C adapter on
the untouched 450-case Stage 5C development split. It selected **checkpoint 30** as the sole
candidate eligible for a later frozen-benchmark run.

The frozen 160-case benchmark was not run. No adapter was copied, promoted, merged, exported,
or used for training.

## Locked evaluation contract

- Development SHA-256: `ea941ca169b3b2003c773620dbee33e9f98fc7f585b9d536ebf6abe4b86d8b17`
- Cases: 450 (397 calculation, 53 multiple choice)
- Sources: 300 family-disjoint Cosimo, 150 document-disjoint FinQA
- All cases require exactly one terminal `FINAL:` value.
- Greedy decoding, thinking disabled, 192 generated-token ceiling
- One base model or adapter loaded at a time
- Atomic per-batch checkpoints with ordered-prefix resume validation
- Inference batch size 8 for every canonical report

The primary metric is exact normalized terminal-answer accuracy. Normalization accepts harmless
numeric presentation differences such as thousands separators and trailing decimal zeroes, but
does not accept an embellished or semantically approximate final value. Format accuracy measures
whether the response emits exactly one terminal `FINAL:` marker.

## Predeclared selection policy

A candidate must strictly improve overall answer accuracy over base while not regressing format
accuracy, calculation accuracy, multiple-choice accuracy, Cosimo accuracy, or FinQA accuracy.
Eligible candidates rank by answer accuracy, then format accuracy, then earlier checkpoint.
If none pass, Stage 7C selects no candidate.

## Measured result

| Model | Answer | Format | Calculation | Multiple choice | Cosimo | FinQA | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Base | 9.78% | 49.33% | 9.07% | 15.09% | 10.00% | 9.33% | reference |
| Checkpoint 15 | 3.56% | 34.00% | 3.27% | 5.66% | 1.00% | 8.67% | fail |
| Checkpoint 30 | **18.67%** | 68.89% | **18.89%** | 16.98% | **23.00%** | 10.00% | pass, selected |
| Checkpoint 45 | 18.44% | 76.89% | 17.88% | **22.64%** | 22.33% | 10.67% | pass |
| Final adapter | 14.67% | **81.56%** | 13.60% | **22.64%** | 16.33% | **11.33%** | pass |

Checkpoint 30 doubled exact answer accuracy relative to base (+8.89 percentage points) and had
the highest primary score. Later candidates improved format compliance more, but their exact
answer accuracy was lower. Checkpoint 15 regressed on every gate dimension, showing again that an
earlier loss snapshot is not automatically safer.

The absolute scores are intentionally strict and should not be read as broad financial quality
scores. They measure exact transfer to family/document-disjoint output contracts. Stage 7C is a
candidate-selection screen, not a release claim.

## Identity and resources

- Selected checkpoint weights SHA-256:
  `4e26f978d10f3d34e94f8b8aa0d328924d8ad2244bbbdbc2da92b44d21aacbc3`
- Final adapter weights SHA-256:
  `8ada68682d359273a1090f47c14acd4138156cf6b8116ab3251956b43ee6f97f`
- Maximum measured total device VRAM across canonical reports: 5,939.6 MiB
- Maximum measured process RAM across canonical reports: 3,793.3 MiB

Batch 8 completed without OOM but left only about 204 MiB of measured device headroom on the
final adapter's longest batches. The reports are complete and should not be rerun merely to
change throughput. If a future explicit rerun is needed under a different desktop GPU baseline,
use `--batch-size 4` and regenerate every compared report with the same batch size.

## Files and reproduction

- `configs/models/qwen3_4b_stage7c_dev.toml`: locked generation configuration
- `src/finpulse_llm/evaluation/stage7c.py`: scoring, gate, selection, and Markdown rendering
- `scripts/run_stage7c_development.py`: one-model-at-a-time resumable runner
- `scripts/compare_stage7c_development.py`: CPU-only comparison and selection
- `results/benchmarks/stage7c_dev_*.json`: complete base/candidate reports
- `results/benchmarks/stage7c_dev_comparison.json`: machine-readable selection
- `results/benchmarks/stage7c_dev_comparison.md`: concise comparison

Rebuild only the comparison from saved reports, without loading a model:

```powershell
.\.venv\Scripts\python.exe scripts\compare_stage7c_development.py
```

## Next boundary

Stop before the frozen benchmark. Checkpoint 30 is the only Stage 7C-selected candidate, but it
is not promoted or approved for export. A later explicitly authorized stage may evaluate this
single checkpoint on the unchanged frozen benchmark, reusing the saved Stage 4 base report.
