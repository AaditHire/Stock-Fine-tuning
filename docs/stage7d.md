# Stage 7D: selected-checkpoint frozen evaluation

## Outcome

The Stage 7C-selected checkpoint 30 regressed on the unchanged frozen benchmark and is rejected.
It must not be promoted, merged, exported, or advanced to Stage 8.

| Evaluation | Base | Checkpoint 30 | Change |
| --- | ---: | ---: | ---: |
| Frozen benchmark | 90.56% (355/392) | 84.44% (331/392) | -6.12 points |

The saved Stage 4 base report was reused; the base model was not rerun. Checkpoint 30 was the only
adapter loaded. The benchmark remained 160 cases with SHA-256
`bfd1b847d2042f6a59f8a8a5f0dfe0826729dc68a0d390a356c2f1fd3b1781fa`.

## Candidate identity

- Stage 7C selection: `checkpoint-30`
- Adapter weights SHA-256:
  `4e26f978d10f3d34e94f8b8aa0d328924d8ad2244bbbdbc2da92b44d21aacbc3`
- Base model: `unsloth/Qwen3-4B-bnb-4bit`
- Base revision: `cad0bedfdd862093a12af478cb974ab2addd0e0a`
- Rank / alpha: 8 / 16

Trainer checkpoints omit the base revision in their adapter configuration. The frozen runner was
therefore tightened to require an explicit expected adapter SHA-256 whenever revision metadata is
absent. An incorrect SHA was verified to fail before loading model weights.

## Category results

| Category | Base | Checkpoint 30 | Change |
| --- | ---: | ---: | ---: |
| Contradictory signals | 93.75% | 92.19% | -1.56 |
| Crypto derivatives | 90.62% | 68.75% | -21.87 |
| Financial calculations | 75.00% | 62.50% | -12.50 |
| Hallucination traps | 87.50% | **100.00%** | +12.50 |
| Macroeconomics | 95.31% | 81.25% | -14.06 |
| Risk management | 96.88% | 85.94% | -10.94 |
| Scenario analysis | 89.06% | 82.81% | -6.25 |
| Stock fundamentals | 96.88% | 82.81% | -14.07 |
| Structured output | 87.50% | **93.75%** | +6.25 |
| Technical analysis | 95.31% | 65.62% | -29.69 |

Instruction following and hallucination resistance improved, but those gains did not offset broad
regressions in reasoning, calculations, risk awareness, and factual finance knowledge. Ten cases
improved, 32 regressed, and 118 were unchanged. The comparison recommendation is
`reject_adapter`.

## Failure analysis

Checkpoint 30 produced 19,905 output tokens, and 63 of 160 responses reached the shared
192-token ceiling. Twenty-four of the 32 regressed cases reached that ceiling. Many showed
repetitive continuations or failed to emit required final markers or evidence before truncation.
Examples included:

- repeated technical-analysis interpretations without a concise conclusion;
- multiple-choice prompts echoed or discussed until the answer was truncated;
- calculation responses degenerating into repeated digits or delimiters; and
- scenario analyses repeating uncertainty language instead of completing all requested points.

Truncation is therefore a major proximate cause, but not a reason to rerun with a different limit:
the frozen comparison must retain the exact saved-base generation contract. The adapter also
regressed on eight cases that did not hit the ceiling, so verbosity is not the only problem.

## Runtime and resources

- Generation time: 3,288.3 seconds
- Aggregate throughput: 6.05 tokens/second
- Peak total device VRAM: 3,873.5 MiB
- Peak process RAM: 3,782.6 MiB
- Inference batch size: 1

The run remained comfortably within the 6 GB VRAM target. Resource exhaustion did not cause the
quality regression.

## Files

- `results/benchmarks/stage7d_qwen3_4b_checkpoint30.json`: complete adapter responses and scoring
- `results/benchmarks/stage7d_comparison.json`: machine-readable saved-base comparison
- `results/benchmarks/stage7d_comparison.md`: rendered comparison
- `scripts/run_stage7_adapter.py`: explicit SHA lock for revision-less trainer checkpoints
- `src/finpulse_llm/evaluation/stage7.py`: accurate limitation text when no general sentinel runs

## Boundary

Stop before export. Checkpoint 30 is rejected, no Stage 6C candidate is promoted, and Stage 8 has
not started. This stage did not merge, copy, convert, publish, or export model artifacts.
