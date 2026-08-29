# Stage 7B: corrective adapter evaluation

## Outcome

The Stage 6B corrective adapter **regressed severely** on the unchanged frozen benchmark and
must not be promoted, exported, or advanced to Stage 8.

The 51-case development screen passed before the frozen benchmark was opened. That screen used
the Stage 5B system prompt, greedy decoding, a 192-token response ceiling, and task-aware scoring.
The adapter improved from 75.00% to 91.13% on development and passed all seven predefined gate
checks. The frozen result did not confirm that improvement.

| Evaluation | Base | Adapter | Change |
| --- | ---: | ---: | ---: |
| Stage 5B development | 75.00% | 91.13% | +16.13 points |
| Frozen benchmark | 90.56% (355/392) | 70.92% (278/392) | -19.64 points |

The frozen benchmark remained 160 cases with SHA-256
`bfd1b847d2042f6a59f8a8a5f0dfe0826729dc68a0d390a356c2f1fd3b1781fa`.
The saved Stage 4 base report was reused; the base model was not rerun.

## Frozen category results

| Category | Base | Adapter | Change |
| --- | ---: | ---: | ---: |
| Contradictory signals | 93.75% | 70.31% | -23.44 |
| Crypto derivatives | 90.62% | 84.38% | -6.24 |
| Financial calculations | 75.00% | 18.75% | -56.25 |
| Hallucination traps | 87.50% | 100.00% | +12.50 |
| Macroeconomics | 95.31% | 84.38% | -10.93 |
| Risk management | 96.88% | 85.94% | -10.94 |
| Scenario analysis | 89.06% | 57.81% | -31.25 |
| Stock fundamentals | 96.88% | 82.81% | -14.07 |
| Structured output | 87.50% | 81.25% | -6.25 |
| Technical analysis | 95.31% | 81.25% | -14.06 |

Seven cases improved, 57 regressed, and 96 were unchanged. Hallucination resistance again
reached 100%, and factual finance multiple-choice checks remained at 100%, but neither offsets
the broad loss in reasoning, calculations, risk awareness, and instruction following. The
three-case general-reasoning sentinel remained unchanged at 2/3.

## Failure analysis

This failure is different from Stage 7. The adapter averaged only 49.7 output tokens and just 2
of 57 regressed cases reached the 192-token ceiling, so truncation is not the primary cause.
Representative calculation failures included:

- ignoring the account-risk percentage during position sizing;
- reversing the reward sign for a profitable short trade;
- producing incorrect valuation arithmetic; and
- emitting variants such as `FINAL, 1000` when an exact `FINAL: 50` marker was required.

The Stage 5B calculation targets use the correct formulas. The model therefore failed to transfer
those patterns reliably to independently worded frozen prompts rather than merely imitating bad
labels.

The development gate was too optimistic because its 51 rows were row-disjoint but not
template-family-disjoint from training. It measured interpolation across familiar generated
templates. The frozen benchmark measured broader phrasing and task transfer, exposing the gap.
This limitation was documented before the gate was run, and Stage 7B confirms it is material.

## Runtime and identity

- Adapter weights SHA-256:
  `4dbfab3baa3fe052b95f8334e7b3657fcf253d2a3947af9f83e18da48e289a56`
- Frozen generation: 7,957 output tokens in 1,310.6 seconds
- Aggregate throughput: 6.07 tokens/second
- Peak total device VRAM: 3,961.5 MiB
- Peak process RAM: 3,689.7 MiB

The adapter remained comfortably within the 6 GB VRAM limit. Resource use was not the cause of
the regression.

## Reproduce from saved artifacts

Run or resume the adapter-only development screen:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage7b_development.py --model adapter --resume
```

Rebuild the development comparison:

```powershell
.\.venv\Scripts\python.exe scripts\compare_stage7b_development.py
```

Run or resume the frozen adapter evaluation without rerunning the base:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage7_adapter.py `
  --adapter-dir models\adapters\finpulse-qwen3-4b-stage5b-v1 `
  --output results\benchmarks\stage7b_qwen3_4b_adapter.json `
  --resume
```

The detailed comparison is in `results/benchmarks/stage7b_comparison.md` and its machine-readable
equivalent. Model weights remain Git-ignored.

## Recommendation

Reject this adapter and stop before Stage 8. A future corrective iteration should first replace
the template-sharing development split with template-family-disjoint validation and development
sets. It should also test a smaller, more diverse corpus or a less aggressive optimization setup,
with independently phrased calculation and exact-format holdouts. No further training should
begin until that next stage is explicitly requested.
