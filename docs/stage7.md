# Stage 7: base versus fine-tuned evaluation

## Outcome

The Stage 6 adapter **regressed overall** and should not be promoted or exported as the finished FinPulse model.

Both models answered the same 160 frozen cases with the same Qwen3-4B base revision, system prompt, tokenizer template, 2,048-token context, 192-token response ceiling, greedy decoding, and deterministic scoring code. The benchmark SHA-256 remained `bfd1b847d2042f6a59f8a8a5f0dfe0826729dc68a0d390a356c2f1fd3b1781fa`.

| Model | Score | Checks passed |
| --- | ---: | ---: |
| Base Qwen3-4B | 90.56% | 355 / 392 |
| Stage 6 adapter | 84.44% | 331 / 392 |
| Change | -6.12 points | -24 checks |

Eight cases improved, 31 regressed, and 121 were unchanged. The result is not close enough to call mixed: the adapter is worse on the primary benchmark.

## Category results

| Category | Base | Adapter | Change |
| --- | ---: | ---: | ---: |
| Contradictory signals | 93.75% | 90.62% | -3.13 |
| Crypto derivatives | 90.62% | 78.12% | -12.50 |
| Financial calculations | 75.00% | 50.00% | -25.00 |
| Hallucination traps | 87.50% | 100.00% | +12.50 |
| Macroeconomics | 95.31% | 89.06% | -6.25 |
| Risk management | 96.88% | 89.06% | -7.82 |
| Scenario analysis | 89.06% | 85.94% | -3.12 |
| Stock fundamentals | 96.88% | 85.94% | -10.94 |
| Structured output | 87.50% | 81.25% | -6.25 |
| Technical analysis | 95.31% | 70.31% | -25.00 |

The strongest result is important: all 32 hallucination-trap checks passed. The adapter stopped supplying stale CPI, unemployment, BTC open-interest, and similar values after admitting it lacked live access. This suggests the Stage 5 behavioral examples successfully reinforced refusal behavior.

That safety improvement does not outweigh the regressions. Financial calculations fell to 50%, factual finance multiple-choice checks fell from 100% to 82.5%, and strict structured-output validity fell from 87.5% to 81.25%.

## Failure analysis

The adapter learned an explanatory response style from the seed but weakened concise instruction compliance. Its mean answer grew from 129.9 to 134.8 tokens. Twenty-four of 31 regressed cases reached the shared 192-token ceiling, often before emitting the required `FINAL:` value or all expected evidence. One strict JSON answer wrapped a correct object in Markdown fences.

This is not only a truncation problem. Some answers also contained incorrect finance mechanics or arithmetic. Increasing `max_new_tokens` after seeing the result would make the comparison unfair and would not fix those reasoning errors.

The likely causes are:

- Only 33 training examples were available, far below the planned 8,000–15,000.
- The seed emphasized balanced explanation but contained too little exact-answer, calculation, multiple-choice, and strict-JSON practice.
- A 33-example run provides very few optimizer updates and can shift style without building broad domain competence.
- Training loss measured imitation of the seed, not success on unseen evaluation tasks.

## General-capability sentinel

The three existing Stage 3 general-reasoning prompts were rerun greedily on both models. Both scored 2/3: they passed multiplication and probability and made the same syllogism error. This shows no change on that tiny sentinel, but three prompts are not enough for a broad general-capability claim.

The frozen financial benchmark also includes a narrow arithmetic and exact-instruction regression proxy. That proxy fell from 81.25% to 65.62%, so regression risk remains clear even though the three general prompts tied.

## Runtime

| Measurement | Base | Adapter |
| --- | ---: | ---: |
| Generated tokens | 20,785 | 21,561 |
| Aggregate tokens/second | 10.70 | 6.16 |
| Peak total device VRAM | 3,809.5 MiB | 3,913.5 MiB |
| Peak process RAM | 3,740.3 MiB | 3,721.0 MiB |

The adapter still fits the 6 GB GPU comfortably. Its slower generation is a measured property of this local PEFT/Unsloth run, not a memory failure.

## Reproduce

Run a small adapter smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage7_adapter.py --limit 5 --output results\benchmarks\stage7_qwen3_4b_adapter_smoke.json
```

Run or resume all frozen cases:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage7_adapter.py --resume
```

Run the secondary general-reasoning sentinel and rebuild comparison reports:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage7_general_regression.py
.\.venv\Scripts\python.exe scripts\compare_stage7_models.py
```

The detailed responses and scores are stored in `results/benchmarks/stage7_qwen3_4b_adapter.json`. The concise comparison is available as both `stage7_comparison.json` and `stage7_comparison.md`.

## What this teaches

Evaluation is the step that tests whether lower training loss became useful behavior on unseen examples. Here it did not: training loss fell, but the held-out benchmark score also fell.

Overfitting is broader than memorizing exact sentences. A tiny dataset can over-specialize response style—such as always explaining at length—while weakening arithmetic, exact formatting, or unrelated knowledge.

A single aggregate score can hide important tradeoffs. The adapter became substantially safer around live-data hallucinations while getting worse overall. Good model development preserves that safety gain in a larger, more balanced dataset rather than declaring the entire fine-tune successful.

Before another training run, the dataset should grow substantially with independently authored examples for concise multiple-choice endings, financial calculations, strict JSON, short answers, and varied response lengths. The frozen evaluation questions and paraphrases must remain excluded.

