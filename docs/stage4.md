# Stage 4: frozen FinPulse evaluation benchmark

## Outcome

The frozen `finpulse_eval_v1` benchmark contains 160 questions: 16 in each of the 10 required categories. The selected Qwen3-4B base model scored **90.56%**, passing 355 of 392 deterministic rubric checks.

This is the pre-training baseline. Stage 7 must use the same dataset hash, system prompt, generation settings, and scoring implementation when comparing the base model with a fine-tuned adapter.

## Frozen-data safeguards

- Dataset: `data/eval/finpulse_eval_v1.jsonl`
- Manifest: `data/eval/finpulse_eval_v1.manifest.json`
- SHA-256: `bfd1b847d2042f6a59f8a8a5f0dfe0826729dc68a0d390a356c2f1fd3b1781fa`
- Split marker on every case: `eval`
- Training exclusion marker on every case: `exclude_from_training: true`
- Provenance: project-original or synthetic; no scraped or copied source text
- Per-prompt normalized SHA-256 fingerprints are stored for later leakage checks
- Exact Stage 3 prompt overlap: zero

The benchmark builder now verifies without writing by default. Rewriting requires the explicit `--write` flag. The questions, expected answers, rubric checks, and close paraphrases must never be included in Stage 5 training data.

## Coverage

| Category | Cases | Base score |
| --- | ---: | ---: |
| Contradictory signals | 16 | 93.75% |
| Crypto derivatives | 16 | 90.62% |
| Financial calculations | 16 | 75.00% |
| Hallucination traps | 16 | 87.50% |
| Macroeconomics | 16 | 95.31% |
| Risk management | 16 | 96.88% |
| Scenario analysis | 16 | 89.06% |
| Stock fundamentals | 16 | 96.88% |
| Structured output | 16 | 87.50% |
| Technical analysis | 16 | 95.31% |

The benchmark mixes multiple-choice finance knowledge, exact calculations, qualitative evidence rubrics, conflicting-signal analysis, live-data refusal traps, and strict JSON-only output.

## Baseline runtime

| Measure | Result |
| --- | ---: |
| Model | `unsloth/Qwen3-4B-bnb-4bit` |
| Quantization | bitsandbytes 4-bit |
| Context limit | 2,048 tokens |
| Maximum response | 192 tokens |
| Decoding | greedy (`do_sample = false`) |
| Generated tokens | 20,785 |
| Measured generation time | 32.4 minutes |
| Aggregate throughput | 10.69 tokens/second |
| Peak total device VRAM | 3,809.5 MiB |
| Peak PyTorch allocation | 2,705.3 MiB |
| Peak process RAM | 3,740.3 MiB |

Only one model was loaded. A five-case smoke test ran first, then the full benchmark used per-answer atomic checkpoints. The long run did not encounter an out-of-memory error.

## Important baseline failures

The overall number is not the whole result. Seven cases scored zero:

- Four calculation cases: position sizing, output-format compliance for EPS, futures-basis sign, and break-even win-rate calculation
- Two strict JSON calculations: share count and P/E
- One hallucination trap: Qwen supplied stale CPI values instead of refusing

The model also supplied a stale unemployment figure after acknowledging it lacked current access. This received partial credit because the refusal check passed but the no-fabricated-value check failed.

These are genuine baseline weaknesses to measure later. The exact benchmark questions must not be inserted into training data to “teach to the test.” Training examples may cover the underlying skills using independently authored scenarios.

## Audit before freezing

The first full run was treated as a benchmark audit, not blindly accepted. Two rubric defects were found:

1. A calculation instruction used a literal `<answer>` placeholder, which encouraged tagged outputs.
2. Four hallucination checks mistook harmless numbers such as “12 months,” “S&P 500,” or numbered steps for fabricated market values.

The wording and rules were corrected before the final hash was declared. Only the 16 changed calculation prompts were rerun with identical model settings; the 144 prompt-identical responses were preserved and verified during the merge. Percentage questions accept a correct numeric percentage with or without the `%` symbol because the final instruction requests a numeric answer.

No further benchmark edits should be made after this point. A future version would require a new benchmark ID and a separate baseline.

## What this teaches

A frozen evaluation set is a sealed exam. Freezing the bytes and recording their hash makes later edits detectable. This matters because changing questions or scoring after fine-tuning would make before/after results incomparable.

Dataset leakage occurs when evaluation questions, expected answers, or close paraphrases enter training data. Leakage can create impressive scores without teaching general financial analysis. The prompt fingerprints provide a first exact-match defense; Stage 5 will need additional normalization and similarity checks for paraphrases.

Greedy decoding disables random sampling, making repeated comparisons more reproducible. The same prompt can still behave differently after fine-tuning, but variation from sampling is removed as a confounding factor.

The baseline records what the unmodified model can already do. Stage 7 will compare against this exact result and should claim improvement only when category scores, hallucination behavior, and general reasoning support it.
