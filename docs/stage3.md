# Stage 3: base-model benchmark

## Outcome

`unsloth/Qwen3-4B-bnb-4bit` is the selected base model for the next stages. It scored 82.8% (24 of 29 automated checks), while `unsloth/Phi-4-mini-instruct-bnb-4bit` scored 75.9% (22 of 29).

This is a small development benchmark for choosing a practical model. It is not the frozen Stage 4 evaluation benchmark and must not be used later to claim that fine-tuning improved the model.

## Fair comparison setup

- Both models received the same 15 prompts, system prompt, sampling settings, 2,048-token context limit, and 256-token response limit.
- Both were loaded through Unsloth with bitsandbytes 4-bit quantization.
- Models ran separately, preventing two checkpoints from competing for the 6 GB GPU.
- The benchmark used 29 deterministic checks covering finance reasoning and knowledge, calculations, hallucination resistance, structured output, instruction following, and general reasoning.
- The full prompt, response, timing, memory, and individual check results are retained in the JSON reports.

Phi's checkpoint is pinned to revision `cece1fd36f04ff79f55ec861f206ca4e16acea6e`. Transformers 5.5 provides native Phi-3 architecture support, so repository remote code is disabled. This avoids a legacy remote model file that attempted to import the removed `SlidingWindowCache` class.

## Results

| Measure | Qwen3-4B | Phi-4-mini-instruct |
| --- | ---: | ---: |
| Automated score | 82.8% | 75.9% |
| Checks passed | 24/29 | 22/29 |
| Aggregate generation speed | 8.83 tok/s | 10.90 tok/s |
| Model load time | 7.64 s | 6.05 s |
| Peak total device VRAM | 3,787.5 MiB | 3,961.5 MiB |
| Peak process RAM | 3,742.2 MiB | 3,974.1 MiB |

Qwen passed all three direct financial calculation cases. Phi passed one of three and made large errors in position sizing and reward/risk. Phi was faster, used about 174 MiB more total device VRAM, and correctly refused both requests for exact live market data. Qwen refused the BTC request but fabricated an Apple P/E value while also claiming it lacked live access.

Both models showed important weaknesses. Qwen reversed the positive-funding payment direction, failed one syllogism, and missed the exact structured risk calculation. Phi did not follow strict JSON-only output, and its RSI invalidation reasoning was confused. These failures are useful targets for the later dataset and frozen evaluation, not evidence to tune on these exact benchmark answers.

## How scoring works

The benchmark file contains declarative checks rather than subjective ratings. Checks validate expected calculations, required or forbidden language, exact JSON shapes, and uncertainty/refusal behavior. Category averages reveal *where* a model differs; the overall score alone does not explain why.

Automated text rules are imperfect. During validation, the rubric was corrected so positive funding tests the correct payment direction, equivalent refusal wording is accepted, and a JSON-only request rejects fenced JSON plus commentary. Both saved model responses were then re-scored with the same final rubric.

The selection rule ranks overall score first, then hallucination resistance, financial calculations, and speed. Memory can override that ranking only if a model is impractical on the target GPU. Both models fit, so Qwen's higher overall result determined the choice.

## Files

- `benchmarks/stage3_base_models.json`: small development benchmark and deterministic checks
- `configs/models/phi4_mini.toml`: pinned Phi loading and matched generation configuration
- `scripts/run_stage3_model.py`: reusable single-model benchmark runner
- `scripts/compare_stage3_models.py`: comparison and deterministic selection report
- `results/benchmarks/stage3_qwen3_4b.json`: Qwen responses, metrics, and scores
- `results/benchmarks/stage3_phi4_mini.json`: Phi responses, metrics, and scores
- `results/benchmarks/stage3_comparison.json`: machine-readable decision
- `results/benchmarks/stage3_comparison.md`: human-readable comparison

## What this teaches

A benchmark is a repeatable exam: models receive the same inputs and are scored against explicit expectations. Identical prompts and settings reduce confounding, while separate category scores prevent a fast or eloquent model from hiding calculation or hallucination failures.

4-bit quantization compresses most model weights enough for both roughly 4-billion-parameter models to fit this laptop. It reduces memory use for inference; it does not fine-tune either model. Tokens per second measures generation throughput, while peak RAM and VRAM show whether the model is practical on the real target hardware.

Stage 4 will create a larger, frozen evaluation set and record Qwen's baseline. “Frozen” means its questions and scoring rules stop changing before training data is built, which is essential for detecting dataset leakage and measuring genuine improvement later.
