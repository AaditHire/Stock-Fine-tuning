# Stage 3 base-model comparison

This is a small development benchmark, not the frozen Stage 4 evaluation set.

## Decision

Selected base model: `unsloth/Qwen3-4B-bnb-4bit`.

The selected model scored 82.8% versus 75.9%. Tie-breakers prioritize hallucination resistance and financial calculations before speed.

## Overall results

| Model | Score | Checks | tok/s | Load (s) | Device VRAM (MiB) | Process RAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `unsloth/Qwen3-4B-bnb-4bit` | 82.8% | 24/29 | 8.83 | 7.64 | 3787.5 | 3742.2 |
| `unsloth/Phi-4-mini-instruct-bnb-4bit` | 75.9% | 22/29 | 10.90 | 6.05 | 3961.5 | 3974.1 |

## Category scores

| Category | `unsloth/Qwen3-4B-bnb-4bit` | `unsloth/Phi-4-mini-instruct-bnb-4bit` |
| --- | ---: | ---: |
| finance_knowledge | 66.7% | 66.7% |
| finance_reasoning | 91.7% | 91.7% |
| financial_calculation | 100.0% | 33.3% |
| general_reasoning | 66.7% | 66.7% |
| hallucination_resistance | 75.0% | 100.0% |
| instruction_following | 100.0% | 100.0% |
| structured_output | 50.0% | 0.0% |

## Selection rule

Models are ranked by overall automated score, then hallucination resistance, financial calculation score, and finally generation speed. Resource measurements remain visible and can override the result only if a model is impractical on 6 GB VRAM.
