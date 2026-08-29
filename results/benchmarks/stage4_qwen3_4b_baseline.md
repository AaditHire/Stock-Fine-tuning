# Stage 4 Qwen3-4B frozen baseline

- Benchmark: `finpulse_eval_v1` (160 cases)
- Overall: **90.6%** (355/392 checks)
- Aggregate generation speed: **10.69 tokens/second**
- Peak total device VRAM: **3809.5 MiB**
- Peak process RAM: **3740.3 MiB**

## Category scores

| Category | Score |
| --- | ---: |
| contradictory_signals | 93.8% |
| crypto_derivatives | 90.6% |
| financial_calculations | 75.0% |
| hallucination_traps | 87.5% |
| macroeconomics | 95.3% |
| risk_management | 96.9% |
| scenario_analysis | 89.1% |
| stock_fundamentals | 96.9% |
| structured_output | 87.5% |
| technical_analysis | 95.3% |

## Weakest categories

- `financial_calculations`: 75.0%
- `hallucination_traps`: 87.5%
- `structured_output`: 87.5%

This is the pre-training baseline. The benchmark is frozen and excluded from all future training data.
