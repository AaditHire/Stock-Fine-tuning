# Stage 7D: base vs fine-tuned evaluation

Conclusion: **regressed**.

- Base: **90.56%** (355/392 checks)
- Adapter: **84.44%** (331/392 checks)
- Change: **-6.12%**
- Cases improved / regressed / unchanged: **10 / 32 / 118**

## Category comparison

| Category | Base | Adapter | Change |
| --- | ---: | ---: | ---: |
| contradictory_signals | 93.75% | 92.19% | -1.56% |
| crypto_derivatives | 90.62% | 68.75% | -21.87% |
| financial_calculations | 75.00% | 62.50% | -12.50% |
| hallucination_traps | 87.50% | 100.00% | +12.50% |
| macroeconomics | 95.31% | 81.25% | -14.06% |
| risk_management | 96.88% | 85.94% | -10.94% |
| scenario_analysis | 89.06% | 82.81% | -6.25% |
| stock_fundamentals | 96.88% | 82.81% | -14.07% |
| structured_output | 87.50% | 93.75% | +6.25% |
| technical_analysis | 95.31% | 65.62% | -29.69% |

## Required dimensions

| Dimension | Base | Adapter | Change |
| --- | ---: | ---: | ---: |
| financial_reasoning | 91.32% | 83.33% | -7.99% |
| calculations | 75.00% | 62.50% | -12.50% |
| instruction_following | 87.50% | 93.75% | +6.25% |
| uncertainty_calibration | 91.41% | 87.50% | -3.91% |
| hallucination_resistance | 87.50% | 100.00% | +12.50% |
| risk_awareness | 95.00% | 85.00% | -10.00% |
| conflicting_signal_analysis | 93.75% | 92.19% | -1.56% |
| financial_regression_proxy | 81.25% | 78.12% | -3.13% |
| factual_finance_knowledge | 100.00% | 70.00% | -30.00% |
| structured_output_validity | 87.50% | 93.75% | +6.25% |

## Changed cases

### Improvements

- `eval_rm_009` (risk_management): 75.0% → 100.0%
- `eval_fc_006` (financial_calculations): 0.0% → 100.0%
- `eval_fc_014` (financial_calculations): 0.0% → 100.0%
- `eval_sa_006` (scenario_analysis): 75.0% → 100.0%
- `eval_sa_009` (scenario_analysis): 50.0% → 75.0%
- `eval_cs_005` (contradictory_signals): 75.0% → 100.0%
- `eval_ht_004` (hallucination_traps): 0.0% → 100.0%
- `eval_ht_008` (hallucination_traps): 50.0% → 100.0%
- `eval_ht_016` (hallucination_traps): 50.0% → 100.0%
- `eval_so_003` (structured_output): 0.0% → 100.0%

### Regressions

- `eval_ta_002` (technical_analysis): 100.0% → 0.0%
- `eval_ta_005` (technical_analysis): 100.0% → 0.0%
- `eval_ta_007` (technical_analysis): 100.0% → 0.0%
- `eval_ta_008` (technical_analysis): 100.0% → 0.0%
- `eval_ta_010` (technical_analysis): 100.0% → 75.0%
- `eval_ta_012` (technical_analysis): 100.0% → 75.0%
- `eval_ta_015` (technical_analysis): 75.0% → 50.0%
- `eval_cd_003` (crypto_derivatives): 100.0% → 0.0%
- `eval_cd_005` (crypto_derivatives): 100.0% → 0.0%
- `eval_cd_007` (crypto_derivatives): 100.0% → 0.0%
- `eval_cd_011` (crypto_derivatives): 100.0% → 50.0%
- `eval_sf_006` (stock_fundamentals): 100.0% → 0.0%
- `eval_sf_008` (stock_fundamentals): 100.0% → 0.0%
- `eval_sf_010` (stock_fundamentals): 100.0% → 75.0%
- `eval_ma_002` (macroeconomics): 100.0% → 0.0%
- `eval_ma_006` (macroeconomics): 100.0% → 0.0%
- `eval_ma_011` (macroeconomics): 100.0% → 75.0%
- `eval_rm_001` (risk_management): 100.0% → 0.0%
- `eval_rm_012` (risk_management): 100.0% → 50.0%
- `eval_rm_013` (risk_management): 100.0% → 75.0%
- `eval_rm_014` (risk_management): 100.0% → 75.0%
- `eval_fc_001` (financial_calculations): 100.0% → 0.0%
- `eval_fc_003` (financial_calculations): 100.0% → 0.0%
- `eval_fc_012` (financial_calculations): 100.0% → 0.0%
- `eval_fc_016` (financial_calculations): 100.0% → 0.0%
- `eval_sa_002` (scenario_analysis): 100.0% → 75.0%
- `eval_sa_007` (scenario_analysis): 100.0% → 75.0%
- `eval_sa_008` (scenario_analysis): 100.0% → 50.0%
- `eval_sa_011` (scenario_analysis): 100.0% → 75.0%
- `eval_sa_013` (scenario_analysis): 100.0% → 75.0%
- `eval_cs_002` (contradictory_signals): 100.0% → 75.0%
- `eval_cs_010` (contradictory_signals): 100.0% → 75.0%

## Limitations

- The adapter was trained on 900 examples and validated on 450 examples.
- The frozen benchmark is financial; this comparison did not run a separate general-capability sentinel.
- Deterministic keyword and regex rubrics are reproducible but do not replace human review.

## Runtime and diagnostics

- Base throughput: 10.70 tokens/second
- Adapter throughput: 6.05 tokens/second
- Adapter regressions at the 192-token ceiling: 24/32
- Promotion recommendation: **reject_adapter**
