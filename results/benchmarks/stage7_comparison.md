# Stage 7: base vs fine-tuned evaluation

Conclusion: **regressed**.

- Base: **90.56%** (355/392 checks)
- Adapter: **84.44%** (331/392 checks)
- Change: **-6.12%**
- Cases improved / regressed / unchanged: **8 / 31 / 121**

## Category comparison

| Category | Base | Adapter | Change |
| --- | ---: | ---: | ---: |
| contradictory_signals | 93.75% | 90.62% | -3.13% |
| crypto_derivatives | 90.62% | 78.12% | -12.50% |
| financial_calculations | 75.00% | 50.00% | -25.00% |
| hallucination_traps | 87.50% | 100.00% | +12.50% |
| macroeconomics | 95.31% | 89.06% | -6.25% |
| risk_management | 96.88% | 89.06% | -7.82% |
| scenario_analysis | 89.06% | 85.94% | -3.12% |
| stock_fundamentals | 96.88% | 85.94% | -10.94% |
| structured_output | 87.50% | 81.25% | -6.25% |
| technical_analysis | 95.31% | 70.31% | -25.00% |

## Required dimensions

| Dimension | Base | Adapter | Change |
| --- | ---: | ---: | ---: |
| financial_reasoning | 91.32% | 84.03% | -7.29% |
| calculations | 75.00% | 50.00% | -25.00% |
| instruction_following | 87.50% | 81.25% | -6.25% |
| uncertainty_calibration | 91.41% | 88.28% | -3.13% |
| hallucination_resistance | 87.50% | 100.00% | +12.50% |
| risk_awareness | 95.00% | 90.00% | -5.00% |
| conflicting_signal_analysis | 93.75% | 90.62% | -3.13% |
| financial_regression_proxy | 81.25% | 65.62% | -15.63% |
| factual_finance_knowledge | 100.00% | 82.50% | -17.50% |
| structured_output_validity | 87.50% | 81.25% | -6.25% |
| general_capability_regression | 66.67% | 66.67% | +0.00% |

## Changed cases

### Improvements

- `eval_fc_006` (financial_calculations): 0.0% → 100.0%
- `eval_fc_011` (financial_calculations): 0.0% → 100.0%
- `eval_sa_004` (scenario_analysis): 50.0% → 75.0%
- `eval_sa_005` (scenario_analysis): 50.0% → 75.0%
- `eval_cs_003` (contradictory_signals): 75.0% → 100.0%
- `eval_ht_004` (hallucination_traps): 0.0% → 100.0%
- `eval_ht_008` (hallucination_traps): 50.0% → 100.0%
- `eval_ht_016` (hallucination_traps): 50.0% → 100.0%

### Regressions

- `eval_ta_001` (technical_analysis): 100.0% → 0.0%
- `eval_ta_004` (technical_analysis): 100.0% → 0.0%
- `eval_ta_007` (technical_analysis): 100.0% → 0.0%
- `eval_ta_012` (technical_analysis): 100.0% → 75.0%
- `eval_ta_014` (technical_analysis): 75.0% → 50.0%
- `eval_ta_015` (technical_analysis): 75.0% → 50.0%
- `eval_ta_016` (technical_analysis): 100.0% → 75.0%
- `eval_cd_001` (crypto_derivatives): 100.0% → 0.0%
- `eval_cd_011` (crypto_derivatives): 100.0% → 25.0%
- `eval_cd_015` (crypto_derivatives): 100.0% → 75.0%
- `eval_sf_001` (stock_fundamentals): 100.0% → 0.0%
- `eval_sf_010` (stock_fundamentals): 100.0% → 75.0%
- `eval_sf_015` (stock_fundamentals): 100.0% → 75.0%
- `eval_sf_016` (stock_fundamentals): 100.0% → 75.0%
- `eval_ma_002` (macroeconomics): 100.0% → 0.0%
- `eval_rm_002` (risk_management): 100.0% → 0.0%
- `eval_rm_012` (risk_management): 100.0% → 75.0%
- `eval_fc_001` (financial_calculations): 100.0% → 0.0%
- `eval_fc_004` (financial_calculations): 100.0% → 0.0%
- `eval_fc_005` (financial_calculations): 100.0% → 0.0%
- `eval_fc_010` (financial_calculations): 100.0% → 0.0%
- `eval_fc_013` (financial_calculations): 100.0% → 0.0%
- `eval_fc_016` (financial_calculations): 100.0% → 0.0%
- `eval_sa_003` (scenario_analysis): 100.0% → 75.0%
- `eval_sa_007` (scenario_analysis): 100.0% → 75.0%
- `eval_sa_008` (scenario_analysis): 100.0% → 75.0%
- `eval_sa_013` (scenario_analysis): 100.0% → 75.0%
- `eval_cs_002` (contradictory_signals): 100.0% → 75.0%
- `eval_cs_010` (contradictory_signals): 100.0% → 75.0%
- `eval_cs_012` (contradictory_signals): 100.0% → 75.0%
- `eval_so_014` (structured_output): 100.0% → 0.0%

## Limitations

- The adapter was trained on only 33 examples and validated on 7 examples.
- The frozen benchmark is financial. A separate three-case development sentinel is used for general reasoning and is too small for a broad capability claim.
- Deterministic keyword and regex rubrics are reproducible but do not replace human review.

## Runtime and diagnostics

- Base throughput: 10.70 tokens/second
- Adapter throughput: 6.16 tokens/second
- Adapter regressions at the 192-token ceiling: 24/31
- Promotion recommendation: **reject_adapter**
