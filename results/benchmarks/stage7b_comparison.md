# Stage 7B: base vs fine-tuned evaluation

Conclusion: **regressed**.

- Base: **90.56%** (355/392 checks)
- Adapter: **70.92%** (278/392 checks)
- Change: **-19.64%**
- Cases improved / regressed / unchanged: **7 / 57 / 96**

## Category comparison

| Category | Base | Adapter | Change |
| --- | ---: | ---: | ---: |
| contradictory_signals | 93.75% | 70.31% | -23.44% |
| crypto_derivatives | 90.62% | 84.38% | -6.24% |
| financial_calculations | 75.00% | 18.75% | -56.25% |
| hallucination_traps | 87.50% | 100.00% | +12.50% |
| macroeconomics | 95.31% | 84.38% | -10.93% |
| risk_management | 96.88% | 85.94% | -10.94% |
| scenario_analysis | 89.06% | 57.81% | -31.25% |
| stock_fundamentals | 96.88% | 82.81% | -14.07% |
| structured_output | 87.50% | 81.25% | -6.25% |
| technical_analysis | 95.31% | 81.25% | -14.06% |

## Required dimensions

| Dimension | Base | Adapter | Change |
| --- | ---: | ---: | ---: |
| financial_reasoning | 91.32% | 69.10% | -22.22% |
| calculations | 75.00% | 18.75% | -56.25% |
| instruction_following | 87.50% | 81.25% | -6.25% |
| uncertainty_calibration | 91.41% | 64.06% | -27.35% |
| hallucination_resistance | 87.50% | 100.00% | +12.50% |
| risk_awareness | 95.00% | 77.50% | -17.50% |
| conflicting_signal_analysis | 93.75% | 70.31% | -23.44% |
| financial_regression_proxy | 81.25% | 50.00% | -31.25% |
| factual_finance_knowledge | 100.00% | 100.00% | +0.00% |
| structured_output_validity | 87.50% | 81.25% | -6.25% |
| general_capability_regression | 66.67% | 66.67% | +0.00% |

## Changed cases

### Improvements

- `eval_sf_009` (stock_fundamentals): 75.0% → 100.0%
- `eval_fc_006` (financial_calculations): 0.0% → 100.0%
- `eval_cs_005` (contradictory_signals): 75.0% → 100.0%
- `eval_ht_004` (hallucination_traps): 0.0% → 100.0%
- `eval_ht_008` (hallucination_traps): 50.0% → 100.0%
- `eval_ht_016` (hallucination_traps): 50.0% → 100.0%
- `eval_so_003` (structured_output): 0.0% → 100.0%

### Regressions

- `eval_ta_009` (technical_analysis): 100.0% → 75.0%
- `eval_ta_011` (technical_analysis): 75.0% → 25.0%
- `eval_ta_012` (technical_analysis): 100.0% → 25.0%
- `eval_ta_013` (technical_analysis): 100.0% → 75.0%
- `eval_ta_014` (technical_analysis): 75.0% → 50.0%
- `eval_ta_016` (technical_analysis): 100.0% → 75.0%
- `eval_cd_010` (crypto_derivatives): 100.0% → 75.0%
- `eval_cd_011` (crypto_derivatives): 100.0% → 75.0%
- `eval_cd_015` (crypto_derivatives): 100.0% → 75.0%
- `eval_cd_016` (crypto_derivatives): 75.0% → 50.0%
- `eval_sf_011` (stock_fundamentals): 75.0% → 25.0%
- `eval_sf_012` (stock_fundamentals): 100.0% → 25.0%
- `eval_sf_015` (stock_fundamentals): 100.0% → 50.0%
- `eval_sf_016` (stock_fundamentals): 100.0% → 25.0%
- `eval_ma_009` (macroeconomics): 100.0% → 50.0%
- `eval_ma_010` (macroeconomics): 100.0% → 75.0%
- `eval_ma_011` (macroeconomics): 100.0% → 75.0%
- `eval_ma_013` (macroeconomics): 75.0% → 50.0%
- `eval_ma_016` (macroeconomics): 100.0% → 50.0%
- `eval_rm_012` (risk_management): 100.0% → 75.0%
- `eval_rm_013` (risk_management): 100.0% → 75.0%
- `eval_rm_014` (risk_management): 100.0% → 50.0%
- `eval_rm_015` (risk_management): 75.0% → 50.0%
- `eval_rm_016` (risk_management): 100.0% → 50.0%
- `eval_fc_001` (financial_calculations): 100.0% → 0.0%
- `eval_fc_003` (financial_calculations): 100.0% → 0.0%
- `eval_fc_004` (financial_calculations): 100.0% → 0.0%
- `eval_fc_005` (financial_calculations): 100.0% → 0.0%
- `eval_fc_009` (financial_calculations): 100.0% → 0.0%
- `eval_fc_010` (financial_calculations): 100.0% → 0.0%
- `eval_fc_012` (financial_calculations): 100.0% → 0.0%
- `eval_fc_013` (financial_calculations): 100.0% → 0.0%
- `eval_fc_015` (financial_calculations): 100.0% → 0.0%
- `eval_fc_016` (financial_calculations): 100.0% → 0.0%
- `eval_sa_002` (scenario_analysis): 100.0% → 50.0%
- `eval_sa_003` (scenario_analysis): 100.0% → 75.0%
- `eval_sa_005` (scenario_analysis): 50.0% → 25.0%
- `eval_sa_007` (scenario_analysis): 100.0% → 75.0%
- `eval_sa_008` (scenario_analysis): 100.0% → 25.0%
- `eval_sa_010` (scenario_analysis): 100.0% → 25.0%
- `eval_sa_011` (scenario_analysis): 100.0% → 75.0%
- `eval_sa_012` (scenario_analysis): 100.0% → 25.0%
- `eval_sa_013` (scenario_analysis): 100.0% → 75.0%
- `eval_sa_015` (scenario_analysis): 100.0% → 50.0%
- `eval_sa_016` (scenario_analysis): 100.0% → 50.0%
- `eval_cs_002` (contradictory_signals): 100.0% → 50.0%
- `eval_cs_003` (contradictory_signals): 75.0% → 25.0%
- `eval_cs_004` (contradictory_signals): 100.0% → 75.0%
- `eval_cs_006` (contradictory_signals): 75.0% → 25.0%
- `eval_cs_007` (contradictory_signals): 100.0% → 50.0%
- `eval_cs_008` (contradictory_signals): 100.0% → 75.0%
- `eval_cs_009` (contradictory_signals): 100.0% → 75.0%
- `eval_cs_010` (contradictory_signals): 100.0% → 25.0%
- `eval_cs_011` (contradictory_signals): 100.0% → 75.0%
- `eval_cs_015` (contradictory_signals): 100.0% → 75.0%
- `eval_so_005` (structured_output): 100.0% → 0.0%
- `eval_so_010` (structured_output): 100.0% → 0.0%

## Limitations

- The adapter was trained on 398 examples and validated on 51 examples.
- The frozen benchmark is financial. A separate three-case development sentinel is used for general reasoning and is too small for a broad capability claim.
- Deterministic keyword and regex rubrics are reproducible but do not replace human review.

## Runtime and diagnostics

- Base throughput: 10.70 tokens/second
- Adapter throughput: 6.07 tokens/second
- Adapter regressions at the 192-token ceiling: 2/57
- Promotion recommendation: **reject_adapter**
