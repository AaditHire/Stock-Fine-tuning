# Stage 7B development gate

Gate passed: **True**

- Base: **75.00%**
- Adapter: **91.13%**
- Change: **+16.13%**

| Task | Base | Adapter | Change |
| --- | ---: | ---: | ---: |
| analysis | 75.00% | 100.00% | +25.00% |
| calculation | 93.33% | 93.33% | +0.00% |
| factual | 54.55% | 59.09% | +4.54% |
| instruction_following | 50.00% | 100.00% | +50.00% |
| multiple_choice | 72.73% | 100.00% | +27.27% |
| refusal | 100.00% | 100.00% | +0.00% |

## Gate checks

- overall_improves_by_5_points: **True**
- calculation_at_least_90_percent: **True**
- calculation_not_worse: **True**
- instruction_following_at_least_90_percent: **True**
- refusal_not_worse: **True**
- analysis_within_5_points: **True**
- factual_within_5_points: **True**

The development split is row-disjoint but shares template families with training; this gate is a screen, not a release-quality result.
