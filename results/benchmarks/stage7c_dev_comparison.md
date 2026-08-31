# Stage 7C development evaluation

Selected candidate: **checkpoint-30**

- Development split: **450 cases**
- Base answer accuracy: **9.78%**
- Base format accuracy: **49.33%**

| Candidate | Answer | Change | Format | Gate |
| --- | ---: | ---: | ---: | --- |
| checkpoint-15 | 3.56% | -6.22% | 34.00% | fail |
| checkpoint-30 | 18.67% | +8.89% | 68.89% | pass |
| checkpoint-45 | 18.44% | +8.66% | 76.89% | pass |
| final | 14.67% | +4.89% | 81.56% | pass |

## Task and source scores

| Model | Calculation | Multiple choice | Cosimo | FinQA |
| --- | ---: | ---: | ---: | ---: |
| base | 9.07% | 15.09% | 10.00% | 9.33% |
| checkpoint-15 | 3.27% | 5.66% | 1.00% | 8.67% |
| checkpoint-30 | 18.89% | 16.98% | 23.00% | 10.00% |
| checkpoint-45 | 17.88% | 22.64% | 22.33% | 10.67% |
| final | 13.60% | 22.64% | 16.33% | 11.33% |

The frozen 160-case benchmark was not run. Stage 7C uses only the locked, family-disjoint Stage 5C development split and stops before the next stage.
