# Stage 5B: corrective instruction dataset

## Outcome

Stage 5B replaces the 40-example training experiment as the next data candidate without changing or deleting the original Stage 5 seed. It adds 500 deterministic, project-original synthetic examples aimed directly at the behavior gaps observed in Stage 7. No model was loaded or trained during this stage.

The corpus passed schema, exact-duplicate, near-duplicate, protected-evaluation leakage, live-data safety, response-format, response-length, and balance checks with 500 accepted and zero rejected records.

## Split policy

The data is split deterministically by category and example ID using seed `5813`:

| Split | Examples | Purpose |
| --- | ---: | --- |
| Train | 398 | Adapter weight updates |
| Validation | 51 | Loss monitoring during training |
| Development | 51 | Behavioral candidate selection before the frozen benchmark |

All three sets are disjoint. The development split is not a replacement for the frozen Stage 4 benchmark. It exists so future iteration does not repeatedly use the frozen benchmark as a tuning target. It is row-disjoint rather than template-family-disjoint, so it is a lightweight behavioral screen and may be optimistic for repeated numeric drill families.

## Corrective balance

| Task type | Count | Share |
| --- | ---: | ---: |
| Financial calculations | 150 | 30% |
| Multiple choice | 100 | 20% |
| Factual finance | 75 | 15% |
| Conditional analysis | 100 | 20% |
| Live-data refusal | 25 | 5% |
| Instruction following | 50 | 10% |

| Output contract | Count | Share |
| --- | ---: | ---: |
| Exact `FINAL:` marker | 250 | 50% |
| Strict JSON only | 75 | 15% |
| Plain response | 175 | 35% |

Response-length labels are also balanced: 250 short, 200 medium, and 50 long. This reverses the Stage 5 seed's near-uniform long explanatory style while retaining conditional reasoning and uncertainty.

The topical distribution is 20% technical analysis, 15% crypto derivatives, 15% stock fundamentals, 12% macroeconomics, 15% risk management, 13% scenario analysis, and 10% terminology/miscellaneous.

## Authorship and limitations

All examples are generated from project-authored rules and concept banks in `scripts/build_stage5b_corpus.py`. They do not contain copied articles, scraped material, or prompts, answers, rubrics, or paraphrases from the frozen benchmark. Calculation answers are produced from deterministic formulas, which reduces arithmetic-label errors.

This is controlled synthetic instruction data, not 500 separately human-written essays. Numeric drills intentionally reuse formula scaffolding with different supplied values, so exact-duplicate rejection remains strict while the Stage 5B near-duplicate threshold permits those deliberate variants. The corpus should be treated as a stronger corrective experiment, not proof that 500 examples are sufficient for a release model.

Before a later release-quality run, representative samples from every task family should receive human financial review. Any future expansion should add new concepts and phrasings rather than merely increasing numeric variants.

## Files and reproduction

- `configs/data/training_pipeline_stage5b.toml`: split and balance contract
- `scripts/build_stage5b_corpus.py`: deterministic source builder
- `data/raw/finpulse_stage5b_v1.jsonl`: complete reviewed source
- `data/train/finpulse_stage5b_v1.jsonl`: training split
- `data/validation/finpulse_stage5b_v1.jsonl`: loss-monitoring split
- `data/development/finpulse_stage5b_v1.jsonl`: independent development holdout
- `data/processed/finpulse_stage5b_v1.quality.json`: quality report
- `data/processed/finpulse_stage5b_v1.manifest.json`: locked source and split hashes

Reproduce the files:

```powershell
.\.venv\Scripts\python.exe scripts\build_stage5b_corpus.py --write
.\.venv\Scripts\python.exe scripts\build_training_dataset.py data\raw\finpulse_stage5b_v1.jsonl --config configs\data\training_pipeline_stage5b.toml --train-output data\train\finpulse_stage5b_v1.jsonl --validation-output data\validation\finpulse_stage5b_v1.jsonl --development-output data\development\finpulse_stage5b_v1.jsonl --quality-output data\processed\finpulse_stage5b_v1.quality.json --manifest-output data\processed\finpulse_stage5b_v1.manifest.json
```

## Next boundary

Stage 6B completed one conservative QLoRA candidate against these locked train and validation splits. It did not train on the development split or frozen benchmark. See `docs/stage6b.md` for the measured run.
