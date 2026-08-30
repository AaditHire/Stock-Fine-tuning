# Stage 5C: pinned external-data collection

## Outcome

Stage 5C collected and curated a 4,800-example candidate corpus from two pinned external datasets plus a small training-only selection of project-authored Stage 5B behavior examples. It did not train, evaluate, export, or load model weights.

The collection completed with 3,900 training, 450 validation, and 450 development examples. Every rendered Qwen conversation fits the 512-token hardware-safe context limit; the observed maxima are 511, 496, and 504 tokens respectively.

## Source mixture

| Source | Pinned revision | License recorded | Train | Validation | Development |
| --- | --- | --- | ---: | ---: | ---: |
| `btech-software/cosimo-cfa-frm-71k` | `42244d29c6b9912683213a08d1a9c5b0373b381b` | MIT | 2,400 | 300 | 300 |
| `bevaya/FinQA` | `3d6a736bc67e06bc15fbf3618d88204a57c5b25e` | MIT; underlying FinTabNet CDLA-Permissive-1.0 | 1,200 | 150 | 150 |
| Stage 5B project behavior | local locked file | project-original | 300 | 0 | 0 |

Cosimo contributes code-verified CFA/FRM-style calculations and multiple choice. FinQA contributes reasoning over annotated financial-report evidence; only its gold evidence is included so records fit the local context window. The project selection preserves refusal, JSON-only, exact-final-marker, analysis, factual, and instruction-following behavior.

`gbharti/finance-alpaca` was audited but excluded. Sampled answers were often verbose or opinionated and were not consistently suitable for exact financial reasoning or the project's output contracts. The exclusion and pinned revision are recorded in the manifest.

## Quality and isolation

- The frozen 160-case benchmark remains unchanged at SHA-256 `bfd1b847d2042f6a59f8a8a5f0dfe0826729dc68a0d390a356c2f1fd3b1781fa`.
- Both the raw source question and the transformed user prompt are screened for exact or fuzzy overlap with protected Stage 3 and Stage 4 prompts.
- Cosimo generator families are assigned wholly to one split.
- FinQA report documents retain their official train/validation/test separation, and only one question per report is selected.
- External source-family overlap is zero across all three splits.
- Exact prompt-and-answer duplicates are rejected and backfilled deterministically within the same split-family assignment.
- The scan rejected 2,997 duplicate Cosimo conversations. This confirms that ingesting nominal source rows without deduplication would substantially overstate effective diversity.

## Important limitation

This is a locked candidate corpus, not approval to train it blindly. It is calculation-heavy: 4,345 of 4,800 examples are labeled calculation, while the 300 project examples carry most of the refusal, JSON, factual, analysis, and instruction-following behavior. Before Stage 6C, the training sampling policy should cap the effective calculation share or up-weight the project behavior examples so the much larger external portion does not drown out the behaviors that Stage 7 showed must be preserved.

Cosimo also originates from only 71 synthetic generator templates, despite its large row count. Family-disjoint holdouts make evaluation more honest, but representative samples still need human financial review before any release-quality claim.

## Files and reproduction

- `configs/data/stage5c_sources.toml`: pinned source, license, count, tokenizer, and seed contract
- `scripts/build_stage5c_dataset.py`: offline-capable acquisition, transformation, filtering, split, and audit builder
- `data/raw/finpulse_stage5c_v1.jsonl`: combined curated corpus
- `data/train/finpulse_stage5c_v1.jsonl`: training candidate
- `data/validation/finpulse_stage5c_v1.jsonl`: loss-monitoring candidate
- `data/development/finpulse_stage5c_v1.jsonl`: family-disjoint development candidate
- `data/processed/finpulse_stage5c_v1.quality.json`: counts, rejection reasons, token statistics, and limitations
- `data/processed/finpulse_stage5c_v1.manifest.json`: pinned revisions and immutable file hashes

After the pinned datasets and tokenizer are cached under the Git-ignored `models/huggingface/` directory, reproduce the collection without network access:

```powershell
$env:HF_HOME = (Join-Path (Get-Location) 'models\huggingface')
$env:HF_DATASETS_CACHE = (Join-Path (Get-Location) 'models\huggingface\datasets')
.\.venv\Scripts\python.exe scripts\build_stage5c_dataset.py --offline
```

## Next boundary

Stage 5C stopped at collection and audit. Stage 5D subsequently created a balanced 900-row training view while preserving these validation and development files unchanged. See `docs/stage5d.md`; no model training was performed.
