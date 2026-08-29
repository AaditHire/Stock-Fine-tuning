# Stage 5: financial instruction-data pipeline

## Outcome

Stage 5 adds the data-engineering system needed before QLoRA training. The pipeline accepts conversational JSONL, normalizes text, validates schema and behavior, rejects duplicates and evaluation leakage, checks category balance, creates deterministic train/validation splits, and records hashes and quality metrics.

The included `finpulse_seed_v1` corpus has 40 project-original, reviewed examples. It is intentionally small and exists to prove the pipeline and establish writing quality. It is not the planned 8,000–15,000-example training corpus.

## Seed results

| Measure | Result |
| --- | ---: |
| Input examples | 40 |
| Accepted | 40 |
| Rejected | 0 |
| Train split | 33 |
| Validation split | 7 |
| Original-source examples | 40 |
| Average user length | 100.7 characters |
| Average assistant length | 596.5 characters |

| Category | Count | Distribution |
| --- | ---: | ---: |
| Technical analysis | 10 | 25% |
| Crypto derivatives | 8 | 20% |
| Stock fundamentals | 6 | 15% |
| Macroeconomics | 6 | 15% |
| Risk management | 4 | 10% |
| Scenario analysis | 4 | 10% |
| Terminology/miscellaneous | 2 | 5% |

The small seed reserves one validation example from each category so every category is represented. That produces a 7/40 validation split rather than exactly 10%. On a large dataset, the same deterministic stratified algorithm converges toward the configured 10% ratio.

## Conversational schema

Each JSONL row has three top-level fields:

```json
{
  "id": "fp_ta_0001",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "category": "technical_analysis",
    "subtopics": ["market_structure"],
    "difficulty": "intermediate",
    "source": {
      "type": "original",
      "reference": "finpulse-llm-stage5-seed",
      "license": "project-original"
    },
    "review": {"status": "reviewed", "reviewer": "project-author"}
  }
}
```

The formal contract is in `schemas/training_example.schema.json`. The Python validator deliberately has no JSON-schema runtime dependency and produces plain-language rejection reasons.

## Quality and safety rules

Before an example can enter either split, the pipeline checks:

- exact top-level, metadata, source, review, and message structure;
- `system → user → assistant` role order;
- configured system-prompt consistency;
- unique, patterned IDs;
- non-empty category, subtopic, difficulty, provenance, and licensing fields;
- reviewed status;
- minimum and maximum message lengths;
- possible credentials or secrets;
- explicit data-access limitations when a prompt asks for live/current data, plus rejection if the response then supplies an unsupported exact value;
- exact duplicate conversations;
- near-duplicate user prompts;
- exact and fuzzy similarity to protected Stage 3 and Stage 4 prompts;
- category-distribution drift beyond the configured tolerance.

Near-duplicate candidate lookup uses three-token shingles, then applies sequence/token similarity only to plausible candidates. This avoids an expensive all-pairs comparison when the dataset grows toward 15,000 examples. It does not use embeddings or a vector database.

## Leakage prevention

Evaluation leakage means an evaluation question, answer, rubric, or close paraphrase appears in training. A model can then score well by memorizing the exam rather than learning a transferable skill.

The pipeline loads every protected Stage 3 and Stage 4 prompt. Exact normalized hashes catch direct copies, while fuzzy text comparison catches close rewrites. Stage 4's frozen SHA-256 is copied into the training manifest, tying each dataset build to the exact protected benchmark version.

The current checks are a strong first boundary, not permission to intentionally paraphrase evaluation cases. New training examples must be authored independently around general financial concepts.

## Cleaning and deduplication

Unicode is normalized with NFKC, Windows and Unix line endings are unified, repeated spaces are collapsed, and paragraph boundaries are preserved. Normalization occurs before validation, hashing, and deduplication so cosmetic formatting cannot create hidden duplicates.

Exact duplicates use SHA-256 fingerprints. Near duplicates use conservative text similarity and keep the first reviewed record in deterministic source order. Deduplication happens before splitting, preventing the same underlying example from appearing in both train and validation.

## Train and validation splits

The train split is used to update LoRA adapter weights. The validation split is not used for weight updates; it measures behavior on held-out examples during training and can reveal overfitting.

The split is deterministic from example ID plus seed `3407`, and it is stratified by category. Rebuilding unchanged inputs produces byte-identical files and hashes.

Validation data is different from the frozen Stage 4 evaluation benchmark:

- training validation helps select and monitor a training run;
- frozen evaluation provides the final independent base-versus-fine-tuned comparison.

## Tokenization

Stage 5 stores readable messages rather than model-specific token IDs. Tokenization will occur in Stage 6 using Qwen's tokenizer and chat template. The tokenizer converts text into integer token IDs, while the chat template adds the model's expected role markers. Keeping raw messages here makes the dataset inspectable and avoids binding it prematurely to one tokenizer version.

## Provenance and copyright

The seed examples are original project text. No articles, books, earnings-call transcripts, or scraped datasets were copied. Future external records must have a documented source reference and license before review. Provenance metadata remains attached after splitting so questionable records can be traced and removed.

## Files and commands

- `configs/data/training_pipeline.toml`: thresholds, seed, system prompt, and target distribution
- `schemas/training_example.schema.json`: human- and machine-readable schema
- `data/raw/finpulse_seed_v1.jsonl`: reviewed source examples
- `data/train/finpulse_seed_v1.jsonl`: deterministic training split
- `data/validation/finpulse_seed_v1.jsonl`: deterministic validation split
- `data/processed/finpulse_seed_v1.quality.json`: quality report
- `data/processed/finpulse_seed_v1.manifest.json`: source/split hashes and protected evaluation hash
- `scripts/build_stage5_seed.py`: verifies the reviewed seed source
- `scripts/build_training_dataset.py`: reusable pipeline command

Verify and rebuild:

```powershell
.\.venv\Scripts\python.exe scripts\build_stage5_seed.py
.\.venv\Scripts\python.exe scripts\build_training_dataset.py data\raw\finpulse_seed_v1.jsonl
```

The generated splits were also loaded with Hugging Face `datasets.load_dataset("json", ...)`, confirming compatibility with the later TRL workflow.

## Next boundary

Stage 6 will tokenize these messages and configure the first QLoRA/Unsloth training run. Before real training, the seed corpus should be expanded substantially with independently authored and reviewed examples. Stage 6 must start conservatively on the 6 GB GPU and must not use the frozen evaluation data.
