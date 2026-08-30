"""Acquire and build the pinned, family-disjoint Stage 5C finance dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from finpulse_llm.data.config import load_data_config
from finpulse_llm.data.leakage import EvaluationLeakageIndex
from finpulse_llm.data.pipeline import file_sha256, load_jsonl
from finpulse_llm.data.text import text_fingerprint
from finpulse_llm.data.validation import validate_example
from finpulse_llm.evaluation.stage4 import verify_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/data/stage5c_sources.toml"
VALIDATION_CONFIG = PROJECT_ROOT / "configs/data/training_pipeline_stage5b.toml"
RAW_OUTPUT = PROJECT_ROOT / "data/raw/finpulse_stage5c_v1.jsonl"
TRAIN_OUTPUT = PROJECT_ROOT / "data/train/finpulse_stage5c_v1.jsonl"
VALIDATION_OUTPUT = PROJECT_ROOT / "data/validation/finpulse_stage5c_v1.jsonl"
DEVELOPMENT_OUTPUT = PROJECT_ROOT / "data/development/finpulse_stage5c_v1.jsonl"
QUALITY_OUTPUT = PROJECT_ROOT / "data/processed/finpulse_stage5c_v1.quality.json"
MANIFEST_OUTPUT = PROJECT_ROOT / "data/processed/finpulse_stage5c_v1.manifest.json"
SPLITS = ("train", "validation", "development")
CATEGORY_PREFIX = {
    "technical_analysis": "ta",
    "crypto_derivatives": "cd",
    "stock_fundamentals": "sf",
    "macroeconomics": "ma",
    "risk_management": "rm",
    "scenario_analysis": "sa",
    "terminology_misc": "tm",
}


@dataclass(frozen=True)
class Candidate:
    record: dict[str, Any]
    source: str
    source_id: str
    family: str
    split: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only the repository-local Hugging Face cache.",
    )
    return parser.parse_args()


def _stable_key(seed: int, *values: object) -> str:
    text = ":".join((str(seed), *(str(value) for value in values)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _response_length(text: str) -> str:
    length = len(text)
    if length <= 280:
        return "short"
    if length <= 700:
        return "medium"
    if length <= 1400:
        return "long"
    raise ValueError("assistant response exceeds Stage 5 response-length contract")


def _category_for_topic(topic: str) -> str:
    if topic in {"Derivatives", "Options", "Financial Markets and Products"}:
        return "crypto_derivatives"
    if topic in {
        "Financial Statement Analysis",
        "Equity Investments",
        "Equity Valuation",
        "Corporate Issuers",
    }:
        return "stock_fundamentals"
    if topic in {"Economics", "Fixed Income"}:
        return "macroeconomics"
    if topic in {
        "Risk Management",
        "Market Risk",
        "Credit Risk",
        "Operational Risk",
        "Liquidity Risk",
        "Valuation and Risk Models",
    }:
        return "risk_management"
    if topic in {
        "Portfolio Management",
        "Performance Evaluation",
        "Private Wealth",
        "Institutional Portfolio",
        "Asset Allocation",
        "Alternative Investments",
    }:
        return "scenario_analysis"
    return "terminology_misc"


def _finqa_category(question: str, evidence: str) -> str:
    text = f"{question} {evidence}".casefold()
    if any(word in text for word in ("hedg", "risk", "default", "exposure")):
        return "risk_management"
    if any(word in text for word in ("exchange rate", "inflation", "interest rate", "libor")):
        return "macroeconomics"
    if any(word in text for word in ("scenario", "sensitivity", "assumption")):
        return "scenario_analysis"
    return "stock_fundamentals"


def _metadata(
    *,
    category: str,
    subtopics: Iterable[str],
    difficulty: str,
    task_type: str,
    response_format: str,
    response_length: str,
    source_reference: str,
    license_name: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "subtopics": list(subtopics),
        "difficulty": difficulty,
        "task_type": task_type,
        "response_format": response_format,
        "response_length": response_length,
        "source": {
            "type": "licensed",
            "reference": source_reference,
            "license": license_name,
        },
        "review": {
            "status": "reviewed",
            "reviewer": "stage5c-deterministic-source-and-quality-audit",
        },
    }


def _cosimo_record(
    row: dict[str, Any], source: dict[str, Any], system_prompt: str, seed: int
) -> dict[str, Any]:
    category = _category_for_topic(str(row["topic"]))
    trace = str(row["reasoning_trace"]).strip().replace("FINAL:", "Answer:")
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    task_type = "calculation"
    if row["question_type"] == "MCQ":
        options = [answer, *(str(value) for value in row["distractors"])]
        options = sorted(options, key=lambda value: _stable_key(seed, row["id"], value))
        labels = "ABCD"
        option_text = "\n".join(
            f"{label}. {value}" for label, value in zip(labels, options, strict=True)
        )
        correct = labels[options.index(answer)]
        user = (
            f"{question}\n\nOptions:\n{option_text}\n"
            "Explain briefly, then end with FINAL: <letter>."
        )
        assistant = f"{trace}\nFINAL: {correct}"
        task_type = "multiple_choice"
    else:
        user = f"{question}\nShow the calculation and end with FINAL: <answer>."
        assistant = f"{trace}\nFINAL: {answer}"
    return {
        "id": "fp_tm_0001",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": _metadata(
            category=category,
            subtopics=(
                str(row["topic"]).casefold().replace(" ", "_"),
                str(row["subtopic"]).casefold().replace(" ", "_"),
            ),
            difficulty=(
                "advanced"
                if "Hard" in row["difficulty"] or "L3" in row["difficulty"]
                else "intermediate"
                if "Medium" in row["difficulty"] or "L2" in row["difficulty"]
                else "beginner"
            ),
            task_type=task_type,
            response_format="final_marker",
            response_length=_response_length(assistant),
            source_reference=(
                f"{source['repo_id']}@{source['revision']}:{row['id']}"
            ),
            license_name=source["license"],
        ),
    }


def _render_finqa_steps(row: dict[str, Any]) -> str:
    steps = row.get("steps") or {}
    operations = steps.get("op") or []
    arg1 = steps.get("arg1") or []
    arg2 = steps.get("arg2") or []
    results = steps.get("res") or []
    lines: list[str] = []
    for index, (operation, left, right, result) in enumerate(
        zip(operations, arg1, arg2, results, strict=False), start=1
    ):
        name = re.sub(r"\d+(?:-\d+)?$", "", str(operation)).replace("_", " ")
        left_text = str(left).replace("#", "step ")
        right_text = str(right).replace("#", "step ")
        lines.append(f"Step {index}: {name}({left_text}, {right_text}) = {result}.")
    if not lines and row.get("program"):
        lines.append(f"Calculation: {row['program']}.")
    return "\n".join(lines)


def _finqa_record(
    row: dict[str, Any], source: dict[str, Any], system_prompt: str
) -> dict[str, Any] | None:
    try:
        gold = json.loads(row["gold_inds"])
    except (TypeError, json.JSONDecodeError):
        return None
    evidence_lines = [str(value).strip() for value in gold.values() if str(value).strip()]
    if not evidence_lines:
        return None
    evidence = "\n".join(f"- {value}" for value in evidence_lines)
    question = str(row["question"]).strip()
    final_answer = str(row.get("exe_ans") or row.get("answer") or "").strip()
    if not final_answer:
        return None
    user = (
        "Use only the supplied financial-report evidence.\n"
        f"Evidence:\n{evidence}\nQuestion: {question}\n"
        "Show concise steps and end with FINAL: <answer>."
    )
    steps = _render_finqa_steps(row)
    assistant = f"{steps}\nFINAL: {final_answer}" if steps else f"FINAL: {final_answer}"
    category = _finqa_category(question, evidence)
    try:
        length_label = _response_length(assistant)
    except ValueError:
        return None
    return {
        "id": "fp_sf_0001",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": _metadata(
            category=category,
            subtopics=("financial_report_reasoning", "numerical_reasoning"),
            difficulty="intermediate",
            task_type="calculation",
            response_format="final_marker",
            response_length=length_label,
            source_reference=(
                f"{source['repo_id']}@{source['revision']}:{row['id']}"
            ),
            license_name=source["license"],
        ),
    }


def _token_length(record: dict[str, Any], tokenizer: Any) -> int:
    rendered = tokenizer.apply_chat_template(
        record["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def _eligible(
    record: dict[str, Any],
    validation_config: Any,
    leakage: EvaluationLeakageIndex,
    tokenizer: Any,
    max_tokens: int,
) -> tuple[bool, str | None, int]:
    validation = validate_example(record, validation_config)
    if validation.errors:
        return False, "; ".join(validation.errors), 0
    prompt = record["messages"][1]["content"]
    match = leakage.find_match(prompt, validation_config.evaluation_leakage_threshold)
    if match:
        return False, f"protected evaluation overlap: {match.source}/{match.case_id}", 0
    tokens = _token_length(record, tokenizer)
    if tokens > max_tokens:
        return False, f"token length {tokens} exceeds {max_tokens}", tokens
    return True, None, tokens


def _conversation_fingerprint(record: dict[str, Any]) -> str:
    prompt = record["messages"][1]["content"]
    answer = record["messages"][2]["content"]
    return text_fingerprint(f"{prompt}\n{answer}")


def _family_split(
    family_categories: dict[str, str], seed: int
) -> dict[str, str]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for family, category in family_categories.items():
        by_category[category].append(family)
    assignments: dict[str, str] = {}
    for category, families in sorted(by_category.items()):
        ranked = sorted(families, key=lambda value: _stable_key(seed, category, value))
        holdout = max(1, round(len(ranked) * 0.15)) if len(ranked) >= 3 else 0
        for family in ranked[:holdout]:
            assignments[family] = "validation"
        for family in ranked[holdout : holdout * 2]:
            assignments[family] = "development"
        for family in ranked[holdout * 2 :]:
            assignments[family] = "train"
    return assignments


def _quotas(total: int, families: list[str]) -> dict[str, int]:
    if not families:
        raise ValueError("Cannot allocate a target without source families")
    base, remainder = divmod(total, len(families))
    return {
        family: base + (index < remainder)
        for index, family in enumerate(sorted(families))
    }


def _load_sources(config: dict[str, Any], offline: bool) -> tuple[Any, Any]:
    from datasets import load_dataset

    cache = PROJECT_ROOT / "models/huggingface/datasets"
    cosimo = config["sources"]["cosimo"]
    finqa = config["sources"]["finqa"]
    common = {"cache_dir": str(cache), "download_mode": "reuse_dataset_if_exists"}
    if offline:
        common["verification_mode"] = "no_checks"
    cosimo_data = load_dataset(
        cosimo["repo_id"], cosimo["config"], revision=cosimo["revision"], **common
    )
    finqa_data = load_dataset(
        finqa["repo_id"], finqa["config"], revision=finqa["revision"], **common
    )
    return cosimo_data, finqa_data


def _select_cosimo(
    data: Any,
    source: dict[str, Any],
    system_prompt: str,
    seed: int,
    validation_config: Any,
    leakage: EvaluationLeakageIndex,
    tokenizer: Any,
    max_tokens: int,
    rejection_counts: Counter[str],
    seen_conversations: set[str],
) -> list[Candidate]:
    refs: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    family_categories: dict[str, str] = {}
    for source_split, dataset in data.items():
        columns = zip(
            dataset["metadata"], dataset["id"], dataset["topic"], strict=True
        )
        for index, (metadata, source_id, topic) in enumerate(columns):
            family = str(metadata["generator"])
            refs[family].append((source_split, index, str(source_id)))
            family_categories[family] = _category_for_topic(str(topic))
    assignments = _family_split(family_categories, seed)
    selected: list[Candidate] = []

    def try_add(
        split: str, family: str, source_split: str, index: int, source_id: str
    ) -> bool:
        row = data[source_split][index]
        raw_match = leakage.find_match(
            str(row["question"]),
            validation_config.evaluation_leakage_threshold,
        )
        if raw_match:
            rejection_counts["cosimo_raw_question_evaluation_overlap"] += 1
            return False
        if not row["verified"] or not row["verification"]["answer_matches_recomputation"]:
            rejection_counts["cosimo_not_verified"] += 1
            return False
        try:
            record = _cosimo_record(row, source, system_prompt, seed)
        except ValueError as exc:
            rejection_counts[f"cosimo_{exc}"] += 1
            return False
        ok, reason, _tokens = _eligible(
            record, validation_config, leakage, tokenizer, max_tokens
        )
        if not ok:
            rejection_counts[f"cosimo_{reason}"] += 1
            return False
        conversation = _conversation_fingerprint(record)
        if conversation in seen_conversations:
            rejection_counts["cosimo_duplicate_conversation"] += 1
            return False
        selected.append(Candidate(record, "cosimo", source_id, family, split))
        seen_conversations.add(conversation)
        return True

    for split in SPLITS:
        families = [family for family, value in assignments.items() if value == split]
        quotas = _quotas(int(source[split]), families)
        attempted: set[tuple[str, int]] = set()
        for family, quota in quotas.items():
            ranked = sorted(
                refs[family], key=lambda value: _stable_key(seed, "cosimo", value[2])
            )
            accepted = 0
            for source_split, index, source_id in ranked:
                attempted.add((source_split, index))
                if try_add(split, family, source_split, index, source_id):
                    accepted += 1
                if accepted == quota:
                    break
            if accepted != quota:
                rejection_counts["cosimo_family_quota_shortfall_redistributed"] += (
                    quota - accepted
                )

        selected_in_split = sum(item.split == split for item in selected)
        deficit = int(source[split]) - selected_in_split
        if deficit:
            fallback = sorted(
                (
                    (family, source_split, index, source_id)
                    for family in families
                    for source_split, index, source_id in refs[family]
                    if (source_split, index) not in attempted
                ),
                key=lambda value: _stable_key(seed, "cosimo-fallback", value[3]),
            )
            for family, source_split, index, source_id in fallback:
                if try_add(split, family, source_split, index, source_id):
                    deficit -= 1
                if not deficit:
                    break
        if deficit:
            raise RuntimeError(f"Cosimo {split} remains short by {deficit} eligible rows")
    return selected


def _select_finqa(
    data: Any,
    source: dict[str, Any],
    system_prompt: str,
    seed: int,
    validation_config: Any,
    leakage: EvaluationLeakageIndex,
    tokenizer: Any,
    max_tokens: int,
    rejection_counts: Counter[str],
    seen_conversations: set[str],
) -> list[Candidate]:
    source_splits = {"train": "train", "validation": "validation", "development": "test"}
    selected: list[Candidate] = []
    for split, source_split in source_splits.items():
        dataset = data[source_split]
        ranked = sorted(
            range(len(dataset)),
            key=lambda index: _stable_key(seed, "finqa", dataset[index]["id"]),
        )
        seen_documents: set[str] = set()
        target = int(source[split])
        selected_count = 0
        for index in ranked:
            row = dataset[index]
            raw_match = leakage.find_match(
                str(row["question"]),
                validation_config.evaluation_leakage_threshold,
            )
            if raw_match:
                rejection_counts["finqa_raw_question_evaluation_overlap"] += 1
                continue
            family = str(row["filename"])
            if family in seen_documents:
                continue
            record = _finqa_record(row, source, system_prompt)
            if record is None:
                rejection_counts["finqa_missing_evidence_or_answer"] += 1
                continue
            ok, reason, _tokens = _eligible(
                record, validation_config, leakage, tokenizer, max_tokens
            )
            if not ok:
                rejection_counts[f"finqa_{reason}"] += 1
                continue
            conversation = _conversation_fingerprint(record)
            if conversation in seen_conversations:
                rejection_counts["finqa_duplicate_conversation"] += 1
                continue
            selected.append(Candidate(record, "finqa", str(row["id"]), family, split))
            seen_conversations.add(conversation)
            seen_documents.add(family)
            selected_count += 1
            if selected_count == target:
                break
        observed = sum(item.source == "finqa" and item.split == split for item in selected)
        if observed != target:
            raise RuntimeError(f"FinQA {source_split} supplied {observed}/{target} eligible rows")
    return selected


def _select_project_behavior(
    source: dict[str, Any],
    seed: int,
    system_prompt: str,
    seen_conversations: set[str],
    rejection_counts: Counter[str],
) -> list[Candidate]:
    rows = load_jsonl([PROJECT_ROOT / source["path"]])
    priority = []
    for row in rows:
        metadata = row["metadata"]
        rank = (
            0
            if metadata["task_type"] in {"refusal", "instruction_following"}
            else 1
            if metadata["response_format"] == "json_only"
            else 2
            if metadata["task_type"] in {"analysis", "factual"}
            else 3
        )
        priority.append((rank, _stable_key(seed, "behavior", row["id"]), row))
    output: list[Candidate] = []
    for _rank, _key, row in sorted(priority, key=lambda item: (item[0], item[1])):
        copied = json.loads(json.dumps(row))
        copied["messages"][0]["content"] = system_prompt
        conversation = _conversation_fingerprint(copied)
        if conversation in seen_conversations:
            rejection_counts["project_behavior_duplicate_conversation"] += 1
            continue
        output.append(
            Candidate(
                copied,
                "project_behavior",
                str(row["id"]),
                f"training-only:{row['metadata']['task_type']}:{row['metadata']['subtopics'][0]}",
                "train",
            )
        )
        seen_conversations.add(conversation)
        if len(output) == int(source["train"]):
            break
    if len(output) != int(source["train"]):
        raise RuntimeError(
            f"Project behavior supplied {len(output)}/{source['train']} unique rows"
        )
    return output


def _assign_ids(candidates: list[Candidate]) -> list[Candidate]:
    counters: Counter[str] = Counter()
    output: list[Candidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (SPLITS.index(item.split), item.source, item.family, item.source_id),
    ):
        category = candidate.record["metadata"]["category"]
        counters[category] += 1
        identifier = f"fp_{CATEGORY_PREFIX[category]}_{counters[category]:04d}"
        output.append(replace(candidate, record={**candidate.record, "id": identifier}))
    return output


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _counts(candidates: list[Candidate], field: str) -> dict[str, int]:
    return dict(
        sorted(Counter(item.record["metadata"][field] for item in candidates).items())
    )


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models/huggingface"))
    os.environ.setdefault(
        "HF_DATASETS_CACHE", str(PROJECT_ROOT / "models/huggingface/datasets")
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["tokenizer_id"],
        revision=config["model"]["revision"],
        trust_remote_code=False,
        local_files_only=True,
    )
    validation_config = load_data_config(VALIDATION_CONFIG)
    stage4_path = PROJECT_ROOT / "data/eval/finpulse_eval_v1.jsonl"
    stage4_manifest = verify_manifest(
        stage4_path, PROJECT_ROOT / "data/eval/finpulse_eval_v1.manifest.json"
    )
    leakage = EvaluationLeakageIndex.from_files(
        PROJECT_ROOT / "benchmarks/stage3_base_models.json", stage4_path
    )
    cosimo_data, finqa_data = _load_sources(config, args.offline)
    rejection_counts: Counter[str] = Counter()
    seen_conversations: set[str] = set()
    candidates = _select_cosimo(
        cosimo_data,
        config["sources"]["cosimo"],
        config["system_prompt"],
        config["seed"],
        validation_config,
        leakage,
        tokenizer,
        config["max_sequence_length"],
        rejection_counts,
        seen_conversations,
    )
    candidates.extend(
        _select_finqa(
            finqa_data,
            config["sources"]["finqa"],
            config["system_prompt"],
            config["seed"],
            validation_config,
            leakage,
            tokenizer,
            config["max_sequence_length"],
            rejection_counts,
            seen_conversations,
        )
    )
    behavior = _select_project_behavior(
        config["sources"]["project_behavior"],
        config["seed"],
        config["system_prompt"],
        seen_conversations,
        rejection_counts,
    )
    for candidate in behavior:
        ok, reason, _tokens = _eligible(
            candidate.record,
            validation_config,
            leakage,
            tokenizer,
            config["max_sequence_length"],
        )
        if not ok:
            raise ValueError(f"Selected behavior row {candidate.source_id} failed: {reason}")
    candidates.extend(behavior)
    candidates = _assign_ids(candidates)

    expected = {
        split: sum(int(source.get(split, 0)) for source in config["sources"].values())
        for split in SPLITS
    }
    observed = Counter(item.split for item in candidates)
    if any(observed[split] != expected[split] for split in SPLITS):
        raise RuntimeError(f"Split counts differ: observed={observed}, expected={expected}")

    ids: set[str] = set()
    conversations: set[str] = set()
    token_lengths: dict[str, list[int]] = defaultdict(list)
    for candidate in candidates:
        result = validate_example(candidate.record, validation_config)
        if result.errors:
            raise ValueError(f"Final row {candidate.record['id']} failed: {result.errors}")
        if candidate.record["id"] in ids:
            raise ValueError(f"Duplicate final ID: {candidate.record['id']}")
        conversation = _conversation_fingerprint(candidate.record)
        if conversation in conversations:
            raise ValueError(f"Duplicate final conversation: {candidate.record['id']}")
        ids.add(candidate.record["id"])
        conversations.add(conversation)
        token_lengths[candidate.split].append(_token_length(candidate.record, tokenizer))

    external_family_sets = {
        split: {
            f"{item.source}:{item.family}"
            for item in candidates
            if item.split == split and item.source != "project_behavior"
        }
        for split in SPLITS
    }
    family_overlaps = {
        f"{left}_{right}": sorted(external_family_sets[left] & external_family_sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    if any(family_overlaps.values()):
        raise ValueError(f"Source families cross splits: {family_overlaps}")

    split_candidates = {
        split: [item for item in candidates if item.split == split] for split in SPLITS
    }
    _write_jsonl(RAW_OUTPUT, (item.record for item in candidates))
    _write_jsonl(TRAIN_OUTPUT, (item.record for item in split_candidates["train"]))
    _write_jsonl(
        VALIDATION_OUTPUT, (item.record for item in split_candidates["validation"])
    )
    _write_jsonl(
        DEVELOPMENT_OUTPUT, (item.record for item in split_candidates["development"])
    )

    def token_stats(values: list[int]) -> dict[str, float | int]:
        return {
            "minimum": min(values),
            "mean": round(sum(values) / len(values), 2),
            "maximum": max(values),
        }

    quality = {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "status": "complete",
        "records": len(candidates),
        "split_counts": dict(observed),
        "source_counts": dict(sorted(Counter(item.source for item in candidates).items())),
        "category_counts": _counts(candidates, "category"),
        "task_type_counts": _counts(candidates, "task_type"),
        "response_format_counts": _counts(candidates, "response_format"),
        "response_length_counts": _counts(candidates, "response_length"),
        "source_family_counts": {
            split: len(external_family_sets[split]) for split in SPLITS
        },
        "source_family_overlaps": family_overlaps,
        "token_counts": {split: token_stats(token_lengths[split]) for split in SPLITS},
        "candidate_rejections_before_quota_fill": dict(sorted(rejection_counts.items())),
        "audited_exclusions": config.get("audited_exclusions", {}),
        "limitations": [
            "Cosimo answers are code-verified but originate from 71 synthetic templates.",
            (
                "FinQA examples are transformed to use only annotated gold evidence that "
                "fits the local context window."
            ),
            (
                "Project behavior examples are training-only; the external holdouts screen "
                "transfer rather than every custom output contract."
            ),
        ],
    }
    _write_json(QUALITY_OUTPUT, quality)
    selected_source_ids = {
        split: hashlib.sha256(
            "\n".join(
                sorted(f"{item.source}:{item.source_id}" for item in split_candidates[split])
            ).encode("utf-8")
        ).hexdigest()
        for split in SPLITS
    }
    manifest = {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "seed": config["seed"],
        "source_config": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "source_config_sha256": file_sha256(config_path),
        "sources": config["sources"],
        "audited_exclusions": config.get("audited_exclusions", {}),
        "selected_source_id_sha256": selected_source_ids,
        "raw_file": RAW_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
        "raw_sha256": file_sha256(RAW_OUTPUT),
        "train_file": TRAIN_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
        "train_sha256": file_sha256(TRAIN_OUTPUT),
        "validation_file": VALIDATION_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
        "validation_sha256": file_sha256(VALIDATION_OUTPUT),
        "development_file": DEVELOPMENT_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
        "development_sha256": file_sha256(DEVELOPMENT_OUTPUT),
        "quality_report": QUALITY_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
        "protected_stage4_sha256": stage4_manifest["dataset_sha256"],
        "leakage_policy": (
            "Reject exact or fuzzy matches against Stage 3 and frozen Stage 4 prompts before "
            "selection."
        ),
        "split_policy": (
            "Cosimo generator families and FinQA report documents are disjoint across splits; "
            "project behavior examples are train-only."
        ),
    }
    _write_json(MANIFEST_OUTPUT, manifest)
    print(json.dumps(quality, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
