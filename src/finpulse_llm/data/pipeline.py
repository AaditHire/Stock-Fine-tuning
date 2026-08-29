"""Deterministic financial instruction-data cleaning and split pipeline."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finpulse_llm.data.config import DataPipelineConfig
from finpulse_llm.data.leakage import EvaluationLeakageIndex
from finpulse_llm.data.text import NearDuplicateIndex, text_fingerprint
from finpulse_llm.data.validation import validate_example


@dataclass(frozen=True)
class Rejection:
    id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PipelineResult:
    accepted: tuple[dict[str, Any], ...]
    train: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    development: tuple[dict[str, Any], ...]
    rejections: tuple[Rejection, ...]


def load_jsonl(paths: list[str | Path]) -> list[Any]:
    """Load records from one or more JSONL sources with useful line errors."""

    records: list[Any] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return records


def _split(
    examples: list[dict[str, Any]], config: DataPipelineConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        groups[example["metadata"]["category"]].append(example)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    development: list[dict[str, Any]] = []
    for _category, group in sorted(groups.items()):
        ranked = sorted(
            group,
            key=lambda item: hashlib.sha256(
                f"{config.seed}:{item['id']}".encode()
            ).hexdigest(),
        )
        validation_count = max(1, round(len(ranked) * config.validation_ratio))
        development_count = (
            max(1, round(len(ranked) * config.development_ratio))
            if config.development_ratio
            else 0
        )
        if len(ranked) == 1:
            validation_count = 0
            development_count = 0
        if validation_count + development_count >= len(ranked):
            raise ValueError(f"Not enough examples in category {_category} for requested splits")
        validation.extend(ranked[:validation_count])
        development.extend(ranked[validation_count : validation_count + development_count])
        train.extend(ranked[validation_count + development_count :])
    return (
        sorted(train, key=lambda item: item["id"]),
        sorted(validation, key=lambda item: item["id"]),
        sorted(development, key=lambda item: item["id"]),
    )


def run_pipeline(
    records: list[Any], config: DataPipelineConfig, leakage: EvaluationLeakageIndex
) -> PipelineResult:
    """Normalize, validate, deduplicate, leakage-check, and split reviewed records."""

    accepted: list[dict[str, Any]] = []
    rejections: list[Rejection] = []
    seen_ids: set[str] = set()
    seen_conversations: set[str] = set()
    near_duplicates = NearDuplicateIndex()

    for position, raw in enumerate(records, start=1):
        validation = validate_example(raw, config)
        example_id = str(raw.get("id", f"record_{position}")) if isinstance(raw, dict) else (
            f"record_{position}"
        )
        reasons = list(validation.errors)
        example = validation.example
        if example is not None and not reasons:
            if example["id"] in seen_ids:
                reasons.append("duplicate id")
            user = example["messages"][1]["content"]
            assistant = example["messages"][2]["content"]
            conversation_hash = text_fingerprint(f"{user}\n{assistant}")
            if conversation_hash in seen_conversations:
                reasons.append("exact duplicate conversation")
            near_match = near_duplicates.find(user, config.near_duplicate_threshold)
            if near_match:
                reasons.append(
                    f"near-duplicate user prompt: {near_match[0]} "
                    f"similarity={near_match[1]:.3f}"
                )
            match = leakage.find_match(user, config.evaluation_leakage_threshold)
            if match:
                reasons.append(
                    f"evaluation leakage: {match.source}/{match.case_id} "
                    f"similarity={match.similarity:.3f}"
                )
        if reasons or example is None:
            rejections.append(Rejection(example_id, tuple(dict.fromkeys(reasons))))
            continue
        accepted.append(example)
        seen_ids.add(example["id"])
        seen_conversations.add(conversation_hash)
        near_duplicates.add(example["id"], user)

    train, validation_split, development = _split(accepted, config)
    return PipelineResult(
        accepted=tuple(accepted),
        train=tuple(train),
        validation=tuple(validation_split),
        development=tuple(development),
        rejections=tuple(rejections),
    )


def write_jsonl(path: str | Path, records: tuple[dict[str, Any], ...]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records)
    output.write_text(text, encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_quality_report(result: PipelineResult, config: DataPipelineConfig) -> dict[str, Any]:
    categories = Counter(item["metadata"]["category"] for item in result.accepted)
    difficulties = Counter(item["metadata"]["difficulty"] for item in result.accepted)
    source_types = Counter(item["metadata"]["source"]["type"] for item in result.accepted)
    task_types = Counter(item["metadata"].get("task_type") for item in result.accepted)
    response_formats = Counter(
        item["metadata"].get("response_format") for item in result.accepted
    )
    response_lengths = Counter(
        item["metadata"].get("response_length") for item in result.accepted
    )
    total = len(result.accepted)
    actual_distribution = {
        category: round(categories.get(category, 0) / total, 4) if total else 0.0
        for category in config.expected_distribution
    }
    distribution_delta = {
        category: round(
            actual_distribution[category] - config.expected_distribution[category], 4
        )
        for category in config.expected_distribution
    }

    def distribution_report(
        counts: Counter, expected: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float], bool]:
        actual = {
            name: round(counts.get(name, 0) / total, 4) if total else 0.0
            for name in expected
        }
        delta = {name: round(actual[name] - expected[name], 4) for name in expected}
        within = all(abs(value) <= config.distribution_tolerance for value in delta.values())
        return actual, delta, within

    task_actual, task_delta, task_within = distribution_report(
        task_types, config.expected_task_type_distribution
    )
    format_actual, format_delta, format_within = distribution_report(
        response_formats, config.expected_response_format_distribution
    )
    length_actual, length_delta, length_within = distribution_report(
        response_lengths, config.expected_response_length_distribution
    )
    user_lengths = [len(item["messages"][1]["content"]) for item in result.accepted]
    assistant_lengths = [len(item["messages"][2]["content"]) for item in result.accepted]

    def length_stats(values: list[int]) -> dict[str, float | int]:
        return {
            "minimum": min(values) if values else 0,
            "maximum": max(values) if values else 0,
            "average": round(sum(values) / len(values), 1) if values else 0.0,
        }

    return {
        "dataset_id": config.dataset_id,
        "input_records": total + len(result.rejections),
        "accepted_records": total,
        "rejected_records": len(result.rejections),
        "train_records": len(result.train),
        "validation_records": len(result.validation),
        "development_records": len(result.development),
        "category_counts": dict(sorted(categories.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "source_type_counts": dict(sorted(source_types.items())),
        "task_type_counts": {
            str(key): task_types[key]
            for key in sorted(key for key in task_types if key is not None)
        },
        "response_format_counts": {
            str(key): response_formats[key]
            for key in sorted(key for key in response_formats if key is not None)
        },
        "response_length_counts": {
            str(key): response_lengths[key]
            for key in sorted(key for key in response_lengths if key is not None)
        },
        "message_character_stats": {
            "user": length_stats(user_lengths),
            "assistant": length_stats(assistant_lengths),
        },
        "actual_distribution": actual_distribution,
        "expected_distribution": config.expected_distribution,
        "distribution_delta": distribution_delta,
        "distribution_within_tolerance": all(
            abs(delta) <= config.distribution_tolerance for delta in distribution_delta.values()
        ),
        "distribution_tolerance": config.distribution_tolerance,
        "task_type_distribution": {
            "actual": task_actual,
            "expected": config.expected_task_type_distribution,
            "delta": task_delta,
            "within_tolerance": task_within,
        },
        "response_format_distribution": {
            "actual": format_actual,
            "expected": config.expected_response_format_distribution,
            "delta": format_delta,
            "within_tolerance": format_within,
        },
        "response_length_distribution": {
            "actual": length_actual,
            "expected": config.expected_response_length_distribution,
            "delta": length_delta,
            "within_tolerance": length_within,
        },
        "all_distributions_within_tolerance": all(
            (
                all(
                    abs(delta) <= config.distribution_tolerance
                    for delta in distribution_delta.values()
                ),
                task_within,
                format_within,
                length_within,
            )
        ),
        "rejections": [asdict(item) for item in result.rejections],
    }
