"""Schema and behavioral quality checks for financial instruction examples."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from finpulse_llm.data.config import DataPipelineConfig
from finpulse_llm.data.text import normalize_text

ID_PATTERN = re.compile(r"^fp_[a-z]{2,4}_\d{4}$")
EXPECTED_ROLES = ("system", "user", "assistant")
DIFFICULTIES = {"beginner", "intermediate", "advanced"}
SOURCE_TYPES = {"original", "synthetic", "licensed"}
LIVE_REQUEST = re.compile(r"\b(current|right now|today|latest|live|real[- ]time)\b", re.I)
UNSUPPORTED_LIVE_VALUE = re.compile(
    r"(?:\$\s*\d+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?\b)"
)
REFUSAL_MARKERS = (
    "no access",
    "do not have access",
    "don't have access",
    "cannot access",
    "can't access",
    "cannot provide",
    "can't provide",
    "data is provided",
    "provide the data",
)
SECRET_PATTERN = re.compile(
    r"(?:api[_-]?key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{12,}", re.I
)


@dataclass(frozen=True)
class ValidationResult:
    example: dict[str, Any] | None
    errors: tuple[str, ...]


def _normalized_copy(raw: dict[str, Any]) -> dict[str, Any]:
    example = dict(raw)
    example["id"] = normalize_text(str(raw.get("id", "")))
    messages = raw.get("messages", [])
    example["messages"] = [
        {
            "role": normalize_text(str(item.get("role", ""))),
            "content": normalize_text(str(item.get("content", ""))),
        }
        for item in messages
        if isinstance(item, dict)
    ]
    metadata = dict(raw.get("metadata", {})) if isinstance(raw.get("metadata"), dict) else {}
    metadata["category"] = normalize_text(str(metadata.get("category", "")))
    subtopics = metadata.get("subtopics", [])
    metadata["subtopics"] = (
        [normalize_text(str(value)) for value in subtopics]
        if isinstance(subtopics, list)
        else []
    )
    metadata["difficulty"] = normalize_text(str(metadata.get("difficulty", "")))
    source = metadata.get("source", {})
    review = metadata.get("review", {})
    metadata["source"] = dict(source) if isinstance(source, dict) else {}
    metadata["review"] = dict(review) if isinstance(review, dict) else {}
    example["metadata"] = metadata
    return example


def validate_example(raw: Any, config: DataPipelineConfig) -> ValidationResult:
    """Normalize one record and return every actionable validation error."""

    if not isinstance(raw, dict):
        return ValidationResult(None, ("record must be a JSON object",))
    example = _normalized_copy(raw)
    errors: list[str] = []
    if set(raw) != {"id", "messages", "metadata"}:
        errors.append("top-level keys must be exactly id, messages, metadata")
    raw_messages = raw.get("messages", [])
    if isinstance(raw_messages, list) and any(
        not isinstance(item, dict) or set(item) != {"role", "content"}
        for item in raw_messages
    ):
        errors.append("each message must contain exactly role and content")
    raw_metadata = raw.get("metadata", {})
    if not isinstance(raw_metadata, dict) or set(raw_metadata) != {
        "category",
        "subtopics",
        "difficulty",
        "source",
        "review",
    }:
        errors.append("metadata keys are invalid")
    elif (
        not isinstance(raw_metadata.get("source"), dict)
        or set(raw_metadata["source"]) != {"type", "reference", "license"}
    ):
        errors.append("source keys are invalid")
    elif (
        not isinstance(raw_metadata.get("review"), dict)
        or set(raw_metadata["review"]) != {"status", "reviewer"}
    ):
        errors.append("review keys are invalid")
    if not ID_PATTERN.fullmatch(example["id"]):
        errors.append("id must match fp_<category>_<four digits>")

    messages = example["messages"]
    if len(messages) != 3:
        errors.append("messages must contain exactly system, user, assistant")
    elif tuple(item["role"] for item in messages) != EXPECTED_ROLES:
        errors.append("message roles must be exactly system, user, assistant")
    if len(messages) == 3:
        system, user, assistant = (item["content"] for item in messages)
        if system != config.system_prompt:
            errors.append("system message does not match configured training prompt")
        if not config.min_user_characters <= len(user) <= config.max_message_characters:
            errors.append("user message length is outside configured bounds")
        if not config.min_assistant_characters <= len(assistant) <= config.max_message_characters:
            errors.append("assistant message length is outside configured bounds")
        if LIVE_REQUEST.search(user):
            if not any(marker in assistant.casefold() for marker in REFUSAL_MARKERS):
                errors.append("live-data request lacks an explicit access limitation")
            if UNSUPPORTED_LIVE_VALUE.search(assistant):
                errors.append("live-data response contains an unsupported exact value")
        if SECRET_PATTERN.search(user) or SECRET_PATTERN.search(assistant):
            errors.append("possible secret or credential detected")

    metadata = example["metadata"]
    category = metadata.get("category")
    if category not in config.expected_distribution:
        errors.append("metadata category is not configured")
    if not isinstance(raw_metadata.get("subtopics"), list):
        errors.append("metadata subtopics must be an array")
    elif not metadata.get("subtopics") or any(not item for item in metadata.get("subtopics", [])):
        errors.append("metadata subtopics must contain non-empty values")
    if metadata.get("difficulty") not in DIFFICULTIES:
        errors.append("metadata difficulty is invalid")
    source = metadata.get("source", {})
    if source.get("type") not in SOURCE_TYPES:
        errors.append("source type must be original, synthetic, or licensed")
    if not source.get("reference") or not source.get("license"):
        errors.append("source reference and license are required")
    review = metadata.get("review", {})
    if review.get("status") != "reviewed" or not review.get("reviewer"):
        errors.append("only explicitly reviewed examples may enter a split")
    return ValidationResult(example, tuple(errors))
