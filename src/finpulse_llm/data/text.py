"""Text normalization and similarity primitives used by dataset quality checks."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher


def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace without flattening useful paragraph boundaries."""

    text = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def comparison_text(value: str) -> str:
    """Produce a conservative normalized form for fingerprints and similarity."""

    return re.sub(r"[^a-z0-9%$]+", " ", normalize_text(value).casefold()).strip()


def text_fingerprint(value: str) -> str:
    return hashlib.sha256(comparison_text(value).encode("utf-8")).hexdigest()


def text_similarity(left: str, right: str) -> float:
    """Combine sequence similarity and token-set overlap without embeddings."""

    left_normalized = comparison_text(left)
    right_normalized = comparison_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    sequence_score = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    left_tokens = set(left_normalized.split())
    right_tokens = set(right_normalized.split())
    jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return max(sequence_score, jaccard)


def word_shingles(value: str, width: int = 3) -> set[tuple[str, ...]]:
    """Create word shingles for efficient near-duplicate candidate retrieval."""

    tokens = comparison_text(value).split()
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


class NearDuplicateIndex:
    """Avoid an all-pairs scan by comparing only prompts sharing a word shingle."""

    def __init__(self) -> None:
        self._prompts: list[tuple[str, str]] = []
        self._shingle_to_indices: dict[tuple[str, ...], set[int]] = defaultdict(set)

    def find(self, prompt: str, threshold: float) -> tuple[str, float] | None:
        shingles = word_shingles(prompt)
        candidates: set[int] = set()
        for shingle in shingles:
            candidates.update(self._shingle_to_indices.get(shingle, ()))
        best: tuple[str, float] | None = None
        for index in candidates:
            example_id, prior_prompt = self._prompts[index]
            similarity = text_similarity(prompt, prior_prompt)
            if similarity >= threshold and (best is None or similarity > best[1]):
                best = (example_id, similarity)
        return best

    def add(self, example_id: str, prompt: str) -> None:
        index = len(self._prompts)
        self._prompts.append((example_id, prompt))
        for shingle in word_shingles(prompt):
            self._shingle_to_indices[shingle].add(index)
