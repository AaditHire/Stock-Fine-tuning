"""Exact and fuzzy evaluation-leakage detection without embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from finpulse_llm.data.text import text_fingerprint, text_similarity


@dataclass(frozen=True)
class LeakageMatch:
    source: str
    case_id: str
    similarity: float


class EvaluationLeakageIndex:
    """Hold protected prompts from every pre-training evaluation benchmark."""

    def __init__(self, prompts: list[tuple[str, str, str]]) -> None:
        self._prompts = prompts
        self._fingerprints = {
            text_fingerprint(prompt): (source, case_id) for source, case_id, prompt in prompts
        }

    @classmethod
    def from_files(cls, stage3_path: str | Path, stage4_path: str | Path) -> EvaluationLeakageIndex:
        stage3 = json.loads(Path(stage3_path).read_text(encoding="utf-8"))
        prompts = [
            ("stage3", str(item["id"]), str(item["prompt"])) for item in stage3["cases"]
        ]
        with Path(stage4_path).open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                prompts.append(("stage4", str(item["id"]), str(item["prompt"])))
        return cls(prompts)

    def find_match(self, prompt: str, threshold: float) -> LeakageMatch | None:
        exact = self._fingerprints.get(text_fingerprint(prompt))
        if exact:
            return LeakageMatch(*exact, similarity=1.0)
        best: LeakageMatch | None = None
        for source, case_id, protected_prompt in self._prompts:
            similarity = text_similarity(prompt, protected_prompt)
            if similarity >= threshold and (best is None or similarity > best.similarity):
                best = LeakageMatch(source, case_id, similarity)
        return best
