"""Loading, validation, fingerprinting, and scoring for the frozen Stage 4 benchmark."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finpulse_llm.evaluation.stage3 import CheckResult, score_check

EXPECTED_CATEGORIES = {
    "technical_analysis",
    "crypto_derivatives",
    "stock_fundamentals",
    "macroeconomics",
    "risk_management",
    "financial_calculations",
    "scenario_analysis",
    "contradictory_signals",
    "hallucination_traps",
    "structured_output",
}


@dataclass(frozen=True)
class FrozenEvalCase:
    """One original evaluation prompt and its deterministic scoring rubric."""

    id: str
    category: str
    prompt: str
    checks: tuple[dict[str, Any], ...]
    tags: tuple[str, ...]
    provenance: dict[str, str]
    split: str
    exclude_from_training: bool


def sha256_file(path: str | Path) -> str:
    """Return a stable content fingerprint used to detect benchmark edits."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_fingerprint(prompt: str) -> str:
    """Fingerprint normalized prompt text for future leakage checks."""

    normalized = " ".join(prompt.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_frozen_benchmark(path: str | Path) -> tuple[FrozenEvalCase, ...]:
    """Load JSONL cases and enforce the frozen-evaluation invariants."""

    cases: list[FrozenEvalCase] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            cases.append(
                FrozenEvalCase(
                    id=str(item["id"]),
                    category=str(item["category"]),
                    prompt=str(item["prompt"]),
                    checks=tuple(item["checks"]),
                    tags=tuple(str(tag) for tag in item["tags"]),
                    provenance=dict(item["provenance"]),
                    split=str(item["split"]),
                    exclude_from_training=bool(item["exclude_from_training"]),
                )
            )
    _validate_cases(cases)
    return tuple(cases)


def _validate_cases(cases: list[FrozenEvalCase]) -> None:
    if not 150 <= len(cases) <= 250:
        raise ValueError(f"Frozen benchmark must contain 150-250 cases, found {len(cases)}")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Frozen benchmark case IDs must be unique")
    fingerprints = [prompt_fingerprint(case.prompt) for case in cases]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("Frozen benchmark prompts must be unique")
    categories = {case.category for case in cases}
    if categories != EXPECTED_CATEGORIES:
        raise ValueError(f"Unexpected category set: {sorted(categories)}")
    for case in cases:
        if case.split != "eval" or not case.exclude_from_training:
            raise ValueError(f"Case {case.id} is not protected from training")
        if not case.checks:
            raise ValueError(f"Case {case.id} has no scoring checks")
        if case.provenance.get("kind") not in {"original", "synthetic"}:
            raise ValueError(f"Case {case.id} has invalid provenance")


def verify_manifest(dataset_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    """Verify that the benchmark bytes and prompt fingerprints match its frozen manifest."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    cases = load_frozen_benchmark(dataset_path)
    actual_hash = sha256_file(dataset_path)
    if actual_hash != manifest["dataset_sha256"]:
        raise ValueError("Frozen benchmark SHA-256 does not match its manifest")
    actual_fingerprints = {case.id: prompt_fingerprint(case.prompt) for case in cases}
    if actual_fingerprints != manifest["prompt_fingerprints"]:
        raise ValueError("Frozen benchmark prompt fingerprints do not match its manifest")
    return manifest


def score_stage4_responses(
    cases: tuple[FrozenEvalCase, ...], responses: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score a complete ordered response set and aggregate categories and check types."""

    if len(cases) != len(responses):
        raise ValueError(f"Expected {len(cases)} responses, received {len(responses)}")

    scored: list[dict[str, Any]] = []
    category_points: dict[str, list[float]] = defaultdict(list)
    check_type_totals: Counter[str] = Counter()
    check_type_passed: Counter[str] = Counter()
    total_passed = 0
    total_checks = 0

    for case, response in zip(cases, responses, strict=True):
        if response.get("case_id") != case.id:
            raise ValueError(f"Response order mismatch at {case.id}")
        results: tuple[CheckResult, ...] = tuple(
            score_check(str(response["response"]), check) for check in case.checks
        )
        passed = sum(result.passed for result in results)
        score = passed / len(results)
        total_passed += passed
        total_checks += len(results)
        category_points[case.category].append(score)
        for result in results:
            check_type_totals[result.type] += 1
            check_type_passed[result.type] += int(result.passed)
        scored.append(
            {
                **response,
                "category": case.category,
                "score": round(score, 4),
                "checks": [asdict(result) for result in results],
            }
        )

    return {
        "overall_score": round(total_passed / total_checks, 4),
        "checks_passed": total_passed,
        "checks_total": total_checks,
        "category_scores": {
            category: round(sum(values) / len(values), 4)
            for category, values in sorted(category_points.items())
        },
        "check_type_scores": {
            check_type: round(check_type_passed[check_type] / count, 4)
            for check_type, count in sorted(check_type_totals.items())
        },
        "cases": scored,
    }
