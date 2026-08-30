"""Build the deterministic balanced Stage 5D view from locked Stage 5C data."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from finpulse_llm.data.config import load_data_config
from finpulse_llm.data.pipeline import file_sha256, load_jsonl
from finpulse_llm.data.validation import validate_example

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/data/stage5d_sampling.toml"
VALIDATION_CONFIG = PROJECT_ROOT / "configs/data/training_pipeline_stage5b.toml"
TRAIN_OUTPUT = PROJECT_ROOT / "data/train/finpulse_stage5d_v1.jsonl"
QUALITY_OUTPUT = PROJECT_ROOT / "data/processed/finpulse_stage5d_v1.quality.json"
MANIFEST_OUTPUT = PROJECT_ROOT / "data/processed/finpulse_stage5d_v1.manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def _source_name(row: dict[str, Any]) -> str:
    reference = row["metadata"]["source"]["reference"]
    if reference.startswith("btech-software/"):
        return "cosimo"
    if reference.startswith("bevaya/"):
        return "finqa"
    return "project_behavior"


def _stable_key(seed: int, *values: str) -> str:
    payload = "\x1f".join((str(seed), *values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select(
    rows: list[dict[str, Any]],
    *,
    source: str,
    task_type: str,
    count: int,
    seed: int,
    category: str | None = None,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if _source_name(row) == source
        and row["metadata"]["task_type"] == task_type
        and (category is None or row["metadata"]["category"] == category)
    ]
    ranked = sorted(
        eligible,
        key=lambda row: _stable_key(seed, source, task_type, category or "all", row["id"]),
    )
    if len(ranked) < count:
        label = f"{source}/{task_type}/{category or 'all'}"
        raise RuntimeError(f"{label} supplied {len(ranked)}/{count} rows")
    return ranked[:count]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    parent_manifest_path = PROJECT_ROOT / config["parent_manifest"]
    if file_sha256(parent_manifest_path) != config["parent_manifest_sha256"]:
        raise ValueError("Stage 5C parent manifest hash does not match Stage 5D config")
    parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    train_input = PROJECT_ROOT / parent["train_file"]
    validation_input = PROJECT_ROOT / parent["validation_file"]
    development_input = PROJECT_ROOT / parent["development_file"]
    if file_sha256(train_input) != parent["train_sha256"]:
        raise ValueError("Stage 5C training file does not match its manifest")
    if file_sha256(validation_input) != parent["validation_sha256"]:
        raise ValueError("Stage 5C validation file does not match its manifest")
    if file_sha256(development_input) != parent["development_sha256"]:
        raise ValueError("Stage 5C development file does not match its manifest")

    rows = load_jsonl([train_input])
    selection = config["selection"]
    selected = _select(
        rows,
        source="project_behavior",
        task_type="analysis",
        count=100,
        seed=config["seed"],
    )
    for task_type, count in (
        ("calculation", 25),
        ("factual", 75),
        ("instruction_following", 50),
        ("multiple_choice", 25),
        ("refusal", 25),
    ):
        selected.extend(
            _select(
                rows,
                source="project_behavior",
                task_type=task_type,
                count=count,
                seed=config["seed"],
            )
        )
    if len([row for row in selected if _source_name(row) == "project_behavior"]) != int(
        selection["project_behavior_total"]
    ):
        raise RuntimeError("Project behavior selection does not match configured total")

    for category, count in selection["cosimo_calculation"].items():
        selected.extend(
            _select(
                rows,
                source="cosimo",
                task_type="calculation",
                category=category,
                count=int(count),
                seed=config["seed"],
            )
        )
    selected.extend(
        _select(
            rows,
            source="cosimo",
            task_type="multiple_choice",
            count=int(selection["cosimo_multiple_choice"]),
            seed=config["seed"],
        )
    )
    for category, count in selection["finqa_calculation"].items():
        selected.extend(
            _select(
                rows,
                source="finqa",
                task_type="calculation",
                category=category,
                count=int(count),
                seed=config["seed"],
            )
        )

    selected = sorted(selected, key=lambda row: _stable_key(config["seed"], "final", row["id"]))
    if len(selected) != 900 or len({row["id"] for row in selected}) != 900:
        raise RuntimeError("Stage 5D must contain exactly 900 unique Stage 5C rows")
    validation_config = load_data_config(VALIDATION_CONFIG)
    failures = {
        row["id"]: validate_example(row, validation_config).errors
        for row in selected
        if validate_example(row, validation_config).errors
    }
    if failures:
        raise ValueError(f"Stage 5D schema failures: {failures}")

    TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    quality = {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "status": "complete",
        "records": len(selected),
        "source_counts": dict(sorted(Counter(_source_name(row) for row in selected).items())),
        "task_type_counts": dict(
            sorted(Counter(row["metadata"]["task_type"] for row in selected).items())
        ),
        "category_counts": dict(
            sorted(Counter(row["metadata"]["category"] for row in selected).items())
        ),
        "response_format_counts": dict(
            sorted(Counter(row["metadata"]["response_format"] for row in selected).items())
        ),
        "response_length_counts": dict(
            sorted(Counter(row["metadata"]["response_length"] for row in selected).items())
        ),
        "calculation_share": round(
            sum(row["metadata"]["task_type"] == "calculation" for row in selected)
            / len(selected),
            4,
        ),
        "policy": (
            "Retain every scarce project-behavior example; cap and category-balance "
            "external calculations; retain 100 external multiple-choice examples."
        ),
        "holdouts": {
            "validation": parent["validation_file"],
            "development": parent["development_file"],
            "unchanged": True,
        },
    }
    _write_json(QUALITY_OUTPUT, quality)
    manifest = {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "seed": config["seed"],
        "sampling_config": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "sampling_config_sha256": file_sha256(config_path),
        "parent_manifest": config["parent_manifest"],
        "parent_manifest_sha256": file_sha256(parent_manifest_path),
        "train_file": TRAIN_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
        "train_sha256": file_sha256(TRAIN_OUTPUT),
        "validation_file": parent["validation_file"],
        "validation_sha256": parent["validation_sha256"],
        "development_file": parent["development_file"],
        "development_sha256": parent["development_sha256"],
        "quality_report": QUALITY_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
        "protected_stage4_sha256": parent["protected_stage4_sha256"],
        "training_approval": "not_granted",
    }
    _write_json(MANIFEST_OUTPUT, manifest)
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
