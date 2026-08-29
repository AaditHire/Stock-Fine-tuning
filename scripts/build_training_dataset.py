"""Run the Stage 5 financial instruction-data pipeline and write reproducible splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finpulse_llm.data.config import load_data_config
from finpulse_llm.data.leakage import EvaluationLeakageIndex
from finpulse_llm.data.pipeline import (
    build_quality_report,
    file_sha256,
    load_jsonl,
    run_pipeline,
    write_jsonl,
)
from finpulse_llm.evaluation.stage4 import verify_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "training_pipeline.toml",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "train" / "finpulse_seed_v1.jsonl",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "validation" / "finpulse_seed_v1.jsonl",
    )
    parser.add_argument(
        "--development-output",
        type=Path,
        help="Optional independent development holdout output.",
    )
    parser.add_argument(
        "--quality-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "finpulse_seed_v1.quality.json",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "finpulse_seed_v1.manifest.json",
    )
    parser.add_argument("--allow-rejections", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def main() -> int:
    args = _parse_args()
    config = load_data_config(args.config)
    stage3_path = PROJECT_ROOT / "benchmarks" / "stage3_base_models.json"
    stage4_path = PROJECT_ROOT / "data" / "eval" / "finpulse_eval_v1.jsonl"
    stage4_manifest_path = (
        PROJECT_ROOT / "data" / "eval" / "finpulse_eval_v1.manifest.json"
    )
    stage4_manifest = verify_manifest(stage4_path, stage4_manifest_path)
    leakage = EvaluationLeakageIndex.from_files(stage3_path, stage4_path)
    result = run_pipeline(load_jsonl(args.inputs), config, leakage)

    write_jsonl(args.train_output, result.train)
    write_jsonl(args.validation_output, result.validation)
    if args.development_output is not None:
        write_jsonl(args.development_output, result.development)
    quality = build_quality_report(result, config)
    _write_json(args.quality_output, quality)
    manifest = {
        "schema_version": 1,
        "dataset_id": config.dataset_id,
        "seed": config.seed,
        "source_files": {
            _portable_path(path): file_sha256(path) for path in sorted(args.inputs)
        },
        "train_file": _portable_path(args.train_output),
        "train_sha256": file_sha256(args.train_output),
        "validation_file": _portable_path(args.validation_output),
        "validation_sha256": file_sha256(args.validation_output),
        "quality_report": _portable_path(args.quality_output),
        "protected_stage4_sha256": stage4_manifest["dataset_sha256"],
        "leakage_policy": "Reject exact or fuzzy matches against Stage 3 and Stage 4 prompts.",
    }
    if args.development_output is not None:
        manifest["development_file"] = _portable_path(args.development_output)
        manifest["development_sha256"] = file_sha256(args.development_output)
    _write_json(args.manifest_output, manifest)

    print(json.dumps(quality, indent=2))
    strict_failure = result.rejections or not quality["all_distributions_within_tolerance"]
    if strict_failure and config.fail_on_rejection and not args.allow_rejections:
        print("Outputs were written for inspection, but strict quality mode rejects this build.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
