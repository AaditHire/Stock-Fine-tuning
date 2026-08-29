from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "results" / "training" / "stage6_qwen3_4b_seed_v1.json"


def test_stage6_report_records_a_successful_memory_safe_run() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["stage"] == 6
    assert report["status"] == "completed"
    assert report["smoke_test"] is False
    assert report["base_model"]["quantization"] == "bitsandbytes 4-bit"
    assert report["dataset"]["train_examples"] == 33
    assert report["dataset"]["validation_examples"] == 7
    assert report["metrics"]["train"]["train_loss"] > 0
    assert report["metrics"]["evaluation"]["eval_loss"] > 0
    assert report["metrics"]["peak_gpu_device_used_mib"] < 6144


def test_stage6_report_records_only_a_small_trainable_fraction() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["parameters"]["trainable"] == 33_030_144
    assert report["parameters"]["total"] == 4_055_498_240
    assert report["parameters"]["trainable_percent"] < 1.0
    assert report["adapter"]["weights_size_bytes"] < 100 * 1024 * 1024
    assert len(report["adapter"]["weights_sha256"]) == 64
