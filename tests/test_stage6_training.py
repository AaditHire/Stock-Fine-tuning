from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "results" / "training" / "stage6_qwen3_4b_seed_v1.json"
STAGE6B_REPORT = ROOT / "results" / "training" / "stage6b_qwen3_4b_stage5b_v1.json"


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


def test_stage6b_report_records_completed_corrective_run() -> None:
    report = json.loads(STAGE6B_REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "completed"
    assert report["run_name"] == "stage6b-qwen3-4b-stage5b-v1"
    assert report["smoke_test"] is False
    assert report["dataset"]["train_examples"] == 398
    assert report["dataset"]["validation_examples"] == 51
    assert report["dataset"]["train_token_max"] <= 512
    assert report["trainer"]["learning_rate"] == 0.0001
    assert report["trainer"]["num_train_epochs"] == 1.0
    assert report["metrics"]["evaluation"]["eval_loss"] > 0
    assert report["metrics"]["peak_gpu_device_used_mib"] < 6144
    assert report["adapter"]["weights_sha256"] == (
        "4dbfab3baa3fe052b95f8334e7b3657fcf253d2a3947af9f83e18da48e289a56"
    )
