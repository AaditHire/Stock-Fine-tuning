"""Report host details relevant to local LLM inference and fine-tuning.

The script intentionally treats PyTorch and NVIDIA tooling as optional so it can
diagnose a fresh environment before the ML stack is installed.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

GIB = 1024**3


@dataclass(frozen=True)
class EnvironmentReport:
    """Serializable snapshot of the software and hardware environment."""

    operating_system: str
    platform: str
    machine: str
    processor: str
    python_version: str
    python_executable: str
    total_ram_gib: float | None
    pytorch_version: str | None
    cuda_available: bool
    pytorch_cuda_version: str | None
    gpu_name: str | None
    gpu_vram_gib: float | None
    nvidia_driver_version: str | None
    nvidia_smi_cuda_version: str | None
    diagnostic_notes: tuple[str, ...]


def _total_ram_gib() -> float | None:
    """Return physical memory in GiB using only platform standard libraries."""

    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return round(status.total_physical / GIB, 2)

        page_size = int(__import__("os").sysconf("SC_PAGE_SIZE"))
        page_count = int(__import__("os").sysconf("SC_PHYS_PAGES"))
        return round((page_size * page_count) / GIB, 2)
    except (AttributeError, OSError, ValueError):
        return None


def _nvidia_smi_details(notes: list[str]) -> tuple[str | None, str | None]:
    """Query driver and maximum supported CUDA version without requiring PyTorch."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        notes.append("nvidia-smi was not found on PATH.")
        return None, None

    command = [
        executable,
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
        driver = completed.stdout.strip().splitlines()[0]

        summary = subprocess.run(
            [executable], capture_output=True, text=True, check=True, timeout=10
        ).stdout
        # Recent Windows drivers label this "CUDA UMD Version" while older
        # nvidia-smi releases use "CUDA Version".
        match = re.search(r"CUDA(?: UMD)? Version:\s*([0-9.]+)", summary)
        cuda_version = match.group(1) if match else None
        return driver, cuda_version
    except (subprocess.SubprocessError, IndexError) as exc:
        notes.append(f"nvidia-smi query failed: {exc}")
        return None, None


def _pytorch_details(notes: list[str]) -> dict[str, Any]:
    """Collect CUDA details from PyTorch when it is installed and importable."""

    details: dict[str, Any] = {
        "pytorch_version": None,
        "cuda_available": False,
        "pytorch_cuda_version": None,
        "gpu_name": None,
        "gpu_vram_gib": None,
    }
    try:
        import torch
    except ImportError:
        notes.append("PyTorch is not installed in this Python environment.")
        return details
    except Exception as exc:  # A broken CUDA/PyTorch binary is itself useful diagnostic data.
        notes.append(f"PyTorch import failed: {exc}")
        return details

    details["pytorch_version"] = torch.__version__
    details["pytorch_cuda_version"] = torch.version.cuda
    details["cuda_available"] = torch.cuda.is_available()

    if not details["cuda_available"]:
        notes.append("PyTorch imported successfully but CUDA is unavailable.")
        return details

    try:
        properties = torch.cuda.get_device_properties(0)
        details["gpu_name"] = properties.name
        details["gpu_vram_gib"] = round(properties.total_memory / GIB, 2)
    except (RuntimeError, AssertionError) as exc:
        notes.append(f"PyTorch could not inspect CUDA device 0: {exc}")

    return details


def collect_environment() -> EnvironmentReport:
    """Collect a side-effect-free environment report."""

    notes: list[str] = []
    torch_details = _pytorch_details(notes)
    driver_version, smi_cuda_version = _nvidia_smi_details(notes)

    return EnvironmentReport(
        operating_system=platform.platform(),
        platform=sys.platform,
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        python_version=platform.python_version(),
        python_executable=sys.executable,
        total_ram_gib=_total_ram_gib(),
        nvidia_driver_version=driver_version,
        nvidia_smi_cuda_version=smi_cuda_version,
        diagnostic_notes=tuple(notes),
        **torch_details,
    )


def _display_value(value: object) -> str:
    if value is None:
        return "not available"
    return str(value)


def format_report(report: EnvironmentReport) -> str:
    """Format a stable, readable report for terminal use."""

    labels = {
        "operating_system": "Operating system",
        "platform": "Python platform",
        "machine": "Machine architecture",
        "processor": "Processor",
        "python_version": "Python version",
        "python_executable": "Python executable",
        "total_ram_gib": "Total RAM (GiB)",
        "pytorch_version": "PyTorch version",
        "cuda_available": "PyTorch CUDA available",
        "pytorch_cuda_version": "PyTorch CUDA runtime",
        "gpu_name": "GPU name",
        "gpu_vram_gib": "GPU VRAM (GiB)",
        "nvidia_driver_version": "NVIDIA driver",
        "nvidia_smi_cuda_version": "Driver-supported CUDA",
    }
    values = asdict(report)
    lines = [f"{label}: {_display_value(values[key])}" for key, label in labels.items()]
    if report.diagnostic_notes:
        lines.append("Notes:")
        lines.extend(f"  - {note}" for note in report.diagnostic_notes)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    report = collect_environment()
    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
