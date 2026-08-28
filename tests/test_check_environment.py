import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "check_environment.py"


def _load_diagnostic_module():
    spec = importlib.util.spec_from_file_location("check_environment", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve annotations through sys.modules while decorating the class.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_environment_report_contains_core_fields() -> None:
    module = _load_diagnostic_module()
    report = module.collect_environment()

    assert report.python_version
    assert report.python_executable
    assert report.operating_system
    assert report.total_ram_gib is None or report.total_ram_gib > 0


def test_human_report_is_readable() -> None:
    module = _load_diagnostic_module()
    rendered = module.format_report(module.collect_environment())

    assert "Python version:" in rendered
    assert "PyTorch CUDA available:" in rendered

