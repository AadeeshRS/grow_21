"""Canonical project paths and directory management for Grow21 benchmark."""

from pathlib import Path

# Package root: grow21_simulation_benchmark/
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIGS_DIR = PACKAGE_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"
SCENARIOS_CONFIG_PATH = CONFIGS_DIR / "scenarios.yaml"

OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
BENCHMARK_OUTPUTS_DIR = OUTPUTS_DIR / "benchmark"
CLOSED_LOOP_OUTPUTS_DIR = OUTPUTS_DIR / "closed_loop"
AUDIT_OUTPUTS_DIR = OUTPUTS_DIR / "audit" / "phase_2"
ARCHIVE_OUTPUTS_DIR = OUTPUTS_DIR / "archive" / "pre_correction_pass"


def ensure_output_directories():
    """Ensures all standard output subdirectories exist."""
    BENCHMARK_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_OUTPUTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    CLOSED_LOOP_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (CLOSED_LOOP_OUTPUTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    AUDIT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
