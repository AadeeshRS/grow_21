"""Canonical project paths and directory management for Grow21 benchmark."""

from pathlib import Path

# Package root: grow21_simulation_benchmark/
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIGS_DIR = PACKAGE_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"
SCENARIOS_CONFIG_PATH = CONFIGS_DIR / "scenarios.yaml"

OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
ARCHIVE_OUTPUTS_DIR = OUTPUTS_DIR / "archive"
DOCS_DIR = PACKAGE_ROOT / "docs"
EVIDENCE_DIR = PACKAGE_ROOT / "evidence"

# Phase 2 canonical experiment outputs
PHASE_2_OUTPUTS_DIR = EXPERIMENTS_DIR / "phase_2"
BENCHMARK_OUTPUTS_DIR = PHASE_2_OUTPUTS_DIR / "benchmark"
CLOSED_LOOP_OUTPUTS_DIR = PHASE_2_OUTPUTS_DIR / "closed_loop"
AUDIT_OUTPUTS_DIR = PHASE_2_OUTPUTS_DIR

# Phase 3 canonical experiment outputs
PHASE_3_OUTPUTS_DIR = EXPERIMENTS_DIR / "phase_3"
PHASE_3_CLOSED_LOOP_DIR = PHASE_3_OUTPUTS_DIR / "closed_loop"
PHASE_3_DIAGNOSTICS_DIR = PHASE_3_OUTPUTS_DIR / "diagnostics"
PHASE_3_PLOTS_DIR = PHASE_3_OUTPUTS_DIR / "plots"
PHASE_3_AUDIT_DIR = PHASE_3_OUTPUTS_DIR

# Phase 4 canonical experiment outputs
PHASE_4_OUTPUTS_DIR = EXPERIMENTS_DIR / "phase_4"
PHASE_4_STATE_ESTIMATION_DIR = PHASE_4_OUTPUTS_DIR / "state_estimation"
PHASE_4_PLOTS_DIR = PHASE_4_OUTPUTS_DIR / "plots"
PHASE_4_AUDIT_DIR = PHASE_4_OUTPUTS_DIR

# Phase 5 canonical experiment outputs
PHASE_5_OUTPUTS_DIR = EXPERIMENTS_DIR / "phase_5"
PHASE_5_OPEN_LOOP_DIR = PHASE_5_OUTPUTS_DIR / "open_loop"
PHASE_5_CLOSED_LOOP_DIR = PHASE_5_OUTPUTS_DIR / "closed_loop"
PHASE_5_INITIALIZERS_DIR = PHASE_5_OUTPUTS_DIR / "initializers"
PHASE_5_PLOTS_DIR = PHASE_5_OUTPUTS_DIR / "plots"
PHASE_5_AUDIT_DIR = PHASE_5_OUTPUTS_DIR


def ensure_output_directories():
    """Ensures all standard output subdirectories exist."""
    BENCHMARK_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_OUTPUTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    CLOSED_LOOP_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (CLOSED_LOOP_OUTPUTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    AUDIT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_3_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_3_CLOSED_LOOP_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_3_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_3_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_4_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_4_STATE_ESTIMATION_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_4_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_5_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_5_OPEN_LOOP_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_5_CLOSED_LOOP_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_5_INITIALIZERS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_5_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
