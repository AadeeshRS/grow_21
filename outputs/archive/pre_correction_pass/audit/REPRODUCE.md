# Grow21 Phase 2 Reproduction Guide (`REPRODUCE.md`)

This guide provides exact commands to set up the environment, run the complete test suite, and regenerate all audited Phase 2 summaries, metrics, and plots.

---

## 1. Environment Setup

### Prerequisites
- Python 3.9+ (Python 3.13.11 was used during the audit)
- Operating System: Windows, macOS, or Linux

### Install Dependencies
```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# Windows (PowerShell):
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies in editable mode
pip install -e grow21_simulation_benchmark
```

Dependencies installed: `torch`, `scikit-learn`, `numpy`, `pandas`, `scipy`, `pyyaml`, `matplotlib`, `pytest`.

---

## 2. Execute Complete Test Suite

Run all unit tests and regression tests:
```bash
pytest -v grow21_simulation_benchmark/tests
```
**Expected Output**: 18 passed in ~5 seconds (100% pass rate).

---

## 3. Regenerate Closed-Loop Simulation Artifacts

To re-run the 60-learner, 60-trial closed-loop experiment:
```bash
python -m grow21_simulation_benchmark.run_closed_loop --scenario base_heterogeneous --num-learners 60 --trials-per-learner 60 --seed 42 --output-dir grow21_simulation_benchmark/outputs
```

**Artifacts Regenerated**:
- `outputs/closed_loop_events.csv` (10,800 event-level trials)
- `outputs/closed_loop_summary.json` (per-controller gains, oscillations, violations)
- `outputs/plots/closed_loop_comparison.png` (mastery gains and oscillation figure)

---

## 4. Regenerate Calibration Tables & 4-Condition Reliability Sweep

To regenerate the calibration bin tables across all scenarios and run the 4-level controlled sensor corruption sweep:
```bash
python -m grow21_simulation_benchmark.audit_calibration_bins --action all
```

**Artifacts Regenerated**:
- `outputs/audit/calibration_bins.csv` (450 calibration bin rows)
- `outputs/audit/reliability_sweep_summary.json` (4-condition sensor corruption sweep)

---

## 5. Regenerate Full Offline Benchmark (Multi-Seed)

To run the complete 5-scenario, 3-seed offline benchmark:
```bash
python -m grow21_simulation_benchmark.run_benchmark --scenario all --seeds 42,123,999 --num-learners 500 --events-per-learner 60 --output-dir grow21_simulation_benchmark/outputs
```

**Small Smoke Run Alternative** (CPU friendly, ~15 seconds):
```bash
python -m grow21_simulation_benchmark.run_benchmark --scenario base_heterogeneous --seeds 42 --num-learners 100 --events-per-learner 40 --output-dir grow21_simulation_benchmark/outputs
```

**Artifacts Regenerated**:
- `outputs/metrics.csv`
- `outputs/summary.json`
- `outputs/plots/calibration_*.png` (5 calibration figures)
- `outputs/plots/scenario_comparison.png`
