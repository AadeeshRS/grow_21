# Grow21 Phase 4 Reproduction Guide: Learner Belief-State Estimation & Robustness

This document provides exact, deterministic commands to reproduce all Phase 4 empirical results, execute the test suite, recompute calibration tables, and verify artifact consistency from a clean environment.

---

## 1. Scientific Disclaimer

All learners, sensor observations, response times, and cognitive trajectories in Grow21 are entirely synthetic simulations designed to stress-test adaptive task scheduling algorithms under noisy, multi-modal interaction constraints inspired by pediatric motor and cognitive profiles (such as those observed in children with Trisomy 21). No human participant data, clinical trials, or developmental validity are claimed.

---

## 2. Environment Setup

### Prerequisites
- Python 3.10 to 3.13 (benchmarked on Python 3.13.11 Windows 64-bit)
- Standard virtual environment tools (`venv` or `conda`)

### Step 1: Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows Powershell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 3. Running the Test Suite

To run all 51 deterministic unit, regression, and documentation integrity tests:

```bash
# Using active environment
pytest -q tests

# Or standalone with explicit PYTHONPATH
PYTHONPATH=src python -m pytest -q tests
```

Expected output:
```text
51 passed in ~30s
```

### Breakdown of Test Modules
- `tests/test_belief_state_estimation.py`: 9 Phase 4 tests validating secondary transfer updating, Bayesian observation likelihood limits, strict no-leakage, invariance of predictions before outcome observation, bound conservation, and summary recomputation.
- `tests/test_outcome_model_alignment.py`: 10 Phase 3 tests verifying canonical clean-success equation parity across all 9 games, non-linear difficulty curves, bisection reachability, and session fatigue dynamics.
- `tests/test_audit_regressions.py`: Phase 2 historical regression suite checking calibration bin conservation, legacy surrogate behavior, and baseline metrics.
- `tests/test_no_leakage.py`: Invariance tests verifying that event $t$ outcomes do not leak into prediction $t$.
- `tests/test_controller_constraints.py`: Safety cooldowns, frustration limits ($\le 3$ failures), and session break rules ($\le 15$ trials).
- `tests/test_documentation_integrity.py`: Zero non-printing control characters, zero literal tabs in documentation, and intact mathematical expressions.
- `tests/test_simulator.py`: Virtual learner profile generation, fatigue/forgetting mechanics, and sensor corruption dynamics.
- `tests/test_reliability.py`: Reliability engine product rule and sensor dropout handling.
- `tests/test_model_shapes.py`: Deep Knowledge Tracing tensor shapes and masking.

---

## 4. Reproducing the Phase 4 Simulation Study

To execute the full staged simulation campaign (540,000 interaction events across 5 scenarios, 9 random seeds, and 4 belief-state estimator variants):

```bash
python -m grow21_simulation_benchmark.run_phase_4_experiments --num-learners 60 --trials-per-learner 60
```

Runtime: approximately 4 minutes on a standard multi-core CPU.

### Generated Artifacts
All generated files will be written to:
1. Primary execution outputs:
   - `outputs/phase_4/state_estimation/phase_4_event_metrics.csv` (540,000 interaction rows)
   - `outputs/phase_4/state_estimation/phase_4_calibration_bins.csv` (200 calibration bin rows)
   - `outputs/phase_4/state_estimation/phase_4_mastery_estimation_metrics.csv` (60 granular mastery rows)
   - `outputs/phase_4/state_estimation/phase_4_scenario_summary.json` (Hierarchical scenario metrics, seed standard deviations, and bootstrap CIs)
   - `outputs/phase_4/plots/*.png` (5 publication-grade figures)
2. Mirrored audit directory:
   - `outputs/audit/phase_4/` (contains exact duplicates of summary JSON, calibration bins, mastery metrics, and plots for archival preservation)

---

## 5. Offline Data Recomputation Verification

To verify that the persisted event-level CSV reproduces the summary JSON metrics:

```python
import json
import pandas as pd
import numpy as np

# Load event metrics and summary JSON
events = pd.read_csv("outputs/phase_4/state_estimation/phase_4_event_metrics.csv")
with open("outputs/phase_4/state_estimation/phase_4_scenario_summary.json") as f:
    summary = json.load(f)

# Recompute base_heterogeneous signed prediction gap for model_aligned_baseline
base_events = events[
    (events["split"].isin(["historical", "held_out"])) &
    (events["scenario"] == "base_heterogeneous") &
    (events["controller"] == "model_aligned_baseline")
]
recomputed_gap = np.mean(base_events["predicted_clean_prob"] - base_events["simulator_p_clean"])
stored_gap = summary["base_heterogeneous"]["policies"]["model_aligned_baseline"]["signed_prediction_gap"]

print(f"Recomputed Gap: {recomputed_gap:.6f}")
print(f"Stored Gap:     {stored_gap:.6f}")
assert abs(recomputed_gap - stored_gap) < 1e-4, "Recomputation check failed!"
print("Verification passed!")
```
