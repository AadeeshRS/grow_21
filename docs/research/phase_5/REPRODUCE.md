# Grow21 Phase 5 Reproduction Guide: Transition-Aware State Estimation, Robustness, & Model Selector

This document provides exact, deterministic commands to reproduce all Phase 5 empirical results, run the test suite, recompute metric summaries, and verify artifact consistency from a clean environment.

---

## 1. Scientific Disclaimer

All learners, sensor observations, response times, and cognitive trajectories in Grow21 are entirely synthetic simulations designed to benchmark adaptive task scheduling algorithms under noisy, multi-modal interaction constraints inspired by pediatric motor and cognitive profiles (such as those observed in children with Trisomy 21). No human participant data, clinical trials, or developmental validity are claimed.

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

To run all 66 deterministic unit, regression, documentation integrity, and Phase 5 belief-state estimation tests:

```bash
# On Windows Powershell:
$env:PYTHONPATH="src"; pytest -v tests

# On Linux/macOS:
PYTHONPATH=src pytest -v tests
```

Expected output:
```text
66 passed in ~145s
```

### Breakdown of Test Modules (66 Total Tests)
- `tests/test_phase_5_state_estimation.py`: 11 Phase 5 tests validating:
  1. Transition-aware primary mastery progression toward 0.95 under both clean success (0.70 feedback) and clean failure (0.30 feedback).
  2. Transition-aware analytical formula matching exact numerical step updates.
  3. Pre-outcome observed response invariance (finalized prior to task execution).
  4. Approx Bayesian filter posterior observation likelihood normalization and asymptotic behavior.
  5. Secondary skill transfer ceiling (0.95) and strict absence of hidden state leakage.
  6. Inter-session forgetting applied strictly once per elapsed event.
  7. Initial mastery prior interface defaults (0.45) and missingness fallback (never penalizes missingness as low ability).
  8. Synthetic parent-proxy report generation and noise perturbation bounds.
  9. Semantic safeguard preventing offline $P(y_{\text{obs}})$ models from driving clean-probability controllers without an adapter.
  10. Open-loop replay exact state tracking and bias reduction across identical exogenous traces.
  11. Offline sequence model selector reproducibility between individual selection and `--models all` via controlled per-model seeds.
- `tests/test_belief_state_estimation.py`: 13 Phase 4 tests validating secondary transfer, observation likelihood, and split separation.
- `tests/test_outcome_model_alignment.py`: 10 Phase 3 tests for canonical equation parity and bisection reachability.
- `tests/test_audit_regressions.py`: 13 Phase 2 historical regression tests.
- `tests/test_no_leakage.py`: 3 invariance tests.
- `tests/test_controller_constraints.py`: 4 tests for safety cooldowns, frustration limits, and session breaks.
- `tests/test_documentation_integrity.py`: 4 tests for ASCII control characters and document consistency.
- `tests/test_simulator.py`: 3 tests for learner generation and fatigue dynamics.
- `tests/test_reliability.py`: 3 tests for sensor uncertainty combinations.
- `tests/test_model_shapes.py`: 2 tests for Deep Knowledge Tracing tensor shapes and masking.

---

## 4. Reproducing the Full Phase 5 Simulation Campaign

The configured simulation cohorts contain **12 seeds in total** across 4 distinct splits (3 seeds each):
- **Development Cohort:** Seeds `[101, 202, 303]`
- **Historical Baseline Cohort:** Seeds `[42, 123, 999]`
- **Phase 4 Evaluation Cohort:** Seeds `[777, 888, 555]`
- **Final Confirmatory Held-Out Cohort:** Seeds `[707, 808, 909]`

### Trial & Interaction Accounting:
- **Learners and Trials:** 60 learners per seed, 60 trials per learner = 3,600 learner trials per scenario per seed.
- **Unique Learner Trials:** Across 12 seeds and 5 scenarios = **216,000 unique learner trials** generated per policy ($54,000$ unique trials per cohort).
- **Closed-Loop Estimator-Policy Interactions:** With 8 estimator policies evaluated on every trial = **1,728,000 closed-loop interaction events** across the entire study ($432,000$ per cohort).
- **Stage 1 Open-Loop Evaluations:** Replaying all traces through 8 estimators = **1,728,000 open-loop evaluation records** ($432,000$ per cohort).

### Reproduction Command:
To execute the complete 4-stage empirical campaign across all cohorts:

```bash
# On Windows Powershell:
$env:PYTHONPATH="src"; python -m grow21_simulation_benchmark.run_phase_5_experiments --cohorts all --num-learners 60 --trials-per-learner 60

# On Linux/macOS:
PYTHONPATH=src python -m grow21_simulation_benchmark.run_phase_5_experiments --cohorts all --num-learners 60 --trials-per-learner 60
```

Runtime: approximately 14 to 15 minutes on a standard multi-core CPU.

### Selective Stage Execution Flags

If selective execution is desired:

```bash
# Run only primary Held-Out cohort (Seeds 707, 808, 909) across all stages (~4 minutes):
python -m grow21_simulation_benchmark.run_phase_5_experiments --cohorts held_out

# Run only Stage 4 (Offline Model Selector verification, ~10 seconds):
python -m grow21_simulation_benchmark.run_phase_5_experiments --skip-open-loop --skip-closed-loop --skip-initializers

# Run Stage 2 Closed-Loop only:
python -m grow21_simulation_benchmark.run_phase_5_experiments --skip-open-loop --skip-initializers --skip-benchmark

# Run Stage 3 Parent-Proxy Sensitivity only:
python -m grow21_simulation_benchmark.run_phase_5_experiments --skip-open-loop --skip-closed-loop --skip-benchmark
```

---

## 5. Offline Model Selector CLI Reproduction

To test and reproduce individual offline sequence models via the modular `--models` selector flag:

```bash
# Train and evaluate only Logistic Baseline:
python -m grow21_simulation_benchmark.run_benchmark --models logistic_baseline --epochs 5

# Train and evaluate only Vanilla DKT:
python -m grow21_simulation_benchmark.run_benchmark --models vanilla_dkt --epochs 5

# Train and evaluate only Reliability-Gated DKT:
python -m grow21_simulation_benchmark.run_benchmark --models reliability_gated_dkt --epochs 5

# Train and evaluate all offline models:
python -m grow21_simulation_benchmark.run_benchmark --models all --epochs 5
```

Because of controlled per-model seeds (`seed + 0` for logistic, `seed + 100` for vanilla DKT, `seed + 200` for reliability-gated DKT), selecting a model individually produces **identical** test metrics to selecting it via `--models all`.

---

## 6. Verification of Persisted Artifacts

### Option A: Direct Verification from Tracked Summary Artifacts (or Packaged Evidence Bundle)
In the repository, compact summary tables and bootstrap results are tracked directly under `outputs/experiments/phase_5/`. (For environments possessing the optional release archive `evidence/phase_5/Grow21_Phase_5_Evidence.zip`, these identical summaries are packaged inside.) To verify empirical findings directly without running the 15-minute simulation:

```python
import pandas as pd

# 1. Load scenario summary CSV directly from tracked experiment outputs
summary = pd.read_csv("outputs/experiments/phase_5/closed_loop/phase_5_scenario_summary.csv")

# Filter for Held-Out cohort
ho = summary[summary["split"] == "held_out"]

# Verify base_heterogeneous metrics
bh_base = ho[(ho["scenario"] == "base_heterogeneous") & (ho["controller"] == "model_aligned_baseline")].iloc[0]
bh_bayes = ho[(ho["scenario"] == "base_heterogeneous") & (ho["controller"] == "approx_bayesian_filter_transfer")].iloc[0]

print("=== HELD-OUT BASE_HETEROGENEOUS ===")
print(f"Baseline Primary Post MAE:       {bh_base['primary_mastery_mae_post']:.4f} (Expected: 0.1039)")
print(f"Approx Bayes Transfer Post MAE:  {bh_bayes['primary_mastery_mae_post']:.4f} (Expected: 0.1002)")
print(f"Min-Bound Saturation (Baseline): {bh_base['proportion_at_min_bound']*100:.1f}% (Expected: 34.8%)")
print(f"Min-Bound Saturation (Bayes):    {bh_bayes['proportion_at_min_bound']*100:.1f}% (Expected: 20.1%)")

assert abs(bh_base["primary_mastery_mae_post"] - 0.1039) < 0.001
assert abs(bh_bayes["primary_mastery_mae_post"] - 0.1002) < 0.001
assert abs(bh_base["proportion_at_min_bound"] - 0.348) < 0.005
assert abs(bh_bayes["proportion_at_min_bound"] - 0.201) < 0.005

# Verify reliability_stress metrics
rel_base = ho[(ho["scenario"] == "reliability_stress") & (ho["controller"] == "model_aligned_baseline")].iloc[0]
rel_bayes = ho[(ho["scenario"] == "reliability_stress") & (ho["controller"] == "approx_bayesian_filter_transfer")].iloc[0]

print("\n=== HELD-OUT RELIABILITY_STRESS ===")
print(f"Baseline Primary Post MAE:       {rel_base['primary_mastery_mae_post']:.4f} (Expected: 0.1224)")
print(f"Approx Bayes Transfer Post MAE:  {rel_bayes['primary_mastery_mae_post']:.4f} (Expected: 0.0986)")

assert abs(rel_base["primary_mastery_mae_post"] - 0.1224) < 0.001
assert abs(rel_bayes["primary_mastery_mae_post"] - 0.0986) < 0.001

print("\nPackaged Summary Artifact Verification Successful!")
```

### Option B: Verification after Full Dataset Regeneration
If the full simulation study was re-executed via `run_phase_5_experiments`, the uncompressed interaction dataset `outputs/phase_5/closed_loop/phase_5_event_metrics.csv` (~876 MB) will be generated. The event-level recomputation can then be verified:

```python
import pandas as pd
import numpy as np
from pathlib import Path

event_path = Path("outputs/experiments/phase_5/closed_loop/phase_5_event_metrics.csv")
if not event_path.exists():
    print(f"File {event_path} is omitted from the git repo due to size (~876 MB).")
    print("Run `python -m grow21_simulation_benchmark.run_phase_5_experiments` to generate it.")
else:
    events = pd.read_csv(event_path)
    ho_rel = events[(events["split"] == "held_out") & (events["scenario"] == "reliability_stress")]
    base_events = ho_rel[ho_rel["controller"] == "model_aligned_baseline"]
    bayes_events = ho_rel[ho_rel["controller"] == "approx_bayesian_filter_transfer"]

    base_mae = float(np.mean(np.abs(base_events["estimated_mastery_primary_post"] - base_events["true_mastery_primary_post"])))
    bayes_mae = float(np.mean(np.abs(bayes_events["estimated_mastery_primary_post"] - bayes_events["true_mastery_primary_post"])))

    assert abs(base_mae - 0.1224) < 0.005
    assert abs(bayes_mae - 0.0986) < 0.005
    print("Full Event-Level Metric Recomputation Successful!")
```
