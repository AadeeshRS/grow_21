# Grow21 Phase 2 Evidence Bundle Index (`BUNDLE_INDEX.md`)

This manifest describes the files included in `Grow21_Phase_2_Audit_Evidence.zip` and explains why each artifact is provided to enable independent verification of the Phase 2 audit.

---

## 1. Documentation & Audit Reports

- **`PHASE_2_AUDIT_REPORT.md`**: Complete Phase 2 technical audit report detailing executive findings, closed-loop investigations, calibration audits, reliability sweeps, test results, and recommendations.
- **`BUNDLE_INDEX.md`**: (This file) Overview of bundle structure, purpose of each file, and omission rationale.
- **`REPRODUCE.md`**: Step-by-step commands to reproduce all audit data, test suites, and figures from scratch.
- **`CHANGELOG_PHASE_2.md`**: Granular list of code, config, test, and documentation changes made during Phase 2.
- **`README.md`**: The top-level project documentation reconciled with verified empirical findings.

---

## 2. Source Code Modules

- **`src/grow21_simulation_benchmark/closed_loop.py`**: Implementation of controller policies (Fixed, Streak Adaptive, and Grow21 Model-Based Adaptive), the Grow21 predicted-success equation $\hat{p} = \sigma(\text{logit}(\theta) - \gamma D^{\nu})$, deadband, step-size limiters, and direction-reversal logic.
- **`src/grow21_simulation_benchmark/simulator.py`**: Virtual learner latent state dynamics, item-response clean probability generation, sensor corruption, practice learning, and exponential forgetting.
- **`src/grow21_simulation_benchmark/reliability.py`**: Grow21 product rule $\hat{\rho} = \prod (1 - \hat{p}_m)$ and non-zero uncertainty penalties for missing sensors.
- **`src/grow21_simulation_benchmark/models/reliability_gated_dkt.py`**: PyTorch implementation of the reliability-gated recurrent state transition $h_t = h_{t-1} + \hat{\rho}_t (\bar{h}_t - h_{t-1})$ with causal prediction timing.
- **`src/grow21_simulation_benchmark/models/dkt.py`**: PyTorch Vanilla DKT baseline with causal prediction timing.
- **`src/grow21_simulation_benchmark/models/logistic_baseline.py`**: Classical scikit-learn Logistic Regression baseline.
- **`src/grow21_simulation_benchmark/features.py`**: Causal feature extraction tracking successes, failures, smoothed rates, recency, streaks, and modality risk strictly before event $t$.
- **`src/grow21_simulation_benchmark/tasks.py`**: Task registry, game metadata, and difficulty parameter mappings.
- **`src/grow21_simulation_benchmark/dataset.py`**: Learner-disjoint dataset generator and sequence collation.
- **`src/grow21_simulation_benchmark/train.py`**: Model training loops with validation-loss early stopping.
- **`src/grow21_simulation_benchmark/evaluate.py`**: Metrics engine (Log loss, Brier, ROC-AUC, ECE, bootstrap CIs).
- **`src/grow21_simulation_benchmark/plots.py`**: Matplotlib plotting routines for calibration curves and scenario comparisons.
- **`src/grow21_simulation_benchmark/run_closed_loop.py`**: CLI script for running closed-loop simulation and logging event traces.
- **`src/grow21_simulation_benchmark/run_benchmark.py`**: CLI runner for multi-seed offline benchmark across scenarios.
- **`src/grow21_simulation_benchmark/audit_calibration_bins.py`**: Script for exporting full calibration bins and running the 4-level reliability sweep.

---

## 3. Configurations

- **`configs/default.yaml`**: Configuration for games, skills, modalities, and model architectures.
- **`configs/scenarios.yaml`**: Parameter configurations for all 5 scenario families.

---

## 4. Test Suite

- **`tests/test_audit_regressions.py`**: Regression tests for oscillation definitions, calibration count sums, and common random numbers.
- **`tests/test_controller_constraints.py`**: Unit tests verifying Grow21 controller equations, deadband, step limits, and break rules.
- **`tests/test_no_leakage.py`**: Unit tests proving event $t$'s outcome cannot influence prediction $\hat{y}_t$.
- **`tests/test_model_shapes.py`**: Unit tests verifying model tensor shapes and gating boundary conditions.
- **`tests/test_reliability.py`**: Unit tests for the Grow21 product rule and missing modality handling.
- **`tests/test_simulator.py`**: Unit tests for latent mastery bounds and forgetting dynamics.

---

## 5. Data Artifacts & Evidence Outputs

- **`outputs/closed_loop_events.csv`**: Full 10,800 event-level trial log from the 60-learner closed-loop simulation (columns: controller, learner_id, step, game, skill, difficulty, gap, observed, clean, reliability).
- **`outputs/closed_loop_summary.json`**: Summary statistics for the verified closed-loop experiment.
- **`outputs/metrics.csv`**: Full per-seed offline benchmark records across 5 scenarios and 3 seeds.
- **`outputs/summary.json`**: Aggregated mean and standard deviation metrics for all models across scenarios.
- **`outputs/audit/calibration_bins.csv`**: 450-row export of all 10 calibration bins across all models, scenarios, and seeds.
- **`outputs/audit/reliability_sweep_summary.json`**: Results of the 4-condition controlled sensor corruption experiment.
- **`outputs/plots/*.png`**: All 7 verified publication figures:
  - `calibration_base_heterogeneous.png`
  - `calibration_reliability_stress.png`
  - `calibration_simple_dynamics.png`
  - `calibration_temporal_dynamics.png`
  - `calibration_distribution_shift.png`
  - `scenario_comparison.png`
  - `closed_loop_comparison.png`

---

## 6. Intentionally Omitted Files

- **Virtual Environment (`.venv/`)**: Omitted to keep the bundle compact; all dependencies are standard packages listed in `requirements.txt`.
- **Large Intermediate Caches (`.pytest_cache/`, `__pycache__/`)**: Omitted as transient build files.
- **Raw multi-megabyte training tensors**: Fully reproducible deterministically via seeds using the included configs and source code.
