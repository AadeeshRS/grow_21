# Grow21 Phase 2 Change Log (`CHANGELOG_PHASE_2.md`)

This log documents all additions, fixes, and updates made during the Phase 2 Audit, categorized by affected area.

---

### 1. Code Changes

1. **`src/grow21_simulation_benchmark/closed_loop.py`**:
   - *Change*: Implemented the true Grow21 model-based controller equation:
     $$\hat{p} = \sigma\left(\text{logit}(\hat{\theta}_k) - \gamma_{gk} \cdot D^{\nu_D}\right)$$
     with target probability $p^* = 0.75$, deadband $\pm 0.04$, step limit $0.08$, and difficulty bounds $[0.08, 0.95]$.
   - *Change*: Replaced single total variation calculation with explicit separation of **direction reversals** ($(D_t - D_{t-1})(D_{t-1} - D_{t-2}) < 0$) and **total variation** ($\sum |D_t - D_{t-1}|$).
   - *Change*: Added tracking and reporting of practiced-skills mastery gain alongside all-skills mean gain.
   - *Change*: Standardized cooldown gap handling uniformly across controllers to maintain fair environmental comparability.
   - *Reason*: Replaces heuristic step sizes with the documented Grow21 controller equation and eliminates unfair time penalty discrepancies.

2. **`src/grow21_simulation_benchmark/run_closed_loop.py`**:
   - *Change*: Updated CLI runner to capture event-level DataFrame, save `outputs/closed_loop_events.csv`, and produce dual-panel stability plots.
   - *Reason*: Provides full event-level transparency and auditability.

3. **`src/grow21_simulation_benchmark/audit_calibration_bins.py`** *(New File)*:
   - *Change*: Added script to export all 10 calibration bins across all models and scenarios, and run the 4-level controlled sensor corruption sweep.
   - *Reason*: Fulfills Phase 2 audit requirements for bin-level transparency and reliability gating stress testing.

---

### 2. Test Changes

1. **`tests/test_controller_constraints.py`**:
   - *Change*: Added tests for the Grow21 controller predicted-success equation, deadband behavior, and step-size limiter bounds. Updated outcome observation method signature.
   - *Reason*: Verifies that the model-based controller strictly respects the mathematical formulation and safety constraints.

2. **`tests/test_audit_regressions.py`** *(New File)*:
   - *Change*: Added regression tests verifying:
     - `test_regression_oscillation_vs_variation`: Direction reversals vs total variation.
     - `test_regression_calibration_bin_sum_equals_total_events`: Bin counts sum to total test events.
     - `test_regression_fairness_common_random_numbers`: Common random numbers across controller initializations.
   - *Reason*: Guards against recurrence of audit-identified defects.

---

### 3. Metric & Data Changes

1. **`outputs/closed_loop_summary.json`**:
   - *Change*: Recomputed with verified metrics: direction reversals (0.0 for target adaptive vs 5.5 for streak), total variation (0.29 vs 2.03), and practiced skill gains (-0.0234 vs -0.0451).
   - *Reason*: Corrects stale and inverted summary statistics.

2. **`outputs/closed_loop_events.csv`** *(New File)*:
   - *Change*: Exported 10,800 event-level trials.
   - *Reason*: Provides granular verification data for independent reviewers.

3. **`outputs/audit/calibration_bins.csv`** *(New File)*:
   - *Change*: Exported full 450-row calibration bin breakdown.
   - *Reason*: Verifies observation counts and documents high-probability bin sparsity under distribution shift.

4. **`outputs/audit/reliability_sweep_summary.json`** *(New File)*:
   - *Change*: Recorded results across 4 controlled sensor-corruption levels.
   - *Reason*: Demonstrates statistically significant benefit of reliability gating under missing modalities ($\Delta = -0.0054$, CI strictly below 0).

---

### 4. Plot Changes

1. **`outputs/plots/closed_loop_comparison.png`**:
   - *Change*: Regenerated dual-panel plot showing practiced skills mastery growth on the left and direction reversals vs total variation on the right.
   - *Reason*: Accurately reflects verified controller stability metrics.

---

### 5. Documentation & Prose Changes

1. **`README.md`**:
   - *Change*: Removed the erroneous claim that "target adaptation reduces oscillations by 40% (2.91 vs 4.88)".
   - *Change*: Documented that Grow21 deadband adaptation eliminates direction reversals (0.0 vs 5.5) and reduces total variation to 0.29.
   - *Change*: Explained the distinction between practiced skill gains and multi-skill forgetting over inter-session rest gaps.
   - *Reason*: Reconciles project documentation with empirical truth.

2. **`outputs/audit/PHASE_2_AUDIT_REPORT.md`** *(New File)*:
   - *Change*: Comprehensive technical audit report addressing all 10 prompt requirements.
