# Grow21 Benchmark — Phase 2 Changelog

All notable changes, bug fixes, parameter alignments, and evidence updates made during the Phase 2 Correction Pass are documented in this file.

---

## [Phase 2 Correction Pass] - October 18, 2026

### Architectural & Directory Restructuring
- **Output Hierarchy Standardization**:
  - Reorganized project outputs into clear canonical subdirectories:
    - `outputs/benchmark/`: Multi-scenario multi-seed knowledge tracing metrics, summaries, and plots.
    - `outputs/closed_loop/`: Closed-loop simulation event logs, summaries, mastery records, and dynamics plots.
    - `outputs/audit/phase_2/`: Authoritative audit report, bundle index, changelog, reproduction guide, calibration bins, paired loss evidence, and evidence ZIP.
    - `outputs/archive/pre_correction_pass/`: Safely archived snapshot of all 19 pre-correction artifacts accompanied by `manifest.json`.
  - Added `src/grow21_simulation_benchmark/paths.py` as the centralized single source of truth for canonical paths.

### Controller & Closed-Loop Physics Fixes
- **Target Reachability & Bound Saturation Diagnosis**:
  - Formally analyzed the Grow21 item-response equation $\hat{p}(D) = \sigma(\text{logit}(\hat{\theta}) - \gamma D^{\nu_D})$.
  - Demonstrated that with target $p^* = 0.75$, minimum difficulty $D_{\min} = 0.08$, $\gamma = 2.4$, and $\nu_D = 1.0$, a learner with initial mastery $\theta \approx 0.45$ can achieve at most $\hat{p}(D_{\min}) = 0.4031 < 0.75$.
  - Confirmed why **93.4% of trials clamped at $D_{\min} = 0.08$**, proving that the reported **0.000 direction reversals** was an artifact of bound saturation rather than active closed-loop tracking.
  - Added trial-by-trial difficulty trajectory plotting (`difficulty_trajectories.png`) to clearly visualize bound flatlining.
- **Oscillation vs. Variation Metrics Disentangled**:
  - Separated two previously conflated metrics in `closed_loop.py` and `closed_loop_summary.json`:
    1. *Immediate Direction Reversals*: $(D_t - D_{t-1})(D_{t-1} - D_{t-2}) < 0$, yielding **4.733 ± 2.007** (fixed) and **5.467 ± 1.658** (streak).
    2. *Filtered Nonzero Reversals*: Sign reversals across non-zero transitions, yielding **8.583 ± 4.076** (fixed) and **17.933 ± 2.949** (streak).
    3. *Total Difficulty Variation*: $\sum |\Delta D|$, yielding **2.392 ± 1.028** (fixed), **2.032 ± 0.217** (streak), and **0.290 ± 0.000** (target adaptive).
- **Session Break Count & Elapsed Rest Gap Logic**:
  - Corrected the break schedule explanation for 60-trial runs with `max_session_trials = 15`.
  - Breaks occur strictly after trials 15, 30, and 45: **exactly 3 breaks = 36.0 hours**, correcting previous claims of 4 breaks / 48 hours.
- **Mastery Accounting & Secondary Transfer Learning**:
  - Fixed misleading labeling of secondary skills as "unpracticed".
  - Verified that in `simulator.py`, games with secondary skills (`trace_glow`, `bubble_pop_brigade`) transfer $30\%$ of learning gains to secondary skills ($\Delta \theta_{\text{sec}} = 0.30 \eta (0.95 - \theta)$).
  - Exported granular `learner_mastery_summary.csv` tracking initial/final mastery, primary practice counts, and secondary transfer counts.

### Calibration & Benchmark Inconsistencies Resolved
- **Evaluation Cohort Reconciliation**:
  - Resolved the sample size discrepancy between `metrics.csv` ($N = 4,500$ test events from 500 learners) and the previous calibration bin export ($N = 2,700$ from 300 learners).
  - Regenerated `calibration_bins.csv` on the canonical 500-learner benchmark cohort. Counts sum to exactly 4,500 across all bins, and recomputed ECE matches `metrics.csv` within $10^{-4}$.
- **Distribution Shift High-Probability Tail Investigation**:
  - Investigated sharp jumps in `calibration_distribution_shift.png`.
  - Confirmed from empirical bin counts that upper probability bins are extremely sparse: for seed 42, $[0.8, 0.9)$ has $N=0$ and $[0.9, 1.0]$ has $N=0$ across all 3 models; $[0.7, 0.8)$ contains $N=71$ (Logistic), $N=45$ (Vanilla DKT), and $N=16$ (Rel-Gated DKT). Across all three seeds, $[0.8, 0.9)$ has 22 (Logistic), 1 (Vanilla DKT), and 0 (Rel-Gated DKT).
  - Documented that curve jumps represent small-sample binomial variance and out-of-distribution calibration drift rather than robust performance.

### Reliability-Gating Sweep & Statistical Claims
- **Reliability Bias Direction Audit**:
  - Audited `uncertainty_bias = +0.12` in `reliability.py`.
  - Confirmed that because $\hat{p} = p_{\text{true}} + \text{bias}$ increases estimated corruption probability, estimated reliability $\hat{\rho} = \prod (1 - \hat{p})$ decreases.
  - Corrected descriptive labels from "under-estimating noise" to **"over-estimating corruption noise / under-estimating reliability"**.
- **Honest Statistical Significance Reporting**:
  - Documented 95% bootstrap confidence intervals for all 4 sweep levels:
    - Level 1: $+0.0010$, 95% CI $[-0.0015, +0.0037]$ (Null)
    - Level 2: $-0.0020$, 95% CI $[-0.0040, +0.0001]$ (Null)
    - Level 3: $-0.0030$, 95% CI $[-0.0061, +0.0005]$ (Null)
    - Level 4: $-0.0054$, 95% CI $[-0.0088, -0.0020]$ (Statistically Significant)
  - Removed unsupported claims of statistical significance for Levels 2 and 3 whose confidence intervals include zero.
  - Preserved the finding that Logistic Regression remains the strongest probabilistic baseline across all levels.
- **Exported Paired Loss Evidence Artifact**:
  - Generated `paired_loss_evidence.csv` containing 216 learner-level paired loss records across all 4 levels, allowing independent verification of bootstrap intervals in 5 lines of Python.

### Test Suite & Regression Coverage
- Expanded test suite from 18 to **24 unit and targeted regression tests** in `tests/test_audit_regressions.py`, covering:
  - Target reachability and controller bound saturation (`test_target_success_reachability_and_bound_saturation`).
  - Flat bound-stuck sequences yielding zero reversals (`test_bound_stuck_zero_reversals_not_active_stability`).
  - Mastery change aggregation and secondary-skill transfer (`test_mastery_change_and_secondary_skill_transfer`).
  - Session break counts and elapsed rest gaps (`test_break_count_and_elapsed_gap_logic`).
  - Calibration bin count conservation and ECE recomputation (`test_calibration_bin_ece_consistency`).
  - Reliability bias direction mechanics (`test_reliability_bias_direction`).
- All 24 tests pass deterministically in 3.68 seconds.

---

## [Phase 2 Final Reconciliation Pass] - October 18, 2026

### 1. Calibration Upper-Bin Count Reconciliation
- Purged stale reference to auxiliary 300-learner run ($N=3$ in $[0.8, 0.9)$).
- Recomputed all upper-bin counts directly from `calibration_bins.csv` for all seeds, models, and scenarios:
  - Seed 42: $[0.7, 0.8)$ has $N=71$ (Logistic), $N=45$ (Vanilla DKT), $N=16$ (Rel-Gated DKT); $[0.8, 0.9)$ has $N=0$ and $[0.9, 1.0]$ has $N=0$ for all models.
  - Seed 123: $[0.7, 0.8)$ has 167 (Logistic), 25 (Vanilla), 34 (Rel-Gated); $[0.8, 0.9)$ has 21 (Logistic), 0 (Vanilla), 0 (Rel-Gated).
  - Seed 999: $[0.7, 0.8)$ has 41 (Logistic), 47 (Vanilla), 65 (Rel-Gated); $[0.8, 0.9)$ has 1 (Logistic), 1 (Vanilla), 0 (Rel-Gated).
  - Across all seeds: In $[0.8, 0.9)$, counts are 22 (Logistic), 1 (Vanilla), 0 (Rel-Gated); in $[0.9, 1.0]$, count is 0 across all models and seeds.
- Verified that all 45 slices conserve exactly 4,500 samples across the 10 bins and recomputed ECE matches `metrics.csv` within $\le 1 \times 10^{-4}$. Added automated regression test `test_persisted_calibration_bins_and_ece_consistency`.

### 2. Closed-Loop Mastery Table & Interpretation Repair
- Resolved table confusion by presenting both aggregation units explicitly:
  - **Per-Learner Averages ($N=60$)**:
    - `fixed_difficulty`: All skills $-0.0319 \pm 0.0590$, Primary skills $-0.0451 \pm 0.0682$, Secondary-only skills $+0.0341 \pm 0.1980$.
    - `streak_adaptive`: All skills $-0.0083 \pm 0.0562$, Primary skills $-0.0221 \pm 0.0656$, Secondary-only skills $+0.0605 \pm 0.1910$.
    - `target_adaptive`: All skills $-0.0092 \pm 0.0562$, Primary skills $-0.0234 \pm 0.0652$, Secondary-only skills $+0.0618 \pm 0.1911$.
  - **Pooled Skill Rows ($N=360$)**: Group means are identical ($-0.04512$, $-0.02209$, $-0.02338$ for primary; $+0.03410$, $+0.06048$, $+0.06179$ for secondary-only; $-0.03192$, $-0.00833$, $-0.00918$ for all skills), with row standard deviations reflecting within-learner skill variance ($\approx 0.16 - 0.20$).
- Disproved the assumption that every skill receives exactly 10 trials: 10 is the across-row mean; observed primary counts per learner-skill are 0 (60 rows, `visual_discrimination`), 6 (60 rows), 7 (60 rows), 13 (120 rows), and 21 (60 rows).
- Clarified secondary-only definition: exactly 60 rows (`visual_discrimination`) received 0 primary trials and 32 secondary transfer events across the 60-trial run. Zero rows had 0 primary and 0 transfer events.

### 3. Controller Target-Reachability Edge Cases Fixed
- Modified `Grow21ModelBasedAdaptiveController.select_next_task` in `src/grow21_simulation_benchmark/closed_loop.py`:
  - **Unreachable-High Target** ($p^* > \hat{p}(D_{\min})$): clamps to $D_{\min}$ as the most achievable difficulty.
  - **Unreachable-Low Target** ($p^* < \hat{p}(D_{\max})$): clamps to $D_{\max}$ as the most achievable difficulty.
  - **Reachable Target** ($\hat{p}(D_{\max}) \le p^* \le \hat{p}(D_{\min})$): numerically solves for $D^*$ using monotonic root finding and applies step limits.
- Added comprehensive deterministic regression tests in `tests/test_audit_regressions.py` covering all three cases, deadband logic, and step bounding.

### 4. Disentangled Reachability Diagnostics from Safety Interventions
- Added `reachability_applicable` boolean to controller telemetry:
  - Fixed Difficulty and Streak Adaptive: `reachability_applicable = False`, `target_reachable = None` (stored as `NaN` / empty in CSV, `null` in JSON, and displayed as `N/A`).
  - Target Adaptive: `reachability_applicable = True`, `target_reachable` evaluated mathematically.
- Decoupled safety intervention from mathematical reachability: safety cooldown branch no longer overrides mathematical `target_reachable`.
- Preserved separate tracking for bound saturation ($D_t == D_{\min}$ or $D_t == D_{\max}$) and safety intervention occurrences.

### 5. Controller-Predictor vs. Simulator Equation Mismatch Clarified
- Formally documented the mathematical divergence between the controller's surrogate model $\hat{p}(D) = \sigma(\text{logit}(\hat{\theta}) - \gamma D^{\nu_D})$ and the simulator's generative IRT process $p_{\text{clean}} = c_g + (1 - c_g - l_g) \sigma(a_g (\theta_{\text{eff}} - D))$.
- Generated diagnostic dataset `outputs/audit/phase_2/predictor_vs_simulator_comparison.csv` demonstrating that the controller's surrogate model underpredicts clean success by 33 to 40 percentage points at low difficulty.
- Clarified that $p^*=0.75$ is unreachable under the controller's internal equation, not the simulated environment. In the actual simulation, clamping at $D_{\min}=0.08$ generated an empirical clean success rate of $76.64\%$.

### 6. Active vs. Provisional Parameter Reconciliations
- Verified that active benchmark settings in `configs/default.yaml` and code are $\gamma = 2.4$ and $\nu_D = 1.0$.
- Reconciled with earlier design documentation values ($\gamma = 3.5, \nu_D = 2.0$), identifying them as early provisional proposals superseded during implementation.

### 7. Direct Implementation Tests & Test Suite Expansion
- Added direct tests exercising production `calculate_direction_reversals` on monotonic, immediate reversals, filtered zero-step sequences, and flat bounds.
- Added tests for unreachable-low target difficulty clamping to $D_{\max}$ and reachability applicability semantics.
- Added persisted artifact ECE and calibration bin conservation verification.
- Test suite expanded from 24 to **27 passed tests** (`pytest -v tests`).

### 8. Archive Manifest Cryptographic Verification
- Recomputed and populated true SHA-256 digests and file sizes for all 19 preserved artifacts in `outputs/archive/pre_correction_pass/manifest.json`.

### 9. Immediate & Filtered Reversals Schema Alignment
- Added dedicated production helper `calculate_immediate_reversals` for immediate consecutive trial reversals ($(D_t - D_{t-1})(D_{t-1} - D_{t-2}) < 0$).
- Defined how zero changes are handled: zero-delta transitions produce a product of 0 and reset immediate direction state (yielding 0 immediate reversals for flat steps), whereas filtered reversals filter zero transitions first.
- Expanded `ClosedLoopRunResult` and `outputs/closed_loop/closed_loop_summary.json` to persist both `immediate_direction_reversals_mean/std` and `filtered_direction_reversals_mean/std`, preserving legacy alias `direction_reversals_mean/std`.
- Verified values from final 60-learner seed-42 run:
  - Immediate reversals: Fixed $4.733 \pm 2.007$, Streak $5.467 \pm 1.658$, Target $0.000 \pm 0.000$.
  - Filtered reversals: Fixed $8.583 \pm 4.076$, Streak $17.933 \pm 2.949$, Target $0.000 \pm 0.000$.
- Reconciled safety intervention counts in documentation to match final bundled data: Fixed 290 (8.06%), Streak 97 (2.69%), Target 78 (2.17%).

### 10. Documentation Control Character Purge & Integrity Test
- Scanned repository text files and purged all non-printing ASCII control characters below 32 (backspace in math symbols, bell characters, form-feeds, and vertical-tabs).
- Added `tests/test_documentation_integrity.py` to deterministically verify that 0 invalid control characters exist in any text file.
- Expanded test suite to **32 passing tests**.

---

## [Phase 2 Documentation & Escape Sequence Repair] - October 18, 2026

### 1. LaTeX Escape Sequence and Multi-Line Fracture Repair
- Repaired fractured LaTeX expressions in `README.md` caused by non-raw string formatting:
  - Intact single-line expressions restored: `$D^{\nu_D}$`, `$\gamma = 2.4, \nu_D = 1.0$`, and `$\gamma = 3.5, \nu_D = 2.0$`.
  - Replaced corrupted tab-prefixed expressions (`\t` followed by identifier) with literal LaTeX commands: `\text{logit}`, `p_{\text{clean}}`, `\hat{\theta}`, `\theta_{\text{eff}}`, and `\times`.
- Repaired identifier and inline code formatting in `outputs/audit/phase_2/CHANGELOG_PHASE_2.md`:
  - Restored intact identifiers: `test_persisted_calibration_bins_and_ece_consistency`, `target_adaptive`, `tests/test_audit_regressions.py`, and `reachability_applicable`.
  - Repaired mathematical symbols: `$p^*`, `$D_{\min}$`, `$D_{\max}$`, and `$p_{\text{clean}}$`.

### 2. Upgraded Documentation Integrity Test Suite
- Enhanced `tests/test_documentation_integrity.py` with multi-level validation:
  - **Zero Tab Invariant**: Rejects all literal tab characters (`\t`) in Markdown and text documentation.
  - **Required Core Expressions**: Asserts that canonical LaTeX commands (`D^{\nu_D}`, `\gamma = 2.4, \nu_D = 1.0$`, `\gamma = 3.5, \nu_D = 2.0$`, `\text{logit}`, `p_{\text{clean}}`, `\theta_{\text{eff}}`) and key code identifiers (`test_persisted_calibration_bins_and_ece_consistency`, `target_adaptive`, `tests/test_audit_regressions.py`) are present and intact.
  - **Fracture Detection**: Detects broken newline splits such as standalone `u_D}` or unescaped commands.
  - **Disallowed ASCII Bytes**: Continues checking for non-printing control characters below ASCII 32.

---

## [Project Root Flattening & Structure Cleanup] - October 18, 2026

### 1. Repository Structure Consolidation
- Flattened the nested `grow21_simulation_benchmark/` project folder into the outer `grow21_simulation/` repository root, eliminating redundant nesting.
- Retained the Python package structure and import namespace `src/grow21_simulation_benchmark/` unchanged.
- All primary project assets (`README.md`, `PROJECT_STRUCTURE.md`, `pyproject.toml`, `requirements.txt`, `configs/`, `src/`, `tests/`, `outputs/`) now reside directly at the project root.

### 2. Archival of Historical Prompts and Original Input Bundle
- Moved obsolete instruction prompts (`GROW21_BENCHMARK_AUDIT_PHASE_2_PROMPT.md` and `GROW21_SIMULATOR_BENCHMARK_ANTIGRAVITY_PROMPT.md`) to `outputs/archive/project_cleanup/obsolete_prompts/` for provenance preservation.
- Moved original input bundle (`grow21_simulation_benchmark_artifacts.zip`) to `outputs/archive/project_cleanup/original_input_bundle/`.
- Generated `outputs/archive/project_cleanup/pre_cleanup_manifest.json` recording paths, sizes, actions, and SHA-256 digests for all 84 files prior to consolidation.

### 3. Path Normalization & Canonical Paths Alignment
- Verified and normalized root-relative paths in `outputs/archive/pre_correction_pass/manifest.json` with cryptographic SHA-256 verification against all 19 preserved artifacts.
- Updated `src/grow21_simulation_benchmark/run_benchmark.py` and `generate_data.py` CLI default paths to use canonical path constants from `src/grow21_simulation_benchmark/paths.py`.
- Re-installed editable package from the unified repository root (`pip install -e .`).

