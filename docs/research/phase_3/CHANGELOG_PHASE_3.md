# Grow21 Phase 3 — Chronological Changelog

This document logs all architectural refactorings, mathematical corrections, test additions, and artifact generations completed during **Phase 3: Target-Success Predictor Alignment**.

---

## 1. Architectural & Mathematical Refactorings

### 1.1 Canonical Outcome Model (`outcome_model.py`)
- **New Module**: `src/grow21_simulation_benchmark/outcome_model.py`
- **Purpose**: Establishes a single, dependency-safe source of truth for clean task-success probability modeling, shared identically between the simulator and target-adaptive controllers.
- **Key Functions**:
  - `compute_difficulty_curve(difficulty)`: Enforces non-linear convex inflection $\text{curve}(D) = 0.25 D + 0.75 D^{1.35}$.
  - `compute_difficulty_threshold(difficulty_offset, difficulty_scale, difficulty)`: Translates abstract difficulty $D \in [0, 1]$ into item threshold.
  - `compute_composite_effective_mastery(primary, secondary)`: Blends 85% primary and 15% secondary mastery for transfer-enabled tasks.
  - `compute_clean_task_success_prob(...)`: Evaluates latent clean probability accounting for discrimination, difficulty threshold, guessing floor, lapse ceiling, and session fatigue.
  - `compute_game_clean_success_prob(game, ...)`: High-level wrapper accepting game configuration objects or dictionaries.
  - `solve_target_difficulty(game, ...)`: Bounded numerical bisection solver on $[D_{\min}, D_{\max}]$ classifying reachability (`reachable`, `unreachable_high`, `unreachable_low`).

### 1.2 Simulator Integration (`simulator.py`)
- **Refactoring**: Delegated `self.calculate_difficulty_threshold`, composite effective mastery calculation, and latent clean success probability evaluation to `outcome_model.py`.
- **Preservation**: Kept synthetic learner dynamics, practice gains, forgetting over session gaps, and multimodal sensor corruption untouched.

### 1.3 Task Registry (`tasks.py`)
- **Refactoring**: Delegated `calculate_difficulty_threshold` to `outcome_model.compute_difficulty_threshold`.

### 1.4 Closed-Loop Controller Redesign (`closed_loop.py`)
- **Legacy Controller Preservation**: Retained `LegacySurrogateTargetAdaptiveController` (with alias `Grow21ModelBasedAdaptiveController`) to maintain Phase 2 test compatibility and enable direct within-experiment 4-policy comparisons.
- **New Model-Aligned Controller**: Added `ModelAlignedTargetAdaptiveController` implementing:
  - Exact IRT canonical outcome prediction;
  - Decoupled reachability analysis and bisection difficulty search;
  - Projected session trial counter tracking;
  - Population-prior belief forgetting decay ($\mu = 0.020$) and fatigue sensitivity ($\mu = 0.020$);
  - Strict absence of simulator ground truth leakage;
  - Independent deadband and step-change constraints.
- **4-Way Closed-Loop Simulation Support**: Updated `run_closed_loop_comparison` to support running 4 policies simultaneously with comprehensive diagnostic tracking (`predicted_clean_prob`, `prediction_gap_clean`, `target_error`, `reachability_status`, `determination_rule`, etc.).

### 1.5 Centralized Paths (`paths.py`)
- **Added Canonical Constants**:
  - `PHASE_3_OUTPUTS_DIR` (`outputs/phase_3`)
  - `PHASE_3_CLOSED_LOOP_DIR` (`outputs/phase_3/closed_loop`)
  - `PHASE_3_DIAGNOSTICS_DIR` (`outputs/phase_3/diagnostics`)
  - `PHASE_3_PLOTS_DIR` (`outputs/phase_3/plots`)
  - `PHASE_3_AUDIT_DIR` (`outputs/audit/phase_3`)

### 1.6 Default Configuration (`configs/default.yaml`)
- **Updated `target_adaptive`**: Cleanly deprecated `gamma` and `nu_D` into `legacy_surrogate_target_adaptive`. Configured active `target_adaptive` with documented population priors `fatigue_sensitivity_prior: 0.02` and `forgetting_rate_prior: 0.02`.

---

## 2. Experimental Scripts & Automation

### 2.1 Full Experimental Campaign Runner (`run_phase_3_experiments.py`)
- **New Script**: `src/grow21_simulation_benchmark/run_phase_3_experiments.py`
- **Features**:
  - Automatically runs 7,938-point formula parity audit across all 9 games.
  - Executes multi-scenario ($5$), multi-seed ($3$), multi-policy ($4$) closed-loop simulations for 60 learners $\times$ 60 trials ($N = 216,000$ events total).
  - Persists event CSVs, mastery summaries, metric tables, and hierarchical JSONs.
  - Automatically renders 4 publication-quality comparison figures in `outputs/phase_3/plots/`.
  - Mirrors core audit deliverables into `outputs/audit/phase_3/`.

### 2.2 Standard Closed-Loop Runner (`run_closed_loop.py`)
- **Safety Update**: Updated default `--output-dir` to `PHASE_3_CLOSED_LOOP_DIR` to prevent accidental overwrites of preserved Phase 2 evidence.

---

## 3. Regression & Unit Tests

### 3.1 New Phase 3 Test Suite (`tests/test_outcome_model_alignment.py`)
Added 10 deterministic tests:
1. `test_shared_clean_success_parity_all_nine_games`: Bitwise parity check across 9 games, 5 mastery levels, 6 difficulties, 6 session counters, and 3 fatigue values.
2. `test_difficulty_curve_nonlinearity_and_no_linear_substitute`: Verifies non-linear convex inflection and rejects linear approximations.
3. `test_guess_lapse_and_fatigue_consistency`: Verifies guessing floors, lapse ceilings, and fatigue onset trial thresholds.
4. `test_target_reachability_and_bisection_solver`: Validates bisection precision on reachable targets and correct clamping on unreachable targets.
5. `test_step_limiting_and_deadband_constraints`: Confirms that deadband and step limits apply after target-solving and preserve stability.
6. `test_session_gap_projection_ordering`: Asserts synchronization of session trial counters between controller and simulator.
7. `test_controller_population_priors_and_no_leakage`: Proves zero access to hidden learner state (true mastery, true fatigue, true forgetting).
8. `test_non_target_diagnostics_and_safety_separation`: Proves that non-target controllers report N/A diagnostics and safety cooldowns do not overwrite mathematical feasibility.
9. `test_closed_loop_four_policy_comparison_and_diagnostics_consistency`: Exercises 4-policy simulation and validates schema completeness.
10. `test_legacy_vs_aligned_bound_saturation_contrast`: Directly contrasts legacy lower-bound saturation against aligned controller behavior.

### 3.2 Test Suite Execution
- Pre-Phase 3 Baseline: 32 tests passed.
- Post-Phase 3 Test Suite: 42 tests passed in 8.72s.

---

## 4. Preservation & Provenance

- **Pre-Phase-3 Source Archive**: Exact snapshots of 14 pre-Phase-3 source and config files preserved in `outputs/archive/pre_phase_3_source/` with SHA-256 digests in `outputs/archive/pre_phase_3_source/manifest.json`.
- **Phase 2 Invariant**: All Phase 2 benchmark, closed-loop, and audit directories preserved completely untouched.
- **Evidence Bundle**: All deliverables originally packaged in `evidence/phase_3/Grow21_Phase_3_Model_Alignment_Evidence.zip` (local/release archive, excluded from Git via `.gitignore`; canonical deliverables tracked directly in repository).
