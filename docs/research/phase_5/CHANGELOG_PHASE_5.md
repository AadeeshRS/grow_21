# Grow21 Phase 5 Changelog: Transition-Aware State Estimation, Robustness, & Model Selector

All modifications in Phase 5 maintain strict zero-leakage, preserve immutable Phase 2, 3, and 4 evidence, and operate purely within synthetic virtual learner simulation. No clinical or developmental validation is asserted.

---

## 1. Summary of Phase 5 Implementations

### A. Transition-Aware Belief-State Estimators
- **Simulator Ground-Truth Analysis**:
  In `simulator.py`, latent learner mastery evolves via practice:
  $$\Delta \theta_p = \mu_{\text{lr}} \cdot (0.95 - \theta_p) \cdot (0.60 + 0.40 \cdot D) \cdot \text{feedback}$$
  where feedback is **$0.70$ on clean success** and **$0.30$ on clean failure**.
  *Defect in Phase 3/4 Heuristic Updater*: The legacy updater pushed mastery towards $0.05$ on observed failure. In the simulator, learners always move upward toward $0.95$ when practicing, even upon failing! This caused massive negative estimation bias ($-0.0819$) and severe artificial minimum-bound difficulty saturation.
- **New Estimator Modes** (implemented in `src/grow21_simulation_benchmark/closed_loop.py`):
  1. `expected_transition`: Uses expected feedback factor based on observed label.
  2. `expected_transition_transfer`: Combines expected feedback with secondary skill transfer updates.
  3. `approx_bayesian_filter`: Calculates posterior soft-label $P(y_{\text{clean}} = 1 \mid y_{\text{obs}}, \hat{\rho}_t)$ using estimated sensor reliability, then applies expectation-based transition update toward $0.95$.
  4. `approx_bayesian_filter_transfer`: Jointly applies Bayesian noise filtering, transition-aware expected feedback, and secondary skill transfer.

### B. Initial Mastery Prior Interface & Parent-Proxy Sensitivity
- **Module Created**: `src/grow21_simulation_benchmark/initializers.py`
  - Defined `SkillPriorEstimate` dataclass (`prior_mean`, `uncertainty`, `source`, `metadata`).
  - Documented that `uncertainty` is metadata-only in Phase 5 (recording questionnaire/proxy noise std). The approximate Bayesian filter initializes its grid distribution using population dispersion $\sigma_0 = 0.18$ centered at `prior_mean`.
  - Implemented `MasteryPriorInitializer` supporting:
    * `fixed_default`: Initial mastery fixed at $0.45$ across all skills.
    * `from_provided_priors`: Accepts arbitrary partial or full skill priors and gracefully falls back to $0.45$ default without penalizing missingness as low ability or distorting initial task selection.
  - Implemented `generate_synthetic_parent_proxy_reports`:
    * Simulates noisy parent questionnaire responses with perturbation noise $\sigma \in \{0.05, 0.15, 0.30\}$ and missingness rates $\{0\%, 30\%, 60\%\}$.

### C. Offline Model Selector CLI, Controlled Seeds, & Architectural Safeguards
- **Modular `--models` Flag**:
  - Updated `train.py`, `evaluate.py`, and `run_benchmark.py` to accept `--models` (`all`, `logistic_baseline`, `vanilla_dkt`, `reliability_gated_dkt`).
  - Added `benchmark.models: "all"` to `configs/default.yaml`.
- **Deterministic Per-Model Random Seeds**:
  - Implemented controlled per-model seeds in `train_benchmark_models`: `seed + 0` for Logistic Regression, `seed + 100` for Vanilla DKT, and `seed + 200` for Reliability-Gated DKT.
  - Assigned dedicated PyTorch `Generator` instances to each model's DataLoader shuffle to eliminate cross-model RNG coupling.
  - Verified that individual model runs produce results identical to `--models all`. Added regression test `test_offline_model_selector_reproducibility_individual_vs_all`.
- **Architectural Safeguard**:
  - Enforced strict semantic barrier in `ModelAlignedTargetAdaptiveController`: offline sequence models trained to predict corrupted observation $P(y_{\text{obs}})$ cannot be passed directly into clean outcome controllers targeting $P(y_{\text{clean}})$ without an explicit calibration adapter. Attempting to do so raises `ValueError`.

### D. Four-Stage Empirical Campaign & Accounting
- **Configured Cohorts (4 cohorts, 12 seeds in total)**:
  * Development Cohort (`dev`): Seeds `[101, 202, 303]`
  * Historical Baseline Cohort (`historical`): Seeds `[42, 123, 999]`
  * Phase 4 Evaluation Cohort (`phase_4_eval`): Seeds `[777, 888, 555]`
  * Final Confirmatory Held-Out Cohort (`held_out`): Seeds `[707, 808, 909]`
- **Trial & Interaction Accounting**:
  * 60 learners per seed, 60 trials per learner = 3,600 trials per scenario per seed.
  * Unique learner trials across 12 seeds and 5 scenarios = **216,000 unique learner trials** generated per policy ($54,000$ unique trials per cohort).
  * Closed-loop estimator-policy interactions with 8 policies = **1,728,000 closed-loop interaction events** across the study ($432,000$ per cohort).
  * Stage 1 open-loop replay evaluations across 8 policies = **1,728,000 open-loop records** ($432,000$ per cohort).
- **Four Stages Executed**:
  * Stage 1: Open-loop replay state tracking on fixed benchmark traces (isolating estimation error from policy feedback).
  * Stage 2: Closed-loop policy comparison across 8 belief-state estimators, 5 scenarios, and 12 random seeds (1,728,000 interactions).
  * Stage 3: Synthetic parent-proxy sensitivity analysis across 9 noise/missingness combinations.
  * Stage 4: Offline model selector benchmark evaluation across 3 sequence architectures.
  * Generated 5 publication-grade figures in `outputs/phase_5/plots/` (and mirrored to `outputs/audit/phase_5/plots/`).

---

## 2. Phase 5.1 Audit Reconciliation & Corrections

### A. Metric Reconciliation for `base_heterogeneous`
- **Issue**: An earlier summary transcription erroneously quoted `base_heterogeneous` closed-loop MAE as `0.1001` (baseline) vs `0.1075` (variant), misreporting an error increase.
- **Resolution**: Audited against saved artifacts (`phase_5_scenario_summary.csv`, `phase_5_paired_bootstrap.csv`, and `PHASE_5_STATE_ESTIMATION_AND_SELECTION_REPORT.md`):
  * Baseline `model_aligned_baseline`: Primary Mastery Post MAE = **0.1039**, Clean Prob Error = **0.0392**, Min-Bound Saturation = **34.8%**, Reachability = **71.7%**.
  * Variant `approx_bayesian_filter_transfer`: Primary Mastery Post MAE = **0.1002**, Clean Prob Error = **0.0386**, Min-Bound Saturation = **20.1%**, Reachability = **82.5%**.
  * Paired difference $\Delta = -0.0037$ (95% Bootstrap CI $[-0.0055, -0.0020]$), confirming a **statistically significant error reduction**.

### B. Cohort & Interaction Count Corrections
- Clarified distinctions between unique learner trials (216,000 across 12 seeds $\times$ 5 scenarios) and estimator-policy interaction events (1,728,000 across 8 controllers).
- Updated documentation across report, changelog, reproduction guide, and bundle index to reflect all 12 seeds and 4 cohorts.

### C. Prior Interface Specification
- Explicitly documented fields: `prior_mean`, `uncertainty`, `source`, `metadata`.
- Clarified that `uncertainty` is metadata-only in Phase 5, preserving existing empirical results while establishing the contract for future questionnaire integration.

### D. Model Selector Reproducibility & Regression Test
- Added per-model random seed offsets (`0`, `100`, `200`) and dedicated DataLoader `Generator` instances in `train_benchmark_models`.
- Added automated regression test `test_offline_model_selector_reproducibility_individual_vs_all` in `tests/test_phase_5_state_estimation.py`. Total test suite expanded to **66 passing tests**.

### E. Reproduction Guide Artifact Alignment
- Updated `REPRODUCE.md` Section 6 to verify against the packaged summary CSV `phase_5_scenario_summary.csv` instead of referencing the unbundled raw event file `phase_5_event_metrics.csv` (~876 MB). Documented the command to regenerate raw events if desired.

---

## 3. Invariants Maintained

1. **Zero Ground-Truth Leaks**: Controller logic has zero access to latent learner state, true learning rate, true sensor corruption parameters, or process noise.
2. **Canonical Formula Parity**: The clean-success outcome equation remains strictly unified across all 9 games via `outcome_model.py`.
3. **Temporal Integrity**: Pre-outcome mastery and clean-probability predictions are finalized strictly before task execution and outcome observation.
4. **Historical Immutability**: All Phase 2, 3, and 4 audit files and evidence archives remain untruncated and unaltered.
