# Grow21 Phase 4 Changelog: Learner Belief-State Estimation & Robustness

All modifications in Phase 4 maintain strict zero-leakage, preserve immutable Phase 2 and Phase 3 evidence, and operate purely within synthetic virtual learner simulation. No clinical or developmental validation is asserted.

---

## 1. Summary of Phase 4 Implementation

### A. Controller Belief-State Estimation Modes
- Enhanced `ModelAlignedTargetAdaptiveController` in `src/grow21_simulation_benchmark/closed_loop.py` to support 4 principled belief-state estimation modes:
  1. `heuristic_baseline`: Preserves the Phase 3 heuristic updater (8% primary step towards 0.95 on success or 0.05 on failure; no secondary transfer updates; hard observed labels). Fully identical to Phase 3 baseline.
  2. `transfer_aware`: Integrates secondary skill transfer updating based on documented game secondary associations and declared population learning rate prior ($\mu_{\text{lr}} = 0.08, \text{ratio} = 0.30 \implies \Delta \hat{\theta}_{\text{sec}} = 0.024 \cdot (0.95 - \hat{\theta}_{\text{sec}})$). Does not access true learner mastery or learning rate.
  3. `reliability_aware`: Uses pre-outcome observable estimated sensor reliability $\hat{\rho}_t$ and controller prior predicted clean probability $\hat{p}_t$ to compute Bayesian posterior expectation $\tilde{y}_t = P(y_{\text{clean}} = 1 \mid y_{\text{obs}}, \hat{\rho}_t)$ under documented sensor slip ($0.85$) and false hit ($0.04$) parameters. Eliminates severe downward belief distortion caused by motor slips.
  4. `combined_transfer_reliability`: Jointly incorporates secondary skill transfer updating and Bayesian observation noise filtering.

### B. Controller Diagnostic and Telemetry Interface
- Extended `observe_outcome` across `ControllerPolicy`, `StreakAdaptiveController`, `LegacySurrogateTargetAdaptiveController`, and `ModelAlignedTargetAdaptiveController` to optionally accept `estimated_reliability: Optional[float] = None`, ensuring complete backwards compatibility with existing 3-argument invocations.
- Stored `_last_predicted_clean_prob` inside `ModelAlignedTargetAdaptiveController` during `select_next_task`, strictly ensuring predictions are finalized prior to task execution and outcome observation.
- Added offline oracle diagnostics tracking in `simulate_closed_loop` and `run_closed_loop_comparison`. These are evaluated after step execution and never exposed to online controller decision logic.

---

## 2. Phase 4 Audit Correction Pass (October 2026)

### A. Temporal Alignment Defect Correction
- **Issue**: Previously, `closed_loop.py` recorded controller belief *after* calling `observe_outcome`, but paired it with `ev.latent_mastery_before`, conflating belief updates with step progression.
- **Correction**:
  - Captured pre-outcome controller belief `est_m_pre` right after task selection and time-gap projection, but before `observe_outcome`. Evaluated against simulator `ev.latent_mastery_before`.
  - Logged explicit fields: `estimated_mastery_primary_pre`, `true_mastery_primary_pre`, `mastery_error_primary_pre`, and corresponding secondary fields.
  - Captured post-update controller belief `est_m_post` immediately after `observe_outcome`. Evaluated against simulator `ev.latent_mastery_after`.
  - Logged explicit fields: `estimated_mastery_primary_post`, `true_mastery_primary_post`, `mastery_error_primary_post`, and corresponding secondary fields.
  - Pre-outcome probability predictions (`predicted_clean_prob`) remain strictly tied to the pre-outcome decision state.

### B. Paired Seed-Aware Cluster Bootstrap Correction
- **Issue**: The previous bootstrap sampled only `learner_id` across pooled seeds and filtered with `.isin`, conflating learners from different seeds and losing replacement multiplicity. Small $p$-values ($p < 10^{-15}$) were mathematically unsupported.
- **Correction**:
  - Replaced with `compute_paired_cluster_bootstrap` in `src/grow21_simulation_benchmark/run_phase_4_experiments.py`.
  - Formed unique cluster identifiers `(seed, learner_id)` ($K = 180$ clusters across 3 seeds and 60 learners).
  - Aligned and verified baseline and variant events on common keys `(scenario, seed, learner_id, step_idx)`.
  - Implemented vectorized cluster resampling with replacement ($B = 2,000$ replicates, fixed seed $42$) strictly preserving multiplicity of repeated draws ($k \times \text{sum}$, $k \times \text{count}$).
  - Reported $95\%$ percentile intervals (`ci_95`) and standard errors; retracted unsupported $p$-values.

### C. Evaluation Cohort Separation
- **Issue**: Prior summary tables pooled Historical and Held-Out seeds into a single headline table.
- **Correction**:
  - Reorganized `phase_4_scenario_summary.json` into distinct top-level keys: `"held_out"`, `"historical"`, `"development"`, and `"combined_eval"`.
  - Made Held-Out seeds `[777, 888, 555]` the primary headline confirmatory evidence.
  - Kept Historical seeds `[42, 123, 999]` distinct as Phase 3 baseline replication evidence.
  - Generated all 5 comparison plots on the primary Held-Out cohort.

### D. Causal Attribution & Bayesian Terminology Alignment
- Identified `temporal_dynamics` mismatch as a compound effect of initial mastery ($0.30$ vs. $0.45$), learning rate ($0.16$ vs. $0.08$), and forgetting rate ($0.07$ vs. $0.02$).
- Clarified that `distribution_shift` jointly shifts initial ability, learning, forgetting, corruption, and modality dropouts.
- Accurately defined reliability-aware updating as a *Bayesian soft-label observation update followed by a heuristic population-rate mastery update*.

### E. Artifact Preservation & Archive Manifest
- Archived all pre-correction Phase 4 artifacts into `outputs/archive/phase_4_pre_correction/`.
- Generated `outputs/archive/phase_4_pre_correction/PHASE_4_PRE_CORRECTION_MANIFEST.json` containing original paths, archived paths, byte counts, and SHA-256 hashes.

### F. Expanded Regression Test Suite
- Added 4 new regression tests to `tests/test_belief_state_estimation.py`:
  - `test_time_alignment_pre_and_post_mastery_estimation`: Verifies pre vs. latent_before and post vs. latent_after identity.
  - `test_outcome_changes_post_update_not_pre_outcome_estimate_or_prediction`: Proves outcome observation modifies post-update belief without retroactively altering pre-outcome prediction.
  - `test_paired_seed_aware_cluster_bootstrap_invariants`: Verifies cluster key distinction across seeds, multiplicity preservation, key validation, and determinism.
  - `test_split_separation_and_cohort_integrity`: Verifies separation of all 4 splits and conservation of 540,000 events.
- Test suite expanded to **55 tests** (all passing in $32.53\text{s}$).

---

## 3. Phase 5 Preflight Verification Pass & Report Errata

- **Report Erratum on Secondary Transfer**: Updated Section 3.A of `PHASE_4_STATE_ESTIMATION_REPORT.md` to document the exact simulator secondary transfer ceiling ($0.95$ rather than $1.0$).
- **Simulator Learning Transition Specification**: Fully documented the primary learning increment including clean outcome feedback factor ($0.70$ success / $0.30$ failure), difficulty engagement multiplier ($0.60 + 0.40 \cdot D$), and mean-zero process noise $\mathcal{N}(0, 0.005^2)$.
- **Bootstrap Output Key Clarification**: Audited sign convention in `run_phase_4_experiments.py`; documented that negative difference indicates error reduction; added explicit key `prob_abs_error_diff_variant_minus_baseline`.
- **Pre-Correction SHA-256 Hashes Preserved**:
  - `PHASE_4_STATE_ESTIMATION_REPORT.md`: `1330ab42bfe4d7d36114fa1efd8317594d00341276d2a4cfb57cfc5d5fa47e98`
  - `CHANGELOG_PHASE_4.md`: `c5d5e08614482e9465b73e67e80b62a50d32e921ced45076ae7da975882a5d28`

