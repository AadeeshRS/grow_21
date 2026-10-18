# Grow21 Phase 4 State-Estimation Audit Correction Summary

**Date:** October 2026  
**Status:** Audit & Correction Pass Complete  
**Scope:** Closed-loop state estimation measurement, paired cluster bootstrap inference, cohort reporting separation, causal interpretations, and documentation alignment.  
**Applicability:** Synthetic research benchmark only; no Grow21 frontend integration; no clinical/developmental validation.

---

## 1. Executive Summary

This audit and correction pass systematically investigated and resolved two empirical implementation defects in the Phase 4 evaluation suite, corrected three interpretive and methodological issues, and regenerated all Phase 4 artifacts:

1. **Temporal Alignment Defect**: Corrected the evaluation timing in `closed_loop.py` to prevent comparing post-outcome controller mastery beliefs against pre-outcome simulator latent mastery. Explicit pre-outcome and post-update fields are now logged and reported separately.
2. **Paired Cluster Bootstrap Defect**: Replaced the flawed learner-ID-only `.isin` bootstrap with a seed-aware cluster bootstrap over unique `(seed, learner_id)` clusters using common event keys `(scenario, seed, learner_id, step_idx)` with $B = 2,000$ replicates preserving draw multiplicity. Retracted unsupported small $p$-values.
3. **Cohort Separation**: Separated results into **Held-Out Evaluation** (seeds `[777, 888, 555]`, $N = 216,000$ events, primary headline confirmatory evidence), **Historical Comparison** (seeds `[42, 123, 999]`, $N = 216,000$ events, Phase 3 baseline replication), and **Development Diagnostics** (seeds `[101, 202, 303]`, $N = 108,000$ events, exploratory).
4. **Causal Interpretation & Scenario Confounding**: Corrected claims attributing prediction gaps in `temporal_dynamics` solely to forgetting mismatch (in reality, initial mastery, learning rates, and forgetting rates simultaneously differ) and in `distribution_shift` solely to initial mastery mismatch. Used associative language ("associated with") for closed-loop policy comparisons.
5. **Clarification of "Bayesian" Terminology**: Explicitly defined the reliability-aware update as a Bayesian soft-label observation update followed by a heuristic population-rate mastery update, rather than a full posterior density over continuous mastery distributions.

---

## 2. Detailed Audit Corrections

### A. Defect 1: Time Alignment in Mastery Estimation

- **Prior Implementation Defect**:
  In `closed_loop.py`, `observe_outcome` was called immediately after `sim.step(...)`. The event row logged `estimated_mastery_primary` from the *already updated* controller, but paired it with `ev.latent_mastery_before`. Consequently, the metric evaluated the controller's post-outcome belief against the learner's pre-outcome ability, conflating belief updates with trial progression.
- **Corrected Implementation**:
  - **Pre-Outcome State**: Captured immediately after task selection and time-gap projection, but strictly *before* `observe_outcome`. Evaluated against simulator `ev.latent_mastery_before`. Logged as `estimated_mastery_primary_pre`, `true_mastery_primary_pre`, and `mastery_error_primary_pre`.
  - **Post-Update State**: Captured immediately *after* `observe_outcome`. Evaluated against simulator `ev.latent_mastery_after`. Logged as `estimated_mastery_primary_post`, `true_mastery_primary_post`, and `mastery_error_primary_post`.
  - Secondary skill fields (`visual_discrimination`) follow identical before/after pairing.
  - Pre-outcome probability prediction (`predicted_clean_prob`) remains strictly finalized prior to outcome observation.

### B. Defect 2: Paired Seed-Aware Cluster Bootstrap

- **Prior Implementation Defect**:
  The previous bootstrap gathered `learner_ids = pol_df["learner_id"].unique()` (0..59) across all seeds and filtered with `.isin(sampled_learners)`. This conflated learner 0 from seed 42 with learner 0 from seed 123, and repeated draws during resampling had zero multiplicity effect because `.isin` returns each row once. The reported small $p$-values (e.g. $p < 10^{-15}$) were mathematically invalid.
- **Corrected Implementation**:
  - The resampling cluster unit is strictly `(seed, learner_id)`. For 3 seeds and 60 learners, $K = 180$ independent clusters.
  - Baseline and variant event sequences are verified to match exactly on `(scenario, seed, learner_id, step_idx)`; any mismatch halts execution.
  - Resampling is performed with replacement for $B = 2,000$ replicates. When a cluster index is drawn multiple times, its complete event sequence is included with exact multiplicity ($k \times \text{sum}$ and $k \times \text{count}$).
  - Final inference reports $95\%$ percentile intervals (`ci_95`) and bootstrap standard errors. Tiny $p$-values are eliminated.

### C. Defect 3: Distinct Evaluation Cohorts

- **Prior Defect**: Historical seeds (`[42, 123, 999]`) and held-out seeds (`[777, 888, 555]`) were pooled as the sole headline evaluation.
- **Corrected Structure**:
  - **HELD-OUT EVALUATION** (`[777, 888, 555]`): Primary generalization headline evidence ($N = 216,000$ events). Fixed population priors with zero seed tuning.
  - **HISTORICAL BASELINE** (`[42, 123, 999]`): Comparison to earlier Phase 3 baseline runs ($N = 216,000$ events).
  - **DEVELOPMENT COHORT** (`[101, 202, 303]`): Diagnostic offline exploration ($N = 108,000$ events).
  - **COMBINED SUMMARY**: Secondary descriptive reference ($N = 432,000$ events), clearly labeled as combined.

### D. Defect 4: Multi-Factor Scenario Dynamics & Causal Language

- **`temporal_dynamics`**: Mismatch is driven jointly by initial mastery ($\mu = 0.30$ vs prior $0.45$), learning rate ($\mu = 0.16$ vs prior $0.08$), and forgetting rate ($\mu = 0.07$ vs prior $0.02$). The prediction gap (+4.50 pp) is a compound dynamic effect, not forgetting alone.
- **`distribution_shift`**: Concurrently varies initial mastery, learning rates, forgetting rates, physical corruption probabilities, modality dropouts, and sensor uncertainty calibration.
- **Closed-Loop Associative Wording**: Controller belief updates alter difficulty selection, generating divergent learner practice trajectories. Results reflect closed-loop system behavior and are reported as "associated with" rather than isolated internal causal mechanisms.

### E. Defect 5: Bayesian Terminology Clarification

- The reliability-aware update calculates a Bayesian soft target for the clean binary outcome:
  $$\tilde{y}_t = P(y_{\text{clean}} = 1 \mid y_{\text{obs}}, \hat{\rho}_t)$$
  using pre-outcome reliability $\hat{\rho}_t$ and confusion rates ($0.85$ false failure on slip, $0.04$ false success on hit).
- It then updates mastery toward $\theta_{\text{target}} = \tilde{y}_t \cdot 0.95 + (1 - \tilde{y}_t) \cdot 0.05$ with heuristic population learning rate $\mu_{\text{lr}} = 0.08$.
- It is accurately designated as a *Bayesian soft-label observation update followed by a heuristic population-rate mastery update*, not a full continuous posterior distribution over mastery $\theta$.

---

## 3. Audit Correction Reconciliation Table

| Metric / Claim Area | Previous Claim (Pre-Correction) | Corrected Finding (Audited Pass) | Evidence File / Artifact | Technical Reason for Correction |
| :--- | :--- | :--- | :--- | :--- |
| **Mastery Error Timing** | Single MAE: $0.1075$ (Baseline), $0.1085$ (Transfer) in `base_heterogeneous` | Pre-MAE: $0.1062$, Post-MAE: $0.1049$ (Baseline); Pre-MAE: $0.1056$, Post-MAE: $0.1046$ (Transfer) | `phase_4_mastery_estimation_metrics.csv` | Resolved temporal mismatch; pre-outcome belief evaluated against `latent_mastery_before`, post-update against `latent_mastery_after`. |
| **Probability Error Significance in Base** | Statistically significant error reduction claimed ($p < 10^{-15}$) | Inconclusive / within noise: Point estimate $+0.0001$, $95\%\text{ CI}: [-0.0005, +0.0007]$ | `phase_4_scenario_summary.json` (`held_out`) | Seed-aware cluster bootstrap with multiplicity shows confidence interval crosses zero. Small $p$-value retracted. |
| **$D_{\min}$ Saturation Reduction** | Pooled reduction $-11.9\text{ pp}$ (Transfer), $-21.6\text{ pp}$ (Combined) | Confirmatory Held-Out reduction: Transfer: $-12.94\text{ pp}$ ($95\%\text{ CI}: [-14.14, -11.78]$); Combined: $-22.06\text{ pp}$ ($95\%\text{ CI}: [-23.58, -20.62]$) | `phase_4_scenario_summary.json` (`held_out`) | Evaluated strictly on fresh held-out cohort `[777, 888, 555]`. Behavioral effect remains robust and significant. |
| **Attribution in `temporal_dynamics`** | Attributed prediction gap (+4.48 pp) solely to forgetting rate mismatch | Multi-factor mismatch: Initial mastery ($0.30$ vs $0.45$), learning rate ($0.16$ vs $0.08$), and forgetting ($0.07$ vs $0.02$) all differ simultaneously | `configs/scenarios.yaml`, `PHASE_4_STATE_ESTIMATION_REPORT.md` | Ablation design does not isolate forgetting alone; initial ability and acquisition speed also mismatch controller priors. |
| **Bayesian Description** | Implied full Bayesian belief updater over mastery | Bayesian soft-label likelihood update on binary outcome followed by heuristic fixed-rate mastery adjustment | `src/grow21_simulation_benchmark/closed_loop.py` | Accurate probabilistic characterization; no continuous probability density over mastery is tracked. |
| **Data Partitioning** | Sole headline table pooled historical (`[42, 123, 999]`) and held-out (`[777, 888, 555]`) seeds | Held-Out seeds presented as primary headline evidence; Historical seeds presented separately as baseline replication | `phase_4_scenario_summary.json`, `PHASE_4_STATE_ESTIMATION_REPORT.md` | Preserves methodological separation between historical comparison and fresh confirmatory validation. |

---

## 4. Verification and Regression Testing

- **Regression Tests Added**:
  - `test_time_alignment_pre_and_post_mastery_estimation`: Verifies exact mathematical identity of pre-outcome error against `latent_mastery_before` and post-update error against `latent_mastery_after`.
  - `test_outcome_changes_post_update_not_pre_outcome_estimate_or_prediction`: Proves observing an outcome updates the post-update belief state but cannot retroactively modify pre-outcome diagnostics or predictions.
  - `test_paired_seed_aware_cluster_bootstrap_invariants`: Verifies distinct cluster indexing across seeds, preservation of draw multiplicity, strict key checking, and seed determinism.
  - `test_split_separation_and_cohort_integrity`: Verifies distinct JSON keys for all four splits, exact event conservation (540,000 total events), and explicit pre/post timing categories in metrics CSV.
- **Suite Execution**:
  - Full test suite passed: **55 passed in 32.53s** (0 failed, 0 skipped).
  - Documentation integrity scan passed: 0 invalid control characters, 0 literal tabs in markdown, intact LaTeX and code identifiers.
