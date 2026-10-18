# Grow21 Phase 5 — Transition-Aware State Estimation, Robustness, and Testing Model Selector Report

**Date:** October 18, 2026  
**Status:** Complete, Confirmatory Evaluated, and Audited  
**Repository Corpus:** `grow21_simulation`  
**Execution Context:** Standalone synthetic simulation benchmark (Python 3.13.11, PyTorch 2.6.0+cpu, NumPy 2.2.3, Pandas 2.2.3, Scikit-Learn 1.6.1)  
**Evidence Artifact:** `evidence/phase_5/Grow21_Phase_5_Evidence.zip` *(local/release archive, excluded from Git)*

---

## 1. Executive Summary & Scientific Disclaimers

### 1.1 Scope and Objective
Phase 5 addresses the primary source of error identified in Phase 4 of the Grow21 simulation project: **learner-state estimation mismatch and population-prior divergence**, especially under severe temporal dynamics and distribution shifts. 

The core objectives accomplished in Phase 5 are:
1. **Transition-Aware Belief Updating:** Formulated, implemented, and empirically compared transition-aligned estimators against legacy heuristic baselines. Evaluated both expected-transition propagation and an approximate Bayesian state filter that rigorously separates pre-outcome observation likelihood updating from dynamical state propagation.
2. **Initial Mastery Prior Interface:** Designed an extensible prior initialization protocol providing a clean modular contract for future caregiver/parent questionnaire integration (`fixed_default` 0.45 baseline and `provided_skill_priors` with fallback to default, strictly ensuring missingness is never treated as low ability). Tested sensitivity to noisy reports via a synthetic proxy measurement experiment.
3. **Testing Model Selector & Semantic Safeguard:** Added an offline benchmark model selector CLI/config switch (`logistic_baseline`, `vanilla_dkt`, `reliability_gated_dkt`, `all`) while enforcing a **strict semantic safeguard** that prohibits routing observed-success classifiers $P(y_{\text{obs}})$ directly into closed-loop controllers targeting clean task success $P(y_{\text{clean}})$.
4. **Open-Loop Replay vs. Closed-Loop Evaluation:** Evaluated all 8 state estimator variants under identical exogenous action sequences (open-loop replay) and adaptive closed-loop control across 4 distinct multi-seed cohorts (Development, Historical, Phase 4 Evaluation, and Final Held-Out).
5. **Preflight Audit & Bootstrap Sign Convention Rectification:** Audited Phase 4 equations, documented the complete simulator learning dynamics, corrected notation on secondary transfer ceiling (0.95), and unified paired cluster bootstrap reporting under a single explicit sign convention (`diff = variant - baseline`, where negative differences represent error reduction).

### 1.2 Mandatory Scientific Invariants & Disclaimers
> [!IMPORTANT]
> **Synthetic-Only Simulation:** All virtual learners, response sequences, physical noise modalities, and caregiver reports evaluated in this project are strictly synthetic computational constructs. This study does NOT constitute clinical, cognitive, or developmental validation of child learning trajectories, and NO real Grow21 child or caregiver data were collected or utilized.

> [!CAUTION]
> **Strict No-Simulator-Truth Leakage:** Under no circumstances did closed-loop controllers or state estimators access hidden simulator ground truth. Pre-task decisions and post-outcome belief updates had access strictly to observable task parameters, observable binary responses ($y_{\text{obs}}$), pre-outcome reliability estimates ($\hat{\rho}$), documented population priors, and elapsed wall-clock time. True latent masteries ($\theta$), per-learner learning/forgetting rates ($\eta, \lambda$), fatigue sensitivities ($\alpha$), clean probabilities ($p_{\text{clean}}$), clean outcomes ($y_{\text{clean}}$), and corruption flags were strictly sequestered.

> [!NOTE]
> **Simulator Invariance:** The ground-truth simulator outcome equations, primary learning increment dynamics, secondary skill transfer rules, exponential forgetting functions, and observation noise distributions in `simulator.py` remained 100% untouched throughout Phase 5.

---

## 2. Phase 4 Preflight Audit & Methodological Rectification

Before launching new algorithmic experiments, an independent preflight audit of Phase 4 artifacts was conducted.

### 2.1 Verification of Historical Artifacts
- **Phase 4 ZIP Verification:** Opened and verified `outputs/audit/phase_4/Grow21_Phase_4_State_Estimation_Evidence.zip` (69 files, 278.3 MB uncompressed). Verification via `zipfile.testzip()` returned `None` (0 corrupted files).
- **Baseline Test Suite Execution:** Ran existing regression suite prior to any modifications: **55 of 55 tests passed** in 56.4 seconds.
- **Pre-Phase-5 SHA-256 Baseline Hashes Recorded:**
  - `src/grow21_simulation_benchmark/simulator.py`: `b5dd48e468d78ad7bc33a3a09992e1e21c7e2b88a399698bea719ec023eb1ee5`
  - `src/grow21_simulation_benchmark/closed_loop.py`: `079e4506e961d5d01b6ea49d4b65478d1bf040caee2bf4fee4a26a59da63bc80`
  - `src/grow21_simulation_benchmark/run_phase_4_experiments.py`: `39bedd07d17aa43d71788a137f858056807ed5f9ebf75452b095b8a47f569dff`
  - `outputs/audit/phase_4/PHASE_4_STATE_ESTIMATION_REPORT.md`: `1330ab42bfe4d7d36114fa1efd8317594d00341276d2a4cfb57cfc5d5fa47e98`
  - `outputs/audit/phase_4/CHANGELOG_PHASE_4.md`: `c5d5e08614482e9465b73e67e80b62a50d32e921ced45076ae7da975882a5d28`

### 2.2 Paired Cluster Bootstrap Sign Convention Rectification
In Phase 4, the paired cluster bootstrap computed differences as:
$$\Delta = \text{Error}_{\text{variant}} - \text{Error}_{\text{baseline}}$$
Under this definition, negative values represent error reduction (the variant achieved lower error than the baseline), while positive values represent error increase. However, the output column in Phase 4 was labelled `prob_abs_error_reduction`, which created semantic ambiguity.

In Phase 5, this convention was formally standardized:
- The output field is explicitly named `prob_abs_error_diff_variant_minus_baseline`.
- An alias `point_estimate` is reported alongside `ci_lower`, `ci_upper`, and an explicit `sign_convention` string.
- An explicit categorical `interpretation` tag is emitted (`significant_reduction`, `significant_increase`, or `not_significant_ci_crosses_zero`). A result is never claimed to be statistically significant if the 95% bootstrap confidence interval crosses zero.
- Section 11 Erratum was appended to `PHASE_4_STATE_ESTIMATION_REPORT.md` and recorded in `CHANGELOG_PHASE_4.md`.

### 2.3 Simulator Ground-Truth Dynamics Specification
The preflight audit verified the exact ground-truth simulator learning dynamics implemented in `simulator.py`:
1. **Primary Transition:**
   $$\Delta \theta_{\text{primary}} = \eta_k \cdot (0.95 - \theta_{\text{primary}}) \cdot \Phi_{\text{feedback}} \cdot \Psi_{\text{difficulty}} + \epsilon_{\text{process}}$$
   where:
   - $\eta_k \sim \mathcal{N}(\mu_{\eta}, \sigma_{\eta}^2)$ is the learner's latent per-skill learning rate.
   - $\Phi_{\text{feedback}} = 0.70$ if clean outcome $y_{\text{clean}} = 1$, and $0.30$ if clean outcome $y_{\text{clean}} = 0$. Both clean success and clean failure produce positive expected learning increments.
   - $\Psi_{\text{difficulty}} = 0.60 + 0.40 \cdot D_t$ scales engagement with task challenge.
   - $\epsilon_{\text{process}} \sim \mathcal{N}(0, 0.005^2)$ is mean-zero stochastic fluctuation.
2. **Secondary Skill Transfer:**
   $$\Delta \theta_{\text{secondary}} = 0.30 \cdot \eta_{\text{secondary}} \cdot (0.95 - \theta_{\text{secondary}})$$
   capped strictly at the mastery ceiling of $0.95$ (correcting earlier report text that loosely referenced $1.0$).
3. **Exponential Forgetting:**
   $$\theta_k(t + \Delta t) = 0.05 + (\theta_k(t) - 0.05) \cdot \exp(-\lambda_k \cdot \Delta t)$$
   decaying toward the minimum skill floor $0.05$ over elapsed hours $\Delta t$.

---

## 3. Transition-Aware State Estimators (Part B)

### 3.1 The Central Modeling Problem
The Phase 3 heuristic updater updated mastery according to:
$$\theta_{\text{new}} = \begin{cases} \theta_{\text{curr}} + 0.08 \cdot (0.95 - \theta_{\text{curr}}) & \text{if } y_{\text{obs}} = 1 \\ \theta_{\text{curr}} - 0.08 \cdot (\theta_{\text{curr}} - 0.05) & \text{if } y_{\text{obs}} = 0 \end{cases}$$
The Phase 4 reliability-aware updater replaced hard observed labels with an estimated clean-success posterior $q_t$, but retained the same directional target mapping:
$$\theta_{\text{new}} = \theta_{\text{curr}} + 0.08 \cdot q_t \cdot (0.95 - \theta_{\text{curr}}) - 0.08 \cdot (1 - q_t) \cdot (\theta_{\text{curr}} - 0.05)$$
This embodied a fundamental structural mismatch: **in the true simulator, primary skills never decay toward 0.05 upon failure.** A clean failure produces an expected increment with feedback factor $0.30$. The heuristic update systematically punished learners for failure, inducing strong negative mastery bias (down to $-0.142$ in noisy scenarios).

### 3.2 Candidate Estimator Formulations
Phase 5 implements and benchmarks 8 distinct estimator policies under identical conditions:

1. `model_aligned_baseline` (Phase 3 Baseline): Heuristic update on primary skill; moves toward 0.95 on success, 0.05 on failure.
2. `transfer_aware`: Adds secondary skill transfer ($0.30 \times 0.08 \times (0.95 - \theta_{\text{sec}})$), retaining legacy primary heuristic.
3. `reliability_aware`: Uses soft-label posterior $q_t$ to interpolate between 0.95 and 0.05 targets.
4. `combined_transfer_reliability`: Combines Phase 4 soft-label updating with secondary transfer.
5. `expected_transition` (Phase 5 Transition-Aligned Baseline):
   Computes posterior clean-success probability $q_t = P(y_{\text{clean}} = 1 \mid y_{\text{obs}}, \hat{\rho}_t)$ under declared observation-noise channels:
   $$q_t = \frac{\hat{p}_{\text{clean}} \cdot P(y_{\text{obs}} \mid y_{\text{clean}}=1, \hat{\rho}_t)}{\hat{p}_{\text{clean}} \cdot P(y_{\text{obs}} \mid y_{\text{clean}}=1, \hat{\rho}_t) + (1 - \hat{p}_{\text{clean}}) \cdot P(y_{\text{obs}} \mid y_{\text{clean}}=0, \hat{\rho}_t)}$$
   Then updates primary mastery using population priors:
   $$\mathbb{E}[\Phi_{\text{feedback}}] = 0.30 + 0.40 \cdot q_t$$
   $$\mathbb{E}[\Psi_{\text{difficulty}}] = 0.60 + 0.40 \cdot D_t$$
   $$\hat{\theta}_{\text{post}} = \operatorname{clip}\left(\hat{\theta}_{\text{pre}} + \bar{\eta} \cdot (0.95 - \hat{\theta}_{\text{pre}}) \cdot \mathbb{E}[\Phi_{\text{feedback}}] \cdot \mathbb{E}[\Psi_{\text{difficulty}}], 0.05, 0.95\right)$$
6. `expected_transition_transfer`: Extends expected transition by adding expected secondary skill transfer:
   $$\hat{\theta}_{\text{sec, post}} = \operatorname{clip}\left(\hat{\theta}_{\text{sec, pre}} + 0.30 \cdot \bar{\eta} \cdot (0.95 - \hat{\theta}_{\text{sec, pre}}), 0.05, 0.95\right)$$
7. `approx_bayesian_filter`:
   Maintains a discrete probability distribution over a 41-point uniform grid on $[0.05, 0.95]$ per skill.
   - **Step 1 (Observation Likelihood Update):** Evaluates clean success likelihood $P(y_{\text{clean}}=1 \mid \theta_g, D_t)$ at each grid point $\theta_g$ using the canonical outcome model. Evaluates observation likelihood:
     $$P(y_{\text{obs}} \mid \theta_g, \hat{\rho}_t) = P(y_{\text{clean}}=1 \mid \theta_g) \cdot P(y_{\text{obs}} \mid y_{\text{clean}}=1, \hat{\rho}_t) + P(y_{\text{clean}}=0 \mid \theta_g) \cdot P(y_{\text{obs}} \mid y_{\text{clean}}=0, \hat{\rho}_t)$$
     Applies Bayes' rule and normalizes grid weights $\mathbf{w}_t \propto \mathbf{w}_{t-} \odot P(y_{\text{obs}} \mid \boldsymbol{\theta}, \hat{\rho}_t)$.
   - **Step 2 (State Transition Propagation):** Propagates grid points through expected primary learning increment $\theta_g' = f(\theta_g, D_t, q_t)$ and projects onto the fixed grid via conservative linear interpolation.
   - **Point Estimate Extraction:** Posterior mean $\hat{\theta} = \sum_g w_g \theta_g$ is extracted for the controller.
8. `approx_bayesian_filter_transfer`: Approximate Bayesian filter combined with expected secondary transfer update.

---

## 4. Initial Mastery Prior Interface & Synthetic Proxy Sensitivity (Part C)

### 4.1 Interface Specification
To support future caregiver/parent questionnaire responses prepared separately by teammates, `initializers.py` provides an isolated protocol:
- Data contract `SkillPriorEstimate`:
  * `prior_mean`: Float bounded in $[0.05, 0.95]$.
  * `uncertainty`: Optional non-negative float recording proxy/questionnaire report standard error or noise. Documented as metadata-only in Phase 5; the approximate Bayesian filter initializes grid dispersion using the known population parameter $\sigma_0 = 0.18$.
  * `source`: Provenance tag (e.g. `"fixed_default"`, `"synthetic_parent_proxy"`).
  * `metadata`: Extensible dictionary for questionnaire question IDs and completion timestamps.
- Initializer `MasteryPriorInitializer`:
  * `MasteryPriorInitializer.fixed_default(0.45)`: Produces the baseline constant belief vector.
  * `MasteryPriorInitializer.from_provided_priors(provided_dict, fallback_default=0.45)`: Accepts bounded skill estimates with uncertainty metadata.
- **Missingness Handling:** Omitted skills explicitly fall back to `0.45`. Missing values are never interpreted as low ability ($0.05$).

### 4.2 Synthetic Parent-Proxy Sensitivity Experiment
To evaluate the theoretical utility of informative priors under realistic noise, a synthetic proxy experiment was conducted across 11 conditions:
- Fixed default baseline ($0.45$)
- Uninformative prior: $U(0.05, 0.95)$
- 9 combinations of Gaussian noise $\sigma \in \{0.05, 0.15, 0.30\}$ and caregiver omission rates $\in \{0\%, 30\%, 60\%\}$.

**Empirical Findings:**
| Condition | Noise Std ($\sigma$) | Missingness | Post Primary Mastery MAE | Probability Error | Target Reachability |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `fixed_default_0.45` | $0.00$ | $0\%$ | **0.1153** | 0.0427 | 73.3% |
| `uninformative_uniform` | $0.26$ | $0\%$ | **0.1426** (+23.7%) | 0.0601 (+40.7%) | 83.9% |
| `proxy_noise_0.05_miss_0.00` | $0.05$ | $0\%$ | **0.1148** (-0.4%) | 0.0431 | 74.5% |
| `proxy_noise_0.05_miss_0.30` | $0.05$ | $30\%$ | **0.1153** (0.0%) | 0.0432 | 73.6% |
| `proxy_noise_0.05_miss_0.60` | $0.05$ | $60\%$ | **0.1162** (+0.8%) | 0.0426 | 72.3% |
| `proxy_noise_0.15_miss_0.00` | $0.15$ | $0\%$ | **0.1150** (-0.3%) | 0.0432 | 75.3% |
| `proxy_noise_0.15_miss_0.30` | $0.15$ | $30\%$ | **0.1156** (+0.2%) | 0.0431 | 73.9% |
| `proxy_noise_0.15_miss_0.60` | $0.15$ | $60\%$ | **0.1179** (+2.3%) | 0.0430 | 71.8% |
| `proxy_noise_0.30_miss_0.00` | $0.30$ | $0\%$ | **0.1158** (+0.5%) | 0.0431 | 75.9% |
| `proxy_noise_0.30_miss_0.30` | $0.30$ | $30\%$ | **0.1160** (+0.6%) | 0.0430 | 73.7% |
| `proxy_noise_0.30_miss_0.60` | $0.30$ | $60\%$ | **0.1210** (+5.0%) | 0.0435 | 70.8% |

**Scientific Interpretation:**
1. A noisy proxy report provides minimal tracking benefit (0.1148 vs 0.1153) only when noise is very small ($\sigma = 0.05$) and completeness is 100%.
2. When report noise reaches $\sigma = 0.30$ and missingness reaches $60\%$, tracking error degrades by 5.0% relative to the uninformative fixed 0.45 baseline.
3. Completely uninformative random priors degrade tracking severely (+23.7% error).
4. **Caregiver Assessment Caution:** These results demonstrate that future parent questionnaires will only benefit adaptive controllers if the parent report error is sufficiently small. Unvalidated, highly noisy questionnaires risk degrading controller precision compared to the fixed 0.45 prior default.

---

## 5. Testing Model Selector & Critical Semantic Safeguard (Part D)

### 5.1 Model Selector Operation
The benchmark runner supports a simple CLI override `--models` and a configuration key in `configs/default.yaml`:
```yaml
benchmark:
  models: "all" # Options: 'all', 'logistic_baseline', 'vanilla_dkt', 'reliability_gated_dkt'
```
CLI usage:
```bash
python -m grow21_simulation_benchmark.run_benchmark --models logistic_baseline --scenario simple_dynamics --seeds 101 --num-learners 60 --events-per-learner 30
```

### 5.2 Benchmark Verification Results
With controlled per-model random seeds (`seed + 0` for logistic, `seed + 100` for vanilla DKT, `seed + 200` for reliability-gated DKT), selecting each model individually produces identical test metrics to selecting it in `--models all` mode:

| Selected Model | Target Semantics | Controller Safe? | Test Log Loss | Test Brier Score | Accuracy | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `logistic_baseline` | $P(y_{\text{obs}}=1 \mid \text{features})$ | **FALSE** | 0.6574 | 0.2314 | 0.6259 | 0.57s |
| `vanilla_dkt` | $P(y_{\text{obs}}=1 \mid \text{features})$ | **FALSE** | 0.7032 | 0.2549 | 0.4852 | 3.16s |
| `reliability_gated_dkt` | $P(y_{\text{obs}}=1 \mid \text{features})$ | **FALSE** | 0.6930 | 0.2499 | 0.5222 | 0.96s |
| `all` (mode: log/van/gat) | $P(y_{\text{obs}}=1 \mid \text{features})$ | **FALSE** | 0.6574 / 0.7032 / 0.6930 | 0.2314 / 0.2549 / 0.2499 | 0.6259 / 0.4852 / 0.5222 | 2.15s |

### 5.3 Critical Semantic Safeguard Implementation
- **The Target Divergence:** Offline predictive benchmark models (Logistic Regression, Vanilla DKT, Reliability-Gated DKT) are trained to predict the probability of *observed* interaction success $P(y_{\text{obs}}=1 \mid \text{features})$.
- In contrast, the closed-loop difficulty controller solves for target difficulty targeting *clean task mastery* $P(y_{\text{clean}}=1 \mid \hat{\theta}, D, \dots)$.
- Under sensor noise and physical slip, $P(y_{\text{obs}}) \neq P(y_{\text{clean}})$. Routing $P(y_{\text{obs}})$ into a clean-success controller induces artificial difficulty suppression because corruptions make observed performance appear lower than true cognitive ability.
- **Code Safeguard:** In `ModelAlignedTargetAdaptiveController`, any instantiation with `predictor_type != "canonical_clean_model"` raises an immediate `ValueError`:
  ```
  ValueError: CRITICAL SEMANTIC SAFEGUARD: Cannot route observed-success predictor P(y_obs) directly into clean-success controller target P(y_clean). Offline benchmark models predict P(y_obs=1|features), whereas the closed-loop controller targets clean task success P(y_clean=1). An explicit, validated observation-noise inversion model is required to convert P(y_obs) to P(y_clean).
  ```
  Tested and verified in `tests/test_phase_5_state_estimation.py::test_model_selector_and_semantic_safeguard`.

---

## 6. Experimental Campaign Evidence & Comparative Evaluation

### 6.1 Cohort Architecture & Preregistration
Experiments were executed across 4 cohorts (12 seeds in total, 60 learners, 60 trials per learner per seed):
1. **Development Seeds `[101, 202, 303]`:** Used for initial algorithmic exploration and parameter verification.
2. **Historical Seeds `[42, 123, 999]`:** Direct replication against Phase 3/4 baseline benchmarks.
3. **Phase 4 Evaluation Seeds `[777, 888, 555]`:** Out-of-sample comparison against Phase 4 reported numbers.
4. **Final Locked Held-Out Seeds `[707, 808, 909]`:** Pre-registered before evaluation; configuration was frozen in `CONFIG_SNAPSHOT.json` and executed once.

**Interaction Accounting Across the 12 Seeds:**
- **Unique Learner Trials:** Across 12 seeds, 5 scenarios, and 60 learners $\times$ 60 trials = **216,000 unique learner trials** generated per policy ($54,000$ per cohort).
- **Closed-Loop Interactions:** With 8 estimator policies evaluated on every trial = **1,728,000 closed-loop estimator-policy interaction events** across the study ($432,000$ per cohort).
- **Open-Loop Replay Evaluations:** Replaying the 216,000 trials across 8 estimators = **1,728,000 open-loop evaluation records** ($432,000$ per cohort).

### 6.2 E1: Open-Loop State Estimation Replay (Isolated Tracking)
Under open-loop replay, all 8 state estimators processed identical exogenous sequences of games, difficulties, time gaps, observed outcomes, and reliability estimates.

**Held-Out Cohort `[707, 808, 909]` Open-Loop Results:**
| Scenario | Estimator Policy | Primary Mastery MAE | Primary Mastery Bias | Probability MAE |
| :--- | :--- | :---: | :---: | :---: |
| **base_heterogeneous** | `model_aligned_baseline` | 0.1264 | -0.0819 | 0.0621 |
| | `transfer_aware` | 0.1180 | -0.0646 | 0.0543 |
| | `reliability_aware` | 0.1162 | -0.0625 | 0.0573 |
| | `combined_transfer_reliability` | 0.1091 | -0.0446 | 0.0507 |
| | `expected_transition` | 0.1049 | -0.0243 | 0.0510 |
| | `expected_transition_transfer` | 0.1021 | -0.0048 | 0.0479 |
| | `approx_bayesian_filter` | 0.1015 | -0.0185 | 0.0494 |
| | **`approx_bayesian_filter_transfer`** | **0.0993** | **-0.0002** | **0.0468** |
| **reliability_stress** | `model_aligned_baseline` | 0.1568 | -0.1365 | 0.0769 |
| | `combined_transfer_reliability` | 0.1120 | -0.0679 | 0.0524 |
| | `expected_transition_transfer` | 0.1002 | -0.0324 | 0.0472 |
| | **`approx_bayesian_filter_transfer`** | **0.0981** | **-0.0265** | **0.0463** |
| **simple_dynamics** | `model_aligned_baseline` | 0.1684 | -0.1122 | 0.0856 |
| | **`approx_bayesian_filter_transfer`** | **0.1431** | **-0.0418** | **0.0675** |
| **temporal_dynamics** | `model_aligned_baseline` | **0.0899** | +0.0087 | **0.0471** |
| | `approx_bayesian_filter_transfer` | 0.1204 | +0.1031 | 0.0726 |
| **distribution_shift** | `model_aligned_baseline` | **0.1294** | +0.1212 | **0.0706** |
| | `approx_bayesian_filter_transfer` | 0.2465 | +0.2454 | 0.1280 |

### 6.3 E2: Closed-Loop Adaptive Evaluation
In closed-loop evaluation, estimators drove difficulty adaptation via the bisection solver.

**Held-Out Cohort `[707, 808, 909]` Closed-Loop Summary:**
| Scenario | Policy | Primary Mastery MAE | Probability Error | Bound Saturation ($D=D_{\text{min}}$) | Reachability Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **base_heterogeneous** | `model_aligned_baseline` | 0.1039 | 0.0392 | 34.8% | 71.7% |
| | `expected_transition` | 0.1048 | 0.0395 | 37.2% | 73.4% |
| | `expected_transition_transfer` | 0.1019 | 0.0392 | 21.8% | 81.8% |
| | `approx_bayesian_filter` | 0.1027 | 0.0391 | 33.2% | 73.2% |
| | **`approx_bayesian_filter_transfer`** | **0.1002** | **0.0386** | **20.1%** | **82.5%** |
| **reliability_stress** | `model_aligned_baseline` | 0.1224 | 0.0446 | 54.0% | 59.5% |
| | `expected_transition_transfer` | 0.1001 | 0.0366 | 30.5% | 80.3% |
| | **`approx_bayesian_filter_transfer`** | **0.0986** | **0.0362** | **27.9%** | **81.0%** |
| **distribution_shift** | `model_aligned_baseline` | **0.1827** | **0.0902** | 60.5% | 53.2% |
| | `expected_transition_transfer` | 0.2440 | 0.1214 | 38.3% | 78.8% |
| | `approx_bayesian_filter_transfer` | 0.2451 | 0.1223 | 37.5% | 78.6% |
| **temporal_dynamics** | `model_aligned_baseline` | **0.1042** | **0.0526** | 45.4% | 65.5% |
| | `approx_bayesian_filter_transfer` | 0.1215 | 0.0672 | 27.2% | 80.4% |

### 6.4 Paired Cluster Bootstrap Evidence (Held-Out Split)
Resampled 180 `(seed, learner_id)` clusters with replacement across 2,000 bootstrap replicates (preserving draw multiplicity).

**Comparison: `approx_bayesian_filter_transfer` vs `model_aligned_baseline`**
| Scenario | Metric | $\Delta$ (Variant - Baseline) | 95% Bootstrap CI | Statistical Conclusion |
| :--- | :--- | :---: | :---: | :--- |
| **reliability_stress** | `primary_mastery_mae_post` | **-0.0238** | **[-0.0270, -0.0205]** | **Significant Reduction** |
| | `prob_abs_error` | **-0.0084** | **[-0.0097, -0.0072]** | **Significant Reduction** |
| | `at_min_bound` | **-0.2613** | **[-0.2785, -0.2435]** | **Significant Reduction** |
| **base_heterogeneous** | `primary_mastery_mae_post` | -0.0037 | [-0.0061, -0.0013] | **Significant Reduction** |
| | `at_min_bound` | **-0.1474** | **[-0.1637, -0.1310]** | **Significant Reduction** |
| **distribution_shift** | `primary_mastery_mae_post` | **+0.0623** | **[+0.0583, +0.0664]** | **Significant Increase (Degraded)** |
| | `prob_abs_error` | **+0.0321** | **[+0.0298, +0.0345]** | **Significant Increase (Degraded)** |
| | `at_min_bound` | **-0.2296** | **[-0.2471, -0.2117]** | **Deceptive Saturation Drop** |
| **temporal_dynamics** | `primary_mastery_mae_post` | **+0.0172** | **[+0.0142, +0.0203]** | **Significant Increase (Degraded)** |
| | `at_min_bound` | **-0.1814** | **[-0.1983, -0.1638]** | **Deceptive Saturation Drop** |

---

## 7. Deep Analytical Findings: The Saturation Trap & Prior Mismatch

### 7.1 Where the Transition-Aware Filter Succeeds
In environments matching population priors (`base_heterogeneous`) and especially under severe physical noise and sensor dropouts (`reliability_stress`), the transition-aligned filter is superior:
- The legacy baseline conflates sensor noise and physical slips with lack of competence, punishing the learner's belief on every corruption and sending difficulty crashing into the floor ($D = 0.08$ on 54% of trials).
- The `approx_bayesian_filter_transfer` maintains belief distributions, correctly discounts low-reliability trials, and updates mastery according to the positive expected learning dynamic. It reduces mastery MAE from $0.1224$ to $0.0986$ ($-19.4\%$), reduces probability error from $0.0446$ to $0.0362$ ($-18.8\%$), and elevates reachability from $59.5\%$ to $81.0\%$.

### 7.2 Where the Estimator Fails: The "Saturation Trap" under Distribution Shift
The prompt issued an explicit scientific instruction:
> *"The current Phase 4 estimator variants can reduce minimum-difficulty saturation while worsening mastery/prediction error under distribution shift. Do not treat lower saturation by itself as improvement. Never select an estimator solely because it produces lower difficulty or fewer bound hits."*

The held-out results provide empirical validation of this exact warning:
1. In `distribution_shift`, learners start with initial mastery around $0.20$ (true population mean $0.20$, well below the prior expectation of $0.45$).
2. The transition-aware updater assumes that clean failures produce an expected feedback increment factor of $0.30$. Because the true learner is struggling far more than the prior assumes, repeatedly applying expected positive increments pushes estimated belief upwards ($\text{bias} = +0.244$).
3. The controller, believing the learner has higher mastery, selects higher task difficulty.
4. As a direct consequence, **minimum bound saturation drops dramatically from 60.5% to 37.5%**.
5. However, the learner is NOT learning faster. The learner faces overly challenging tasks, producing a $34.1\%$ increase in mastery error and a $35.6\%$ increase in prediction error!

**Scientific Conclusion:** A controller that exhibits lower minimum-difficulty bound saturation is NOT necessarily better calibrated. Under distribution shift, lower bound saturation was driven by **systematic positive belief inflation**, creating an illusion of competence that exposed vulnerable learners to inappropriately difficult tasks.

---

## 8. Test Suite & Verification Results

The complete test suite was executed against the repository:
```bash
pytest
```
**Results:** **66 passed in 148.5 seconds** (100% pass rate).
- `tests/test_audit_regressions.py` (13 tests): Passed
- `tests/test_belief_state_estimation.py` (13 tests): Passed
- `tests/test_controller_constraints.py` (4 tests): Passed
- `tests/test_documentation_integrity.py` (4 tests): Passed
- `tests/test_model_shapes.py` (2 tests): Passed
- `tests/test_no_leakage.py` (3 tests): Passed
- `tests/test_outcome_model_alignment.py` (10 tests): Passed
- `tests/test_phase_5_state_estimation.py` (11 Phase 5 tests including model selector reproducibility): Passed
- `tests/test_reliability.py` (3 tests): Passed
- `tests/test_simulator.py` (3 tests): Passed

---

## 9. Limitations & Future Directions

1. **Synthetic-Only Dynamics:** Real neurodiverse learners exhibit idiosyncratic, non-stationary learning curves, motor fatigue spikes, and sensory overstimulation that cannot be fully captured by Gaussian process noise or single-parameter exponential forgetting.
2. **Fixed Population Priors:** The primary vulnerability of both `expected_transition` and `approx_bayesian_filter` is their dependence on static population learning rates ($\bar{\eta} = 0.08$) and forgetting rates ($\bar{\lambda} = 0.02$). Under `distribution_shift` and `temporal_dynamics`, this static assumption breaks down.
3. **Future Hierarchical Bayesian Modeling:** A necessary extension for future phases is an online hierarchical learner-rate estimator (e.g. meta-learning or particle filter over $\eta_i$ and $\lambda_i$), allowing the filter to adapt its expected transition dynamics to the individual learner.
4. **Caregiver Questionnaire Integration:** When the teammate finalizes the questionnaire schema, the internal contract `SkillPriorEstimate` implemented in Phase 5 will seamlessly accept questionnaire mappings without altering controller architecture.
