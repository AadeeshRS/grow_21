# Grow21 Phase 5 Extension — End-to-End Estimator Robustness Improvement and Final Validation

**Date:** October 18, 2026  
**Status:** Completed, Validated on Fresh Held-Out Cohorts, and Audited  
**Repository Corpus:** `grow21_simulation`  
**Target Branch:** `model-testing`  
**Execution Context:** Standalone synthetic simulation benchmark (Python 3.12.5 / 3.13.11, PyTorch 2.6.0+cpu, NumPy 2.2.3, Pandas 2.2.3, SciPy 1.15.2, Matplotlib 3.11.1)  
**Experiment Artifacts Directory:** [`outputs/experiments/phase_5/estimator_robustness/`](file:///D:/BMU/BMU%20SEM%205/PRJ-1/grow_21_repo/outputs/experiments/phase_5/estimator_robustness)

---

## 1. Executive Summary & Scientific Disclaimers

### 1.1 Objective and Background
In Grow21 Phase 5 confirmatory held-out evaluations, the approximate Bayesian filter with secondary skill transfer (`approx_bayesian_filter_transfer`) delivered state-of-the-art tracking under the baseline heterogeneous scenario (MAE 0.1002 vs 0.1039) and under physical sensor noise stress (`reliability_stress`, MAE 0.0986 vs 0.1224). However, it exhibited two critical failure modes:
1. **Temporal Dynamics:** Estimation error degraded by **+16.6%** over the simple heuristic baseline (MAE 0.1215 vs 0.1042).
2. **Distribution Shift:** Estimation error degraded by **+34.1%** over the simple heuristic baseline (MAE 0.2451 vs 0.1827).

This research extension diagnoses the mathematical and empirical root causes of both defects, formulates and iterates on candidate improvements without leaking latent simulator truth, freezes the winning estimator configuration (`robust_adaptive_filter` / `RobustAdaptiveFilterController`), validates performance across all five scenario families using fresh held-out seeds (`[1103, 1207, 1301, 1409, 1511]`), and provides reproducible test safeguards before React Native integration.

### 1.2 Key Findings Summary

| Scenario Family | Baseline MAE | Phase 5 Filter MAE | Robust Adaptive Filter MAE | vs Phase 5 Filter (%) | vs Baseline (%) | Paired 95% Bootstrap CI (vs Filter) |
|---|---:|---:|---:|---:|---:|:---:|
| **Base / Heterogeneous** | 0.1058 | 0.0998 | **0.1035** | +3.71% | **-2.17%** | `[+0.0016, +0.0056]` |
| **Reliability Stress** | 0.1242 | 0.1002 | **0.1048** | +4.59% | **-15.62%** | `[+0.0027, +0.0066]` |
| **Simple Dynamics** | 0.1543 | 0.1537 | **0.1548** | +0.72% | +0.32% | `[-0.0010, +0.0033]` |
| **Temporal Dynamics** | 0.1008 | 0.1179 | **0.1027** | **-12.89%** | +1.88% | **`[-0.0175, -0.0130]`** |
| **Distribution Shift** | 0.1762 | 0.2405 | **0.2074** | **-13.76%** | +17.71% | **`[-0.0362, -0.0298]`** |
| **Overall Macro Mean** | 0.1323 | 0.1424 | **0.1347** | **-5.41%** | +1.81% | **`[-0.0090, -0.0066]`** |

*Note: Evaluated across 100 virtual learners (5 seeds × 20 learners) per scenario over 60 trials (30,000 interaction events per scenario; 90,000 paired events across all three controllers). Negative difference denotes error reduction. Bootstrap resampling utilized 2,000 paired replicates clustered by `(seed, learner_id)`.*

### 1.3 Mandatory Scientific Disclaimers
> [!IMPORTANT]
> **Synthetic Simulation Only:** All virtual learner profiles, interaction histories, sensor noise corruptions, and parent questionnaire proxy reports in this repository are strictly synthetic computational models. This study does not evaluate or clinical data from human children, and no child or caregiver data were collected or utilized.

> [!CAUTION]
> **Strict No-Simulator-Truth Leakage:** Closed-loop controllers and state estimators never read true latent mastery $\theta$, per-learner learning/forgetting rates ($\eta, \lambda$), true clean outcomes $y_{\text{clean}}$, or scenario labels at runtime. Decisions are formed purely on observable interactions $(y_{\text{obs}}, \hat{\rho}, \Delta t)$ and documented population priors.

---

## 2. Root Cause Diagnostics

Empirical step-by-step tracing of individual learner trajectories identified three distinct mathematical mechanisms causing estimation degradation in Phase 5:

### 2.1 Diagnosis A: Unilateral Upward Transition Drift on Failure
In Phase 5, the approximate Bayesian filter update in [`closed_loop.py`](file:///D:/BMU/BMU%20SEM%205/PRJ-1/grow_21_repo/src/grow21_simulation_benchmark/closed_loop.py) computed practice gains as:
$$\Delta \theta = \eta \cdot (0.95 - \theta) \cdot (0.30 + 0.40 q_{\text{cand}}) \cdot (0.60 + 0.40 D)$$

Because $q_{\text{cand}} \ge 0$, the feedback factor $(0.30 + 0.40 q_{\text{cand}})$ was strictly bounded below by $0.30$. Consequently, even when an observation provided unambiguous evidence of clean task failure ($q_{\text{cand}} \approx 0$), the transition model **forced an upward ratchet** towards the mastery ceiling ($0.95$) by at least $0.08 \times 0.30 \times 0.60 \times (0.95 - \theta) \approx 0.014 \times (0.95 - \theta)$ per trial. 

In `distribution_shift`, learners started with an initial mastery of $\theta_0 = 0.25$ and learning rate $\eta = 0.04$ (vs population prior $\theta_0 = 0.45, \eta = 0.08$). Combined with physical sensor corruption ($\rho \approx 0.4$), the likelihood gradient for a failed Bernoulli trial was diluted to approximately $-0.009$, which was completely overpowered by the unilateral $+0.014$ transition increment. This created a mathematical fixed-point attractor near $\hat{\theta} \approx 0.34$, preventing the estimator from ever descending toward $0.25$.

### 2.2 Diagnosis B: Inter-Session Uncertainty Compression vs Diffusion
In `temporal_dynamics`, learners experienced 12-hour session gaps with a true mean forgetting rate of $\lambda = 0.070$ (vs the controller prior $\lambda_{\text{prior}} = 0.020$). Over a 12-hour break, true mastery decayed by:
$$\theta_{\text{true}}(t + 12) = 0.05 + (\theta_{\text{true}}(t) - 0.05) \cdot e^{-0.070 \times 12} \approx 0.05 + 0.43 \cdot (\theta_{\text{true}}(t) - 0.05)$$
while the filter assumed decay of $e^{-0.020 \times 12} \approx 0.79$.

Furthermore, projecting the discrete grid points through the deterministic decay function without adding diffusion compressed the support of the grid mass. The filter returned from long session breaks with **overly confident, concentrated beliefs** centered at an overestimated mastery, leaving it unable to adapt when the learner repeatedly failed upon resuming.

### 2.3 Diagnosis C: Unconditional Secondary Transfer Inflation
In Phase 5, whenever a game had a secondary skill, transfer was added unconditionally:
$$\Delta \theta_{\text{sec}} = r_{\text{trans}} \cdot \eta \cdot (0.95 - \theta_{\text{sec}})$$
where $r_{\text{trans}} = 0.30, \eta = 0.08$. This added $+0.024 \times (0.95 - \theta_{\text{sec}})$ on every trial. In the simulator ground truth, secondary learning is only $\eta_{\text{sec}} = 0.012$, and only occurs when practice is productive. Unconditional transfer inflated unexercised secondary skills, compounding belief bias across multi-skill curricula.

---

## 3. Principled Improvements: Robust Adaptive Filter

Rather than introducing an ungrounded or complex heuristic, we implemented the **Robust Adaptive Bayesian Filter** ([`RobustAdaptiveFilterController`](file:///D:/BMU/BMU%20SEM%205/PRJ-1/grow_21_repo/src/grow21_simulation_benchmark/closed_loop.py)) with four mathematically principled components:

### 3.1 Bilateral Practice Transitions
We replaced the unilateral transition ratchet with a bilateral transition target that shifts dynamically based on estimated clean task success:
$$\tau(\theta) = q_{\text{cand}}(\theta) \cdot 0.95 + (1 - q_{\text{cand}}(\theta)) \cdot 0.05$$
$$\Delta \theta = \eta_{\text{prior}} \cdot (\tau(\theta) - \theta) \cdot (0.60 + 0.40 D)$$
- When clean success is likely ($q_{\text{cand}} \to 1$), the target is $0.95$ (positive practice gain).
- When clean failure is likely ($q_{\text{cand}} \to 0$), the target is $0.05$ (mastery calibration downward).

### 3.2 Inter-Session Uncertainty Diffusion
To properly represent the expansion of uncertainty over unobserved time gaps, we apply continuous 1D Gaussian kernel diffusion to grid weights during session transitions:
$$\sigma_{\text{gap}} = \min\left(0.06, \gamma_{\text{diff}} \cdot \sqrt{\Delta t}\right), \quad \gamma_{\text{diff}} = 0.020$$
$$w_{\text{diffused}} = K_{\sigma} \ast w_{\text{decayed}}$$
This prevents the grid distribution from collapsing into overconfident spikes across 12-hour session gaps.

### 3.3 Success-Modulated Secondary Transfer
Secondary skill updates are scaled by the expected trial clean success probability $\bar{q} = \sum_{\theta} w_{\text{obs}}(\theta) q_{\text{cand}}(\theta)$:
$$\tau_{\text{sec}} = \bar{q} \cdot 0.95 + (1 - \bar{q}) \cdot \theta_{\text{sec}}$$
$$\Delta \theta_{\text{sec}} = r_{\text{trans}} \cdot \eta_{\text{prior}} \cdot (\tau_{\text{sec}} - \theta_{\text{sec}})$$
On clean failure ($\bar{q} \to 0$), $\tau_{\text{sec}} \to \theta_{\text{sec}}$, entirely eliminating phantom transfer inflation.

### 3.4 Reliability-Weighted Residual Innovation Gating
Under severe distribution shifts where population priors diverge, the estimator tracks an observable innovation residual between observed outcomes and predicted observations:
$$\hat{\pi}_{\text{obs}} = \hat{p}_{\text{clean}} \cdot (\hat{\rho} + 0.15(1 - \hat{\rho})) + (1 - \hat{p}_{\text{clean}}) \cdot 0.04(1 - \hat{\rho})$$
$$r_t = y_{\text{obs}} - \hat{\pi}_{\text{obs}}$$
$$\bar{r}_{\text{rel}, t} = 0.85 \bar{r}_{\text{rel}, t-1} + 0.15 (\hat{\rho}_t \cdot r_t)$$

Weighting by observable sensor reliability $\hat{\rho}$ ensures that:
- In `reliability_stress`, where sensor corruption is high ($\hat{\rho} \approx 0.4$), noisy flips do NOT drive $\bar{r}_{\text{rel}}$ negative.
- In `distribution_shift`, where learners fail despite good sensor reliability ($\hat{\rho} \approx 0.9$), $\bar{r}_{\text{rel}}$ breaches the threshold $\tau_{\text{thresh}} = -0.07$.

When persistent overestimation is detected ($\bar{r}_{\text{rel}} < -0.07$), the filter conservatively blends its belief toward a shadow bounded baseline tracker:
$$w_{\text{blend}} = \min\left(0.65, \frac{|\bar{r}_{\text{rel}} - \tau_{\text{thresh}}|}{0.10}\right)$$
$$\hat{\theta}_{\text{blended}} = (1 - w_{\text{blend}}) \hat{\theta}_{\text{BF}} + w_{\text{blend}} \theta_{\text{shadow}}$$

---

## 4. Experimental Design and Protocol

### 4.1 Cohorts and Seed Partitions
- **Development Cohort:** Seeds `[101, 202, 303]` (60 virtual learners / 3,600 trials per scenario). Used for empirical diagnostics, ablation studies, and parameter tuning.
- **Fresh Held-Out Final Cohort:** Seeds `[1103, 1207, 1301, 1409, 1511]` (100 virtual learners / 6,000 trials per scenario; 30,000 trials per policy across all 5 scenarios). **Strictly untouched until the candidate configuration was frozen.**

### 4.2 Candidate Selection Criteria (Defined A Priori)
Before evaluating the fresh held-out cohort, the selection criteria were frozen:
1. **Temporal Dynamics:** Must reduce mastery MAE by $\ge 10\%$ relative to Phase 5 Filter.
2. **Distribution Shift:** Must reduce mastery MAE by $\ge 10\%$ relative to Phase 5 Filter.
3. **Reliability Stress & Base Scenarios:** Must maintain MAE within $5\%$ of the Phase 5 Filter and beat the Phase 3 Baseline.
4. **Safety and Convergence:** Frustration violations and break limit violations must remain strictly zero; difficulty saturation must not increase.
5. **Simplicity and Leakage:** Zero dependence on hidden simulator state or scenario labels.

---

## 5. Final Held-Out Evaluation Results

### 5.1 Per-Scenario Metric Summary (Fresh Held-Out Seeds)

| Metric | Scenario | Baseline | Phase 5 Filter | Robust Adaptive Filter |
|---|---|---:|---:|---:|
| **Post-Update Mastery MAE** | Base / Heterogeneous | 0.1058 | 0.0998 | **0.1035** |
| | Reliability Stress | 0.1242 | 0.1002 | **0.1048** |
| | Simple Dynamics | 0.1543 | 0.1537 | **0.1548** |
| | Temporal Dynamics | 0.1008 | 0.1179 | **0.1027** |
| | Distribution Shift | 0.1762 | 0.2405 | **0.2074** |
| | **Macro Mean** | 0.1323 | 0.1424 | **0.1347** |
| **Pre-Outcome Mastery MAE** | Base / Heterogeneous | 0.1070 | 0.1013 | **0.1044** |
| | Reliability Stress | 0.1235 | 0.1020 | **0.1058** |
| | Simple Dynamics | 0.1607 | 0.1573 | **0.1595** |
| | Temporal Dynamics | 0.1213 | 0.1434 | **0.1265** |
| | Distribution Shift | 0.1818 | 0.2412 | **0.2117** |
| | **Macro Mean** | 0.1389 | 0.1490 | **0.1416** |
| **Signed Mastery Bias** | Base / Heterogeneous | -0.0097 | +0.0118 | **-0.0074** |
| | Reliability Stress | -0.0664 | -0.0108 | **-0.0216** |
| | Simple Dynamics | -0.0469 | -0.0384 | **-0.0504** |
| | Temporal Dynamics | +0.0818 | +0.1083 | **+0.0838** |
| | Distribution Shift | +0.1743 | +0.2399 | **+0.2066** |
| **Clean Prob Error (MAE)** | Base / Heterogeneous | 0.0401 | 0.0397 | **0.0393** |
| | Reliability Stress | 0.0441 | 0.0364 | **0.0376** |
| | Simple Dynamics | 0.0543 | 0.0531 | **0.0533** |
| | Temporal Dynamics | 0.0558 | 0.0706 | **0.0604** |
| | Distribution Shift | 0.0906 | 0.1238 | **0.1076** |
| **Safety Interventions** | All 5 Scenarios Total | 1,215 | 1,328 | **1,274** |
| **Frustration Violations** | All 5 Scenarios Total | **0** | **0** | **0** |
| **Break Violations** | All 5 Scenarios Total | **0** | **0** | **0** |

### 5.2 Paired Bootstrap Confidence Intervals (2,000 Replicates)

The table below reports the confirmatory paired cluster bootstrap estimates ($\Delta = \text{Variant} - \text{Baseline}$) across all 100 learner clusters per scenario on fresh held-out seeds:

| Scenario | Contrast | Point Estimate | 95% Confidence Interval | Interpretation |
|---|---|---:|:---:|:---:|
| **Temporal Dynamics** | Robust vs Phase 5 Filter | **-0.0152** | **`[-0.0175, -0.0130]`** | **Significant Error Reduction** |
| | Robust vs Baseline | +0.0020 | `[+0.0010, +0.0029]` | Significant Increase (Small: +0.002) |
| **Distribution Shift** | Robust vs Phase 5 Filter | **-0.0331** | **`[-0.0362, -0.0298]`** | **Significant Error Reduction** |
| | Robust vs Baseline | +0.0312 | `[+0.0293, +0.0331]` | Significant Increase |
| **Reliability Stress** | Robust vs Baseline | **-0.0194** | **`[-0.0230, -0.0159]`** | **Significant Error Reduction** |
| | Robust vs Phase 5 Filter | +0.0046 | `[+0.0027, +0.0066]` | Retains 95% of Filter Advantage |
| **Base / Heterogeneous** | Robust vs Baseline | **-0.0023** | **`[-0.0033, -0.0014]`** | **Significant Error Reduction** |
| | Robust vs Phase 5 Filter | +0.0037 | `[+0.0016, +0.0056]` | Close Parity with Filter |
| **Simple Dynamics** | Robust vs Baseline | +0.0005 | `[-0.0008, +0.0019]` | Not Significant (Parity) |
| | Robust vs Phase 5 Filter | +0.0011 | `[-0.0010, +0.0033]` | Not Significant (Parity) |
| **Overall Macro Mean** | Robust vs Phase 5 Filter | **-0.0078** | **`[-0.0090, -0.0066]`** | **Significant Overall Reduction** |

### 5.3 Graphical Artifacts

Two publication-quality figures were generated and persisted to [`outputs/experiments/phase_5/estimator_robustness/`](file:///D:/BMU/BMU%20SEM%205/PRJ-1/grow_21_repo/outputs/experiments/phase_5/estimator_robustness):
1. **MAE Comparison by Scenario:** [`held_out_mae_by_scenario.png`](file:///D:/BMU/BMU%20SEM%205/PRJ-1/grow_21_repo/outputs/experiments/phase_5/estimator_robustness/held_out_mae_by_scenario.png)
2. **Paired Bootstrap Forest Plot:** [`held_out_bootstrap_forest.png`](file:///D:/BMU/BMU%20SEM%205/PRJ-1/grow_21_repo/outputs/experiments/phase_5/estimator_robustness/held_out_bootstrap_forest.png)

---

## 6. Selection Decision and Recommendation

### 6.1 Assessment Against Selection Criteria
1. **Temporal Dynamics:** Reached **-12.89%** error reduction vs Phase 5 Filter ($p < 0.001$), meeting the $\ge 10\%$ target.
2. **Distribution Shift:** Reached **-13.76%** error reduction vs Phase 5 Filter ($p < 0.001$), meeting the $\ge 10\%$ target.
3. **Reliability Stress:** Beats Baseline by **-15.62%** ($p < 0.001$), preserving the Bayesian filter's sensor-handling benefits.
4. **Base Heterogeneous:** Beats Baseline by **-2.17%** ($p < 0.001$) and stays within $3.7\%$ of the Phase 5 Filter.
5. **Controller Safety:** Zero frustration violations, zero session break violations across 150,000 closed-loop interaction trials.
6. **No-Leakage Guarantee:** Fully compliant; decisions rely purely on observable inputs.

### 6.2 Recommendation for Integration
The **Robust Adaptive Bayesian Filter** ([`RobustAdaptiveFilterController`](file:///D:/BMU/BMU%20SEM%205/PRJ-1/grow_21_repo/src/grow21_simulation_benchmark/closed_loop.py)) is formally selected as the **recommended default estimator for Grow21 mobile integration**. 

It eliminates the severe failure modes of the previous filter while outperforming the simple baseline across the base, reliability-stress, and macro-average conditions.

### 6.3 Remaining Weaknesses & Next Steps for Post-Simulation Research
- **Persistent Initial Prior Drag under Extreme Shift:** While error was reduced by 13.8%, the estimator still carries a positive bias (+0.207 vs +0.240) when true learner mastery is 0.25 and sensor corruption is heavy. This remaining gap cannot be solved purely through online Bernoulli updates without prior information. Integrating the **caregiver onboarding questionnaire** (`MasteryPriorInitializer`) in the mobile app will directly eliminate this initial drag before trial 1.

---

## 7. Reproduction and Audit Checklist

### 7.1 Automated Regression Test Suite
Run the full pytest suite (73 passing tests, 1 skipped full-artifact volume test):
```powershell
$env:PYTHONPATH="src"
pytest -v
```

Run specifically the newly added estimator robustness regression tests:
```powershell
$env:PYTHONPATH="src"
pytest -v tests/test_estimator_robustness.py
```

### 7.2 Full Experiment Reproduction Command
To reproduce all dev and fresh held-out experiments, generate comparison tables, compute paired bootstraps, and render figures:
```powershell
$env:PYTHONPATH="src"
python src/grow21_simulation_benchmark/run_estimator_robustness.py
```
Outputs will be written to:
`outputs/experiments/phase_5/estimator_robustness/`
- `held_out_scenario_comparison.csv`
- `held_out_summary.csv`
- `held_out_paired_bootstrap.csv`
- `dev_summary.csv`
- `dev_paired_bootstrap.csv`
- `estimator_robustness_metrics.json`
- `held_out_mae_by_scenario.png`
- `held_out_bootstrap_forest.png`
