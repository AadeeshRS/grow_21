# Grow21 Phase 4 Research Report: Learner Belief-State Estimation & Robustness (Audited)

**Phase:** Phase 4 — Learner Belief-State Estimation and Robustness (Audit Correction Pass)  
**Environment:** Python 3.13.11, PyTorch 2.6.0+cpu, NumPy, Pandas, Scikit-Learn, Matplotlib, PyYAML  
**Author:** Pair Programming Research Assistant  
**Date:** October 2026  
**Artifact Directory:** `outputs/phase_4/`  
**Audit Directory:** `outputs/audit/phase_4/`  

---

## 1. Executive Summary

Phase 3 established complete mathematical equation parity between the canonical simulator outcome model and the model-aligned target controller, eliminating structural equation discrepancies. However, empirical analysis revealed residual prediction gaps between the controller's predicted clean success probability $\hat{p}$ and the simulator's latent clean probability $p_{\text{clean}}$ (e.g., $-0.80\text{ pp}$ in `base_heterogeneous`, $-2.94\text{ pp}$ in `reliability_stress`, $+4.50\text{ pp}$ in `temporal_dynamics`, and $+9.09\text{ pp}$ in `distribution_shift` on held-out seeds).

This audited Phase 4 evaluation investigates the causes of these residual gaps through a rigorous three-stage experimental campaign ($N = 540,000$ interaction events across 5 scenarios, 9 random seeds, and 4 principled controller ablations), resolving previous measurement and inferential defects:

1. **Temporal Alignment Correction**: Prior Phase 4 metrics compared post-outcome controller mastery beliefs against pre-outcome simulator latent mastery. This audit explicitly separates and reports **pre-outcome state estimation** (controller belief immediately before observation vs. simulator `latent_mastery_before`) and **post-update state estimation** (controller belief after update vs. simulator `latent_mastery_after`). Pre-outcome probability predictions remain strictly tied to the trial's decision point.
2. **Paired Cluster Bootstrap Correction**: The prior evaluation used a flawed bootstrap that sampled only learner IDs (0..59) across pooled seeds and filtered with `.isin`, conflating learners from distinct seeds and losing draw multiplicity. We implement a seed-aware cluster bootstrap over $K = 180$ unique `(seed, learner_id)` clusters with $B = 2,000$ replicates preserving draw multiplicity and matching events on `(scenario, seed, learner_id, step_idx)`. Unsupported small $p$-values have been removed.
3. **Cohort Partitioning**: Confirmatory generalization claims are based strictly on **Held-Out Evaluation Seeds** (`[777, 888, 555]`, $N = 216,000$ events) evaluated with fixed population priors and zero seed tuning. Historical seeds (`[42, 123, 999]`, $N = 216,000$ events) are reported separately as baseline replication evidence, and development seeds (`[101, 202, 303]`, $N = 108,000$ events) are retained for offline diagnostics.
4. **Secondary-Skill Tracking**: In `base_heterogeneous`, `visual_discrimination` (present as a secondary skill in 5 of 9 games, but never primary) decayed towards $0.05$ under the baseline updater. Introducing a population-prior-based secondary transfer update ($\Delta \hat{\theta}_{\text{sec}} = 0.30 \cdot \mu_{\text{lr}} \cdot (0.95 - \hat{\theta}_{\text{sec}})$) reduced secondary mastery estimation pre-MAE from $0.1450$ to $0.1070$ and eliminated secondary underestimation bias (from $-0.0848$ to $-0.0048$).
5. **Observation Corruption Robustness**: In `reliability_stress`, sensor slips ($P(\text{slip} \mid \text{corruption}) = 0.85$) severely dragged down heuristic baseline beliefs. Implementing a Bayesian soft-label observation update $\tilde{y} = P(y_{\text{clean}} = 1 \mid y_{\text{obs}}, \hat{\rho}_t)$ using pre-outcome observable reliability $\hat{\rho}_t$ shrunk the held-out prediction gap from $-2.94\text{ pp}$ to $-0.88\text{ pp}$, reduced pre-MAE from $0.1189$ to $0.1019$, and reduced lower-bound saturation ($D_{\min} = 0.08$) by $-25.27\text{ pp}$ (from $51.7\%$ to $26.4\%$).
6. **Limits of Static Priors Under Shift**: In `temporal_dynamics` and `distribution_shift`, multi-factor parameter shifts (initial mastery, learning rate, forgetting rate, and corruption) conflict with fixed controller priors. Softening failure penalties under low reliability prevents rapid discovery of lower ability, increasing prediction gaps (+11.91 pp in reliability-aware and +13.28 pp in combined vs. baseline +9.09 pp). Fixed population priors cannot compensate for cohort distribution shifts without adaptive prior estimation.
7. **Reproducibility & Integrity**: All 55 regression and documentation integrity tests pass in $32.53\text{s}$, verifying complete absence of simulator truth leakage, valid cluster bootstrap properties, temporal alignment invariants, and strict absence of literal tabs.

---

## 2. Recomputed Baseline Gaps: Held-Out vs. Historical Cohorts

To ensure methodological rigor, baseline results are evaluated separately across the Held-Out and Historical cohorts.

### Table 1: Model-Aligned Baseline Across Scenarios (Held-Out vs. Historical)

| Scenario | Cohort | Sampled Clean Rate | Mean Sim $p_{\text{clean}}$ | Mean Pred $\hat{p}$ | Signed Gap $(\hat{p} - p_{\text{clean}})$ | Pre-MAE | Post-MAE | At $D_{\min}$ | Reachable |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`base_heterogeneous`** | Held-Out | $74.63\%$ | $75.10\%$ | $74.30\%$ | **$-0.80\text{ pp}$** | $0.1062$ | $0.1049$ | $36.4\%$ | $71.4\%$ |
| | Historical | $74.90\%$ | $75.14\%$ | $74.26\%$ | **$-0.88\text{ pp}$** | $0.1082$ | $0.1070$ | $35.2\%$ | $71.6\%$ |
| **`reliability_stress`** | Held-Out | $76.24\%$ | $76.72\%$ | $73.77\%$ | **$-2.94\text{ pp}$** | $0.1189$ | $0.1196$ | $51.7\%$ | $60.1\%$ |
| | Historical | $77.22\%$ | $76.75\%$ | $73.86\%$ | **$-2.89\text{ pp}$** | $0.1193$ | $0.1199$ | $52.7\%$ | $60.5\%$ |
| **`simple_dynamics`** | Held-Out | $76.01\%$ | $76.61\%$ | $74.37\%$ | **$-2.24\text{ pp}$** | $0.1547$ | $0.1479$ | $26.6\%$ | $75.6\%$ |
| | Historical | $76.44\%$ | $76.73\%$ | $74.36\%$ | **$-2.37\text{ pp}$** | $0.1588$ | $0.1514$ | $26.8\%$ | $75.6\%$ |
| **`temporal_dynamics`** | Held-Out | $68.81\%$ | $69.54\%$ | $74.03\%$ | **$+4.50\text{ pp}$** | $0.1202$ | $0.1062$ | $45.4\%$ | $65.0\%$ |
| | Historical | $69.62\%$ | $69.59\%$ | $74.07\%$ | **$+4.48\text{ pp}$** | $0.1208$ | $0.1062$ | $45.8\%$ | $65.3\%$ |
| **`distribution_shift`** | Held-Out | $64.31\%$ | $64.30\%$ | $73.39\%$ | **$+9.09\text{ pp}$** | $0.1842$ | $0.1843$ | $61.0\%$ | $51.9\%$ |
| | Historical | $64.38\%$ | $64.48\%$ | $73.32\%$ | **$+8.84\text{ pp}$** | $0.1809$ | $0.1807$ | $62.8\%$ | $51.6\%$ |

Baseline performance is exceptionally stable between Historical and Held-Out seeds (signed prediction gaps within $\pm 0.25\text{ pp}$ across all scenarios), verifying that the synthetic benchmark environment reproduces deterministically without seed variance artifacts.

---

## 3. Confirmed Code Mechanisms vs. Hypotheses

### A. Secondary Skill Omission & Asymmetry
In `configs/default.yaml`, `visual_discrimination` is configured as a secondary skill in 5 of 9 tasks (`trace_glow`, `bubble_pop_brigade`, `category_sorting`, `visual_flip_match`, `pattern_train`), but is primary in none.
- **Simulator Dynamic**: Every trial of a task with a secondary skill awards practice transfer: $\Delta \theta_{\text{sec}} = 0.30 \cdot \mu_{\text{lr}} \cdot (1 - \theta_{\text{sec}})$.
- **Baseline Controller Defect**: In Phase 3, the controller updated only the primary skill and applied inter-session forgetting decay to all skills. Visual discrimination received zero practice updates and decayed toward $0.05$. Because effective mastery is $\theta_{\text{eff}} = 0.85 \theta_{\text{prim}} + 0.15 \theta_{\text{sec}}$, this induced systematic clean probability underestimation.

### B. Asymmetric Physical Observation Corruption
In `src/grow21_simulation_benchmark/simulator.py`:
- Physical corruption flips clean successes into observed failures with probability **$0.85$** (motor tremors, slips, vocal cutoffs).
- Physical corruption flips clean failures into observed successes with probability **$0.04$** (accidental hits).
- **Baseline Controller Defect**: When sensor reliability is degraded (`reliability_stress`), observed failures were treated as skill deficits, penalizing belief by $8\%$ toward $0.05$. In reality, $85\%$ of corrupted successes were false failures. This drove the controller into excessive lower-bound saturation ($51.7\%$ at $D_{\min} = 0.08$).

### C. Multi-Factor Scenario Confounding
- **`temporal_dynamics`**: The prediction gap cannot be attributed to forgetting mismatch alone. The scenario jointly shifts:
  1. Initial mastery: mean $\approx 0.30$ vs. controller prior $0.45$.
  2. Learning rate: mean $\approx 0.16$ vs. controller prior $0.08$.
  3. Forgetting rate: mean $\approx 0.07$ vs. controller prior $0.02$.
  The $+4.50\text{ pp}$ gap reflects the combined impact of lower initial mastery, faster learning, and faster forgetting.
- **`distribution_shift`**: Concurrently varies initial mastery ($\mu \approx 0.25$), learning rate ($\mu \approx 0.05$), forgetting rate ($\mu \approx 0.05$), sensor noise, and modality dropouts. The prediction gap (+9.09 pp baseline, +13.28 pp combined) is a compound system-level discrepancy.

---

## 4. Estimator Variants and Explicit Formulations

All controllers operate under the strict zero-leakage invariant: controllers never access true latent mastery, $p_{\text{clean}}$, or $y_{\text{clean}}$.

### 1. `model_aligned_baseline`
- **Primary update**: Hard heuristic update on observed binary outcome:
  $$\Delta \hat{\theta}_{\text{prim}} = \begin{cases} +0.08 \cdot (0.95 - \hat{\theta}_{\text{prim}}), & \text{if } y_{\text{obs}} = 1 \\ -0.08 \cdot (\hat{\theta}_{\text{prim}} - 0.05), & \text{if } y_{\text{obs}} = 0 \end{cases}$$
- **Secondary update**: None (decays via inter-session forgetting).

### 2. `transfer_aware`
- **Primary update**: Same heuristic update as baseline.
- **Secondary update**: Tracks secondary transfer using fixed population priors ($\mu_{\text{lr}} = 0.08$, ratio $0.30$):
  $$\Delta \hat{\theta}_{\text{sec}} = 0.30 \cdot 0.08 \cdot (0.95 - \hat{\theta}_{\text{sec}}) = 0.024 \cdot (0.95 - \hat{\theta}_{\text{sec}})$$

### 3. `reliability_aware`
- **Observation Update**: Calculates the posterior probability of a clean success $\tilde{y}_t = P(y_{\text{clean}} = 1 \mid y_{\text{obs}}, \hat{\rho}_t)$ using pre-outcome observable reliability $\hat{\rho}_t \in [0, 1]$ and prior predicted clean probability $p = \hat{p}_t$:
  $$\text{If } y_{\text{obs}} = 1: \quad \tilde{y}_t = \frac{p \cdot (\hat{\rho}_t + 0.15(1 - \hat{\rho}_t))}{p \cdot (\hat{\rho}_t + 0.15(1 - \hat{\rho}_t)) + (1 - p) \cdot (0.04(1 - \hat{\rho}_t))}$$
  $$\text{If } y_{\text{obs}} = 0: \quad \tilde{y}_t = \frac{p \cdot (0.85(1 - \hat{\rho}_t))}{p \cdot (0.85(1 - \hat{\rho}_t)) + (1 - p) \cdot (\hat{\rho}_t + 0.96(1 - \hat{\rho}_t))}$$
- **Mastery Update**: Heuristic population-rate adjustment toward soft target $\theta_{\text{target}} = \tilde{y}_t \cdot 0.95 + (1 - \tilde{y}_t) \cdot 0.05$:
  $$\Delta \hat{\theta}_{\text{prim}} = 0.08 \cdot (\theta_{\text{target}} - \hat{\theta}_{\text{prim}})$$
- **Methodological Note**: This is a *Bayesian soft-label observation update followed by a heuristic population-rate mastery update*, not a continuous posterior density over mastery.

### 4. `combined_transfer_reliability`
- Jointly applies the Bayesian soft-label observation update to the primary skill and the population-prior transfer update to the secondary skill.

---

## 5. Primary Headline Experimental Results: Held-Out Cohort

The headline confirmatory evaluation is based strictly on the **Held-Out Evaluation Seeds** (`[777, 888, 555]`, 180 virtual learners, 60 trials/learner, $N = 216,000$ events across 5 scenarios and 4 policies).

### Table 2: Primary Held-Out Confirmatory Evaluation Across Scenarios

| Scenario | Policy | Clean Succ | Sim $p_{\text{clean}}$ | Pred $\hat{p}$ | Signed Gap [pp] | Prim Pre-MAE | Prim Post-MAE | Sec Pre-MAE | Sec Post-MAE | At $D_{\min}$ | Reachable |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`base_heterogeneous`** | Baseline | $74.63\%$ | $75.10\%$ | $74.30\%$ | $-0.80\text{ pp}$ | $0.1062$ | $0.1049$ | $0.1450$ | $0.1487$ | $36.4\%$ | $71.4\%$ |
| | Transfer-Aware | $73.71\%$ | $74.28\%$ | $74.50\%$ | $+0.22\text{ pp}$ | $0.1056$ | $0.1046$ | **$0.1070$** | **$0.1053$** | $23.5\%$ | $79.5\%$ |
| | Reliability-Aware | $73.83\%$ | $74.44\%$ | $74.51\%$ | **$+0.07\text{ pp}$** | **$0.1050$** | **$0.1038$** | $0.1443$ | $0.1473$ | $24.2\%$ | $78.6\%$ |
| | Combined | $72.84\%$ | $73.53\%$ | $74.63\%$ | $+1.10\text{ pp}$ | $0.1064$ | $0.1057$ | **$0.1075$** | **$0.1058$** | **$14.4\%$** | **$85.6\%$** |
| **`reliability_stress`** | Baseline | $76.24\%$ | $76.72\%$ | $73.77\%$ | $-2.94\text{ pp}$ | $0.1189$ | $0.1196$ | $0.1544$ | $0.1606$ | $51.7\%$ | $60.1\%$ |
| | Transfer-Aware | $75.65\%$ | $76.18\%$ | $74.28\%$ | $-1.90\text{ pp}$ | $0.1129$ | $0.1143$ | **$0.1069$** | **$0.1052$** | $41.5\%$ | $69.7\%$ |
| | Reliability-Aware | $74.88\%$ | $75.49\%$ | $74.62\%$ | **$-0.88\text{ pp}$** | **$0.1019$** | **$0.1000$** | $0.1483$ | $0.1527$ | **$26.4\%$** | $79.6\%$ |
| | Combined | $74.13\%$ | $74.70\%$ | $74.86\%$ | **$+0.15\text{ pp}$** | **$0.1015$** | **$0.1006$** | **$0.1041$** | **$0.1025$** | **$18.2\%$** | **$85.8\%$** |
| **`simple_dynamics`** | Baseline | $76.01\%$ | $76.61\%$ | $74.37\%$ | $-2.24\text{ pp}$ | $0.1547$ | $0.1479$ | $0.1941$ | $0.1944$ | $26.6\%$ | $75.6\%$ |
| | Transfer-Aware | $75.25\%$ | $75.84\%$ | $74.52\%$ | $-1.32\text{ pp}$ | $0.1493$ | $0.1440$ | **$0.1551$** | **$0.1524$** | $16.3\%$ | $82.7\%$ |
| | Reliability-Aware | $75.64\%$ | $76.21\%$ | $74.46\%$ | $-1.75\text{ pp}$ | $0.1513$ | $0.1451$ | $0.1924$ | $0.1927$ | $19.9\%$ | $79.7\%$ |
| | Combined | $74.81\%$ | $75.40\%$ | $74.59\%$ | **$-0.81\text{ pp}$** | **$0.1467$** | **$0.1420$** | **$0.1540$** | **$0.1514$** | **$11.2\%$** | **$86.4\%$** |
| **`temporal_dynamics`** | Baseline | $68.81\%$ | $69.54\%$ | $74.03\%$ | **$+4.50\text{ pp}$** | **$0.1202$** | **$0.1062$** | **$0.1143$** | **$0.1076$** | $45.4\%$ | $65.0\%$ |
| | Transfer-Aware | $68.02\%$ | $68.73\%$ | $74.44\%$ | $+5.71\text{ pp}$ | $0.1324$ | $0.1153$ | $0.1332$ | $0.1192$ | $33.9\%$ | $73.9\%$ |
| | Reliability-Aware | $67.94\%$ | $68.64\%$ | $74.49\%$ | $+5.85\text{ pp}$ | $0.1429$ | $0.1294$ | $0.1228$ | $0.1144$ | $30.7\%$ | $76.2\%$ |
| | Combined | $66.95\%$ | $67.55\%$ | $74.66\%$ | $+7.11\text{ pp}$ | $0.1566$ | $0.1409$ | $0.1427$ | $0.1278$ | **$20.2\%$** | **$83.5\%$** |
| **`distribution_shift`** | Baseline | $64.31\%$ | $64.30\%$ | $73.39\%$ | **$+9.09\text{ pp}$** | **$0.1842$** | **$0.1843$** | **$0.1538$** | **$0.1457$** | $61.0\%$ | $51.9\%$ |
| | Transfer-Aware | $63.60\%$ | $63.64\%$ | $74.03\%$ | $+10.39\text{ pp}$ | $0.2006$ | $0.1990$ | $0.2294$ | $0.2328$ | $51.7\%$ | $59.9\%$ |
| | Reliability-Aware | $62.73\%$ | $62.75\%$ | $74.66\%$ | $+11.91\text{ pp}$ | $0.2451$ | $0.2575$ | $0.1758$ | $0.1676$ | $37.0\%$ | $71.7\%$ |
| | Combined | $61.60\%$ | $61.70\%$ | $74.98\%$ | $+13.28\text{ pp}$ | $0.2610$ | $0.2720$ | $0.2510$ | $0.2540$ | **$28.5\%$** | **$78.2\%$** |

---

## 6. Corrected Paired Cluster-Bootstrap Statistical Inference

Statistical differences are computed using the corrected seed-aware cluster bootstrap ($B = 2,000$ replicates, random seed $42$, clustering unit: `(seed, learner_id)`, $K = 180$ clusters per comparison). Confidence intervals are $95\%$ percentile intervals (`ci_95`).

### Table 3: Paired Bootstrap Differences vs. Baseline (Held-Out Seeds `[777, 888, 555]`)

| Scenario | Policy Variant | Prob Abs Error Diff (95% CI) | Pre-MAE Diff (95% CI) | Post-MAE Diff (95% CI) | $D_{\min}$ Saturation Diff (95% CI) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`base_heterogeneous`** | Transfer-Aware | $+0.0001$ $[-0.0005, +0.0007]$ | $-0.0006$ $[-0.0018, +0.0004]$ | $-0.0003$ $[-0.0013, +0.0007]$ | **$-0.1294$ $[-0.1414, -0.1178]$** |
| | Reliability-Aware | $-0.0003$ $[-0.0009, +0.0004]$ | $-0.0012$ $[-0.0030, +0.0007]$ | $-0.0011$ $[-0.0032, +0.0010]$ | **$-0.1219$ $[-0.1340, -0.1098]$** |
| | Combined | $+0.0016$ $[+0.0005, +0.0027]$ | $+0.0002$ $[-0.0023, +0.0026]$ | $+0.0008$ $[-0.0018, +0.0033]$ | **$-0.2206$ $[-0.2358, -0.2062]$** |
| **`reliability_stress`** | Transfer-Aware | **$-0.0040$ $[-0.0046, -0.0035]$** | **$-0.0060$ $[-0.0071, -0.0049]$** | **$-0.0053$ $[-0.0063, -0.0043]$** | **$-0.1013$ $[-0.1127, -0.0908]$** |
| | Reliability-Aware | **$-0.0065$ $[-0.0078, -0.0051]$** | **$-0.0170$ $[-0.0209, -0.0131]$** | **$-0.0196$ $[-0.0240, -0.0151]$** | **$-0.2527$ $[-0.2681, -0.2391]$** |
| | Combined | **$-0.0061$ $[-0.0078, -0.0043]$** | **$-0.0174$ $[-0.0218, -0.0129]$** | **$-0.0190$ $[-0.0239, -0.0141]$** | **$-0.3345$ $[-0.3506, -0.3187]$** |
| **`simple_dynamics`** | Transfer-Aware | **$-0.0018$ $[-0.0024, -0.0012]$** | **$-0.0054$ $[-0.0065, -0.0042]$** | **$-0.0039$ $[-0.0050, -0.0028]$** | **$-0.1030$ $[-0.1152, -0.0914]$** |
| | Reliability-Aware | **$-0.0008$ $[-0.0012, -0.0003]$** | **$-0.0033$ $[-0.0047, -0.0020]$** | **$-0.0028$ $[-0.0043, -0.0013]$** | **$-0.0679$ $[-0.0786, -0.0580]$** |
| | Combined | **$-0.0020$ $[-0.0030, -0.0010]$** | **$-0.0080$ $[-0.0100, -0.0061]$** | **$-0.0058$ $[-0.0079, -0.0038]$** | **$-0.1547$ $[-0.1695, -0.1398]$** |
| **`temporal_dynamics`** | Transfer-Aware | $+0.0094$ $[+0.0091, +0.0098]$ | $+0.0122$ $[+0.0116, +0.0128]$ | $+0.0091$ $[+0.0084, +0.0098]$ | **$-0.1156$ $[-0.1279, -0.1044]$** |
| | Reliability-Aware | $+0.0100$ $[+0.0094, +0.0105]$ | $+0.0227$ $[+0.0213, +0.0240]$ | $+0.0232$ $[+0.0215, +0.0249]$ | **$-0.1470$ $[-0.1590, -0.1360]$** |
| | Combined | $+0.0211$ $[+0.0204, +0.0218]$ | $+0.0364$ $[+0.0348, +0.0379]$ | $+0.0347$ $[+0.0327, +0.0365]$ | **$-0.2525$ $[-0.2668, -0.2389]$** |
| **`distribution_shift`** | Transfer-Aware | $+0.0128$ $[+0.0126, +0.0131]$ | $+0.0164$ $[+0.0160, +0.0168]$ | $+0.0147$ $[+0.0142, +0.0151]$ | **$-0.0931$ $[-0.1028, -0.0833]$** |
| | Reliability-Aware | $+0.0279$ $[+0.0274, +0.0285]$ | $+0.0609$ $[+0.0596, +0.0625]$ | $+0.0732$ $[+0.0716, +0.0749]$ | **$-0.2401$ $[-0.2530, -0.2274]$** |
| | Combined | $+0.0415$ $[+0.0410, +0.0420]$ | $+0.0768$ $[+0.0753, +0.0784]$ | $+0.0877$ $[+0.0860, +0.0896]$ | **$-0.3252$ $[-0.3397, -0.3111]$** |

### Key Inferential Corrections
1. **Shrinkage & Inconclusive Significance in `base_heterogeneous`**:
   The previously reported claim of statistically significant probability error reduction ($p < 10^{-15}$) for transfer-aware and reliability-aware was an artifact of the flawed bootstrap. With seed-aware cluster sampling, the $95\%$ confidence intervals for probability absolute error difference cross zero (Transfer-Aware: $[-0.0005, +0.0007]$; Reliability-Aware: $[-0.0009, +0.0004]$). **There is no statistically significant change in probability prediction error in the standard baseline scenario.**
2. **Robust Behavioral Effect on Bound Saturation**:
   In contrast, the reduction in lower-bound saturation ($D_{\min} = 0.08$) is substantial and unambiguous across all scenarios. In `base_heterogeneous`, lower-bound saturation falls by **$-12.94\text{ pp}$** for transfer-aware and by **$-22.06\text{ pp}$** for combined, with tightly bounded non-zero intervals. By maintaining secondary skill mastery and filtering sensor slips, the controller avoids misattributing failures to low skill, preventing artificial collapse into $D_{\min}$.
3. **Statistically Confirmed Gains in `reliability_stress`**:
   Under high sensor corruption, probability error reduction, mastery pre-MAE reduction, and $D_{\min}$ saturation reduction are all strictly significant (all $95\%$ CIs exclude zero). Reliability-aware reduces probability absolute error by **$-0.0065$** ($95\%\text{ CI}: [-0.0078, -0.0051]$) and reduces pre-MAE by **$-0.0170$** ($95\%\text{ CI}: [-0.0209, -0.0131]$).

---

## 7. Controller Behavioral Invariants and Safety Integrity

| Policy Variant | Frustration Violations | Session Break Violations | Safety Interventions (`base_heterogeneous`) | Safety Interventions (`distribution_shift`) |
| :--- | :---: | :---: | :---: | :---: |
| `model_aligned_baseline` | **0** | **0** | 265 | 737 |
| `transfer_aware` | **0** | **0** | 276 | 754 |
| `reliability_aware` | **0** | **0** | 269 | 771 |
| `combined_transfer_reliability` | **0** | **0** | 296 | 794 |

- **Zero Safety Violations**: Across all 540,000 interaction events, every controller variant strictly maintained zero frustration violations ($\le 3$ consecutive failures) and zero session break violations ($\le 15$ continuous trials).
- **Intervention vs. Saturation Trade-Off**: Because the combined controller tracks higher, time-aligned belief states, it challenges learners with higher difficulties ($D = 0.189$ vs. $0.152$ in baseline), reducing lower-bound collapse from $36.4\%$ to $14.4\%$. In exchange, slightly more cooldown safety interventions were triggered ($296$ vs. $265$) when learners encountered consecutive challenging trials.

---

## 8. Verification and Test Suite Execution

All tests were executed in the repository environment using:
```powershell
$env:PYTHONPATH="src"; .venv\Scripts\pytest -v
```

Output:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.11, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\BMU\BMU SEM 5\PRJ-1\grow21_simulation
configfile: pyproject.toml
testpaths: tests
collected 55 items

tests/test_audit_regressions.py (13 tests) .............                 [ 23%]
tests/test_belief_state_estimation.py (13 tests) .............           [ 47%]
tests/test_controller_constraints.py (4 tests) ....                     [ 54%]
tests/test_documentation_integrity.py (4 tests) ....                    [ 61%]
tests/test_model_shapes.py (2 tests) ..                                  [ 65%]
tests/test_no_leakage.py (3 tests) ...                                   [ 70%]
tests/test_outcome_model_alignment.py (10 tests) ..........              [ 89%]
tests/test_reliability.py (3 tests) ...                                  [ 94%]
tests/test_simulator.py (3 tests) ...                                    [100%]

============================= 55 passed in 32.53s =============================
```

All 55 tests passed with zero failures, zero warnings, and zero skips.

---

## 9. Limitations and Recommended Phase 5 Directions

1. **Static Initial Ability Prior ($\hat{\theta}_0 = 0.45$)**: The largest residual prediction gap occurs under `distribution_shift` ($+13.28\text{ pp}$ in combined), where virtual learners have initial mastery $\mu \approx 0.25$. A static population prior cannot compensate for cohort distribution shift.
   - *Phase 5 Recommendation*: Implement an empirical Bayes initial calibration phase (e.g., initial 5 trials of diagnostic probes) to establish a personalized initial prior $\hat{\theta}_0$ before engaging adaptive target control.
2. **Fixed Forgetting Rate Prior ($\mu_{\text{decay}} = 0.02$)**: In `temporal_dynamics`, learners forget at rate $\mu = 0.07$. Inferring the forgetting parameter from post-break recovery trials would eliminate temporal over-prediction.
3. **Replay vs. Closed-Loop Ablations**: Closed-loop comparisons allow belief updates to alter task difficulty and practice trajectories. While valid for evaluating complete adaptive policies, isolated causal mechanism attribution requires a fixed-trajectory offline replay study.

---

## 10. Mandatory Scientific Disclaimer

**All learners, interaction events, task parameters, cognitive dynamics, and telemetry streams in this repository are entirely synthetic.** No human clinical trials, pediatric behavioral studies, or developmental validations were conducted. Grow21 is a computational benchmark for algorithmic evaluation and must not be interpreted as establishing clinical efficacy or developmental validation for children with Trisomy 21 or any other neurodevelopmental profile.

---

## 11. Errata & Verification Addendum (Phase 5 Preflight Audit)

1. **Secondary Transfer Equation Ceiling Notation**:
   - In Section 3.A, the simulator secondary transfer equation was informally written as $\Delta \theta_{\text{sec}} = 0.30 \cdot \mu_{\text{lr}} \cdot (1 - \theta_{\text{sec}})$.
   - In the live canonical implementation (`src/grow21_simulation_benchmark/simulator.py`, line 202), the upper asymptote is governed by the declared mastery ceiling $0.95$:
     $$\Delta \theta_{\text{sec}} = 0.30 \cdot \text{learning\_rate}_{\text{sec}} \cdot (0.95 - \theta_{\text{sec}})$$
   - The controller implementation (`src/grow21_simulation_benchmark/closed_loop.py`, line 695) has always correctly used $(0.95 - \hat{\theta}_{\text{sec}})$. This erratum aligns the written report equation with both the simulator source and controller code.

2. **Complete Simulator Primary Learning Transition Specification**:
   - The ground-truth simulator primary skill progression (`simulator.py`, lines 180-197) is governed by:
     $$\Delta m_{\text{prim}} = \text{learn\_rate} \cdot (0.95 - m_{\text{prim}}) \cdot f_{\text{feedback}} \cdot E_D + \epsilon$$
     where:
     - Feedback factor: $f_{\text{feedback}} = 0.70 \cdot y_{\text{clean}} + 0.30 \cdot (1 - y_{\text{clean}})$. Clean success reinforces slightly more than clean failure, but clean failure still produces a non-negative expected learning increment ($0.30$).
     - Difficulty engagement: $E_D = 0.60 + 0.40 \cdot D_t$ (clamped to $[0.60, 1.00]$).
     - Stochastic process noise: $\epsilon \sim \mathcal{N}(0, 0.005^2)$ (mean zero).
     - Memory decay across session gap $\Delta t$: $m(t + \Delta t) = 0.05 + (m(t) - 0.05) \cdot e^{-\text{forgetting\_rate} \cdot \Delta t}$.

3. **Bootstrap Difference Sign Convention Clarification**:
   - The paired cluster bootstrap computes `variant_error - baseline_error`.
   - A negative difference indicates the policy variant *reduced* error relative to the baseline.
   - A positive difference indicates the policy variant *increased* error relative to the baseline.
   - The JSON key previously labeled `prob_abs_error_reduction` represents this signed difference (`variant - baseline`). In Phase 5, this is explicitly exported as `prob_abs_error_diff_variant_minus_baseline` (with the legacy alias preserved for backwards compatibility).

