# Grow21 Benchmark Audit — Phase 2 Final Reconciliation Report

**Date**: October 18, 2026  
**Auditor**: Antigravity Technical Audit Agent  
**Environment**: Windows 11 Enterprise (10.0.26100), Python 3.13.11, PyTorch 2.14.1+cpu, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, pytest 9.1.1 (32 passed)  
**Project**: Grow21 Standalone Synthetic Simulation & Knowledge-Tracing Benchmark  
**Canonical Output Directory**: `outputs/audit/phase_2/`

---

> ### Mandatory Research & Clinical Disclaimer
> This report and codebase pertain strictly to a **synthetic virtual learner simulation and offline benchmarking framework**. It is **not** a clinical or developmental validation of educational or therapeutic interventions for children with Trisomy 21 (Down syndrome). Virtual learner agents are parameterized mathematical abstractions combining item-response formulations, exponential memory decay, and simulated sensor noise models. They have **never been trained, fitted, or validated on real Grow21 child data**, nor do they capture the biological, medical, or neuropsychological complexities of real learners. Furthermore, this system is a **standalone benchmarking codebase** that is **not integrated into any Grow21 frontend or application**.

---

## 1. Executive Summary & Correction Overview

This Phase 2 final reconciliation pass resolves all outstanding discrepancies between audit documentation, simulation code, and empirical output data. No parameters have been artificially tuned to favor any model, nor has the simulator been altered to force synthetic alignment. All reported values trace directly to persisted, verifiable code and artifact records.

### Summary of Reconciled Corrections

| Domain | Prior Claim / Stale Value | Verified Value / Reality | Evidence Source / Run | Action Taken |
| :--- | :--- | :--- | :--- | :--- |
| **Distribution Shift Upper Bins** | Stale claim: seed 42 has $N=3$ in $[0.8, 0.9)$ for Logistic, $N=0$ for DKT models | **Seed 42 has $N=0$ in $[0.8, 0.9)$ and $N=0$ in $[0.9, 1.0]$ for all 3 models**; $[0.7, 0.8)$ has $N=71$ (Logistic), $N=45$ (Vanilla DKT), $N=16$ (Rel-Gated DKT) | Recalculated from `calibration_bins.csv` across all 45 slices ($N=4,500$ each) | Corrected report, changelog, and bundle index. Added deterministic persistence validation test `test_persisted_calibration_bins_and_ece_consistency` in `tests/test_audit_regressions.py` comparing all slices to `metrics.csv`. |
| **Closed-Loop Mastery Accounting** | Stale table cited $-0.0092 \pm 0.0562$ as "Primary Practiced Skills" and fabricated counts | **Primary practiced skills: $-0.0234 \pm 0.0652$ (Target), $-0.0221 \pm 0.0656$ (Streak), $-0.0451 \pm 0.0682$ (Fixed)** | Verified from `closed_loop_summary.json` and `learner_mastery_summary.csv` | Rebuilt mastery table with separate per-learner averages and pooled skill-record rows. Documented true primary count distribution (0:60, 6:60, 7:60, 13:120, 21:60). |
| **Target Reachability Edge Cases** | Unreachable targets defaulted to $D_{\min}=0.08$, failing unreachable-low targets | **Unreachable-high targets move to $D_{\min}$; unreachable-low targets move to $D_{\max}=0.95$** | Controller code in `closed_loop.py` & regression tests | Explicitly branched unreachable-high ($p^* > p_{\text{high}}$) vs unreachable-low ($p^* < p_{\text{low}}$). Added deterministic regression test. |
| **Reachability Diagnostic Semantics** | Reported 100% reachability for Fixed & Streak; safety cooldown set `target_reachable=False` | **Fixed & Streak do not evaluate target reachability (N/A); safety intervention decoupled from reachability** | Event records in `closed_loop_events.csv` & `closed_loop_summary.json` | Stored `target_reachable=None` and `reachability_applicable=0` for Fixed/Streak; separated `safety_intervention` flag. |
| **Predictor vs Simulator Mismatch** | Target $p^*=0.75$ described as environmentally unreachable | **$p^*=0.75$ is unreachable under controller surrogate model, not simulator generative process** | Mathematical comparison between `closed_loop.py` and `simulator.py` | Distinguished surrogate reachability from simulated clean success ($76.64\%$). Saved compact diagnostic `predictor_vs_simulator_comparison.csv`. |
| **Gamma / Nu Parameter Specification** | Mismatch between math design doc ($\gamma=3.5, \nu_D=2$) and code ($\gamma=2.4, \nu_D=1.0$) | **Active benchmark settings are $\gamma=2.4, \nu_D=1.0$**; $\gamma=3.5, \nu_D=2$ were provisional design values | `configs/default.yaml` and `closed_loop.py` initialization | Documented mismatch and rationale clearly without changing benchmark code. Updated README and reproduction guides. |
| **Production Helper Test Coverage** | Regression tests manually reimplemented reversal loop; lacked artifact persistence validation | **Tests now call production helper `calculate_direction_reversals` directly** | Full test suite (`pytest -v tests`) — 32 tests passed | Added tests for monotonic, immediate-reversal, filtered-zero-step, and flat-bound sequences, plus persisted artifact ECE check. |
| **Artifact Provenance & Archive Hashes** | Archive manifest lacked SHA-256 hashes; benchmark run provenance was unstated | **Computed SHA-256 for all 19 archived files; clarified benchmark outputs are preserved from Phase 2 run** | `outputs/archive/pre_correction_pass/manifest.json` | Populated SHA-256 and byte counts for all archived files. Transparently documented inherited vs newly rerun outputs. |
| **Reversals Schema & Diagnostics** | `closed_loop_summary.json` omitted separate immediate-reversal fields; safety counts misquoted in draft text | **Serialized both `immediate_direction_reversals_mean/std` and `filtered_direction_reversals_mean/std` (plus legacy alias); safety counts match final data (290, 97, 78)** | `closed_loop_summary.json` & `closed_loop_events.csv` | Added production helper `calculate_immediate_reversals`; verified summary matches event log; purged invalid non-printing control characters across all docs. |


---

## 2. Environment, Versions & Verification Commands

### Execution Platform
- **Operating System**: Windows 11 Enterprise (Build 10.0.26100)
- **CPU Architecture**: x86_64
- **Python Version**: 3.13.11 (`.\.venv\Scripts\python.exe`)
- **Key Package Versions**:
  - `torch`: 2.14.1+cpu
  - `scikit-learn`: 1.9.1
  - `numpy`: 2.5.4
  - `pandas`: 3.0.6
  - `scipy`: 1.18.1
  - `pytest`: 9.1.1
  - `pyyaml`: 6.0.3
  - `matplotlib`: 3.10.8

### Exact Verified Commands
```bash
# Activate virtual environment
.\.venv\Scripts\activate

# 1. Run all 32 unit and targeted regression tests
python -m pytest -v tests

# 2. Run multi-seed benchmark across all scenario families
python -m grow21_simulation_benchmark.run_benchmark \
    --config configs/default.yaml \
    --scenarios configs/scenarios.yaml \
    --scenario all \
    --seeds 42,123,999 \
    --num-learners 500 \
    --events-per-learner 60 \
    --output-dir outputs/benchmark

# 3. Run closed-loop adaptive simulation with event-level tracking
python -m grow21_simulation_benchmark.run_closed_loop \
    --config configs/default.yaml \
    --scenarios configs/scenarios.yaml \
    --scenario base_heterogeneous \
    --num-learners 60 \
    --trials-per-learner 60 \
    --seed 42 \
    --output-dir outputs/closed_loop

# 4. Run controlled reliability gating sweep and paired loss evidence generation
python -m grow21_simulation_benchmark.audit_calibration_bins --action reliability_sweep

# 5. Export canonical calibration bins for the 500-learner benchmark cohort
python -m grow21_simulation_benchmark.audit_calibration_bins --action calibration_bins
```

---

## 3. Closed-Loop Adaptive Simulation & Mastery Reporting

### A. Regenerated Comparison Metrics
The closed-loop simulation was executed on scenario `base_heterogeneous` with 60 virtual learners, 60 trials per learner, and random seed 42 under strictly identical initial profiles (common random numbers).

| Controller Strategy | Observed Success Rate | Clean Success Rate | Mean Difficulty ($\bar{D}$) | Immediate Reversals ($(D_t-D_{t-1})(D_{t-1}-D_{t-2})<0$) | Filtered Reversals (Nonzero transitions) | Total Difficulty Variation ($\sum \|\Delta D\|$) | Proportion at $D_{\min}=0.08$ | Reachable Target Proportion | Safety Interventions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fixed_difficulty`** (nominal $D=0.50$ + safety drops) | 47.14% | 52.75% | 0.480 | 4.733 ± 2.007 | 8.583 ± 4.076 | 2.392 ± 1.028 | 0.0% | **N/A** | 290 |
| **`streak_adaptive`** (2-up / 1-down heuristic) | 63.56% | 71.86% | 0.192 | 5.467 ± 1.658 | 17.933 ± 2.949 | 2.032 ± 0.217 | 30.6% | **N/A** | 97 |
| **`target_adaptive`** (Grow21 Model-Based, $p^*=0.75$) | 67.69% | 76.64% | 0.091 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.290 ± 0.000 | **93.4%** | **0.08%** (3 / 3,600) | 78 |

*Note on Reachability Diagnostic*: Fixed Difficulty and Streak Adaptive policies do not formulate or target a success probability $p^*$. Reporting $100\%$ target reachability for these policies was an artifact of a default boolean diagnostic. In this final reconciliation pass, target reachability for non-target controllers is formally designated **Not Applicable (N/A)** (`reachability_applicable = 0`, `target_reachable = null`).

### B. Controller-Predictor vs. Simulator Generative Equation Mismatch

A central architectural finding of this audit is that **the controller's internal success predictor and the simulator's true generative process are formulated differently**:

1. **Controller Internal Predictor** (`closed_loop.py`):
   $$\hat{p}_{\text{ctrl}}(D) = \sigma\left(\text{logit}(\hat{\theta}) - \gamma \cdot D^{\nu_D}\right)$$
   With active parameters $\gamma = 2.4$ and $\nu_D = 1.0$.

2. **Simulator Generative Model** (`simulator.py`):
   $$D_{\text{threshold}} = \text{offset} + \text{scale} \cdot D$$
   $$\text{logit\_diff} = a_g \cdot (\theta_{\text{eff}} - D_{\text{threshold}})$$
   $$p_{\text{clean}} = c_g + (1 - c_g - l_g) \cdot \sigma(\text{logit\_diff})$$
   Where $a_g$ is game discrimination ($3.0 - 3.5$), $c_g$ is guess floor ($0.05 - 0.10$), $l_g$ is lapse ceiling ($0.05 - 0.08$), and effective mastery $\theta_{\text{eff}} = 0.85 \theta_{\text{prim}} + 0.15 \theta_{\text{sec}}$.

#### The Source of Controller Saturation
Under the controller's surrogate formula, at baseline learner mastery ($\hat{\theta} \approx 0.45 - 0.55$) and minimum difficulty $D_{\min} = 0.08$:
$$\hat{p}_{\text{ctrl}}(D_{\min}) = \sigma(\text{logit}(0.55) - 2.4 \times 0.08) = \sigma(0.2007 - 0.1920) = \sigma(0.0087) \approx 0.5022$$
Because $\hat{p}_{\text{ctrl}}(D_{\min}) \approx 0.502 \ll 0.75$, the controller's internal model calculates that target $p^* = 0.75$ is **unreachable-high**. The controller therefore clamps difficulty down to $D_{\min} = 0.08$ on virtually all trials ($93.4\%$).

However, **this does not establish that $p^* = 0.75$ is unreachable in the actual simulation environment**.
Inside the simulator, for the same learner at $D = 0.08$ on `trace_glow` ($D_{\text{threshold}} = -0.2 + 1.4(0.08) = -0.088$):
$$\text{logit\_diff} = 3.5 \times (0.55 - (-0.088)) = 3.5 \times 0.638 = 2.233 \implies \sigma(2.233) \approx 0.903$$
$$p_{\text{clean}} = 0.05 + 0.87 \times 0.903 \approx 0.8358$$

#### Diagnostic Comparison: Controller Predictor vs. Simulator Clean Success
The table below (persisted in `outputs/audit/phase_2/predictor_vs_simulator_comparison.csv`) illustrates this structural divergence for `trace_glow`:

| Learner Mastery ($\theta$) | Difficulty ($D$) | Controller Predicted $\hat{p}$ | Simulator Clean $p_{\text{clean}}$ | Prediction Gap ($\hat{p} - p_{\text{clean}}$) | Controller Reachability ($p^*=0.75$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.45 | 0.08 ($D_{\min}$) | 0.4031 | 0.8051 | -0.4020 | **Unreachable-High** |
| 0.45 | 0.50 | 0.1977 | 0.4470 | -0.2493 | Unreachable-High |
| 0.55 | 0.08 ($D_{\min}$) | 0.5022 | 0.8358 | -0.3336 | **Unreachable-High** |
| 0.55 | 0.50 | 0.2691 | 0.5230 | -0.2539 | Unreachable-High |
| 0.75 | 0.08 ($D_{\min}$) | 0.7123 | 0.8760 | -0.1637 | **Unreachable-High** |
| 0.75 | 0.50 | 0.4747 | 0.6640 | -0.1894 | Unreachable-High |
| 0.85 | 0.08 ($D_{\min}$) | 0.8238 | 0.8885 | -0.0647 | **Reachable** ($D^* = 0.272$) |
| 0.85 | 0.50 | 0.6306 | 0.7225 | -0.0919 | Reachable |

**Key Takeaways**:
1. The controller under-predicts clean task success by 33 to 40 percentage points at low difficulties.
2. Clamping at $D_{\min} = 0.08$ results in an observed empirical clean success rate of **76.64%** across the 60 learners, purely because the simulator environment is far more forgiving at $D_{\min}$ than the controller expects.
3. The empirical success rate of $76.64\%$ is an artifact of bound clamping, **not** dynamic closed-loop ZPD target tracking.

### C. Resolution of Controller Target-Reachability Edge Cases

In the prior implementation of `Grow21ModelBasedAdaptiveController.select_next_task`, any unreachable target triggered:
`target_D = self.min_difficulty`
This handling is mathematically valid only when the target is **unreachable-high** ($p^* > \hat{p}(D_{\min})$). If the target is **unreachable-low** ($p^* < \hat{p}(D_{\max})$), choosing $D_{\min}$ was an inverted logic error: it made the task easiest when the policy should have made it hardest.

In this reconciliation pass, `select_next_task` distinguishes three mutually exclusive cases:
1. **Unreachable-High Target** ($p^* > \hat{p}(D_{\min})$): Controller targets $D_{\min}$ (easiest available difficulty to maximize probability of success).
2. **Unreachable-Low Target** ($p^* < \hat{p}(D_{\max})$): Controller targets $D_{\max}$ (hardest available difficulty to minimize probability of success toward $p^*$).
3. **Reachable Target** ($\hat{p}(D_{\max}) \le p^* \le \hat{p}(D_{\min})$): Controller solves $D^* = \left(\frac{\text{logit}(\hat{\theta}) - \text{logit}(p^*)}{\gamma}\right)^{1/\nu_D}$, bounded by $[D_{\min}, D_{\max}]$, with deadband ($\pm 0.04$) and step limits ($\pm 0.08$).

Furthermore, **safety cooldown interventions are decoupled from target reachability**: when consecutive failures trigger a cooldown step-down, the controller records `safety_intervention = True`, but evaluates `target_reachable` according to the actual mathematical feasibility of $p^*$ under the current belief.

These behaviors are rigorously verified by `test_target_success_reachability_and_bound_saturation` and `test_controller_reachability_applicability_and_safety_separation`.

### D. Mastery Accounting: Primary Practice vs. Secondary Transfer

Mastery dynamics in the Grow21 simulator obey three distinct regimes:
1. **Direct Primary Practice**: When a game is played, its primary skill receives direct learning:
   $$\theta_{i, \text{prim}} \leftarrow \theta_{i, \text{prim}} + \eta_i \cdot (0.95 - \theta_{i, \text{prim}})$$
2. **Secondary Skill Transfer**: If the game designates a secondary skill, it receives 30% transfer practice:
   $$\theta_{i, \text{sec}} \leftarrow \theta_{i, \text{sec}} + 0.30 \cdot \eta_i \cdot (0.95 - \theta_{i, \text{sec}})$$
3. **Inter-Session Forgetting**: During scheduled rest breaks ($\Delta t = 12.0$ hours), all skills undergo exponential decay toward baseline ($0.05$):
   $$\theta_{i, k} \leftarrow 0.05 + (\theta_{i, k} - 0.05) \cdot \exp(-\lambda_i \cdot \Delta t)$$

#### Reconciled Mastery Gains by Controller Policy
Below are the verified empirical mastery changes, presented both as **Per-Learner Averages** (unit: virtual learner, $N=60$ per policy, as summarized in `closed_loop_summary.json`) and as **Pooled Skill-Record Rows** (unit: learner-skill row, $N=360$ per policy from `learner_mastery_summary.csv`):

| Controller Policy | Metric Aggregation Unit | All Skills Combined ($\Delta \theta_{\text{all}}$) | Primary Practiced Skills ($\Delta \theta_{\text{prim}}$) | Secondary-Only Transfer Skills ($\Delta \theta_{\text{sec}}$) |
| :--- | :--- | :---: | :---: | :---: |
| **`fixed_difficulty`** | **Per-Learner Average** ($N=60$) | **-0.0319 ± 0.0590** | **-0.0451 ± 0.0682** | **+0.0341 ± 0.1980** |
| | Pooled Skill Rows ($N=360$) | -0.03192 ± 0.1771 ($N=360$) | -0.04512 ± 0.1695 ($N=300$) | +0.03410 ± 0.1996 ($N=60$) |
| **`streak_adaptive`** | **Per-Learner Average** ($N=60$) | **-0.0083 ± 0.0562** | **-0.0221 ± 0.0656** | **+0.0605 ± 0.1910** |
| | Pooled Skill Rows ($N=360$) | -0.00833 ± 0.1718 ($N=360$) | -0.02209 ± 0.1643 ($N=300$) | +0.06048 ± 0.1926 ($N=60$) |
| **`target_adaptive`** | **Per-Learner Average** ($N=60$) | **-0.0092 ± 0.0562** | **-0.0234 ± 0.0652** | **+0.0618 ± 0.1911** |
| | Pooled Skill Rows ($N=360$) | -0.00918 ± 0.1704 ($N=360$) | -0.02338 ± 0.1623 ($N=300$) | +0.06179 ± 0.1927 ($N=60$) |

#### Structure of the Practice Distribution
1. **Primary Trial Distribution**: In a 60-trial curriculum across 6 skills, primary trials are **not evenly distributed at 10 trials per skill**. Across the 360 learner-skill rows per policy, observed primary-task frequencies are:
   - **0 primary trials**: 60 rows (`visual_discrimination`, which serves exclusively as a secondary transfer skill in `trace_glow` and `bubble_pop_brigade`).
   - **6 primary trials**: 60 rows (`categorization`, from `sorting_spree`).
   - **7 primary trials**: 60 rows (`phonological_awareness`, from `echo_safari`).
   - **13 primary trials**: 120 rows (60 for `working_memory` from `memory_match`, 60 for `sequential_reasoning` from `pattern_train`).
   - **21 primary trials**: 60 rows (`fine_motor`, from `trace_glow`, `bubble_pop_brigade`, and `block_stacker`).
   While the arithmetic mean across all 360 rows is $(0\times 60 + 6\times 60 + 7\times 60 + 13\times 120 + 21\times 60)/360 = 3600 / 360 = 10.0$ trials/row, individual skills receive between 0 and 21 direct trials.
2. **Definition of Secondary-Only Skills**: "Secondary-only" designates skills with `secondary_transfer_counts > 0` AND `primary_task_counts == 0`. In this schedule, this applies exclusively to `visual_discrimination`, which received 32 transfer events and 0 primary practice events per learner. Because it was engaged frequently (32 transfer opportunities) without suffering high-difficulty primary failures, it demonstrated positive net mastery growth ($+0.0618$).
3. **No Unpracticed Skills Across Full Run**: In the 60-trial run, **zero** learner-skill rows had both `primary_task_counts == 0` and `secondary_transfer_counts == 0`. Every skill was engaged either directly or through transfer. The negative net gains observed on primary practiced skills reflect inter-session forgetting across 3 mandatory 12-hour rest breaks (36 hours total), which naturally outpaced sparse within-session practice (6–7 trials for categorization and phonological awareness).

### E. Session Break Verification
In a 60-trial run with `max_session_trials = 15`:
- Trial 0–14: Within-session interactions (gap = 0.02h)
- Trial 15: Mandatory rest gap (12.0h)
- Trial 16–29: Within-session interactions
- Trial 30: Mandatory rest gap (12.0h)
- Trial 31–44: Within-session interactions
- Trial 45: Mandatory rest gap (12.0h)
- Trial 46–59: Within-session interactions
- Trial 60: Run concludes.
Total scheduled session rest gaps: **exactly 3 breaks = 36.0 hours**, NOT 4 breaks / 48 hours. Verified deterministically in `test_break_count_and_elapsed_gap_logic`.

---

## 4. Calibration, Evaluation Cohorts & Distribution Shift

### A. Resolution of the Cohort Size & ECE Mismatch
The Phase 2 audit discovered that the prior report cited ECE values derived from an auxiliary 300-learner run ($N=2,700$ test events) rather than the primary 500-learner benchmark ($N=4,500$ test events).
- In the primary benchmark (`metrics.csv`), each scenario/seed has 500 learners, with a 70/15/15 split:
  $$\text{Held-out test set} = 75 \text{ learners} \times 60 \text{ events} = 4,500 \text{ test events}$$
- All calibration bins have now been regenerated on this exact 4,500-event cohort. Bin counts sum to exactly 4,500 across all 10 probability bins.
- Manual ECE recomputation matches reported ECE values exactly within rounding precision ($\le 10^{-4}$).

### B. Primary Benchmark Metrics Across All Scenario Families (500 Learners, 3 Seeds)

| Scenario Family | Model | Log Loss (Mean ± Std) | ROC-AUC (Mean ± Std) | Brier Score (Mean ± Std) | Accuracy (Mean ± Std) | ECE (Mean ± Std) | Test Events Count |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`base_heterogeneous`** | Logistic Baseline | **0.6256 ± 0.0026** | **0.6769 ± 0.0056** | **0.2179 ± 0.0012** | 0.6474 ± 0.0029 | 0.0151 ± 0.0015 | 4,500 |
| | Vanilla DKT | 0.6275 ± 0.0026 | 0.6735 ± 0.0033 | 0.2188 ± 0.0011 | 0.6470 ± 0.0007 | 0.0131 ± 0.0018 | 4,500 |
| | Reliability-Gated DKT | 0.6270 ± 0.0024 | 0.6744 ± 0.0049 | 0.2186 ± 0.0011 | **0.6484 ± 0.0035** | 0.0174 ± 0.0021 | 4,500 |
| **`reliability_stress`** | Logistic Baseline | **0.5969 ± 0.0024** | **0.6625 ± 0.0046** | **0.2049 ± 0.0011** | 0.6823 ± 0.0056 | 0.0125 ± 0.0026 | 4,500 |
| | Vanilla DKT | 0.5976 ± 0.0008 | 0.6611 ± 0.0071 | 0.2054 ± 0.0003 | 0.6779 ± 0.0091 | 0.0182 ± 0.0015 | 4,500 |
| | Reliability-Gated DKT | 0.5975 ± 0.0011 | 0.6615 ± 0.0058 | 0.2052 ± 0.0005 | 0.6811 ± 0.0060 | 0.0145 ± 0.0045 | 4,500 |
| **`simple_dynamics`** | Logistic Baseline | **0.6452 ± 0.0061** | **0.6760 ± 0.0117** | **0.2268 ± 0.0029** | **0.6308 ± 0.0096** | 0.0179 ± 0.0009 | 4,500 |
| | Vanilla DKT | 0.6453 ± 0.0063 | 0.6745 ± 0.0118 | 0.2269 ± 0.0030 | 0.6294 ± 0.0094 | 0.0177 ± 0.0065 | 4,500 |
| | Reliability-Gated DKT | 0.6458 ± 0.0055 | 0.6748 ± 0.0105 | 0.2271 ± 0.0026 | 0.6288 ± 0.0087 | 0.0189 ± 0.0045 | 4,500 |
| **`temporal_dynamics`** | Logistic Baseline | **0.5880 ± 0.0052** | **0.6758 ± 0.0024** | **0.2010 ± 0.0024** | **0.6888 ± 0.0069** | 0.0118 ± 0.0011 | 4,500 |
| | Vanilla DKT | 0.5898 ± 0.0051 | 0.6722 ± 0.0010 | 0.2017 ± 0.0023 | 0.6875 ± 0.0090 | 0.0160 ± 0.0048 | 4,500 |
| | Reliability-Gated DKT | 0.5889 ± 0.0049 | 0.6738 ± 0.0007 | 0.2014 ± 0.0023 | 0.6876 ± 0.0047 | 0.0127 ± 0.0024 | 4,500 |
| **`distribution_shift`** | Logistic Baseline | **0.5867 ± 0.0147** | 0.6595 ± 0.0056 | **0.2006 ± 0.0065** | **0.6912 ± 0.0128** | **0.1102 ± 0.0222** | 4,500 |
| | Vanilla DKT | 0.5957 ± 0.0160 | 0.6574 ± 0.0032 | 0.2050 ± 0.0072 | 0.6702 ± 0.0182 | 0.1262 ± 0.0282 | 4,500 |
| | Reliability-Gated DKT | 0.6059 ± 0.0180 | 0.6583 ± 0.0024 | 0.2095 ± 0.0083 | 0.6555 ± 0.0275 | 0.1443 ± 0.0263 | 4,500 |

### C. Distribution Shift Tail Analysis
Under `distribution_shift`, learners in the test distribution possess substantially lower initial mastery ($\mu = 0.25$ vs $\mu = 0.55$) and higher corruption ($2.2\times$ vs $0.8\times$).
- **Sample Distribution Across Bins**:
  - In `base_heterogeneous`, bins $[0.2, 0.6]$ contain $\approx 90\%$ of test predictions.
  - In `distribution_shift`, predictions shift heavily leftward into $[0.1, 0.4]$.
  - The upper bins $[0.7, 0.8)$, $[0.8, 0.9)$, and $[0.9, 1.0]$ are **extremely sparse**:
    - **Seed 42**: Bin $[0.7, 0.8)$ contains **71 observations** for Logistic Regression, **45 observations** for Vanilla DKT, and **16 observations** for Reliability-Gated DKT. Bins $[0.8, 0.9)$ and $[0.9, 1.0]$ contain **0 observations** across all three models.
    - **Seed 123**: Bin $[0.7, 0.8)$ has 167 (Logistic), 25 (Vanilla DKT), and 34 (Rel-Gated DKT). Bin $[0.8, 0.9)$ has 21 (Logistic), 0 (Vanilla DKT), and 0 (Rel-Gated DKT). Bin $[0.9, 1.0]$ has 0 across all models.
    - **Seed 999**: Bin $[0.7, 0.8)$ has 41 (Logistic), 47 (Vanilla DKT), and 65 (Rel-Gated DKT). Bin $[0.8, 0.9)$ has 1 (Logistic), 1 (Vanilla DKT), and 0 (Rel-Gated DKT). Bin $[0.9, 1.0]$ has 0 across all models.
    - **Across All Seeds**: In $[0.8, 0.9)$, total counts are 22 for Logistic Regression, 1 for Vanilla DKT, and 0 for Reliability-Gated DKT. In $[0.9, 1.0]$, count is 0 across all models and all seeds.
- **Root Cause of High-Probability Jumps**:
  In `calibration_distribution_shift.png`, the curve for Seed 42 terminates at bin $[0.7, 0.8)$, where empirical accuracy jumps to $0.6250$ for Reliability-Gated DKT based on just 16 observations (10 successes vs 6 failures). This upward step represents small-sample binomial variance under severe covariate shift combined with model overconfidence when trained on high-mastery learners ($\mu=0.55$) and evaluated on low-mastery learners ($\mu=0.25$). It should not be interpreted as robust calibration in the tail.

---

## 5. Reliability-Gating Sweep & Statistical Claims

### A. Reliability Bias Sign Audit
In `reliability.py`:
$$\hat{p}_{\text{modality}} = \text{clip}(p_{\text{true}} + \text{Normal}(\text{bias}, \sigma_{\text{noise}}), 0.01, 0.95)$$
$$\hat{\rho} = \prod_{m \in \text{active}} (1 - \hat{p}_m)$$
When `uncertainty_bias = +0.12`:
- $\hat{p} > p_{\text{true}}$ (the model estimates higher corruption probability than physical reality).
- $1 - \hat{p} < 1 - p_{\text{true}} \implies \hat{\rho} < \rho_{\text{true}}$ (the model underestimates sensor reliability).
- Therefore, a positive bias represents an **overestimation of observation noise / underestimation of reliability**, not an underestimation of noise. The code has been retained for comparability, and the descriptive label has been corrected.

### B. Controlled Reliability Sweep Results (Composite Stress Conditions)
The sweep evaluates models across 4 conditions where physical corruption multipliers and estimator stress vary simultaneously (composite stress levels):

| Condition | Corruption Multiplier | Estimator Stress Parameters | Vanilla DKT Log Loss | Reliability-Gated DKT Log Loss | Logistic Baseline Log Loss | Paired Diff ($\text{Gated} - \text{Vanilla}$) | 95% Bootstrap CI | Statistically Significant? |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 1: Accurate** | $1.2\times$ | Noise $\sigma=0.01$, Bias $=0.00$, Dropout $=0.00$ | 0.6322 | 0.6332 | 0.6345 | +0.0010 | $[-0.0015, +0.0037]$ | **No** (CI includes 0) |
| **Level 2: Noisy** | $1.5\times$ | Noise $\sigma=0.15$, Bias $=0.00$, Dropout $=0.00$ | 0.6289 | 0.6269 | 0.6288 | -0.0020 | $[-0.0040, +0.0001]$ | **No** (CI includes 0) |
| **Level 3: Miscalibrated** | $1.8\times$ | Noise $\sigma=0.05$, Bias $=+0.12$, Dropout $=0.00$ | 0.6142 | 0.6112 | **0.6096** | -0.0030 | $[-0.0061, +0.0005]$ | **No** (CI includes 0) |
| **Level 4: Missing Modalities** | $1.8\times$ | Noise $\sigma=0.05$, Bias $=+0.02$, Dropout $=0.35$ | 0.6073 | 0.6019 | **0.6003** | **-0.0054** | **$[-0.0088, -0.0020]$** | **Yes** ($\Delta < 0$) |

#### Statistical Findings
1. **Gating Advantage Confirmed Under Missingness**: Reliability gating provides a statistically significant improvement over Vanilla DKT **only in Level 4** (high sensor missingness, 95% CI $[-0.0088, -0.0020]$).
2. **Null Results for Noise & Miscalibration**: Under noisy estimation (Level 2) and miscalibrated bias (Level 3), the point differences favor gated DKT (-0.0020 and -0.0030), but their 95% confidence intervals cross zero ($+0.0001$ and $+0.0005$). They **cannot be claimed as statistically significant**.
3. **Logistic Baseline Dominance**: Across all 4 levels, feature-engineered Logistic Regression matches or outperforms both deep sequence models (LogLoss 0.6003 in Level 4 vs. 0.6019 for Gated DKT and 0.6073 for Vanilla DKT).

### C. Information Causal Invariance Verification
Targeted regression tests (`test_dkt_no_leakage_invariant`, `test_reliability_gated_dkt_no_leakage_invariant`, `test_logistic_feature_extractor_no_leakage`) confirm that:
- Prediction $\hat{y}_t$ is computed strictly using history through $t-1$ and pre-response task features.
- Modifying target outcome $y_t$, clean outcome $y_{\text{clean}, t}$, or future interaction features has **zero effect** on predicted probability $\hat{p}_t$.
- True latent mastery $\theta_{ik}$ is never exposed to baseline feature extractors or sequence models.

---

## 6. Structure Cleanup, Provenance & Parameter Reconciliation

### A. Repository Architecture & Canonical Relative Paths
The workspace has been organized into a consistent canonical hierarchy:
```
grow21_simulation_benchmark/
  configs/
    default.yaml               # Canonical task, game, and controller configurations
    scenarios.yaml             # Canonical scenario parameters (5 scenario families)
  src/
    grow21_simulation_benchmark/
      __init__.py
      paths.py                 # Central source of truth for canonical output paths
      tasks.py                 # TaskRegistry and game definitions
      simulator.py             # Virtual learner agent and simulation dynamics
      reliability.py           # Product-rule uncertainty and sensor noise engine
      features.py              # Strict causal feature extractor for Logistic baseline
      models/
        logistic_baseline.py   # L2 regularized logistic regression baseline
        dkt.py                 # Recurrent Vanilla DKT
        reliability_gated_dkt.py # Grow21 Reliability-Gated DKT
      dataset.py               # Learner-disjoint train/val/test sequence splitters
      evaluate.py              # Metrics, calibration tables, paired bootstrap CIs
      plots.py                 # Calibration curves and scenario comparisons
      run_benchmark.py         # Multi-scenario multi-seed benchmark CLI
      run_closed_loop.py       # Closed-loop adaptive simulation CLI
      audit_calibration_bins.py # Calibration export and 4-level reliability sweep CLI
  tests/
    test_simulator.py          # Latent mastery and forgetting tests
    test_reliability.py        # Uncertainty product rule and missingness tests
    test_no_leakage.py         # Causal timing and information isolation tests
    test_model_shapes.py       # Tensor dimensions and boundary behavior tests
    test_controller_constraints.py # Safety rules and session break limits
    tests/test_audit_regressions.py  # 13 targeted regression tests for all audit issues
    test_documentation_integrity.py # Scan verifying 0 control characters in text files
  outputs/
    benchmark/
      metrics.csv              # Full 45-run multi-seed benchmark metrics (preserved)
      summary.json             # Aggregated benchmark metrics (mean ± std, preserved)
      plots/                   # Calibration curves and scenario comparison charts
    closed_loop/
      closed_loop_events.csv   # 10,800 event-level trial records (60 learners × 60 trials × 3 policies)
      closed_loop_summary.json # Summary metrics including immediate/filtered reversals and safety counts
      learner_mastery_summary.csv # Per-learner per-skill initial, final, and gain records
      plots/
        closed_loop_comparison.png # Practiced mastery gains and stability comparison
        difficulty_trajectories.png # Trial-by-trial difficulty trajectories showing saturation
    audit/
      phase_2/
        PHASE_2_AUDIT_REPORT.md # This authoritative audit report
        BUNDLE_INDEX.md        # Complete index of all bundle artifacts and verified hashes
        REPRODUCE.md           # Step-by-step reproduction instructions
        CHANGELOG_PHASE_2.md   # Chronological log of code and document changes
        calibration_bins.csv   # 10-bin calibration tables for canonical 500-learner cohort (45 slices)
        reliability_sweep_summary.json # 4-level controlled sweep summary with bootstrap CIs
        learner_mastery_summary.csv    # Granular per-skill mastery tracking records
        paired_loss_evidence.csv       # Learner-level paired losses for independent CI verification
        predictor_vs_simulator_comparison.csv # Diagnostic comparison of controller vs simulator equations
        Grow21_Phase_2_Audit_Evidence.zip # Validated evidence bundle
    archive/
      pre_correction_pass/     # Preserved pre-correction outputs with SHA-256 verified manifest.json
```

### B. Parameter Reconciliation ($\gamma$ and $\nu_D$)
An earlier mathematical draft for Grow21 suggested provisional values $\gamma = 3.5$ and $\nu_D = 2.0$.
In the active benchmark:
- Active configuration in `configs/default.yaml`: `gamma: 2.4`, `nu_D: 1.0`.
- Active code initialization in `Grow21ModelBasedAdaptiveController`: default `gamma = 2.4`, `nu_D = 1.0`.
**Resolution**: The active benchmark settings are confirmed to be $\gamma = 2.4, \nu_D = 1.0$. The values $\gamma = 3.5, \nu_D = 2.0$ are documented as early conceptual design proposals that were superseded before benchmarking. Parameters were strictly preserved without retuning.

### C. Provenance of Benchmark Outputs (Inherited vs. Regenerated)
To ensure absolute provenance integrity:
1. **Preserved Benchmark Outputs**: `outputs/benchmark/metrics.csv` and `outputs/benchmark/summary.json` represent the complete 45-run evaluation (5 scenario families $\times$ 3 random seeds $\times$ 3 models on 500 learners). These files were preserved and verified identical (SHA-256: `5d1c6383...` for `metrics.csv`, `17d2ddb2...` for `summary.json`) from the pre-correction archive. They were not rerun in this pass, which focused on closed-loop reconciliation and calibration bin verification.
2. **Newly Regenerated Outputs in this Pass**:
   - `outputs/closed_loop/closed_loop_events.csv`: Regenerated with explicit `reachability_applicable`, nullable `target_reachable`, and decoupled `safety_intervention` fields.
   - `outputs/closed_loop/closed_loop_summary.json`: Regenerated with `target_reachable_proportion_mean = null` for Fixed/Streak, and explicit safety intervention counts.
   - `outputs/closed_loop/learner_mastery_summary.csv`: Regenerated with complete per-skill tracking records.
   - `outputs/closed_loop/plots/`: Trajectory and comparison plots regenerated from updated event records.
   - `outputs/audit/phase_2/predictor_vs_simulator_comparison.csv`: Newly created diagnostic artifact.
3. **Archive Manifest Verification**: All 19 files in `outputs/archive/pre_correction_pass/` now possess cryptographically verified SHA-256 digests in `outputs/archive/pre_correction_pass/manifest.json`. The evidence ZIP bundles this verified manifest; the archived raw payloads remain preserved in the local project workspace repository.

---

## 7. Evidence Bundle Verification

The evidence bundle `Grow21_Phase_2_Audit_Evidence.zip` has been rebuilt directly from canonical project paths.

> [!NOTE]
> Under the project evidence exclusion policy, `Grow21_Phase_2_Audit_Evidence.zip` is preserved in local project storage (`evidence/phase_2/`) and excluded from Git tracking via `.gitignore`. All active canonical outputs, source code, configs, and reports exist uncompressed in the repository.

- **ZIP Path**: `evidence/phase_2/Grow21_Phase_2_Audit_Evidence.zip` *(local/release asset)*
- **Integrity Status**: Validated using `zipfile.testzip()` (CRC32 match, zero corrupt entries).
- **Canonical Relative Structure**: All files inside the archive preserve their package-relative paths (`src/`, `configs/`, `tests/`, `outputs/benchmark/`, `outputs/closed_loop/`, `outputs/audit/phase_2/`).
- **Exclusions Enforced**: Virtual environments (`.venv`), Python byte-code caches (`__pycache__`, `.pytest_cache`), git/IDE metadata, and temporary files are strictly excluded.

---

## 8. Limitations & Recommendations for Phase 3

1. **Alignment of Controller Predictor and Simulator**:
   The current target controller surrogate equation $\hat{p}(D) = \sigma(\text{logit}(\hat{\theta}) - \gamma D^{\nu_D})$ underestimates clean simulator success by $33 - 40\%$ at low difficulty. Phase 3 should calibrate the controller predictor parameters ($\gamma, \nu_D$) or adopt an item-response threshold predictor aligned with the game mechanics to enable true dynamic difficulty adaptation rather than lower-bound saturation.
2. **Dynamic ZPD Targeting**:
   Rather than a static target $p^* = 0.75$, an adaptive controller should set personalized ZPD targets conditioned on baseline learner mastery (e.g., $p^* = \max(0.55, \hat{p}(D_{\min}) + 0.10)$).
3. **Synthetic-Only Scope**:
   All findings in this project are synthetic. No real Grow21 child data are present in this repository. These mathematical results must not be construed as clinical or developmental validation for children with Trisomy 21.

