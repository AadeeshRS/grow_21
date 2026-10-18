# Grow21 Synthetic Learner Simulator & Knowledge-Tracing Benchmark

> **MANDATORY CLINICAL & RESEARCH DISCLAIMER**: This module is a **synthetic simulation and benchmarking environment** for evaluating knowledge-tracing algorithms under controlled latent dynamics and sensor noise. It is **NOT** a clinical or developmental validation on children with Trisomy 21 (Down syndrome), and no models in this package were trained on real-world child interaction data. It is a standalone Python evaluation module and is strictly not integrated into any user-facing frontend or application.

---

## 1. Overview & Research Grounding

This benchmark addresses key open questions in knowledge tracing (KT) for special education game-based learning:
1. **Latent-State Generative Simulation vs Marginal Sampling**: Following [Pagonis et al. (2024)](https://arxiv.org/abs/2401.16832), independent correctness sampling destroys temporal learning dependencies. Our simulator implements a true latent-state agent with heterogeneous mastery, practice learning with diminishing returns, exponential forgetting over inter-session gaps, fatigue, secondary-skill transfer, and observation noise.
2. **Controlled Evaluation with Virtual Learners**: Following [Zhang & MacLellan (2021)](https://chrismaclellan.com/media/publications/Investigating-Knowledge-Tracing-Models-using-Simulated-Students.pdf), virtual learners support controlled comparisons and expose model limitations before human pilot studies.
3. **Deep Learning vs Competent Historical Baselines**: As shown by [Gervet et al. (2020)](https://jedm.educationaldatamining.org/index.php/JEDM/article/view/451), logistic regression with well-engineered history features often rivals or outperforms deep recurrent models (DKT) on moderate-sized datasets. We compare a competent **Performance Factors Analysis (PFA)** logistic baseline against **Vanilla DKT** and **Grow21 Reliability-Gated DKT**.

---

## 2. Models Evaluated

All three models are evaluated on strictly **learner-disjoint held-out test splits** using identical test events and causal timing invariants.

### A. Logistic Regression Baseline (`LogisticBaselineModel`)
- Uses `scikit-learn` with standardized numerical features and categorical one-hot encodings fitted on training learners only.
- Rich causal feature set strictly prior to event $t$:
  - Current task metadata: one-hot game ID, one-hot primary skill ID, continuous difficulty $D \in [0, 1]$, session gap $\Delta t$, pre-response estimated reliability $\hat{\rho}_t$.
  - History features strictly from trials $0 \dots t-1$: global smoothed success rate, skill-specific attempts, successes, failures (PFA), recency (last outcome, windowed 3 & 5 success rates), streak counts, and modality missingness indicators.

### B. Vanilla DKT (`VanillaDKT`)
- Recurrent neural network (`nn.GRUCell`, hidden dim 48, embedding dim 24) running on CPU.
- **Strict Causal Timing**:
  At time step $t$, recurrent state $h_{t-1}$ and pre-response metadata $e_{\text{meta}, t}$ form prediction:
  $$\hat{y}_t = \sigma(W_{\text{pred}} [h_{t-1}, e_{\text{meta}, t}] + b_{\text{pred}})$$
  Only *after* $\hat{y}_t$ and loss are formed does the observed outcome $y_t$ update the recurrent state:
  $$h_t = \text{GRUCell}([e_{\text{meta}, t}, y_t], h_{t-1})$$

### C. Reliability-Gated DKT (`ReliabilityGatedDKT`)
- Identical architecture, capacity, and causal prediction timing as Vanilla DKT.
- State update incorporates Grow21's reliability gate:
  $$\bar{h}_t = \text{GRUCell}([e_{\text{meta}, t}, y_t], h_{t-1})$$
  $$h_t = h_{t-1} + \hat{\rho}_t \cdot (\bar{h}_t - h_{t-1})$$
  where $\hat{\rho}_t = \prod_{m \in \text{active}} (1 - \hat{p}_{t,m})$.
- When physical sensor noise is severe ($\hat{\rho}_t \to 0$), state corruption is attenuated ($h_t \to h_{t-1}$).
- When confidence is high ($\hat{\rho}_t \to 1$), the candidate update is accepted ($h_t \to \bar{h}_t$).
- Missing sensor modalities are conservatively penalized and **never imputed as zero uncertainty**.

---

## 3. Project Structure

For detailed descriptions of directory roles and canonical paths, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

```text
grow21_simulation/
├── README.md
├── PROJECT_STRUCTURE.md        # File organization reference
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── default.yaml            # Game, skill, and controller configurations
│   └── scenarios.yaml          # 5 scenario family definitions
├── src/grow21_simulation_benchmark/
│   ├── paths.py                # Centralized canonical paths helper
│   ├── tasks.py                # Skills, games, abstract difficulty mapping
│   ├── reliability.py          # Grow21 product rule & uncertainty modeling
│   ├── simulator.py            # Latent mastery dynamics, IRT responses, forgetting & transfer
│   ├── features.py             # Causal feature extraction (no leakage)
│   ├── dataset.py              # Learner-disjoint splitting & PyTorch collation
│   ├── train.py                # BCE training with validation early stopping
│   ├── evaluate.py             # Log loss, Brier, ROC-AUC, ECE, bootstrap CIs
│   ├── closed_loop.py          # Closed-loop simulation policies & constraint tracking
│   ├── plots.py                # Calibration and scenario comparison figures
│   ├── run_benchmark.py        # CLI offline benchmark entry point
│   ├── run_closed_loop.py      # CLI closed-loop experiment entry point
│   └── audit_calibration_bins.py # CLI calibration export and reliability sweep
├── tests/                      # 32 unit and regression tests (pytest)
└── outputs/
    ├── benchmark/              # Multi-seed benchmark metrics, summary, plots
    ├── closed_loop/            # Closed-loop events, summary, mastery, trajectory plots
    ├── audit/phase_2/          # Audit report, bundle index, changelog, calibration bins, ZIP
    └── archive/                # Preserved pre-correction and pre-cleanup snapshots
        ├── pre_correction_pass/ # Preserved pre-correction output snapshot
        └── project_cleanup/    # Preserved historical prompts and pre-cleanup manifest
```

---

## 4. Setup & Quickstart

### Virtual Environment Setup
```bash
# In the repository root
python -m venv .venv
.\.venv\Scripts\activate  # On Windows PowerShell
pip install -e .
```

### Run All Unit & Targeted Regression Tests
```bash
python -m pytest -v tests
```
*Result: 32/32 tests pass, including reachability (reachable, unreachable-high, unreachable-low), reachability applicability, bound saturation, oscillation metrics, break schedules, calibration ECE consistency (`test_persisted_calibration_bins_and_ece_consistency` in `tests/test_audit_regressions.py`), documentation escape integrity (`test_documentation_integrity.py`), and no-leakage verification.*

### Run End-to-End Multi-Seed Offline Benchmark
```bash
python -m grow21_simulation_benchmark.run_benchmark \
    --config configs/default.yaml \
    --scenarios configs/scenarios.yaml \
    --scenario all \
    --seeds 42,123,999 \
    --num-learners 500 \
    --events-per-learner 60 \
    --output-dir outputs/benchmark
```

### Run Closed-Loop Adaptive Simulation
```bash
python -m grow21_simulation_benchmark.run_closed_loop \
    --config configs/default.yaml \
    --scenarios configs/scenarios.yaml \
    --scenario base_heterogeneous \
    --num-learners 60 \
    --trials-per-learner 60 \
    --seed 42 \
    --output-dir outputs/closed_loop
```

---

## 5. Actual Benchmark Results (500 Learners, 3 Seeds, N=4,500 Test Events/Seed)

Evaluated across **5 scenario families**, **3 random seeds** (42, 123, 999), **500 virtual learners** per run (70% train / 15% val / 15% test learner-disjoint split), and **60 events per learner** ($N = 4,500$ held-out test events per model/seed).

| Scenario Family | Model | Test Log Loss (Mean ± Std) | Test ROC-AUC (Mean ± Std) | Brier Score (Mean ± Std) | CPU Train Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`base_heterogeneous`** | Logistic Baseline | **0.6256 ± 0.0026** | **0.6769 ± 0.0056** | **0.2179 ± 0.0012** | 0.2s |
| | Vanilla DKT | 0.6275 ± 0.0026 | 0.6735 ± 0.0033 | 0.2188 ± 0.0011 | 25.2s |
| | Reliability-Gated DKT | 0.6270 ± 0.0024 | 0.6744 ± 0.0049 | 0.2186 ± 0.0011 | 31.2s |
| **`reliability_stress`** | Logistic Baseline | **0.5969 ± 0.0024** | **0.6625 ± 0.0046** | **0.2049 ± 0.0011** | 0.2s |
| | Vanilla DKT | 0.5976 ± 0.0008 | 0.6611 ± 0.0071 | 0.2054 ± 0.0003 | 28.5s |
| | Reliability-Gated DKT | 0.5975 ± 0.0011 | 0.6615 ± 0.0058 | 0.2052 ± 0.0005 | 34.2s |
| **`simple_dynamics`** | Logistic Baseline | **0.6452 ± 0.0061** | **0.6760 ± 0.0117** | **0.2268 ± 0.0029** | 0.2s |
| | Vanilla DKT | 0.6453 ± 0.0063 | 0.6745 ± 0.0118 | 0.2269 ± 0.0030 | 20.1s |
| | Reliability-Gated DKT | 0.6458 ± 0.0055 | 0.6748 ± 0.0105 | 0.2271 ± 0.0026 | 18.2s |
| **`temporal_dynamics`** | Logistic Baseline | **0.5880 ± 0.0052** | **0.6758 ± 0.0024** | **0.2010 ± 0.0024** | 0.2s |
| | Vanilla DKT | 0.5898 ± 0.0051 | 0.6722 ± 0.0010 | 0.2017 ± 0.0023 | 32.7s |
| | Reliability-Gated DKT | 0.5889 ± 0.0049 | 0.6738 ± 0.0007 | 0.2014 ± 0.0023 | 21.7s |
| **`distribution_shift`** | Logistic Baseline | **0.5867 ± 0.0147** | **0.6595 ± 0.0056** | **0.2006 ± 0.0065** | 0.2s |
| | Vanilla DKT | 0.5957 ± 0.0160 | 0.6574 ± 0.0032 | 0.2050 ± 0.0072 | 25.5s |
| | Reliability-Gated DKT | 0.6059 ± 0.0180 | 0.6583 ± 0.0024 | 0.2095 ± 0.0083 | 28.0s |

### Key Scientific Findings
1. **Did DKT beat Logistic Regression?**
   **No.** Across all 5 scenario families, the competent **Logistic Regression baseline achieved the lowest log loss and highest ROC-AUC**, while training over 100× faster. This directly replicates the empirical findings of **Gervet et al. (2020)**. When a linear baseline is equipped with proper Performance Factors Analysis history features (running successes, failures, smoothed rates, recent outcomes, and session gaps), it presents a formidable benchmark.
2. **Did Reliability Gating improve upon Vanilla DKT?**
   **Yes, specifically under sensor missingness.** In a 4-level controlled sweep:
   - Level 4 (35% sensor dropout): Reliability-Gated DKT significantly outperforms Vanilla DKT (paired $\Delta \text{LogLoss} = -0.0054$, 95% bootstrap CI $[-0.0088, -0.0020]$).
   - In Level 1, 2, and 3, the 95% confidence intervals cross zero and do not reach statistical significance.
3. **Distribution Shift Behavior:**
   When tested on an out-of-distribution learner population with lower mastery and higher noise, the Logistic Baseline demonstrated superior resilience (Log Loss 0.5867 vs 0.5957 / 0.6059).

---

## 6. Closed-Loop Adaptive Simulation Results

Simulated on `base_heterogeneous` with 60 virtual learners, 60 trials per learner, and seed 42 using common random numbers:

| Controller Strategy | Observed Success Rate | Clean Success Rate | Mean Difficulty ($\bar{D}$) | Immediate Reversals | Filtered Reversals | Total Variation ($\sum \|\Delta D\|$) | % Trials at $D_{\min}=0.08$ | Reachable Target Trials | Safety Interventions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fixed_difficulty`** (nominal $D=0.50$ + safety drops) | 47.14% | 52.75% | 0.480 | 4.733 ± 2.007 | 8.583 ± 4.076 | 2.392 ± 1.028 | 0.0% | **N/A** | 290 (8.06%) |
| **`streak_adaptive`** (2-up / 1-down heuristic) | 63.56% | 71.86% | 0.192 | 5.467 ± 1.658 | 17.933 ± 2.949 | 2.032 ± 0.217 | 30.6% | **N/A** | 97 (2.69%) |
| **`target_adaptive`** (Grow21 Model-Based, $p^*=0.75$) | 67.69% | 76.64% | 0.091 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.290 ± 0.000 | **93.4%** | **0.08%** (3/3600) | 78 (2.17%) |

### Critical Investigation: Target Controller Dynamics & Modeling Reconciliations

1. **Controller-Predictor vs. Simulator Equation Mismatch**:
   - The model-based controller uses internal surrogate equation $\hat{p}(D) = \sigma(\text{logit}(\hat{\theta}) - \gamma D^{\nu_D})$ with active simulation settings **$\gamma = 2.4, \nu_D = 1.0$** (reconciling with earlier provisional design document values $\gamma = 3.5, \nu_D = 2.0$, which were not active benchmark parameters).
   - Under this surrogate predictor, for an average learner ($\hat{\theta} \approx 0.50$) at minimum difficulty $D_{\min} = 0.08$:
     $$\hat{p}(D_{\min}) = \sigma(\text{logit}(0.50) - 2.4 \times 0.08) \approx 0.5022 < 0.75$$
     Because $\hat{p}(D_{\min}) \ll 0.75$, target $p^* = 0.75$ is **unreachable under the controller's internal predictor**.
   - In contrast, the simulator's true generative process (`simulator.py`) computes clean success using Item Response Theory with game discrimination $a_g \in [3.0, 3.5]$, guess floor $c_g \in [0.05, 0.10]$, lapse ceiling $l_g \in [0.05, 0.08]$, and effective mastery including secondary skills:
     $$p_{\text{clean}} = c_g + (1 - c_g - l_g) \sigma(a_g (\theta_{\text{eff}} - D))$$
     At $D_{\min} = 0.08$, the simulator yields $p_{\text{clean}} \approx 83.58\%$. The aggregate clean success rate across the run was **76.64%**.
   - **Crucial Distinction**: The target is unreachable under the controller's internal predictor, which causes the controller to clamp at $D_{\min} = 0.08$ for **93.4% of all interaction events**. Clamping is an artifact of the controller's surrogate model underpredicting success by 33 to 40 percentage points, not an inability of the simulator environment to achieve 75% success. The zero direction reversals reflects bound flatlining, not dynamic tracking.

2. **Target Reachability Edge Cases & Diagnostic Semantics**:
   - In `Grow21ModelBasedAdaptiveController.select_next_task`, three explicit cases are enforced:
     - *Unreachable-High Target* ($p^* > \hat{p}(D_{\min})$): clamps to $D_{\min}$ as the most achievable difficulty.
     - *Unreachable-Low Target* ($p^* < \hat{p}(D_{\max})$): clamps to $D_{\max}$ as the most achievable difficulty.
     - *Reachable Target* ($\hat{p}(D_{\max}) \le p^* \le \hat{p}(D_{\min})$): solves for $D^*$ numerically and respects maximum step bounds ($|\Delta D| \le 0.08$).
   - **Reachability Applicability**: Fixed and Streak policies do not evaluate target reachability; their reachability is strictly **N/A** (not applicable; stored as `NaN` / empty in CSV, `null` in JSON). For Target Adaptive, only 3 out of 3,600 trials (0.08%) were mathematically reachable without safety intervention.
   - **Decoupled Safety Intervention**: Safety cooldown drops (78 events, 2.17% for Target; 97 events, 2.69% for Streak; 290 events, 8.06% for Fixed) are tracked in a dedicated `safety_intervention` field and are no longer conflated with mathematical reachability.

3. **Mastery Accounting & Primary Practice Distribution**:
   - **Per-Learner Means ($N=60$ learners)**:
     - Fixed: All skills $-0.0319 \pm 0.0590$, Primary skills $-0.0451 \pm 0.0682$, Secondary-only skills $+0.0341 \pm 0.1980$.
     - Streak: All skills $-0.0083 \pm 0.0562$, Primary skills $-0.0221 \pm 0.0656$, Secondary-only skills $+0.0605 \pm 0.1910$.
     - Target: All skills $-0.0092 \pm 0.0562$, Primary skills $-0.0234 \pm 0.0652$, Secondary-only skills $+0.0618 \pm 0.1911$.
   - **Pooled Skill Rows ($N=360$ rows)**: Group means are identical ($-0.04512$, $-0.02209$, $-0.02338$ for primary; $+0.03410$, $+0.06048$, $+0.06179$ for secondary-only; $-0.03192$, $-0.00833$, $-0.00918$ for all skills), with standard deviations reflecting cross-skill variance ($\approx 0.16 - 0.20$).
   - **Trial Distribution**: The 10.0 trials per skill figure represents the across-row mean ($60 \text{ trials} / 6 \text{ skills} = 10.0$). The empirical primary trial counts per learner-skill are: 0 trials (60 rows, `visual_discrimination`, which received 32 transfer events), 6 trials (60 rows), 7 trials (60 rows), 13 trials (120 rows), and 21 trials (60 rows). Zero skills were entirely unpracticed/untransferred.
   - **Secondary Skills**: Secondary skills receive transfer learning ($0.30 \eta (0.95 - \theta)$); only non-involved skills purely decay.

4. **Oscillation Metrics & Session Breaks**:
   - *Immediate Reversals* ($(D_t - D_{t-1})(D_{t-1} - D_{t-2}) < 0$): 4.733 for Fixed, 5.467 for Streak, 0.000 for Target.
   - *Filtered Reversals* (consecutive nonzero changes): 8.583 for Fixed, 17.933 for Streak, 0.000 for Target.
   - *Total Difficulty Variation* ($\sum |\Delta D|$): 2.392 for Fixed, 2.032 for Streak, 0.290 for Target.
   - *Session Breaks*: Exactly **3 breaks = 36.0 hours** occur in a 60-trial run with 15-trial limit (after trials 15, 30, 45).


## 7. Known Limitations

1. **Synthetic Nature**: Virtual agents follow mathematical IRT and exponential forgetting equations; they do not capture the full neurodiversity, emotional factors, or multi-modal motor intricacies of human children.
2. **Linear Baseline Strength**: Feature engineering captured the primary linear and recency effects; further scaling DKT to larger sequences (e.g. 500+ trials per learner) may reveal regimes where deep sequence learning scales better.
3. **No App Integration**: As specified in the project contract, these scripts run independently and are not connected to any React/React Native frontend.
