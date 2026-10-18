# Grow21 Synthetic Learner Simulator & Knowledge-Tracing Benchmark

> [!CAUTION]
> **MANDATORY CLINICAL & RESEARCH DISCLAIMER**  
> This software repository is a **computational synthetic simulation and benchmarking environment** designed to evaluate adaptive difficulty algorithms, state estimators, and knowledge-tracing models under controlled latent dynamics and simulated sensor noise.  
> - It is **NOT** a clinical, cognitive, or developmental validation on children with Trisomy 21 (Down syndrome).  
> - **NO real-world child interaction data, clinical trial observations, or human user studies were collected or used.**  
> - Passing unit and regression tests confirms **deterministic software invariants**, NOT clinical suitability or real-world educational efficacy.  
> - This module is a standalone Python package and is **strictly not integrated into any user-facing frontend or application** at this stage.

---

## 1. Overview & Research Grounding

This benchmark investigates adaptive difficulty control and knowledge tracing (KT) in serious games for specialized cognitive learning:
1. **Generative Latent-State Modeling vs Marginal Sampling**: Following [Pagonis et al. (2024)](https://arxiv.org/abs/2401.16832), independent coin-flip sampling destroys temporal learning dependencies. Our simulator implements a true latent-state agent with heterogeneous mastery, diminishing-returns practice, exponential forgetting over inter-session gaps, cognitive fatigue, secondary-skill transfer, and multi-modal observation noise.
2. **Controlled Evaluation with Virtual Learners**: Following [Zhang & MacLellan (2021)](https://chrismaclellan.com/media/publications/Investigating-Knowledge-Tracing-Models-using-Simulated-Students.pdf), virtual learners enable controlled multi-seed comparisons that expose edge-case failure modes and model mismatches prior to human pilot studies.
3. **Deep Sequence Models vs Competent Historical Baselines**: As demonstrated by [Gervet et al. (2020)](https://jedm.educationaldatamining.org/index.php/JEDM/article/view/451), logistic regression with well-engineered Performance Factors Analysis (PFA) history features often rivals or outperforms deep recurrent models (Vanilla DKT) on interaction sequences.
4. **Sensor-Aware Gating**: Under physical sensor noise (touch slip, audio dropouts, pose loss), the Grow21 Reliability-Gated DKT attenuates hidden state corruption using multi-modal observable sensor confidence.

---

## 2. Models & Adaptive Controllers

### A. Offline Predictive Sequence Models (`src/grow21_simulation_benchmark/models/`)
Evaluated on learner-disjoint held-out test splits under strict causal timing ($y_t$ predicted before state update):
- **Logistic Regression Baseline (`LogisticBaselineModel`)**: L2-regularized linear model with standardized features strictly prior to event $t$: primary skill, task difficulty $D$, session gap $\Delta t$, estimated reliability $\hat{\rho}_t$, and running PFA counts (attempts, successes, failures, recency, streak).
- **Vanilla DKT (`VanillaDKT`)**: Recurrent neural network (`nn.GRUCell`, hidden dim 48, embedding dim 24) running on CPU.
- **Reliability-Gated DKT (`ReliabilityGatedDKT`)**: Incorporates Grow21's reliability gate $\hat{\rho}_t = \prod_{m \in \text{active}} (1 - \hat{p}_{t,m})$. Missing modalities are penalized and **never imputed as zero uncertainty**.

### B. Closed-Loop Difficulty Controllers (`src/grow21_simulation_benchmark/closed_loop.py`)
- **Fixed Difficulty**: Nominal difficulty $D = 0.50$ with safety drops.
- **Streak Adaptive**: Empirical 2-up / 1-down heuristic stepping.
- **Model-Based Target Adaptive (`target_adaptive`)**: Uses a canonical Item Response Theory (IRT) model to numerically solve for target difficulty $D^*$ achieving clean success probability $p^* = 0.75$, equipped with rate limits ($|\Delta D| \le 0.08$), deadband ($0.02$), frustration cooldowns, and session breaks.
- **8 Belief-State Estimators**: From Phase 3 heuristic baselines to Phase 5 transition-aware analytical expectations and 41-point uniform grid approximate Bayesian state filters.

---

## 3. Project Documentation & Navigation

Comprehensive documentation has been organized into dedicated modules:

- [**System Architecture Overview**](docs/architecture/ARCHITECTURE_OVERVIEW.md): Complete component architecture, data flow, and causal timing invariants.
- [**Testing & Validation Methodology**](docs/testing/TESTING_AND_VALIDATION.md): 67-test suite breakdown, verification levels, metric definitions, and clean-checkout procedures.
- [**Phase Validation Matrix**](docs/testing/PHASE_VALIDATION_MATRIX.md): Detailed comparative matrix across Phases 1–5 (research questions, seeds, findings, and limitations).
- [**Frontend Integration Preparation**](docs/integration/FRONTEND_INTEGRATION_PREPARATION.md): Provisional integration boundaries, contracts, and open questions.
- [**Restructuring Move Manifest**](docs/RESTRUCTURING_MOVE_MANIFEST.md): Complete audit log of 236 original repository files, new paths, and SHA-256 hashes.
- [**Project Structure Guide**](PROJECT_STRUCTURE.md): Regenerated repository hierarchy and active vs archive breakdown.
- **Phase Research Reports**:
  - [Phase 2 Technical Audit Report](docs/research/phase_2/PHASE_2_AUDIT_REPORT.md)
  - [Phase 3 Model Alignment Report](docs/research/phase_3/PHASE_3_MODEL_ALIGNMENT_REPORT.md)
  - [Phase 4 State Estimation & Robustness Report](docs/research/phase_4/PHASE_4_STATE_ESTIMATION_REPORT.md)
  - [Phase 5 State Estimation, Robustness, & Model Selector Report](docs/research/phase_5/PHASE_5_STATE_ESTIMATION_AND_SELECTION_REPORT.md)

---

## 4. Setup & Quickstart

### 4.1 Environment Setup
```bash
# Clone or navigate to the repository root
python -m venv .venv
.\.venv\Scripts\activate   # Windows PowerShell
# source .venv/bin/activate # Linux / macOS

pip install -e .
```

### 4.2 Run All Automated Tests
```bash
# Clean-checkout self-contained test suite (66 tests across 10 modules)
python -m pytest -v tests
```
*Current Result:* **66 passed, 1 skipped in ~11s** on clean checkout (or **67 passed in ~34s** when full experimental artifacts are present) on Python 3.13 / pytest 9.1.

> [!NOTE]
> **Artifact Availability & Evidence Archives**: All four historical evidence ZIP archives (`evidence/phase_N/*.zip`) and large multi-hundred-megabyte raw event interaction logs are excluded from public Git tracking via `.gitignore`. The core test suite runs 100% self-contained using deterministic test fixtures in `tests/fixtures/`. Full raw interaction logs can be regenerated at any time using the simulation scripts in Sections 4.3–4.5.



### 4.3 Run Offline Sequence Benchmark
```bash
python -m grow21_simulation_benchmark.run_benchmark \
    --config configs/default.yaml \
    --scenarios configs/scenarios.yaml \
    --scenario all \
    --seeds 42,123,999 \
    --num-learners 500 \
    --events-per-learner 60 \
    --output-dir outputs/experiments/phase_2/benchmark
```

### 4.4 Run Closed-Loop Adaptive Simulation
```bash
python -m grow21_simulation_benchmark.run_closed_loop \
    --config configs/default.yaml \
    --scenarios configs/scenarios.yaml \
    --scenario base_heterogeneous \
    --num-learners 60 \
    --trials-per-learner 60 \
    --seed 42 \
    --output-dir outputs/experiments/phase_2/closed_loop
```

### 4.5 Run Phase 5 Multi-Stage Campaign (Help & Options)
```bash
python -m grow21_simulation_benchmark.run_phase_5_experiments --help
```

---

## 5. Summary of Key Research Findings Across Phases

### Phase 1: Offline Sequence Benchmark (500 Learners, 3 Seeds, N=4,500 Test Events/Seed)
- **Logistic Regression Beat DKT**: Across all 5 scenario families, the competent Logistic Regression baseline achieved the lowest Log Loss (0.5867–0.6452) and highest ROC-AUC (0.6595–0.6769) while training >100× faster.
- **Reliability Gating Benefit**: Reliability-Gated DKT demonstrated statistically significant improvements over Vanilla DKT primarily under severe sensor dropout (35% dropout: paired $\Delta \text{LogLoss} = -0.0054$, 95% CI $[-0.0088, -0.0020]$).

### Phase 2: Closed-Loop Controller Equation Mismatch Audit
- **The Equation Mismatch**: The controller used internal surrogate predictor $\hat{p}(D) = \sigma(\text{logit}(\hat{\theta}) - \gamma D^{\nu_D})$ with active simulation settings **$\gamma = 2.4, \nu_D = 1.0$** (reconciling with earlier design notes mentioning $\gamma = 3.5, \nu_D = 2.0$).
- Under this surrogate, for an average learner ($\hat{\theta} \approx 0.50$), $\hat{p}(D_{\min} = 0.08) \approx 0.5022 \ll 0.75$.
- Because the target $p^* = 0.75$ was unreachable under the controller's surrogate model, the controller **clamped at $D_{\min} = 0.08$ for 93.4% of all trials**. The observed zero direction reversals was an artifact of bound flatlining, not dynamic stability.
- In contrast, the simulator's true generative process computes clean success via Item Response Theory with game parameters $a_g, c_g, l_g$:
  $$p_{\text{clean}} = c_g + (1 - c_g - l_g) \sigma(a_g (\theta_{\text{eff}} - D))$$
  yielding $p_{\text{clean}} \approx 83.58\%$ at $D_{\min}$. Clamping was caused entirely by surrogate underprediction.

### Phase 3: Canonical IRT Model Alignment & Parity
- **Parity Audited**: Factored the canonical 3-parameter IRT model into shared module [`outcome_model.py`](src/grow21_simulation_benchmark/outcome_model.py). Across 7,938 grid points across all 9 games, maximum absolute error between canonical calculations and simulator evaluation is strictly **0.00e+00**.
- **Bound Clamping Resolved**: Lower-bound saturation plummeted from **93.4% to 35.2%**, and feasible target reachability surged from **0.1% to 71.6%**, tracking the $p^* = 0.7500$ clean-success target within 0.10 percentage points.

### Phase 4: State Estimation, Temporal Alignment, & Cluster Bootstrap
- **Pre/Post Temporal Mastery Alignment**: Clarified evaluation timing by splitting metrics into pre-outcome belief estimation (against `latent_mastery_before`) and post-outcome belief update (against `latent_mastery_after`).
- **Paired Seed-Aware Cluster Bootstrap**: Replaced single-row bootstrapping with cluster resampling over $K = 180$ unique `(seed, learner_id)` clusters ($B = 2,000$ replicates), strictly preserving draw multiplicity.
- **Secondary Transfer Tracking**: Tracking transfer on unpracticed secondary skills (`visual_discrimination`) reduced MAE from 0.145 to 0.107 and eliminated underestimation bias.

### Phase 5: Transition Awareness, Priors, & The "Saturation Trap"
- **Noise Robustness**: In `reliability_stress`, the approximate Bayesian filter combined with secondary transfer achieved the lowest mastery MAE ($0.0986$, $-19.4\%$) and probability error ($0.0362$, $-18.8\%$).
- **The Saturation Trap**: Under `distribution_shift`, lower-bound saturation dropped deceptively from 60.5% to 37.5%, but mastery MAE increased by +34.1%. Estimators must **never be selected solely because they reduce minimum-difficulty bound saturation**.
- **Parent-Proxy Sensitivity**: Noisy caregiver priors ($\sigma \ge 0.30$, missingness $\ge 60\%$) degrade tracking accuracy (+5.0% error) compared to the uninformative fixed 0.45 baseline.
- **Critical Semantic Safeguard**: Explicitly prohibits routing offline observed-success sequence models $P(y_{\text{obs}})$ directly into clean-success difficulty controllers $P(y_{\text{clean}})$.

---

## 6. Frontend Integration Status

- **Status:** **NOT COMPLETE / PREPARATION STAGE ONLY**.
- The minigames and user interface belong to a teammate and are developed in a separate repository.
- No frontend code, web server, or production database has been implemented here.
- See [`docs/integration/FRONTEND_INTEGRATION_PREPARATION.md`](docs/integration/FRONTEND_INTEGRATION_PREPARATION.md) for provisional data structures, architectural boundaries, and open schema questions.
