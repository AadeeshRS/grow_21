# Grow21 System Architecture Overview

**Status:** Audited & Consolidated Architecture  
**Import Package Namespace:** `grow21_simulation_benchmark` (under `src/grow21_simulation_benchmark/`)  
**Execution Context:** Standalone Python research simulation and knowledge-tracing benchmark  

---

> [!IMPORTANT]
> **SYNTHETIC-ONLY SIMULATION DISCLAIMER**  
> This software system is a **purely computational synthetic simulation benchmark** designed to evaluate adaptive difficulty algorithms and knowledge-tracing models under controlled latent dynamics and simulated sensor noise. **No real-child data, clinical trials, or developmental validations on human subjects (with or without Down syndrome / Trisomy 21) have been conducted or used.** All learner trajectories, sensor dropouts, and caregiver questionnaire responses are generated computationally.

---

## 1. System High-Level Architecture

The Grow21 simulation benchmark comprises five tightly decoupled sub-systems operating under strict causal timing boundaries:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SIMULATION ENVIRONMENT                          │
│                                                                        │
│   VirtualLearner (simulator.py)                                        │
│   ├── Latent Mastery Dynamics (diminishing-returns practice, η_k)       │
│   ├── Secondary Skill Transfer (30% coupling, 0.95 ceiling)            │
│   ├── Inter-Session Forgetting (exponential decay, λ_k)                │
│   ├── Cognitive Fatigue Accumulation & Recovery                        │
│   └── Canonical Clean Outcome Model (outcome_model.py, IRT)            │
│                 │                                                      │
│                 ▼ Clean Outcome (y_clean ∈ {0, 1}) [HIDDEN]            │
│   Reliability & Noise Engine (reliability.py)                          │
│   ├── Multi-Modal Sensor Uncertainty (Touch, Audio, Pose)              │
│   ├── Grow21 Product Rule: ρ̂ = Π (1 - p_m)                             │
│   └── Observation Distortion: y_obs = Flip(y_clean, ρ)                 │
└─────────────────┬──────────────────────────────────────────────────────┘
                  │ OBSERVABLE CHANNEL ONLY (y_obs, ρ̂_t, Task Metadata, Δt)
                  │ [STRICT NO-LEAKAGE BOUNDARY]
                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  LEARNER BELIEF STATE & ADAPTATION                     │
│                                                                        │
│   Prior Initialization (initializers.py)                               │
│   ├── SkillPriorEstimate (prior_mean, uncertainty, source)             │
│   └── MasteryPriorInitializer (fixed_default=0.45, provided_priors)    │
│                 │                                                      │
│                 ▼ Prior Belief Vector θ̂_0                              │
│   Belief-State Estimators (closed_loop.py)                             │
│   ├── Heuristic Baselines (Phase 3 Model-Aligned)                      │
│   ├── Reliability-Aware Soft-Label Estimators (Phase 4)                │
│   ├── Expected-Transition Dynamical Estimators (Phase 5)               │
│   └── Approx Bayesian State Filters (41-point uniform grid)            │
│                 │                                                      │
│                 ▼ Current Estimated Belief θ̂_t                         │
│   Closed-Loop Adaptive Controllers (closed_loop.py)                    │
│   ├── Fixed Difficulty Policy (nominal D=0.50 + safety)                │
│   ├── Streak Adaptive Policy (2-up / 1-down heuristic)                 │
│   └── Model-Based Target Adaptive Controller                           │
│       ├── Canonical IRT Solver: Bisection for p* = 0.75                │
│       ├── Rate Limiting (|ΔD| ≤ 0.08) & Deadband (0.02)                │
│       └── Frustration Safety Cooldowns & Session Breaks                │
└────────────────────────────────────────────────────────────────────────┘
                  ▲
                  │ CRITICAL SEMANTIC SAFEGUARD
                  │ (P(y_obs) ≠ P(y_clean); cannot route directly)
┌─────────────────┴──────────────────────────────────────────────────────┐
│                    OFFLINE PREDICTIVE BENCHMARK                        │
│                                                                        │
│   Predictive Sequence Models (models/)                                 │
│   ├── Logistic Regression Baseline (PFA history features)             │
│   ├── Vanilla Deep Knowledge Tracing (DKT, recurrent GRU)              │
│   └── Reliability-Gated DKT (Grow21 sensor-gated GRU)                  │
│                                                                        │
│   Target Semantics: P(y_obs = 1 | features)                            │
│   Used strictly for sequence modeling benchmarks, NOT direct control.  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Modules & Responsibilities

### 2.1 Virtual Learner Simulator (`src/grow21_simulation_benchmark/simulator.py`)
Implements the generative latent-state ground truth for simulated children:
- **Latent Mastery Vector $\boldsymbol{\theta} \in [0.05, 0.95]^K$**: Each learner maintains latent ability across $K=6$ cognitive skills (`phoneme_identification`, `auditory_discrimination`, `visual_discrimination`, `pattern_matching`, `number_sense`, `motor_coordination`).
- **Diminishing-Returns Practice Increment**:
  $$\Delta \theta_{\text{primary}} = \eta_k \cdot (0.95 - \theta_{\text{primary}}) \cdot \Phi_{\text{feedback}} \cdot \Psi_{\text{difficulty}} + \epsilon_{\text{process}}$$
  where feedback factor $\Phi_{\text{feedback}} = 0.70$ on clean success ($y_{\text{clean}}=1$) and $0.30$ on clean failure ($y_{\text{clean}}=0$); difficulty factor $\Psi = 0.60 + 0.40 D_t$.
- **Secondary Skill Transfer**:
  $$\Delta \theta_{\text{secondary}} = 0.30 \cdot \eta_{\text{secondary}} \cdot (0.95 - \theta_{\text{secondary}})$$
  strictly capped at the mastery ceiling of $0.95$.
- **Inter-Session Exponential Forgetting**:
  $$\theta_k(t + \Delta t) = 0.05 + (\theta_k(t) - 0.05) \cdot \exp(-\lambda_k \cdot \Delta t)$$
  decaying toward the minimum skill floor $0.05$ over elapsed gap hours $\Delta t$.
- **Cognitive Fatigue**: Tracks continuous trial engagement, elevating lapse probabilities when sessions exceed 15 uninterrupted trials.

### 2.2 Canonical Clean Outcome Model (`src/grow21_simulation_benchmark/outcome_model.py`)
The unified mathematical specification for clean task success across all 9 games:
- **3-Parameter Item Response Theory (IRT)**:
  $$p_{\text{clean}}(\theta_{\text{eff}}, D) = c_g + (1 - c_g - l_g) \cdot \sigma\left(a_g \cdot (\theta_{\text{eff}} - D)\right)$$
  where:
  - $a_g \in [3.0, 3.5]$ is game discrimination.
  - $c_g \in [0.05, 0.10]$ is guessing floor.
  - $l_g \in [0.05, 0.08]$ is lapse ceiling (elevated by fatigue).
  - $\theta_{\text{eff}} = \theta_{\text{primary}} + 0.30 \sum_{s \in \text{secondary}} \theta_s$ represents composite skill capability.
- **Root Solver**: Inverts the IRT equation via numerical bisection to find the target difficulty $D^*$ achieving clean success probability $p^* = 0.75$.
- **Parity Verification**: Verified to achieve strictly **0.00e+00** error against the simulator generative equation across 7,938 grid evaluations in Phase 3.

### 2.3 Sensor Uncertainty & Reliability Engine (`src/grow21_simulation_benchmark/reliability.py`)
Simulates multi-modal physical sensor noise during child gameplay:
- **Active Modalities**: Touch accuracy, vocalization confidence, head pose/gaze tracking.
- **Grow21 Product Rule**:
  $$\hat{\rho}_t = \prod_{m \in \text{active}} (1 - \hat{p}_{t, m})$$
  where $\hat{p}_{t, m} \in [0, 1]$ is the estimated failure probability of modality $m$.
- **Missing Modality Safeguard**: When a modality drops out or disconnects, the system incurs an explicit uncertainty penalty and **never imputes zero uncertainty**.
- **Response Distortion**: Observable response $y_{\text{obs}}$ flips with probability $(1 - \rho_t) \cdot \text{noise\_flip\_rate}$.

### 2.4 Prior Initialization Protocol (`src/grow21_simulation_benchmark/initializers.py`)
Modular contract preparing for future caregiver questionnaire responses:
- **Data Contract `SkillPriorEstimate`**: Bounded prior mean $\in [0.05, 0.95]$, report uncertainty standard error, provenance tag (`"fixed_default"`, `"synthetic_parent_proxy"`), and metadata dictionary.
- **Initializer `MasteryPriorInitializer`**:
  - `fixed_default(0.45)`: Constant baseline prior for uninitialized learners.
  - `from_provided_priors(dict, fallback_default=0.45)`: Ingests external skill dictionaries; omitted skills safely fall back to $0.45$ and are never penalized as zero ability.
- **Synthetic Parent-Proxy Generator**: Simulates questionnaire reports under controlled noise $\sigma \in \{0.05, 0.15, 0.30\}$ and omission rates $\in \{0\%, 30\%, 60\%\}$.

### 2.5 Closed-Loop Controllers & Estimators (`src/grow21_simulation_benchmark/closed_loop.py`)
Coordinates online belief-state tracking and difficulty adaptation:
- **8 Belief-State Estimators**:
  1. `model_aligned_baseline`: Phase 3 heuristic updater (+0.08 on success, -0.08 on failure).
  2. `transfer_aware`: Adds secondary skill transfer to heuristic baseline.
  3. `reliability_aware`: Soft-label Bayesian clean posterior update ($q_t$).
  4. `combined_transfer_reliability`: Phase 4 dual soft-label + secondary transfer.
  5. `expected_transition`: Analytical expected transition under population priors ($\bar{\eta}=0.08$).
  6. `expected_transition_transfer`: Analytical expected transition + secondary transfer.
  7. `approx_bayesian_filter`: 41-point uniform grid discrete filter separating observation likelihood from dynamical state propagation.
  8. `approx_bayesian_filter_transfer`: Approximate Bayesian filter + secondary transfer.
- **Controllers**:
  - `FixedDifficultyController`: Constant nominal difficulty $D=0.50$ with safety drops.
  - `StreakAdaptiveController`: Heuristic 2-up / 1-down difficulty adjustment.
  - `ModelAlignedTargetAdaptiveController`: Bisection solver targeting $p^* = 0.75$, with step limits ($|\Delta D| \le 0.08$), deadband ($0.02$), frustration cooldowns, and session breaks.
- **Critical Semantic Safeguard**: Explicit runtime exception preventing offline models $P(y_{\text{obs}})$ from directly driving clean-success controllers $P(y_{\text{clean}})$.

### 2.6 Offline Sequence Predictive Models (`src/grow21_simulation_benchmark/models/`)
Sequence models evaluated on held-out interaction sequences:
- **Logistic Regression Baseline (`logistic_baseline.py`)**: L2-regularized linear model with rich Performance Factors Analysis (PFA) history features (attempts, successes, failures, recency, streak, session gaps).
- **Vanilla DKT (`dkt.py`)**: Recurrent neural network (`nn.GRUCell`, hidden dim 48) predicting $y_t$ strictly before consuming $y_t$ into state $h_t$.
- **Reliability-Gated DKT (`reliability_gated_dkt.py`)**: State update attenuated by observable reliability: $h_t = h_{t-1} + \hat{\rho}_t \cdot (\bar{h}_t - h_{t-1})$.

### 2.7 Directory & Path Management (`src/grow21_simulation_benchmark/paths.py`)
Centralized single source of truth for all project paths, organizing outputs cleanly by phase and isolating research documentation into `docs/` and evidence archives into `evidence/`.

---

## 3. Strict Causal Timing & Information Flow

At every interaction step $t$, operations execute in an invariant causal order:

```
Step 1: Inter-Session Elapsed Time Applied
        θ_latent(t) = Forgetting(θ_latent(t-1), Δt)
        θ̂_belief(t) = Forgetting(θ̂_belief(t-1), Δt)

Step 2: Task Selection & Difficulty Decision
        D_t = Controller.select_next_task(θ̂_belief(t))
        [Difficulty D_t is finalized BEFORE trial starts]

Step 3: Pre-Outcome Prediction Formed
        ŷ_pred(t) = Predictor.predict(θ̂_belief(t), D_t)
        [Prediction is logged BEFORE outcome occurs]

Step 4: True Latent Trial Execution in Simulator
        y_clean(t) ~ IRT(θ_latent(t), D_t)       [GROUND TRUTH, HIDDEN]
        ρ̂_t ~ ReliabilityModel(ActiveSensors)     [OBSERVABLE]
        y_obs(t) = Corrupt(y_clean(t), ρ̂_t)       [OBSERVABLE]

Step 5: Simulator Latent Mastery Transition
        θ_latent(t+1) = SimulatorTransition(θ_latent(t), y_clean(t), D_t)

Step 6: Belief State Update (Observable Data Only)
        θ̂_belief(t+1) = Estimator.update(θ̂_belief(t), D_t, y_obs(t), ρ̂_t)
        [Estimator NEVER observes θ_latent, y_clean, or true noise]
```

Under this timing contract:
1. Pre-outcome predictions $\hat{y}_t$ cannot leak information from event $t$ outcome.
2. Belief estimates $\hat{\theta}_t$ at trial selection time reflect strictly past history $0 \dots t-1$.
3. All evaluated algorithms interact strictly through observable channels ($y_{\text{obs}}$, $\hat{\rho}_t$, $D_t$, $\Delta t$).
