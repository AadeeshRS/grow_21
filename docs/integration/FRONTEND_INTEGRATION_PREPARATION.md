# Frontend Integration Preparation & Architectural Boundary

**Document Type:** Provisional Architectural Note  
**Audience:** Grow21 Simulation Developers, Frontend Engineers, and Game Integration Teammates  
**Status:** **Superseded by the implemented integration**: see [`FRONTEND_BACKEND_API.md`](FRONTEND_BACKEND_API.md). This note is kept as the historical pre-integration boundary analysis.  

---

> [!WARNING]
> **PROVISIONAL STATUS & SCOPE LIMITATION**  
> This document specifies the **provisional architectural boundary and contract expectations** between the Grow21 simulation/adaptive control package and a future game frontend.  
> - **DO NOT implement a premature web server, REST API, or WebSocket daemon in this repository.**
> - **DO NOT modify controller semantics or Item Response Theory parameters to fit uninspected frontend schemas.**
> - The actual mobile/web frontend and Unity/React Native minigames belong to a teammate and are developed in a separate repository. Final field names and transmission protocols must be resolved jointly after reviewing the frontend codebase.

---

## 1. Architectural System Boundary

The proposed integration boundary divides responsibilities into four cleanly decoupled layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. GAME FRONTEND LAYER (Teammate's UI / Minigames)                     │
│    • Renders interactive tasks (Sound Safari, Visual Match, etc.)      │
│    • Collects multi-modal user interactions (touch coords, audio, gaze)│
│    • Dispatches raw telemetry events upon trial completion             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Raw Interaction Telemetry
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. ADAPTER & TELEMETRY SERVICE LAYER (Integration Bridge)              │
│    • Validates JSON schemas and sanitizes inputs                       │
│    • Computes sensor reliability vector via Grow21 Product Rule        │
│    • Serializes interaction history into LearnerInteractionEvent       │
│    • Manages session breaks, timeouts, and offline persistence         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Sanitized Observable Event (y_obs, ρ̂_t, Δt)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. LEARNER STATE ESTIMATOR (grow21_simulation_benchmark)               │
│    • Ingests initial questionnaire prior via SkillPriorEstimate        │
│    • Tracks latent mastery vector θ̂_t ∈ [0.05, 0.95]^K                │
│    • Updates belief state via Approx Bayesian Filter / Transition      │
│    • Sequestered: Never accesses hidden simulator truth                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Current Mastery Belief Vector θ̂_t
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. DIFFICULTY CONTROLLER (grow21_simulation_benchmark)                 │
│    • Canonical IRT Bisection Solver: Finds D* for target p* = 0.75     │
│    • Safety Filters: Rate limiting (|ΔD| ≤ 0.08), deadband (0.02)      │
│    • Frustration Cooldown Drops: -0.15 on 3 consecutive failures       │
│    • Returns next TaskSpecification (game_id, difficulty D_{t+1})      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Established In-Memory Data Contracts

The Python package `grow21_simulation_benchmark` already provides clean, tested data structures ready for adaptation:

### 2.1 Task Specification (`tasks.py`)
```python
class TaskSpecification:
    game_id: str           # Unique identifier of the game (e.g., 'bubble_pop', 'sound_safari')
    primary_skill: str     # Main cognitive skill targeted (e.g., 'phoneme_identification')
    secondary_skills: list # Coupled transfer skills (e.g., ['motor_coordination'])
    difficulty: float      # Continuous difficulty parameter D ∈ [0.08, 0.92]
```

### 2.2 Observable Interaction Event (`closed_loop.py`)
```python
class LearnerInteractionEvent:
    learner_id: str        # Anonymous identifier for the learner
    game_id: str           # Game played
    difficulty: float      # Task difficulty assigned
    observed_response: int # Binary response: 1 (success) or 0 (failure)
    reliability_hat: float # Observable reliability scalar ρ̂ ∈ [0, 1]
    elapsed_hours: float   # Inter-trial / inter-session elapsed wall-clock hours
```

### 2.3 Initial Skill Prior Contract (`initializers.py`)
```python
class SkillPriorEstimate:
    prior_mean: float      # Bounded initial estimate θ_0 ∈ [0.05, 0.95]
    uncertainty: float     # Standard error of report (metadata-only in Phase 5)
    source: str            # Provenance tag ('fixed_default', 'questionnaire')
    metadata: dict         # Extensible dictionary for question IDs and timestamps
```

---

## 3. Critical Semantic Safeguard: Observed vs. Clean Predictions

> [!CAUTION]
> **STRICT ARCHITECTURAL REQUIREMENT**  
> **Offline predictive sequence models (Logistic Regression Baseline, Vanilla DKT, Reliability-Gated DKT) output observed-outcome predictions $P(y_{\text{obs}}=1 \mid \text{features})$. They MUST NOT be routed directly into closed-loop controllers targeting clean task success $P(y_{\text{clean}}=1)$.**

### Why Direct Routing Causes Failure
1. Under physical motor slip and sensor noise, a child often understands a concept ($y_{\text{clean}} = 1$) but the sensor registers a miss ($y_{\text{obs}} = 0$).
2. Models trained on observed outcomes predict low success probabilities when sensor noise is high.
3. If an observed-success model drives difficulty selection, it forces task difficulty down to the absolute minimum ($D_{\min} = 0.08$), falsely believing the child is incompetent.
4. **Enforced Safeguard:** Instantiating `ModelAlignedTargetAdaptiveController` with any predictor other than the canonical IRT clean-outcome model raises an immediate runtime `ValueError`. Any future connection between sequence models and difficulty controllers must include an explicit, validated observation-noise inversion layer.

---

## 4. Open Questions to Resolve with Frontend Teammate

Before finalizing any integration code or API contract, the following technical details must be reviewed against the teammate's frontend codebase:

1. **Telemetry Schema & Units**:
   - What timestamp resolution does the frontend emit (UNIX epoch milliseconds vs ISO-8601 strings)?
   - How does the frontend represent sensor confidence? Are raw modality probabilities ($p_{\text{touch}}, p_{\text{audio}}, p_{\text{gaze}}$) emitted, or pre-computed composite confidence?
2. **Break & Fatigue UX**:
   - The simulation controller enforces a mandatory session gap after 15 consecutive trials. How does the frontend UX present this (e.g. celebration screen, badge reward, or locked pause)?
3. **Caregiver Questionnaire Mapping**:
   - What specific question items exist in the caregiver onboarding flow?
   - How are Likert-scale responses (e.g. 1–5 scale) mathematically calibrated to bounded mastery priors $\in [0.05, 0.95]$?
   - How will missing questions be represented (null, omitted, or skipped)? *(Note: Simulation safeguards require falling back to 0.45 default, never zero ability).*
4. **Network & Offline Synchronization**:
   - Does gameplay require real-time difficulty calculation on a local device (e.g. lightweight Python/ONNX runtime on device), or does it operate via a remote cloud API?
   - If offline, how does the frontend queue and synchronize batched trial events after internet restoration?
5. **Target Clean Success Tuning**:
   - The current research baseline targets $p^* = 0.75$ clean success. Does game design require distinct targets per game family (e.g. $p^* = 0.85$ for confidence-building intro games, $p^* = 0.65$ for exploratory puzzle games)?
