# Grow21

**Adaptive Multimodal Learning Platform for Children with Trisomy 21**

---

## 1. Overview & Purpose

Grow21 is an adaptive, multimodal learning and cognitive-motor support system designed specifically for children with Trisomy 21 (Down syndrome). Children with Trisomy 21 exhibit distinct learning trajectories characterized by varied motor response latencies, cognitive processing profiles, session fatigue, and multimodal input variability. 

Grow21's core mission is to provide an engaging, supportive digital learning environment that dynamically tailors task difficulty to each child's current ability level, maintaining optimal challenge and preventing frustration without clinical over-fatigue.

---

## 2. Architecture & Component Separation

The Grow21 system consists of two independently developed core components:

```
Grow21 Platform
├── React Native Mobile Application (In Separate Development)
│   ├── Touch & gesture interaction engine
│   ├── Vocalization and auditory response processing
│   └── 9 gamified cognitive-motor mini-games
│
└── Python Learner Model & Simulation Benchmark (Branch: model-testing)
    ├── Latent cognitive mastery state estimation
    ├── Sensor uncertainty and reliability gating
    ├── Item Response Theory (IRT) outcome formulation
    └── Closed-loop difficulty adaptation controllers
```

1. **React Native Mobile Application** (Client):
   - The user-facing mobile client developed for tablet and touch-enabled devices.
   - Captures multimodal responses (touch accuracy, reaction times, voice tokens).
   - Designed for intuitive, accessible interaction under pediatric motor constraints.
   - *Note: The mobile application frontend is developed independently and will integrate with the learner model via dedicated API endpoints.*

2. **Python Learner-Model Benchmark** (Backend / Research Engine):
   - The algorithmic modeling and simulation core housed in the **[`model-testing`](https://github.com/AadeeshRS/grow_21/tree/model-testing)** branch.
   - Evaluates adaptive task difficulty scheduling, belief-state estimation (Approximate Bayesian filtering and Deep Knowledge Tracing), and sensor reliability engines under severe observational noise and missingness.

---

## 3. Scientific Status & Disclaimers

> [!IMPORTANT]
> **Experimental Research Status**: The Grow21 learner-state estimation models, item-response dynamics, and difficulty controllers are currently **experimental** and evaluated strictly on **computational synthetic learner simulations**.

- **No Clinical Claims**: The simulation parameters and virtual learner cohorts are inspired by pediatric motor and cognitive research literature, but they do **not** constitute clinical trials, developmental diagnostic tools, or therapeutic recommendations.
- **No Real-Child Data**: No real human participant or clinical data were collected or utilized in developing these benchmark models. All benchmarks reflect synthetic computational experiments designed to stress-test adaptive control algorithms under noise, session fatigue, and population distribution shifts.

---

## 4. The `model-testing` Branch

The complete Python model implementation, test suite, and research documentation are maintained on the **`model-testing`** branch:

- **Source Code**: Full Python package (`src/grow21_simulation_benchmark/`) implementing the virtual learner simulator, reliability engines, canonical IRT clean-outcome models, and 8 belief-state estimators.
- **Verification Suite**: Complete 67-test deterministic test suite validating causal timing, equation parity, likelihood limits, safety cooldowns, and regression invariants.
- **Empirical Artifacts**: Compact experiment outputs, calibration tables, paired cluster bootstrap evaluations, and publication figures across all five research phases.
- **Research Documentation**: Comprehensive Phase 1–5 research audit reports, methodology guides, and reproduction instructions organized under `docs/research/`.

### Accessing the Model Benchmark
To checkout and run the Python model benchmark:

```bash
# Clone the repository
git clone https://github.com/AadeeshRS/grow_21.git
cd grow_21

# Switch to the model-testing branch
git checkout model-testing

# Create and activate virtual environment
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

# Install package and run core test suite
pip install -e .
python -m pytest tests
```

---

## 5. License & Attribution

This project is developed for academic research and evaluation in adaptive cognitive modeling. Refer to individual component documentation for license terms and dependency attributions.
