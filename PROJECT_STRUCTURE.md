# Grow21 Benchmark Project Structure

**Audited Status:** Fully Restructured, Validated, and Documented  
**Import Package Namespace:** `grow21_simulation_benchmark` (under `src/grow21_simulation_benchmark/`)  
**Test Suite Status:** 111 tests across 12 modules (110 passing self-contained tests on clean checkout, 1 full-artifact test)  

This document provides the authoritative layout of source code, configuration files, automated test suites, human-facing research documentation, immutable evidence packages, canonical experiment outputs, and archived historical material.

---

## 1. Directory Tree Hierarchy

```
grow21_simulation/
│
├── configs/                                # Experiment and scenario configuration files
│   ├── default.yaml                        # Task registry, game parameterizations, initializers, and controllers
│   └── scenarios.yaml                      # 5 scenario family definitions (base, stress, dynamics, shift)
│
├── src/grow21_simulation_benchmark/        # Source package (grow21_simulation_benchmark namespace)
│   ├── __init__.py                         # Package initialization and public API exports
│   ├── paths.py                            # Centralized canonical paths helper (single source of truth)
│   ├── tasks.py                            # TaskRegistry and game/skill modality definitions
│   ├── outcome_model.py                    # Canonical IRT clean-success model & bisection solver (0.00e+00 error)
│   ├── simulator.py                        # Virtual learner agent, latent mastery, IRT responses, forgetting & transfer
│   ├── reliability.py                      # Sensor uncertainty engine (Grow21 product rule) and observation distortion
│   ├── initializers.py                     # SkillPriorEstimate, MasteryPriorInitializer, and synthetic parent-proxy
│   ├── closed_loop.py                      # Controller policies, 8 belief-state estimators, semantic safeguard
│   ├── service/                            # Live learner service: runtime.py (state owner), schemas.py, app.py (FastAPI)
│   ├── features.py                         # Causal feature extraction for linear baseline (strictly pre-outcome)
│   ├── dataset.py                          # Sequence generation and learner-disjoint splitting
│   ├── generate_data.py                    # Benchmark dataset generation CLI
│   ├── train.py                            # Model training routines with modular --models and controlled seeds
│   ├── evaluate.py                         # Evaluation metrics, calibration tables, and paired bootstrap CIs
│   ├── plots.py                            # Plotting routines for curves and scenario comparisons
│   ├── run_benchmark.py                    # Offline predictive model benchmark runner (--models flag)
│   ├── run_closed_loop.py                  # Paired closed-loop adaptive simulation runner
│   ├── run_phase_3_experiments.py          # Phase 3 full multi-scenario simulation & parity runner
│   ├── run_phase_4_experiments.py          # Phase 4 3-stage belief estimation & robustness runner
│   ├── run_phase_5_experiments.py          # Phase 5 4-stage empirical orchestrator across 4 cohorts / 12 seeds
│   ├── audit_calibration_bins.py           # Calibration export and 4-level reliability sweep CLI
│   └── models/                             # Offline predictive sequence models
│       ├── __init__.py                     # Model exports
│       ├── logistic_baseline.py            # L2-regularized logistic regression baseline with PFA history features
│       ├── dkt.py                          # Vanilla recurrent Deep Knowledge Tracing (GRU)
│       └── reliability_gated_dkt.py        # Grow21 Reliability-Gated DKT (sensor-gated GRU)
│
├── tests/                                  # Complete pytest test suite (111 tests across 12 modules)
│   ├── __init__.py                         # Test package marker
│   ├── fixtures/                           # Deterministic test fixtures for clean checkout
│   │   └── phase_4_event_metrics_fixture.csv # 27,000-event slice for offline belief estimation tests (5.36 MB)
│   ├── test_simulator.py                   # 3 tests: Latent mastery bounds, game params, forgetting dynamics
│   ├── test_reliability.py                 # 3 tests: Product rule, missing modalities, vocalization
│   ├── test_no_leakage.py                  # 3 tests: Strict causal timing and information isolation
│   ├── test_model_shapes.py                # 2 tests: Tensor dimensions and extreme gating conditions
│   ├── test_controller_constraints.py      # 4 tests: Frustration safety, cooldowns, and session break rules
│   ├── test_audit_regressions.py           # 13 targeted regression tests for Phase 2 findings
│   ├── test_outcome_model_alignment.py      # 10 deterministic tests for Phase 3 IRT alignment and parity
│   ├── test_belief_state_estimation.py      # 14 tests: 13 self-contained tests + 1 full-volume test
│   ├── test_phase_5_state_estimation.py     # 11 deterministic tests for Phase 5 transition awareness & safeguards
│   └── test_documentation_integrity.py      # 4 tests: ASCII control chars, tab-free Markdown, and formatting
│

├── docs/                                   # Human-facing technical documentation and research reports
│   ├── architecture/
│   │   └── ARCHITECTURE_OVERVIEW.md        # Complete system architecture, component breakdown, and data flow
│   ├── testing/
│   │   ├── TESTING_AND_VALIDATION.md       # Full suite commands, module invariants, and metric definitions
│   │   └── PHASE_VALIDATION_MATRIX.md      # Matrix of Phases 1-5 (research questions, seeds, findings, limits)
│   ├── integration/
│   │   └── FRONTEND_INTEGRATION_PREPARATION.md # Provisional integration boundaries and semantic safeguards
│   ├── research/
│   │   ├── phase_2/                        # Canonical Phase 2 Audit Package
│   │   │   ├── PHASE_2_AUDIT_REPORT.md     # Authoritative Phase 2 technical audit report
│   │   │   ├── CHANGELOG_PHASE_2.md        # Detailed log of Phase 2 modifications
│   │   │   ├── BUNDLE_INDEX.md             # Evidence bundle manifest and checksums
│   │   │   └── REPRODUCE.md                # Phase 2 reproduction guide
│   │   ├── phase_3/                        # Canonical Phase 3 Audit Package
│   │   │   ├── PHASE_3_MODEL_ALIGNMENT_REPORT.md # Technical report on IRT outcome model parity
│   │   │   ├── CHANGELOG_PHASE_3.md        # Phase 3 changes log
│   │   │   ├── BUNDLE_INDEX.md             # Evidence manifest
│   │   │   └── REPRODUCE.md                # Phase 3 reproduction guide
│   │   ├── phase_4/                        # Canonical Phase 4 Audit Package
│   │   │   ├── PHASE_4_STATE_ESTIMATION_REPORT.md # Audited 10-section research report
│   │   │   ├── PHASE_4_CORRECTION_SUMMARY.md # Executive audit correction summary (bootstrap & timing)
│   │   │   ├── CHANGELOG_PHASE_4.md        # Phase 4 changes log
│   │   │   ├── BUNDLE_INDEX.md             # Evidence manifest
│   │   │   └── REPRODUCE.md                # Phase 4 reproduction guide
│   │   └── phase_5/                        # Canonical Phase 5 Audit Package
│   │       ├── PHASE_5_STATE_ESTIMATION_AND_SELECTION_REPORT.md # Authoritative 10-section Phase 5 report
│   │       ├── CHANGELOG_PHASE_5.md        # Detailed log of Phase 5 audit modifications
│   │       ├── BUNDLE_INDEX.md             # Complete evidence manifest and file inventory
│   │       ├── REPRODUCE.md                # Deterministic reproduction commands (12 seeds)
│   │       └── CONFIG_SNAPSHOT.json        # Pre-execution frozen configuration snapshot
│   └── RESTRUCTURING_MOVE_MANIFEST.md      # Full 236-file restructuring move manifest and SHA-256 audit
│
├── evidence/                               # Local-only evidence ZIP bundles (excluded from Git via .gitignore)
│   ├── phase_2/
│   │   └── Grow21_Phase_2_Audit_Evidence.zip # Local/release archive: 1.25 MB (56 entries, testzip verified)
│   ├── phase_3/
│   │   └── Grow21_Phase_3_Model_Alignment_Evidence.zip # Local/release archive: 5.36 MB (56 entries, testzip verified)
│   ├── phase_4/
│   │   └── Grow21_Phase_4_State_Estimation_Evidence.zip # Local/release archive: 57.31 MB (69 entries, testzip verified)
│   └── phase_5/
│       └── Grow21_Phase_5_Evidence.zip     # Local/release archive: 66.64 MB (161 entries, testzip verified)

│
├── outputs/                                # Generated empirical simulation outputs and historical archives
│   ├── experiments/                        # Canonical active experiment outputs grouped by phase
│   │   ├── phase_2/                        # Phase 2 simulation outputs and audit sweep
│   │   │   ├── benchmark/                  # 45-row per-seed evaluation metrics (N=4,500 test events/seed), summary, plots
│   │   │   ├── closed_loop/                # 10,800 event records (60 learners x 60 trials x 3 policies), summary, plots
│   │   │   ├── calibration_bins.csv        # Calibration table (verified by test_audit_regressions.py)
│   │   │   ├── audit_sweep_learner_mastery_summary.csv # Granular 90-row audit sweep
│   │   │   ├── paired_loss_evidence.csv    # Paired loss records
│   │   │   ├── predictor_vs_simulator_comparison.csv # Discrepancy analysis
│   │   │   └── reliability_sweep_summary.json # 4-level noise sweep metrics
│   │   ├── phase_3/                        # Phase 3 simulation and formula parity outputs
│   │   │   ├── closed_loop/                # 216,000 trial events (5 scenarios x 3 seeds x 4 policies), summaries
│   │   │   ├── diagnostics/                # 7,938 grid parity verification evaluations (0.00e+00 error)
│   │   │   └── plots/                      # Publication figures (calibration, difficulty, reachability)
│   │   ├── phase_4/                        # Phase 4 belief-state estimation outputs
│   │   │   ├── state_estimation/           # 540,000 trial events, calibration bins, scenario summaries
│   │   │   └── plots/                      # Calibration curves, prediction gaps, mastery errors
│   │   └── phase_5/                        # Phase 5 active generated artifacts (1,728,000 interactions across 12 seeds)
│   │       ├── CONFIG_SNAPSHOT.json        # Pre-execution frozen configuration snapshot
│   │       ├── open_loop/                  # Stage 1: Open-loop state estimation replay on fixed traces
│   │       ├── closed_loop/                # Stage 2: Closed-loop policy comparison across 8 estimators
│   │       ├── initializers/               # Stage 3: Mastery prior interface & synthetic proxy sensitivity
│   │       ├── benchmark/                  # Stage 4: Testing model selector benchmarks (individual vs all)
│   │       └── plots/                      # Publication figures (e1-e3)
│   │
│   └── archive/                            # Historical snapshots and pre-correction artifacts
│       ├── phase_4_pre_correction/         # 17 files: Preserved pre-correction Phase 4 outputs & manifest
│       ├── pre_correction_pass/            # 20 files: Preserved snapshots of pre-correction outputs
│       ├── pre_phase_3_source/             # 15 files: Pre-Phase-3 source snapshot with SHA-256 manifest
│       ├── project_cleanup/                # 4 files: Preserved pre-cleanup historical prompts & manifest
│       └── scratch_model_test/             # 6 files: Preserved scratch benchmark test run
│
├── .gitignore                              # Git exclusion rules (caches, venvs, and >25MB event logs)
├── pyproject.toml                          # Build specification (setuptools, pytest pythonpath = ["src"])
├── README.md                               # Project overview, quickstart, and scientific findings
└── requirements.txt                        # Python dependencies
```

---

## 2. Active Results vs. Archived Historical Artifacts

To maintain absolute provenance and avoid ambiguity during audits:

1. **Active Results (`outputs/experiments/`)**:
   - Contains the authoritative, actively validated outputs of Phases 2, 3, 4, and 5.
   - All regression tests and evaluation scripts point exclusively to these canonical directories via `src/grow21_simulation_benchmark/paths.py`.
   - Mirrored duplicate files that previously existed across `outputs/audit/` have been consolidated into `outputs/experiments/` without data loss (saving ~1.5 GB).

2. **Archived Material (`outputs/archive/`)**:
   - Contains immutable snapshots of pre-correction code, intermediate prompts, and early baseline outputs preserved strictly for historical audit trail.
   - Files in `outputs/archive/` are never imported by active source code or modified by active simulation scripts.

3. **Research Reports (`docs/research/`)**:
   - Human-facing documentation (technical reports, changelogs, reproduction guides, and bundle indexes) is separated from numerical experiment outputs.

4. **Evidence Bundles (`evidence/`)**:
   - Standalone ZIP archives representing self-contained historical evidence snapshots for each audited phase (verified with `zipfile.testzip() == None`).
   - **Local-Only & Excluded from Git:** All four evidence archives are excluded from GitHub tracking via `.gitignore`. They are preserved locally for provenance and GitHub Release distribution, ensuring the Git repository remains lean and fast to clone.


---

## 3. Canonical Path Helper (`paths.py`)

All directory paths are centralized in [`src/grow21_simulation_benchmark/paths.py`](src/grow21_simulation_benchmark/paths.py):

| Constant | Relative Path | Role |
| :--- | :--- | :--- |
| `CONFIGS_DIR` | `configs/` | Experiment and scenario YAML configuration |
| `DOCS_DIR` | `docs/` | Architecture, testing, research, and integration documentation |
| `EVIDENCE_DIR` | `evidence/` | Immutable evidence ZIP archives |
| `OUTPUTS_DIR` | `outputs/` | Root generated output directory |
| `EXPERIMENTS_DIR` | `outputs/experiments/` | Active canonical experiment outputs |
| `ARCHIVE_OUTPUTS_DIR` | `outputs/archive/` | Historical pre-correction and cleanup snapshots |
| `PHASE_2_OUTPUTS_DIR` | `outputs/experiments/phase_2/` | Phase 2 benchmark, closed-loop, and audit sweep |
| `PHASE_3_OUTPUTS_DIR` | `outputs/experiments/phase_3/` | Phase 3 closed-loop, diagnostics, and plots |
| `PHASE_4_OUTPUTS_DIR` | `outputs/experiments/phase_4/` | Phase 4 state estimation and plots |
| `PHASE_5_OUTPUTS_DIR` | `outputs/experiments/phase_5/` | Phase 5 open-loop, closed-loop, initializers, benchmark, plots |
