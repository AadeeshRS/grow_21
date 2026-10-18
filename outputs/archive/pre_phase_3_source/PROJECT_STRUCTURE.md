# Grow21 Benchmark Project Structure

This document outlines the organization of code, configuration files, tests, generated outputs, audit evidence, and archival snapshots in the consolidated Grow21 simulation benchmark repository.

---

## Directory Hierarchy

```
grow21_simulation/
│
├── configs/                          # Experiment configuration files
│   ├── default.yaml                  # Task registry, game parameterizations, and controllers
│   └── scenarios.yaml                # 5 scenario definitions (base, stress, dynamics, shift)
│
├── src/grow21_simulation_benchmark/  # Source code package (unchanged namespace)
│   ├── __init__.py                   # Package initialization
│   ├── paths.py                      # Canonical directory paths helper (single source of truth)
│   ├── tasks.py                      # TaskRegistry and task/modality definitions
│   ├── simulator.py                  # Virtual learner agent, IRT responses, forgetting & transfer
│   ├── reliability.py                # Sensor uncertainty engine (product rule) and noise
│   ├── features.py                   # Causal feature extraction for Logistic Regression baseline
│   ├── dataset.py                    # Sequence generation and learner-disjoint splitting
│   ├── train.py                      # Model training routines
│   ├── evaluate.py                   # Evaluation metrics, calibration tables, paired bootstrap CIs
│   ├── plots.py                      # Plotting routines for curves and scenario comparisons
│   ├── run_benchmark.py              # Multi-scenario multi-seed offline benchmark runner
│   ├── run_closed_loop.py            # Paired closed-loop adaptive simulation runner
│   ├── audit_calibration_bins.py     # Calibration export and 4-level reliability sweep CLI
│   └── models/                       # Model architectures
│       ├── logistic_baseline.py      # L2-regularized logistic regression baseline
│       ├── dkt.py                    # Vanilla recurrent Deep Knowledge Tracing (GRU)
│       └── reliability_gated_dkt.py  # Grow21 Reliability-Gated DKT
│
├── tests/                            # Complete pytest test suite (32 passing tests)
│   ├── test_simulator.py             # Latent mastery and forgetting dynamics
│   ├── test_reliability.py           # Product rule and sensor dropout mechanics
│   ├── test_no_leakage.py            # Strict causal timing and information isolation
│   ├── test_model_shapes.py          # Tensor dimensions and boundary conditions
│   ├── test_controller_constraints.py# Frustration safety and session break rules
│   ├── test_audit_regressions.py     # 13 targeted regression tests for Phase 2 audit and reconciliation findings
│   └── test_documentation_integrity.py# Documentation control character and formatting integrity
│
├── outputs/                          # Active and archived generated artifacts
│   ├── benchmark/                    # Benchmark metrics and plots
│   │   ├── metrics.csv               # 45-row per-seed evaluation table (500 learners, N=4,500 test events)
│   │   ├── summary.json              # Aggregated metrics (mean ± std) across seeds
│   │   └── plots/                    # Calibration curves and cross-scenario charts
│   │
│   ├── closed_loop/                  # Closed-loop simulation results
│   │   ├── closed_loop_events.csv    # 10,800 event records (60 learners × 60 trials × 3 policies)
│   │   ├── closed_loop_summary.json  # Controller performance and stability summary
│   │   ├── learner_mastery_summary.csv # Granular per-skill initial/final mastery and counts
│   │   └── plots/
│   │       ├── closed_loop_comparison.png # Practiced skill mastery gains and difficulty dynamics
│   │       └── difficulty_trajectories.png # Trial difficulty paths showing bound saturation
│   │
│   ├── audit/
│   │   └── phase_2/                  # Canonical Phase 2 Audit Package
│   │       ├── PHASE_2_AUDIT_REPORT.md # Authoritative technical audit report
│   │       ├── BUNDLE_INDEX.md       # Comprehensive evidence manifest and checksums
│   │       ├── REPRODUCE.md          # Step-by-step reproduction guide
│   │       ├── CHANGELOG_PHASE_2.md  # Detailed log of all audit fixes and changes
│   │       ├── calibration_bins.csv  # 10-bin calibration table for canonical 500-learner cohort
│   │       ├── reliability_sweep_summary.json # 4-level controlled sweep summary with bootstrap CIs
│   │       ├── learner_mastery_summary.csv # Mirrored granular mastery evidence
│   │       ├── paired_loss_evidence.csv # Learner-level paired losses for bootstrap verification
│   │       ├── predictor_vs_simulator_comparison.csv # Controller-predictor vs. simulator comparison
│   │       └── Grow21_Phase_2_Audit_Evidence.zip # Self-contained validated evidence bundle
│   │
│   └── archive/                      # Historical and cleanup archival storage
│       ├── pre_correction_pass/      # Preserved snapshots of pre-correction outputs
│       │   ├── manifest.json         # SHA-256 manifest of archived pre-correction outputs
│       │   ├── metrics.csv
│       │   ├── summary.json
│       │   ├── closed_loop_events.csv
│       │   ├── closed_loop_summary.json
│       │   └── audit/
│       │
│       └── project_cleanup/          # Preserved historical prompts and pre-cleanup artifacts
│           ├── pre_cleanup_manifest.json # Complete pre-cleanup SHA-256 manifest of all 84 files
│           ├── obsolete_prompts/     # Historical instruction prompts moved from active root
│           │   ├── GROW21_BENCHMARK_AUDIT_PHASE_2_PROMPT.md
│           │   └── GROW21_SIMULATOR_BENCHMARK_ANTIGRAVITY_PROMPT.md
│           └── original_input_bundle/# Original input archive
│               └── grow21_simulation_benchmark_artifacts.zip
│
├── README.md                         # Project overview, quickstart, and key findings
├── PROJECT_STRUCTURE.md              # This directory and file reference
├── pyproject.toml                    # Package build configuration
└── requirements.txt                  # Python dependency specifications
```

---

## Output Path Canonical Source of Truth

All output paths are programmatically managed by `src/grow21_simulation_benchmark/paths.py`:
- `BENCHMARK_OUTPUTS_DIR`: `outputs/benchmark/`
- `CLOSED_LOOP_OUTPUTS_DIR`: `outputs/closed_loop/`
- `AUDIT_OUTPUTS_DIR`: `outputs/audit/phase_2/`
- `ARCHIVE_OUTPUTS_DIR`: `outputs/archive/pre_correction_pass/`
