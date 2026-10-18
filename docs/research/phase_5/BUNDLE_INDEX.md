# Grow21 Phase 5 Evidence Bundle Index

**Bundle File:** `evidence/phase_5/Grow21_Phase_5_Evidence.zip` *(local/release archive, excluded from Git)*  
**Phase:** Phase 5 — Transition-Aware Learner State Estimation, Robustness, & Testing Model Selector (Audited)  
**Date:** October 2026  
**Status:** Self-Contained, Verified, Passing All Core Tests on Clean Checkout  

> [!NOTE]
> Under the project evidence exclusion policy, `Grow21_Phase_5_Evidence.zip` is excluded from Git tracking via `.gitignore`. All canonical compact outputs, model selector metrics, calibration tables, and figures are tracked uncompressed under `outputs/experiments/phase_5/`, ensuring fresh GitHub clones have immediate access to complete research findings without downloading large archives.

---

## 1. Bundle Manifest and File Inventory

The evidence ZIP archive contains the complete, self-contained Grow21 simulation and benchmarking project originally packaged for Phase 5:

| Relative Path in Archive | Category | Description |
| :--- | :--- | :--- |
| `pyproject.toml` | Build | PEP 518/621 build configuration and package metadata. |
| `requirements.txt` | Build | Exact pinned Python dependencies. |
| `README.md` | Documentation | Repository overview, quickstart guide, and core equations. |
| `PROJECT_STRUCTURE.md` | Documentation | Comprehensive repository tree and directory schema. |
| `configs/default.yaml` | Config | Canonical 9-game task definitions, initializers, model selector, and population parameters. |
| `configs/scenarios.yaml` | Config | Stress scenario definitions (noise, dropout, drift, temporal). |
| `src/grow21_simulation_benchmark/__init__.py` | Source | Package initialization and public API exports. |
| `src/grow21_simulation_benchmark/initializers.py` | Source | `SkillPriorEstimate` (`prior_mean`, `uncertainty` metadata), `MasteryPriorInitializer`, and synthetic parent-proxy generator. |
| `src/grow21_simulation_benchmark/closed_loop.py` | Source | 8 belief-state estimators, open-loop replay, and semantic model safeguard. |
| `src/grow21_simulation_benchmark/outcome_model.py` | Source | Canonical clean-success outcome model and bisection difficulty solver. |
| `src/grow21_simulation_benchmark/simulator.py` | Source | Virtual learner simulator, sensory corruption, and practice updates. |
| `src/grow21_simulation_benchmark/tasks.py` | Source | Task registry, game parameter mappings, and sensor modalities. |
| `src/grow21_simulation_benchmark/reliability.py` | Source | Sensor uncertainty engine, product rule, and missingness handling. |
| `src/grow21_simulation_benchmark/evaluate.py` | Source | Modular `--models` evaluation, calibration tables, and Brier scores. |
| `src/grow21_simulation_benchmark/train.py` | Source | Modular `--models` model training pipeline with controlled per-model seeds and dedicated generators. |
| `src/grow21_simulation_benchmark/dataset.py` | Source | Synthetic event sequence generation. |
| `src/grow21_simulation_benchmark/features.py` | Source | Interaction feature extraction. |
| `src/grow21_simulation_benchmark/models/__init__.py` | Source | Model exports. |
| `src/grow21_simulation_benchmark/models/logistic_baseline.py` | Source | Logistic regression benchmark model. |
| `src/grow21_simulation_benchmark/models/dkt.py` | Source | Vanilla Deep Knowledge Tracing architecture. |
| `src/grow21_simulation_benchmark/models/reliability_gated_dkt.py` | Source | Reliability-Gated Deep Knowledge Tracing architecture. |
| `src/grow21_simulation_benchmark/paths.py` | Source | Canonical directory path resolution including Phase 5 paths. |
| `src/grow21_simulation_benchmark/plots.py` | Source | Visualization routines and style utilities. |
| `src/grow21_simulation_benchmark/audit_calibration_bins.py` | Source | Calibration table export CLI. |
| `src/grow21_simulation_benchmark/generate_data.py` | Source | Benchmark dataset generator CLI. |
| `src/grow21_simulation_benchmark/run_benchmark.py` | Source | Offline sequence model benchmark runner with `--models` flag. |
| `src/grow21_simulation_benchmark/run_closed_loop.py` | Source | Closed-loop policy comparison runner. |
| `src/grow21_simulation_benchmark/run_phase_3_experiments.py` | Source | Phase 3 simulation and equation parity runner. |
| `src/grow21_simulation_benchmark/run_phase_4_experiments.py` | Source | Phase 4 runner with seed-aware cluster bootstrap and split separation. |
| `src/grow21_simulation_benchmark/run_phase_5_experiments.py` | Source | Phase 5 4-stage empirical orchestrator, open-loop replay, and plotting across 12 seeds (4 cohorts). |
| `tests/__init__.py` | Test | Test suite init. |
| `tests/test_phase_5_state_estimation.py` | Test | 11 Phase 5 tests for transition awareness, priors, model selector, per-model seed consistency, and safeguards. |
| `tests/test_belief_state_estimation.py` | Test | 13 Phase 4 tests for belief updating, likelihood limits, and time-alignment. |
| `tests/test_outcome_model_alignment.py` | Test | 10 Phase 3 tests for canonical equation parity and bisection reachability. |
| `tests/test_audit_regressions.py` | Test | 13 Phase 2 historical regression suite checking calibration conservation. |
| `tests/test_no_leakage.py` | Test | 3 invariance tests verifying event $t$ outcome does not leak into prediction $t$. |
| `tests/test_controller_constraints.py` | Test | 4 tests for safety cooldowns, frustration limits, and session breaks. |
| `tests/test_documentation_integrity.py` | Test | 4 tests for zero non-printing control characters and zero literal tabs in docs. |
| `tests/test_simulator.py` | Test | 3 tests for learner generation, fatigue dynamics, and sensor corruption. |
| `tests/test_reliability.py` | Test | 3 tests for sensor uncertainty engine and dropout rules. |
| `tests/test_model_shapes.py` | Test | 2 tests for Deep Knowledge Tracing tensor shapes and masking. |
| `outputs/audit/phase_2/*` | Historical Audit | Phase 2 audit report, changelog, and calibration bins required for regression tests. |
| `outputs/benchmark/metrics.csv` | Historical Data | Phase 2 benchmark evaluation metrics required for regression tests. |
| `outputs/closed_loop/*` | Historical Data | Phase 2 closed-loop events and summary required for regression tests. |
| `outputs/audit/phase_3/*` | Historical Audit | Phase 3 model alignment report and verification outputs. |
| `outputs/phase_3/closed_loop/*` | Historical Data | Phase 3 closed loop metrics and summary JSON. |
| `outputs/audit/phase_4/*` | Historical Audit | Phase 4 audited report, changelog, bundle index, calibration bins, and summary. |
| `outputs/phase_4/plots/*` | Historical Plots | Phase 4 evaluation figures. |
| `outputs/phase_4/state_estimation/*` | Phase 4 State Data | Phase 4 calibration bins, mastery metrics, and event metrics required for regression tests. |
| `outputs/audit/phase_5/CONFIG_SNAPSHOT.json` | Phase 5 Audit | Pre-execution frozen configuration snapshot. |
| `outputs/audit/phase_5/PHASE_5_STATE_ESTIMATION_AND_SELECTION_REPORT.md` | Phase 5 Audit | Comprehensive 10-section Phase 5 research and validation report. |
| `outputs/audit/phase_5/REPRODUCE.md` | Phase 5 Audit | Deterministic reproduction commands, 12-seed cohort accounting, and verification procedures. |
| `outputs/audit/phase_5/CHANGELOG_PHASE_5.md` | Phase 5 Audit | Detailed log of all Phase 5 and Phase 5.1 modifications, equations, and invariant checks. |
| `outputs/audit/phase_5/BUNDLE_INDEX.md` | Phase 5 Audit | This bundle manifest and verification index. |
| `outputs/audit/phase_5/open_loop/open_loop_summary.csv` | Phase 5 Data | Stage 1 open-loop replay summary metrics across scenarios, cohorts, and estimators (480 rows). |
| `outputs/audit/phase_5/open_loop/open_loop_paired_bootstrap.csv` | Phase 5 Data | Stage 1 open-loop paired cluster bootstrap differences and 95% CIs. |
| `outputs/audit/phase_5/closed_loop/phase_5_scenario_summary.csv` | Phase 5 Data | Stage 2 closed-loop metrics across 8 estimators, 5 scenarios, and 4 cohorts (160 rows, 1,728,000 interactions). |
| `outputs/audit/phase_5/closed_loop/phase_5_paired_bootstrap.csv` | Phase 5 Data | Stage 2 closed-loop paired cluster bootstrap differences and 95% CIs. |
| `outputs/audit/phase_5/closed_loop/phase_5_calibration_table.csv` | Phase 5 Data | Stage 2 empirical calibration table by probability decile bin. |
| `outputs/audit/phase_5/initializers/parent_proxy_sensitivity_metrics.csv` | Phase 5 Data | Stage 3 synthetic parent-proxy noise sensitivity metrics across 9 grids. |
| `outputs/audit/phase_5/benchmark/model_selector_benchmark_metrics.csv` | Phase 5 Data | Stage 4 offline model selector benchmark metrics with verified individual vs all-mode identity. |
| `outputs/audit/phase_5/benchmark/selector_runs/*` | Phase 5 Data | Stage 4 sub-model runs, calibration plots, metrics, and summaries. |
| `outputs/audit/phase_5/plots/e1_open_loop_mastery_error_by_scenario.png` | Phase 5 Figure | Stage 1 open-loop mastery MAE and bias across scenarios. |
| `outputs/audit/phase_5/plots/e2_clean_prob_calibration_curves.png` | Phase 5 Figure | Stage 2 closed-loop clean-probability calibration curves. |
| `outputs/audit/phase_5/plots/e2_closed_loop_difficulty_and_saturation.png` | Phase 5 Figure | Stage 2 difficulty trajectory and min-bound saturation comparison. |
| `outputs/audit/phase_5/plots/e2_closed_loop_mastery_error_comparison.png` | Phase 5 Figure | Stage 2 closed-loop primary mastery MAE by scenario and estimator. |
| `outputs/audit/phase_5/plots/e3_parent_proxy_noise_sensitivity.png` | Phase 5 Figure | Stage 3 parent-proxy noise/missingness sensitivity curves. |

---

## 2. Optional: Local Archive Verification and Fresh Extraction Test

> [!NOTE]
> The evidence archive is an optional local-only / release asset excluded from Git by `.gitignore`. The procedure below is for environments possessing `evidence/phase_5/Grow21_Phase_5_Evidence.zip`.

Every evidence bundle is deterministically validated using:
1. `zipfile.is_zipfile()` and `zipfile.testzip() is None` (zero archive corruption).
2. Fresh temporary directory extraction.
3. Execution of `$env:PYTHONPATH="src"; pytest -v tests` inside the freshly extracted tree.
4. All 66 tests pass with zero warnings and zero failures.
