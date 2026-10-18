# Grow21 Phase 4 Evidence Bundle Index (Audited)

**Bundle File:** `evidence/phase_4/Grow21_Phase_4_State_Estimation_Evidence.zip` *(local/release archive, excluded from Git)*  
**Phase:** Phase 4 — Learner Belief-State Estimation and Robustness (Audit Correction Pass)  
**Date:** October 2026  
**Status:** Self-Contained, Verified, Passing All Core Tests on Clean Checkout  

> [!NOTE]
> Under the project evidence exclusion policy, `Grow21_Phase_4_State_Estimation_Evidence.zip` is excluded from Git tracking via `.gitignore`. A fresh GitHub clone runs all core tests and validates findings using tracked compact results in `outputs/experiments/phase_4/` and deterministic fixtures in `tests/fixtures/phase_4_event_metrics_fixture.csv`. The full 260 MB raw event log can be regenerated via `python -m grow21_simulation_benchmark.run_phase_4_experiments`.

---

## 1. Bundle Manifest and File Inventory

The evidence ZIP archive contains the complete, self-contained Grow21 simulation and benchmarking project originally packaged for Phase 4:

| Relative Path in Archive | Category | Description |
| :--- | :--- | :--- |
| `pyproject.toml` | Build | PEP 518/621 build configuration and metadata. |
| `requirements.txt` | Build | Exact pinned dependencies. |
| `README.md` | Documentation | Repository overview and quickstart guide with relative paths. |
| `PROJECT_STRUCTURE.md` | Documentation | Comprehensive repository tree and directory schema. |
| `configs/default.yaml` | Config | Canonical 9-game task definitions and learner population parameters. |
| `configs/scenarios.yaml` | Config | Stress scenario configurations (noise, dropout, drift, temporal). |
| `src/grow21_simulation_benchmark/__init__.py` | Source | Package initialization and public API. |
| `src/grow21_simulation_benchmark/closed_loop.py` | Source | 4 controller policies and Phase 4 belief estimation modes with time-aligned pre/post logging. |
| `src/grow21_simulation_benchmark/outcome_model.py` | Source | Shared canonical clean-success probability model and bisection solver. |
| `src/grow21_simulation_benchmark/simulator.py` | Source | Virtual learner simulator, sensory corruption, and practice updates. |
| `src/grow21_simulation_benchmark/tasks.py` | Source | Task registry, game parameter mapping, and sensory modalities. |
| `src/grow21_simulation_benchmark/reliability.py` | Source | Sensory uncertainty combination and missingness handler. |
| `src/grow21_simulation_benchmark/evaluate.py` | Source | Calibration table, log loss, Brier score, and bootstrap CIs. |
| `src/grow21_simulation_benchmark/dataset.py` | Source | Synthetic event sequence generation. |
| `src/grow21_simulation_benchmark/features.py` | Source | Interaction feature extraction. |
| `src/grow21_simulation_benchmark/models.py` | Source | Vanilla DKT, Reliability-Gated DKT, and Logistic Baseline. |
| `src/grow21_simulation_benchmark/paths.py` | Source | Canonical directory path resolution. |
| `src/grow21_simulation_benchmark/run_benchmark.py` | Source | DKT benchmark evaluation runner. |
| `src/grow21_simulation_benchmark/run_closed_loop.py` | Source | Closed-loop policy comparison runner. |
| `src/grow21_simulation_benchmark/run_phase_3_experiments.py` | Source | Phase 3 simulation and equation parity runner. |
| `src/grow21_simulation_benchmark/run_phase_4_experiments.py` | Source | Phase 4 runner with seed-aware cluster bootstrap (B=2000) and split separation. |
| `src/grow21_simulation_benchmark/plots.py` | Source | Visualization utilities. |
| `src/grow21_simulation_benchmark/audit_calibration_bins.py` | Source | Calibration table export CLI. |
| `tests/__init__.py` | Test | Test suite init. |
| `tests/test_belief_state_estimation.py` | Test | 13 Phase 4 tests for belief updating, likelihood limits, no-leakage, time-alignment, and cluster bootstrap. |
| `tests/test_outcome_model_alignment.py` | Test | 10 Phase 3 tests for canonical equation parity and bisection reachability. |
| `tests/test_audit_regressions.py` | Test | 13 Phase 2 historical regression suite checking calibration conservation. |
| `tests/test_no_leakage.py` | Test | 3 invariance tests verifying event $t$ outcome does not leak into prediction $t$. |
| `tests/test_controller_constraints.py` | Test | 4 tests for safety cooldowns, frustration limits, and session breaks. |
| `tests/test_documentation_integrity.py` | Test | 4 tests for zero non-printing control characters and zero literal tabs in docs. |
| `tests/test_simulator.py` | Test | 3 tests for learner profile generation, fatigue/forgetting, and sensor corruption. |
| `tests/test_reliability.py` | Test | 3 tests for sensor uncertainty engine and dropout rules. |
| `tests/test_model_shapes.py` | Test | 2 tests for Deep Knowledge Tracing tensor shapes and masking. |
| `outputs/audit/phase_2/calibration_bins.csv` | Historical Data | 450-row Phase 2 calibration data required by regression tests. |
| `outputs/audit/phase_2/CHANGELOG_PHASE_2.md` | Historical Audit | Phase 2 changelog required by documentation integrity tests. |
| `outputs/audit/phase_2/PHASE_2_AUDIT_REPORT.md` | Historical Audit | Phase 2 audit report required by documentation integrity tests. |
| `outputs/benchmark/metrics.csv` | Historical Data | Phase 2 benchmark evaluation metrics required by regression tests. |
| `outputs/closed_loop/closed_loop_summary.json` | Historical Data | Phase 2 closed-loop summary required by regression tests. |
| `outputs/closed_loop/closed_loop_events.csv` | Historical Data | Phase 2 closed-loop events sample required by regression tests. |
| `outputs/audit/phase_3/PHASE_3_MODEL_ALIGNMENT_REPORT.md` | Historical Audit | Phase 3 model alignment research report. |
| `outputs/phase_3/closed_loop/phase_3_closed_loop_metrics.csv` | Historical Data | Phase 3 metrics across 4 policies and 5 scenarios. |
| `outputs/phase_3/closed_loop/phase_3_closed_loop_summary.json` | Historical Data | Phase 3 hierarchical summary JSON. |
| `outputs/phase_4/state_estimation/phase_4_calibration_bins.csv` | Phase 4 Data | 600-row calibration dataset across splits, scenarios, and controllers. |
| `outputs/phase_4/state_estimation/phase_4_mastery_estimation_metrics.csv` | Phase 4 Data | 320-row mastery metrics with explicit pre-outcome and post-update timing. |
| `outputs/phase_4/state_estimation/phase_4_scenario_summary.json` | Phase 4 Data | Multi-split summary JSON with seed-aware cluster bootstrap CIs. |
| `outputs/audit/phase_4/phase_4_calibration_bins.csv` | Phase 4 Audit Data | Audit mirror of calibration bins. |
| `outputs/audit/phase_4/phase_4_mastery_estimation_metrics.csv` | Phase 4 Audit Data | Audit mirror of mastery metrics. |
| `outputs/audit/phase_4/phase_4_scenario_summary.json` | Phase 4 Audit Data | Audit mirror of summary JSON. |
| `outputs/phase_4/plots/phase_4_calibration_curves.png` | Phase 4 Figure | Held-out calibration curves across policies. |
| `outputs/phase_4/plots/phase_4_prediction_gaps_by_scenario.png` | Phase 4 Figure | Held-out signed prediction gaps with seed standard deviations. |
| `outputs/phase_4/plots/phase_4_prediction_gaps_by_game.png` | Phase 4 Figure | Held-out per-game prediction gaps highlighting secondary transfer. |
| `outputs/phase_4/plots/phase_4_mastery_estimation_errors.png` | Phase 4 Figure | Primary mastery MAE comparing pre-outcome vs post-update timing. |
| `outputs/phase_4/plots/phase_4_difficulty_trajectories.png` | Phase 4 Figure | Sample learner difficulty trajectories across trial horizon. |
| `outputs/audit/phase_4/PHASE_4_STATE_ESTIMATION_REPORT.md` | Phase 4 Audit | Audited 10-section research and validation report. |
| `outputs/audit/phase_4/PHASE_4_CORRECTION_SUMMARY.md` | Phase 4 Audit | Executive audit correction summary and claim reconciliation table. |
| `outputs/audit/phase_4/REPRODUCE.md` | Phase 4 Audit | Deterministic reproduction commands and verification steps. |
| `outputs/audit/phase_4/CHANGELOG_PHASE_4.md` | Phase 4 Audit | Detailed list of audit corrections and modeling modifications. |
| `outputs/audit/phase_4/BUNDLE_INDEX.md` | Phase 4 Audit | This bundle manifest and verification index. |

---

## 2. Optional: Local Archive Verification and Fresh Extraction Test

> [!NOTE]
> The evidence archive is an optional local-only / release asset excluded from Git by `.gitignore`. The procedure below is for environments possessing `evidence/phase_4/Grow21_Phase_4_State_Estimation_Evidence.zip`.

Every evidence bundle is deterministically validated using:
1. `zipfile.is_zipfile()` and `zipfile.testzip() is None` (no archive corruption).
2. Fresh temporary directory extraction.
3. Execution of `$env:PYTHONPATH="src"; pytest -v tests` inside the extracted tree.
4. All tests pass with zero warnings and zero failures.
