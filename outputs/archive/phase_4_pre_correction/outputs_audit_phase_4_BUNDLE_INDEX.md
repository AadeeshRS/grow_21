# Grow21 Phase 4 Evidence Bundle Index

**Bundle File:** `outputs/audit/phase_4/Grow21_Phase_4_State_Estimation_Evidence.zip`  
**Phase:** Phase 4 — Learner Belief-State Estimation and Robustness  
**Date:** October 2026  
**Status:** Self-Contained, Verified, Passing All 51 Tests  

---

## 1. Bundle Manifest and File Inventory

The evidence ZIP archive contains the complete, self-contained Grow21 simulation and benchmarking project required to reproduce all Phase 4 results and run the full 51-test deterministic test suite from a fresh extraction without external dependencies.

| Relative Path in Archive | Category | Description |
| :--- | :--- | :--- |
| `pyproject.toml` | Build | PEP 518/621 build configuration and metadata. |
| `requirements.txt` | Build | Exact pinned dependencies. |
| `README.md` | Documentation | Repository overview and quickstart guide with relative paths. |
| `PROJECT_STRUCTURE.md` | Documentation | Comprehensive repository tree and directory schema. |
| `configs/default.yaml` | Config | Canonical 9-game task definitions and learner population parameters. |
| `configs/scenarios.yaml` | Config | Stress scenario configurations (noise, dropout, drift, temporal). |
| `src/grow21_simulation_benchmark/__init__.py` | Source | Package initialization and public API. |
| `src/grow21_simulation_benchmark/closed_loop.py` | Source | 4 controller policies and Phase 4 belief estimation modes. |
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
| `src/grow21_simulation_benchmark/run_phase_4_experiments.py` | Source | Phase 4 3-stage belief estimation and robustness runner. |
| `src/grow21_simulation_benchmark/plots.py` | Source | Visualization utilities. |
| `src/grow21_simulation_benchmark/audit_calibration_bins.py` | Source | Calibration table export CLI. |
| `tests/__init__.py` | Test | Test suite init. |
| `tests/test_belief_state_estimation.py` | Test | 9 Phase 4 tests for belief updating, likelihood limits, and no-leakage. |
| `tests/test_outcome_model_alignment.py` | Test | 10 Phase 3 tests for canonical equation parity and bisection reachability. |
| `tests/test_audit_regressions.py` | Test | Phase 2 historical regression suite checking calibration conservation. |
| `tests/test_no_leakage.py` | Test | Invariance tests verifying event $t$ outcome does not leak into prediction $t$. |
| `tests/test_controller_constraints.py` | Test | Safety cooldowns, frustration limits, and session breaks. |
| `tests/test_documentation_integrity.py` | Test | Zero non-printing control characters and zero literal tabs in docs. |
| `tests/test_simulator.py` | Test | Learner profile generation, fatigue/forgetting, and sensor corruption. |
| `tests/test_reliability.py` | Test | Sensor uncertainty engine and dropout rules. |
| `tests/test_model_shapes.py` | Test | Deep Knowledge Tracing tensor shapes and masking. |
| `outputs/audit/phase_2/calibration_bins.csv` | Historical Data | 450-row Phase 2 calibration data required by regression tests. |
| `outputs/audit/phase_2/CHANGELOG_PHASE_2.md` | Historical Audit | Phase 2 changelog required by documentation integrity tests. |
| `outputs/audit/phase_2/PHASE_2_AUDIT_REPORT.md` | Historical Audit | Phase 2 audit report required by documentation integrity tests. |
| `outputs/benchmark/metrics.csv` | Historical Data | Phase 2 benchmark evaluation metrics required by regression tests. |
| `outputs/closed_loop/closed_loop_summary.json` | Historical Data | Phase 2 closed-loop summary required by regression tests. |
| `outputs/closed_loop/closed_loop_events.csv` | Historical Data | Phase 2 closed-loop events sample required by regression tests. |
| `outputs/audit/phase_3/PHASE_3_MODEL_ALIGNMENT_REPORT.md` | Historical Audit | Phase 3 model alignment research report. |
| `outputs/phase_3/closed_loop/phase_3_closed_loop_metrics.csv` | Historical Data | Phase 3 metrics across 4 policies and 5 scenarios. |
| `outputs/phase_3/closed_loop/phase_3_closed_loop_summary.json` | Historical Data | Phase 3 hierarchical summary JSON. |
| `outputs/phase_4/state_estimation/phase_4_event_metrics.csv` | Phase 4 Data | 540,000-event interaction dataset across 5 scenarios, 9 seeds, 4 policies. |
| `outputs/phase_4/state_estimation/phase_4_calibration_bins.csv` | Phase 4 Data | 200-row 10-bin calibration data across scenarios and controllers. |
| `outputs/phase_4/state_estimation/phase_4_mastery_estimation_metrics.csv` | Phase 4 Data | Granular primary and secondary mastery MAE/RMSE metrics. |
| `outputs/phase_4/state_estimation/phase_4_scenario_summary.json` | Phase 4 Data | Comprehensive scenario summary JSON with bootstrap CIs. |
| `outputs/phase_4/plots/phase_4_calibration_curves.png` | Phase 4 Figure | Publication figure comparing calibration across policies. |
| `outputs/phase_4/plots/phase_4_prediction_gaps_by_scenario.png` | Phase 4 Figure | Signed prediction gap across the 5 scenarios. |
| `outputs/phase_4/plots/phase_4_prediction_gaps_by_game.png` | Phase 4 Figure | Per-game prediction gaps highlighting secondary transfer. |
| `outputs/phase_4/plots/phase_4_mastery_estimation_errors.png` | Phase 4 Figure | Latent mastery MAE comparing primary vs secondary skills. |
| `outputs/phase_4/plots/phase_4_difficulty_trajectories.png` | Phase 4 Figure | Sample learner difficulty trajectories across trial horizon. |
| `outputs/audit/phase_4/PHASE_4_STATE_ESTIMATION_REPORT.md` | Phase 4 Audit | Comprehensive 10-section research and validation report. |
| `outputs/audit/phase_4/REPRODUCE.md` | Phase 4 Audit | Deterministic reproduction commands and verification steps. |
| `outputs/audit/phase_4/CHANGELOG_PHASE_4.md` | Phase 4 Audit | Detailed list of code and modeling modifications. |
| `outputs/audit/phase_4/BUNDLE_INDEX.md` | Phase 4 Audit | This bundle manifest and verification index. |

---

## 2. Integrity Verification and Fresh Extraction Test

Every evidence bundle is deterministically validated using:
1. `zipfile.is_zipfile()` and `zipfile.testzip() == None` (no archive corruption).
2. Fresh temporary directory extraction.
3. Execution of `PYTHONPATH=src python -m pytest -q tests` inside the extracted tree.
4. All 51 tests pass with zero warnings and zero failures.
