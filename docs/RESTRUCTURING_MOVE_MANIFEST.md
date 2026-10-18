# Grow21 Simulation Repository Restructuring Move Manifest

**Date:** October 18, 2026  
**Status:** Complete and Verified  
**Verification Status:** 67 pytest tests validated (66 passing self-contained tests on clean checkout, 1 full-volume test); all moved artifacts SHA-256 verified; 44 duplicate mirrors safely removed; all 4 evidence ZIPs verified with zipfile.testzip() == None.

---

## 1. Executive Summary of Restructure

The Grow21 simulation and benchmarking repository has been restructured from a flat/mixed `outputs/` structure into a clean, modular, research-grade architecture:
- **Human-Facing Research Documentation** (`docs/research/phase_N/`): Phase reports, audit changelogs, bundle indexes, and reproduction instructions organized by phase.
- **Architecture, Testing, & Integration Docs** (`docs/architecture/`, `docs/testing/`, `docs/integration/`): Consolidated technical design, testing methodologies, validation matrices, and frontend integration preparation.
- **Immutable Evidence Bundles** (`evidence/phase_N/`): Pre-computed self-contained ZIP evidence bundles preserved byte-for-byte in local project storage / release assets, verified with `zipfile.testzip() == None`, and explicitly excluded from public Git tracking via `.gitignore`.
- **Canonical Experiment Outputs** (`outputs/experiments/phase_N/`): Experiment event logs, summary tables, calibration curves, and publication plots.
- **Historical Archive** (`outputs/archive/`): Preserved historical snapshots (`phase_4_pre_correction`, `pre_correction_pass`, `pre_phase_3_source`, `project_cleanup`, `scratch_model_test`).
- **Clean Import Namespace** (`src/grow21_simulation_benchmark/`): Untouched package namespace and source code, preserving all model algorithms, formulas, and CLI runners.

---

## 2. Summary of File Operations

| Category | Count | Action Taken | Rationale |
| :--- | :---: | :--- | :--- |
| `kept_in_place` | 97 | Untouched in place | Active Python package source (`src/`), tests (`tests/`), configs (`configs/`), historical archives (`outputs/archive/`), and root packaging files. |
| `research_doc` | 18 | Moved to `docs/research/phase_N/` | Human-facing markdown reports, changelogs, manifests, and config snapshots. |
| `evidence_zip` | 4 | Moved to `evidence/phase_N/` | Standalone ZIP evidence bundles containing audited empirical artifacts. |
| `p2_unique_audit` | 5 | Moved to `outputs/experiments/phase_2/` | Phase 2 unique calibration bins, paired loss evidence, and audit sweep summaries. |
| `canonical_experiment` | 62 | Moved to `outputs/experiments/phase_N/` | Active simulation outputs, event metrics, scenario summaries, and plots for Phases 2, 3, 4, and 5. |
| `archived_scratch` | 6 | Moved to `outputs/archive/scratch_model_test/` | Preserved scratch benchmark test run. |
| `deduplicated_mirror` | 44 | Verified SHA-256 & Removed | Exact byte-for-byte duplicates previously mirrored into `outputs/audit/phase_3`, `phase_4`, and `phase_5` (including duplicate 876MB and 570MB CSV files, saving ~1.5 GB). |
| **Total Original Files** | **236** | **All accounted for** | **Zero unique data or code lost.** |

---

## 3. Detailed Move Manifest

### A. Human-Facing Research Documentation (`docs/research/`)

| Original Path | Target Path | SHA-256 (first 12 chars) | Size (bytes) |
| :--- | :--- | :---: | :---: |
| `outputs/audit/phase_2/BUNDLE_INDEX.md` | `docs/research/phase_2/BUNDLE_INDEX.md` | `d8149f5ba689` | 8722 |
| `outputs/audit/phase_2/CHANGELOG_PHASE_2.md` | `docs/research/phase_2/CHANGELOG_PHASE_2.md` | `0575aaebbc73` | 17242 |
| `outputs/audit/phase_2/PHASE_2_AUDIT_REPORT.md` | `docs/research/phase_2/PHASE_2_AUDIT_REPORT.md` | `90bb11f0a9a1` | 36413 |
| `outputs/audit/phase_2/REPRODUCE.md` | `docs/research/phase_2/REPRODUCE.md` | `880acd04ec6f` | 7521 |
| `outputs/audit/phase_3/BUNDLE_INDEX.md` | `docs/research/phase_3/BUNDLE_INDEX.md` | `e9667ea2c1f8` | 7456 |
| `outputs/audit/phase_3/CHANGELOG_PHASE_3.md` | `docs/research/phase_3/CHANGELOG_PHASE_3.md` | `d4ddeb4f4414` | 6840 |
| `outputs/audit/phase_3/PHASE_3_MODEL_ALIGNMENT_REPORT.md` | `docs/research/phase_3/PHASE_3_MODEL_ALIGNMENT_REPORT.md` | `70983c4e1a64` | 23540 |
| `outputs/audit/phase_3/REPRODUCE.md` | `docs/research/phase_3/REPRODUCE.md` | `70bc43308efd` | 3399 |
| `outputs/audit/phase_4/BUNDLE_INDEX.md` | `docs/research/phase_4/BUNDLE_INDEX.md` | `04b1bee7c849` | 8096 |
| `outputs/audit/phase_4/CHANGELOG_PHASE_4.md` | `docs/research/phase_4/CHANGELOG_PHASE_4.md` | `814c77a0d12d` | 7717 |
| `outputs/audit/phase_4/PHASE_4_CORRECTION_SUMMARY.md` | `docs/research/phase_4/PHASE_4_CORRECTION_SUMMARY.md` | `7797b9448522` | 10839 |
| `outputs/audit/phase_4/PHASE_4_STATE_ESTIMATION_REPORT.md` | `docs/research/phase_4/PHASE_4_STATE_ESTIMATION_REPORT.md` | `6c713331bfaa` | 26529 |
| `outputs/audit/phase_4/REPRODUCE.md` | `docs/research/phase_4/REPRODUCE.md` | `b9022d190f79` | 5623 |
| `outputs/audit/phase_5/BUNDLE_INDEX.md` | `docs/research/phase_5/BUNDLE_INDEX.md` | `a15049df7d7a` | 9679 |
| `outputs/audit/phase_5/CHANGELOG_PHASE_5.md` | `docs/research/phase_5/CHANGELOG_PHASE_5.md` | `291892d17156` | 8341 |
| `outputs/audit/phase_5/CONFIG_SNAPSHOT.json` | `docs/research/phase_5/CONFIG_SNAPSHOT.json` | `123d587c8f2d` | 1556 |
| `outputs/audit/phase_5/PHASE_5_STATE_ESTIMATION_AND_SELECTION_REPORT.md` | `docs/research/phase_5/PHASE_5_STATE_ESTIMATION_AND_SELECTION_REPORT.md` | `22d0bcd29d54` | 28802 |
| `outputs/audit/phase_5/REPRODUCE.md` | `docs/research/phase_5/REPRODUCE.md` | `2289f63aa60b` | 10562 |

### B. Evidence Archives (`evidence/`)

> [!NOTE]
> **GitHub Tracking Policy:** Under the Grow21 evidence-archive exclusion policy, all four ZIP archives listed below are explicitly excluded from Git tracking via `.gitignore` (combined size ~130.56 MB). They remain preserved unchanged in local project storage (`evidence/phase_N/`) for archival provenance, and can be distributed as GitHub Release assets if desired. A clean checkout does not require these archives because all canonical outputs, figures, and self-contained test fixtures are tracked directly in `outputs/experiments/` and `tests/fixtures/`.

| Original Path | Target Path | SHA-256 (first 12 chars) | Size (MB) | Git Tracking Status |
| :--- | :--- | :---: | :---: | :---: |
| `outputs/audit/phase_2/Grow21_Phase_2_Audit_Evidence.zip` | `evidence/phase_2/Grow21_Phase_2_Audit_Evidence.zip` | `09d93c94405c` | 1.25 MB | Excluded (`.gitignore`) |
| `outputs/audit/phase_3/Grow21_Phase_3_Model_Alignment_Evidence.zip` | `evidence/phase_3/Grow21_Phase_3_Model_Alignment_Evidence.zip` | `a4baeb97a3fb` | 5.36 MB | Excluded (`.gitignore`) |
| `outputs/audit/phase_4/Grow21_Phase_4_State_Estimation_Evidence.zip` | `evidence/phase_4/Grow21_Phase_4_State_Estimation_Evidence.zip` | `c67a16222940` | 57.31 MB | Excluded (`.gitignore`) |
| `outputs/audit/phase_5/Grow21_Phase_5_Evidence.zip` | `evidence/phase_5/Grow21_Phase_5_Evidence.zip` | `99a3045590fd` | 66.64 MB | Excluded (`.gitignore`) |

### C. Canonical Experiment Outputs (`outputs/experiments/`)

| Original Path | Target Path | SHA-256 (first 12 chars) | Size (bytes/MB) |
| :--- | :--- | :---: | :---: |
| `outputs/audit/phase_2/calibration_bins.csv` | `outputs/experiments/phase_2/calibration_bins.csv` | `64a26d836965` | 55456 B |
| `outputs/audit/phase_2/learner_mastery_summary.csv` | `outputs/experiments/phase_2/audit_sweep_learner_mastery_summary.csv` | `1190cf5e927e` | 5960 B |
| `outputs/audit/phase_2/paired_loss_evidence.csv` | `outputs/experiments/phase_2/paired_loss_evidence.csv` | `0be80f65fffd` | 13094 B |
| `outputs/audit/phase_2/predictor_vs_simulator_comparison.csv` | `outputs/experiments/phase_2/predictor_vs_simulator_comparison.csv` | `dc83f17205a1` | 8416 B |
| `outputs/audit/phase_2/reliability_sweep_summary.json` | `outputs/experiments/phase_2/reliability_sweep_summary.json` | `48e63f3e527e` | 4141 B |
| `outputs/benchmark/metrics.csv` | `outputs/experiments/phase_2/benchmark/metrics.csv` | `5d1c6383914d` | 4686 B |
| `outputs/benchmark/plots/calibration_base_heterogeneous.png` | `outputs/experiments/phase_2/benchmark/plots/calibration_base_heterogeneous.png` | `7c7a6cc06795` | 141676 B |
| `outputs/benchmark/plots/calibration_distribution_shift.png` | `outputs/experiments/phase_2/benchmark/plots/calibration_distribution_shift.png` | `bcc4464cb967` | 152359 B |
| `outputs/benchmark/plots/calibration_reliability_stress.png` | `outputs/experiments/phase_2/benchmark/plots/calibration_reliability_stress.png` | `b2ac47465eb7` | 131682 B |
| `outputs/benchmark/plots/calibration_simple_dynamics.png` | `outputs/experiments/phase_2/benchmark/plots/calibration_simple_dynamics.png` | `46c33fefc44a` | 136151 B |
| `outputs/benchmark/plots/calibration_temporal_dynamics.png` | `outputs/experiments/phase_2/benchmark/plots/calibration_temporal_dynamics.png` | `3640cf49b4db` | 139897 B |
| `outputs/benchmark/plots/scenario_comparison.png` | `outputs/experiments/phase_2/benchmark/plots/scenario_comparison.png` | `d78df81562ee` | 150123 B |
| `outputs/benchmark/summary.json` | `outputs/experiments/phase_2/benchmark/summary.json` | `17d2ddb219a7` | 5905 B |
| `outputs/closed_loop/closed_loop_events.csv` | `outputs/experiments/phase_2/closed_loop/closed_loop_events.csv` | `b867c78cb7f3` | 1.09 MB |
| `outputs/closed_loop/closed_loop_summary.json` | `outputs/experiments/phase_2/closed_loop/closed_loop_summary.json` | `3f6e30f3c33a` | 3146 B |
| `outputs/closed_loop/learner_mastery_summary.csv` | `outputs/experiments/phase_2/closed_loop/learner_mastery_summary.csv` | `0d22bc97a0c2` | 72333 B |
| `outputs/closed_loop/plots/closed_loop_comparison.png` | `outputs/experiments/phase_2/closed_loop/plots/closed_loop_comparison.png` | `41018277dc9a` | 113238 B |
| `outputs/closed_loop/plots/difficulty_trajectories.png` | `outputs/experiments/phase_2/closed_loop/plots/difficulty_trajectories.png` | `8a75e5f562d1` | 244979 B |
| `outputs/phase_3/closed_loop/phase_3_closed_loop_events.csv` | `outputs/experiments/phase_3/closed_loop/phase_3_closed_loop_events.csv` | `23a30846c4e6` | 41.66 MB |
| `outputs/phase_3/closed_loop/phase_3_closed_loop_metrics.csv` | `outputs/experiments/phase_3/closed_loop/phase_3_closed_loop_metrics.csv` | `f36f6e950653` | 12267 B |
| `outputs/phase_3/closed_loop/phase_3_closed_loop_summary.json` | `outputs/experiments/phase_3/closed_loop/phase_3_closed_loop_summary.json` | `6911f7c890ad` | 18601 B |
| `outputs/phase_3/closed_loop/phase_3_learner_mastery_summary.csv` | `outputs/experiments/phase_3/closed_loop/phase_3_learner_mastery_summary.csv` | `1d022de8f934` | 1.92 MB |
| `outputs/phase_3/closed_loop/phase_3_per_game_summary.csv` | `outputs/experiments/phase_3/closed_loop/phase_3_per_game_summary.csv` | `998d65f17288` | 19827 B |
| `outputs/phase_3/diagnostics/formula_parity_results.csv` | `outputs/experiments/phase_3/diagnostics/formula_parity_results.csv` | `02050d856063` | 444848 B |
| `outputs/phase_3/diagnostics/formula_parity_summary.json` | `outputs/experiments/phase_3/diagnostics/formula_parity_summary.json` | `6701398fcf02` | 143 B |
| `outputs/phase_3/plots/calibration_predicted_clean_success.png` | `outputs/experiments/phase_3/plots/calibration_predicted_clean_success.png` | `65ac2ffab7b6` | 116351 B |
| `outputs/phase_3/plots/difficulty_trajectories_comparison.png` | `outputs/experiments/phase_3/plots/difficulty_trajectories_comparison.png` | `3819ceb17092` | 273028 B |
| `outputs/phase_3/plots/policy_comparison_metrics.png` | `outputs/experiments/phase_3/plots/policy_comparison_metrics.png` | `279496ad0b43` | 118173 B |
| `outputs/phase_3/plots/target_reachability_and_saturation.png` | `outputs/experiments/phase_3/plots/target_reachability_and_saturation.png` | `85a2dfec2b6b` | 108345 B |
| `outputs/phase_4/plots/phase_4_calibration_curves.png` | `outputs/experiments/phase_4/plots/phase_4_calibration_curves.png` | `6ec1107b020c` | 421190 B |
| `outputs/phase_4/plots/phase_4_difficulty_trajectories.png` | `outputs/experiments/phase_4/plots/phase_4_difficulty_trajectories.png` | `ddedc899b049` | 356360 B |
| `outputs/phase_4/plots/phase_4_mastery_estimation_errors.png` | `outputs/experiments/phase_4/plots/phase_4_mastery_estimation_errors.png` | `2640c2806735` | 229095 B |
| `outputs/phase_4/plots/phase_4_prediction_gaps_by_game.png` | `outputs/experiments/phase_4/plots/phase_4_prediction_gaps_by_game.png` | `e5d4035c2730` | 170018 B |
| `outputs/phase_4/plots/phase_4_prediction_gaps_by_scenario.png` | `outputs/experiments/phase_4/plots/phase_4_prediction_gaps_by_scenario.png` | `752e28d35795` | 151891 B |
| `outputs/phase_4/state_estimation/phase_4_calibration_bins.csv` | `outputs/experiments/phase_4/state_estimation/phase_4_calibration_bins.csv` | `404a1a494b2a` | 54078 B |
| `outputs/phase_4/state_estimation/phase_4_event_metrics.csv` | `outputs/experiments/phase_4/state_estimation/phase_4_event_metrics.csv` | `2ae167a01dbd` | 260.56 MB |
| `outputs/phase_4/state_estimation/phase_4_mastery_estimation_metrics.csv` | `outputs/experiments/phase_4/state_estimation/phase_4_mastery_estimation_metrics.csv` | `37efcb53a6b8` | 44842 B |
| `outputs/phase_4/state_estimation/phase_4_scenario_summary.json` | `outputs/experiments/phase_4/state_estimation/phase_4_scenario_summary.json` | `ea16b4bcf035` | 266074 B |
| `outputs/phase_5/CONFIG_SNAPSHOT.json` | `outputs/experiments/phase_5/CONFIG_SNAPSHOT.json` | `123d587c8f2d` | 1556 B |
| `outputs/phase_5/benchmark/model_selector_benchmark_metrics.csv` | `outputs/experiments/phase_5/benchmark/model_selector_benchmark_metrics.csv` | `a341b97e44ae` | 694 B |
| `outputs/phase_5/benchmark/selector_runs/all/metrics.csv` | `outputs/experiments/phase_5/benchmark/selector_runs/all/metrics.csv` | `470cecfcb24f` | 420 B |
| `outputs/phase_5/benchmark/selector_runs/all/plots/calibration_simple_dynamics.png` | `outputs/experiments/phase_5/benchmark/selector_runs/all/plots/calibration_simple_dynamics.png` | `cea18a0c7293` | 135989 B |
| `outputs/phase_5/benchmark/selector_runs/all/summary.json` | `outputs/experiments/phase_5/benchmark/selector_runs/all/summary.json` | `90039728240b` | 1144 B |
| `outputs/phase_5/benchmark/selector_runs/logistic_baseline/metrics.csv` | `outputs/experiments/phase_5/benchmark/selector_runs/logistic_baseline/metrics.csv` | `f7cd4d5132d2` | 227 B |
| `outputs/phase_5/benchmark/selector_runs/logistic_baseline/plots/calibration_simple_dynamics.png` | `outputs/experiments/phase_5/benchmark/selector_runs/logistic_baseline/plots/calibration_simple_dynamics.png` | `19bdbf37f854` | 118226 B |
| `outputs/phase_5/benchmark/selector_runs/logistic_baseline/summary.json` | `outputs/experiments/phase_5/benchmark/selector_runs/logistic_baseline/summary.json` | `1501cbf23cc0` | 418 B |
| `outputs/phase_5/benchmark/selector_runs/reliability_gated_dkt/metrics.csv` | `outputs/experiments/phase_5/benchmark/selector_runs/reliability_gated_dkt/metrics.csv` | `e7cf9d642865` | 233 B |
| `outputs/phase_5/benchmark/selector_runs/reliability_gated_dkt/plots/calibration_simple_dynamics.png` | `outputs/experiments/phase_5/benchmark/selector_runs/reliability_gated_dkt/plots/calibration_simple_dynamics.png` | `443485dacf2e` | 99181 B |
| `outputs/phase_5/benchmark/selector_runs/reliability_gated_dkt/summary.json` | `outputs/experiments/phase_5/benchmark/selector_runs/reliability_gated_dkt/summary.json` | `c4656d17db1f` | 424 B |
| `outputs/phase_5/benchmark/selector_runs/vanilla_dkt/metrics.csv` | `outputs/experiments/phase_5/benchmark/selector_runs/vanilla_dkt/metrics.csv` | `0b629e4344d1` | 223 B |
| `outputs/phase_5/benchmark/selector_runs/vanilla_dkt/plots/calibration_simple_dynamics.png` | `outputs/experiments/phase_5/benchmark/selector_runs/vanilla_dkt/plots/calibration_simple_dynamics.png` | `3d99654f9555` | 99946 B |
| `outputs/phase_5/benchmark/selector_runs/vanilla_dkt/summary.json` | `outputs/experiments/phase_5/benchmark/selector_runs/vanilla_dkt/summary.json` | `f481e5a46d01` | 415 B |
| `outputs/phase_5/closed_loop/phase_5_calibration_table.csv` | `outputs/experiments/phase_5/closed_loop/phase_5_calibration_table.csv` | `eb71d5df60cc` | 145254 B |
| `outputs/phase_5/closed_loop/phase_5_event_metrics.csv` | `outputs/experiments/phase_5/closed_loop/phase_5_event_metrics.csv` | `61901acfab81` | 835.93 MB |
| `outputs/phase_5/closed_loop/phase_5_learner_summary.csv` | `outputs/experiments/phase_5/closed_loop/phase_5_learner_summary.csv` | `a2388e5a6432` | 16.91 MB |
| `outputs/phase_5/closed_loop/phase_5_paired_bootstrap.csv` | `outputs/experiments/phase_5/closed_loop/phase_5_paired_bootstrap.csv` | `1b99f847dc27` | 147170 B |
| `outputs/phase_5/closed_loop/phase_5_scenario_summary.csv` | `outputs/experiments/phase_5/closed_loop/phase_5_scenario_summary.csv` | `7ab5b927baa2` | 86438 B |
| `outputs/phase_5/initializers/parent_proxy_events.csv` | `outputs/experiments/phase_5/initializers/parent_proxy_events.csv` | `de7c58fc955e` | 19.37 MB |
| `outputs/phase_5/initializers/parent_proxy_sensitivity_metrics.csv` | `outputs/experiments/phase_5/initializers/parent_proxy_sensitivity_metrics.csv` | `5023fea59243` | 6316 B |
| `outputs/phase_5/open_loop/open_loop_events.csv` | `outputs/experiments/phase_5/open_loop/open_loop_events.csv` | `5525684d7e24` | 570.23 MB |
| `outputs/phase_5/open_loop/open_loop_paired_bootstrap.csv` | `outputs/experiments/phase_5/open_loop/open_loop_paired_bootstrap.csv` | `4b7e01d01384` | 73138 B |
| `outputs/phase_5/open_loop/open_loop_summary.csv` | `outputs/experiments/phase_5/open_loop/open_loop_summary.csv` | `50d431747e5e` | 149611 B |
| `outputs/phase_5/plots/e1_open_loop_mastery_error_by_scenario.png` | `outputs/experiments/phase_5/plots/e1_open_loop_mastery_error_by_scenario.png` | `89c514ccbe72` | 204397 B |
| `outputs/phase_5/plots/e2_clean_prob_calibration_curves.png` | `outputs/experiments/phase_5/plots/e2_clean_prob_calibration_curves.png` | `c7a365eecbd6` | 248837 B |
| `outputs/phase_5/plots/e2_closed_loop_difficulty_and_saturation.png` | `outputs/experiments/phase_5/plots/e2_closed_loop_difficulty_and_saturation.png` | `c01543d53c2b` | 265492 B |
| `outputs/phase_5/plots/e2_closed_loop_mastery_error_comparison.png` | `outputs/experiments/phase_5/plots/e2_closed_loop_mastery_error_comparison.png` | `7be2f34422e9` | 248047 B |
| `outputs/phase_5/plots/e3_parent_proxy_noise_sensitivity.png` | `outputs/experiments/phase_5/plots/e3_parent_proxy_noise_sensitivity.png` | `41294aebbb10` | 175011 B |

### D. Archived Scratch Runs (`outputs/archive/`)

| Original Path | Target Path | SHA-256 (first 12 chars) |
| :--- | :--- | :---: |
| `outputs/scratch_model_test/all/metrics.csv` | `outputs/archive/scratch_model_test/all/metrics.csv` | `b80f496dd495` |
| `outputs/scratch_model_test/all/plots/calibration_simple_dynamics.png` | `outputs/archive/scratch_model_test/all/plots/calibration_simple_dynamics.png` | `85c76538dd09` |
| `outputs/scratch_model_test/all/summary.json` | `outputs/archive/scratch_model_test/all/summary.json` | `7d516c123938` |
| `outputs/scratch_model_test/ind/metrics.csv` | `outputs/archive/scratch_model_test/ind/metrics.csv` | `3586189959df` |
| `outputs/scratch_model_test/ind/plots/calibration_simple_dynamics.png` | `outputs/archive/scratch_model_test/ind/plots/calibration_simple_dynamics.png` | `338cfe450de1` |
| `outputs/scratch_model_test/ind/summary.json` | `outputs/archive/scratch_model_test/ind/summary.json` | `2b67f138a590` |

### E. Deduplicated Mirrors (Consolidated to Canonical)

| Redundant Mirror Path | Canonical Active Path | SHA-256 Verified Match |
| :--- | :--- | :---: |
| `outputs/audit/phase_3/calibration_predicted_clean_success.png` | `outputs/experiments/phase_3/plots/calibration_predicted_clean_success.png` | `65ac2ffab7b6` |
| `outputs/audit/phase_3/difficulty_trajectories_comparison.png` | `outputs/experiments/phase_3/plots/difficulty_trajectories_comparison.png` | `3819ceb17092` |
| `outputs/audit/phase_3/formula_parity_summary.json` | `outputs/experiments/phase_3/diagnostics/formula_parity_summary.json` | `6701398fcf02` |
| `outputs/audit/phase_3/phase_3_closed_loop_metrics.csv` | `outputs/experiments/phase_3/closed_loop/phase_3_closed_loop_metrics.csv` | `f36f6e950653` |
| `outputs/audit/phase_3/phase_3_closed_loop_summary.json` | `outputs/experiments/phase_3/closed_loop/phase_3_closed_loop_summary.json` | `6911f7c890ad` |
| `outputs/audit/phase_3/phase_3_per_game_summary.csv` | `outputs/experiments/phase_3/closed_loop/phase_3_per_game_summary.csv` | `998d65f17288` |
| `outputs/audit/phase_3/policy_comparison_metrics.png` | `outputs/experiments/phase_3/plots/policy_comparison_metrics.png` | `279496ad0b43` |
| `outputs/audit/phase_3/target_reachability_and_saturation.png` | `outputs/experiments/phase_3/plots/target_reachability_and_saturation.png` | `85a2dfec2b6b` |
| `outputs/audit/phase_4/phase_4_calibration_bins.csv` | `outputs/experiments/phase_4/state_estimation/phase_4_calibration_bins.csv` | `404a1a494b2a` |
| `outputs/audit/phase_4/phase_4_calibration_curves.png` | `outputs/experiments/phase_4/plots/phase_4_calibration_curves.png` | `6ec1107b020c` |
| `outputs/audit/phase_4/phase_4_difficulty_trajectories.png` | `outputs/experiments/phase_4/plots/phase_4_difficulty_trajectories.png` | `ddedc899b049` |
| `outputs/audit/phase_4/phase_4_mastery_estimation_errors.png` | `outputs/experiments/phase_4/plots/phase_4_mastery_estimation_errors.png` | `2640c2806735` |
| `outputs/audit/phase_4/phase_4_mastery_estimation_metrics.csv` | `outputs/experiments/phase_4/state_estimation/phase_4_mastery_estimation_metrics.csv` | `37efcb53a6b8` |
| `outputs/audit/phase_4/phase_4_prediction_gaps_by_game.png` | `outputs/experiments/phase_4/plots/phase_4_prediction_gaps_by_game.png` | `e5d4035c2730` |
| `outputs/audit/phase_4/phase_4_prediction_gaps_by_scenario.png` | `outputs/experiments/phase_4/plots/phase_4_prediction_gaps_by_scenario.png` | `752e28d35795` |
| `outputs/audit/phase_4/phase_4_scenario_summary.json` | `outputs/experiments/phase_4/state_estimation/phase_4_scenario_summary.json` | `ea16b4bcf035` |
| `outputs/audit/phase_5/benchmark/model_selector_benchmark_metrics.csv` | `outputs/experiments/phase_5/benchmark/model_selector_benchmark_metrics.csv` | `a341b97e44ae` |
| `outputs/audit/phase_5/benchmark/selector_runs/all/metrics.csv` | `outputs/experiments/phase_5/benchmark/selector_runs/all/metrics.csv` | `470cecfcb24f` |
| `outputs/audit/phase_5/benchmark/selector_runs/all/summary.json` | `outputs/experiments/phase_5/benchmark/selector_runs/all/summary.json` | `90039728240b` |
| `outputs/audit/phase_5/benchmark/selector_runs/all/plots/calibration_simple_dynamics.png` | `outputs/experiments/phase_5/benchmark/selector_runs/all/plots/calibration_simple_dynamics.png` | `cea18a0c7293` |
| `outputs/audit/phase_5/benchmark/selector_runs/logistic_baseline/metrics.csv` | `outputs/experiments/phase_5/benchmark/selector_runs/logistic_baseline/metrics.csv` | `f7cd4d5132d2` |
| `outputs/audit/phase_5/benchmark/selector_runs/logistic_baseline/summary.json` | `outputs/experiments/phase_5/benchmark/selector_runs/logistic_baseline/summary.json` | `1501cbf23cc0` |
| `outputs/audit/phase_5/benchmark/selector_runs/logistic_baseline/plots/calibration_simple_dynamics.png` | `outputs/experiments/phase_5/benchmark/selector_runs/logistic_baseline/plots/calibration_simple_dynamics.png` | `19bdbf37f854` |
| `outputs/audit/phase_5/benchmark/selector_runs/reliability_gated_dkt/metrics.csv` | `outputs/experiments/phase_5/benchmark/selector_runs/reliability_gated_dkt/metrics.csv` | `e7cf9d642865` |
| `outputs/audit/phase_5/benchmark/selector_runs/reliability_gated_dkt/summary.json` | `outputs/experiments/phase_5/benchmark/selector_runs/reliability_gated_dkt/summary.json` | `c4656d17db1f` |
| `outputs/audit/phase_5/benchmark/selector_runs/reliability_gated_dkt/plots/calibration_simple_dynamics.png` | `outputs/experiments/phase_5/benchmark/selector_runs/reliability_gated_dkt/plots/calibration_simple_dynamics.png` | `443485dacf2e` |
| `outputs/audit/phase_5/benchmark/selector_runs/vanilla_dkt/metrics.csv` | `outputs/experiments/phase_5/benchmark/selector_runs/vanilla_dkt/metrics.csv` | `0b629e4344d1` |
| `outputs/audit/phase_5/benchmark/selector_runs/vanilla_dkt/summary.json` | `outputs/experiments/phase_5/benchmark/selector_runs/vanilla_dkt/summary.json` | `f481e5a46d01` |
| `outputs/audit/phase_5/benchmark/selector_runs/vanilla_dkt/plots/calibration_simple_dynamics.png` | `outputs/experiments/phase_5/benchmark/selector_runs/vanilla_dkt/plots/calibration_simple_dynamics.png` | `3d99654f9555` |
| `outputs/audit/phase_5/closed_loop/phase_5_calibration_table.csv` | `outputs/experiments/phase_5/closed_loop/phase_5_calibration_table.csv` | `eb71d5df60cc` |
| `outputs/audit/phase_5/closed_loop/phase_5_event_metrics.csv` | `outputs/experiments/phase_5/closed_loop/phase_5_event_metrics.csv` | `61901acfab81` |
| `outputs/audit/phase_5/closed_loop/phase_5_learner_summary.csv` | `outputs/experiments/phase_5/closed_loop/phase_5_learner_summary.csv` | `a2388e5a6432` |
| `outputs/audit/phase_5/closed_loop/phase_5_paired_bootstrap.csv` | `outputs/experiments/phase_5/closed_loop/phase_5_paired_bootstrap.csv` | `1b99f847dc27` |
| `outputs/audit/phase_5/closed_loop/phase_5_scenario_summary.csv` | `outputs/experiments/phase_5/closed_loop/phase_5_scenario_summary.csv` | `7ab5b927baa2` |
| `outputs/audit/phase_5/initializers/parent_proxy_events.csv` | `outputs/experiments/phase_5/initializers/parent_proxy_events.csv` | `de7c58fc955e` |
| `outputs/audit/phase_5/initializers/parent_proxy_sensitivity_metrics.csv` | `outputs/experiments/phase_5/initializers/parent_proxy_sensitivity_metrics.csv` | `5023fea59243` |
| `outputs/audit/phase_5/open_loop/open_loop_events.csv` | `outputs/experiments/phase_5/open_loop/open_loop_events.csv` | `5525684d7e24` |
| `outputs/audit/phase_5/open_loop/open_loop_paired_bootstrap.csv` | `outputs/experiments/phase_5/open_loop/open_loop_paired_bootstrap.csv` | `4b7e01d01384` |
| `outputs/audit/phase_5/open_loop/open_loop_summary.csv` | `outputs/experiments/phase_5/open_loop/open_loop_summary.csv` | `50d431747e5e` |
| `outputs/audit/phase_5/plots/e1_open_loop_mastery_error_by_scenario.png` | `outputs/experiments/phase_5/plots/e1_open_loop_mastery_error_by_scenario.png` | `89c514ccbe72` |
| `outputs/audit/phase_5/plots/e2_clean_prob_calibration_curves.png` | `outputs/experiments/phase_5/plots/e2_clean_prob_calibration_curves.png` | `c7a365eecbd6` |
| `outputs/audit/phase_5/plots/e2_closed_loop_difficulty_and_saturation.png` | `outputs/experiments/phase_5/plots/e2_closed_loop_difficulty_and_saturation.png` | `c01543d53c2b` |
| `outputs/audit/phase_5/plots/e2_closed_loop_mastery_error_comparison.png` | `outputs/experiments/phase_5/plots/e2_closed_loop_mastery_error_comparison.png` | `7be2f34422e9` |
| `outputs/audit/phase_5/plots/e3_parent_proxy_noise_sensitivity.png` | `outputs/experiments/phase_5/plots/e3_parent_proxy_noise_sensitivity.png` | `41294aebbb10` |
