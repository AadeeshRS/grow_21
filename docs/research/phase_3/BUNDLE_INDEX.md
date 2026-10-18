# Grow21 Benchmark — Evidence Bundle Index (Phase 3)

This index provides a comprehensive manifest of all files, empirical logs, diagnostic tables, source code, regression tests, and publication-ready figures originally packaged within the Grow21 Phase 3 Evidence Bundle (`Grow21_Phase_3_Model_Alignment_Evidence.zip`).

> [!NOTE]
> Under the project evidence exclusion policy, `Grow21_Phase_3_Model_Alignment_Evidence.zip` (located in `evidence/phase_3/`) is a local-only / release archive excluded from public Git tracking by `.gitignore`. All canonical experiment outputs, source files, and research documentation are tracked uncompressed in `outputs/experiments/phase_3/`, `src/`, and `docs/research/phase_3/`.

---

## 1. Primary Reports & Audit Documentation

| Canonical Project & Bundle Path | Type | Description |
| :--- | :---: | :--- |
| `outputs/audit/phase_3/PHASE_3_MODEL_ALIGNMENT_REPORT.md` | Markdown | Comprehensive Phase 3 technical audit report detailing mathematical mismatch resolution, formula parity audits, target reachability bisection, closed-loop comparisons, and statistical findings. |
| `outputs/audit/phase_3/BUNDLE_INDEX.md` | Markdown | This bundle index and artifact inventory. |
| `outputs/audit/phase_3/REPRODUCE.md` | Markdown | Step-by-step reproduction guide with environment specifications, exact terminal commands, and verification scripts. |
| `outputs/audit/phase_3/CHANGELOG_PHASE_3.md` | Markdown | Chronological changelog documenting all code changes, new modules, and regression test additions. |

---

## 2. Verified Empirical Evidence Artifacts

| Canonical Project & Bundle Path | Type | Records | Key Empirical Content |
| :--- | :---: | :---: | :--- |
| `outputs/phase_3/closed_loop/phase_3_closed_loop_metrics.csv` (and mirrored in `outputs/audit/phase_3/`) | CSV | 60 rows | Full metrics across 5 scenarios $\times$ 3 random seeds $\times$ 4 policies ($N = 216,000$ trials total). Records clean success rate, observed success rate, mean difficulty, bound saturation, reachability rate, reversals, and prediction gaps. |
| `outputs/phase_3/closed_loop/phase_3_closed_loop_summary.json` (and mirrored in `outputs/audit/phase_3/`) | JSON | 5 scenarios | Multi-seed aggregated means and standard deviations across seeds [42, 123, 999] for each scenario and policy. |
| `outputs/phase_3/closed_loop/phase_3_per_game_summary.csv` (and mirrored in `outputs/audit/phase_3/`) | CSV | 180 rows | Granular diagnostic breakdown by scenario, policy, and mini-game, verifying per-game clean success rates, difficulty occupancy, and target feasibility. |
| `outputs/phase_3/diagnostics/formula_parity_summary.json` (and mirrored in `outputs/audit/phase_3/`) | JSON | 1 summary | Summary of 7,938 grid parity evaluations across all 9 games, confirming maximum absolute error of strictly 0.00e+00. |
| `outputs/phase_3/diagnostics/formula_parity_results.csv` | CSV | 7,938 rows | Full grid evaluation table comparing canonical outcome model against simulator item-response equations. |
| `outputs/phase_3/closed_loop/phase_3_learner_mastery_summary.csv` | CSV | 43,200 rows | Granular per-learner, per-skill initial mastery, final mastery, and delta mastery across all 15 scenario-seed runs. |
| `outputs/archive/pre_phase_3_source/manifest.json` | JSON | 14 files | SHA-256 pre-Phase-3 baseline snapshot manifest mapping original relative paths to archived copies. |
| `outputs/audit/phase_2/calibration_bins.csv` | CSV | 450 rows | Phase 2 historical calibration bins required for regression test verification. |
| `outputs/audit/phase_2/CHANGELOG_PHASE_2.md` | Markdown | 1 file | Phase 2 historical changelog for documentation regression assertions. |
| `outputs/audit/phase_2/PHASE_2_AUDIT_REPORT.md` | Markdown | 1 file | Phase 2 authoritative audit report for documentation regression assertions. |
| `outputs/benchmark/metrics.csv` | CSV | 45 rows | Phase 2 historical benchmark metrics for ECE regression testing. |
| `outputs/closed_loop/closed_loop_summary.json` | JSON | 3 policies | Phase 2 closed-loop summary for reversal and safety regression checks. |
| `outputs/closed_loop/closed_loop_events.csv` | CSV | 10,800 rows | Phase 2 event-level log for reversal and safety recomputation assertions. |

---

## 3. Visualization Artifacts

| Canonical Project & Bundle Path | Type | Description |
| :--- | :---: | :--- |
| `outputs/phase_3/plots/difficulty_trajectories_comparison.png` (and mirrored in `outputs/audit/phase_3/`) | PNG | Sample trial-by-trial difficulty trajectories comparing Fixed (0.50), Streak Adaptive, Phase 2 Legacy Surrogate (clamped at $D_{\min}$), and Phase 3 Model-Aligned Target Adaptive (healthy tracking). |
| `outputs/phase_3/plots/target_reachability_and_saturation.png` (and mirrored in `outputs/audit/phase_3/`) | PNG | Bar charts illustrating the dramatic reduction in lower-bound saturation (from 93.4% down to 35.2%) and the corresponding surge in feasible target reachability (from 0.1% to 71.6%). |
| `outputs/phase_3/plots/policy_comparison_metrics.png` (and mirrored in `outputs/audit/phase_3/`) | PNG | Side-by-side comparison of latent clean task success vs. observed corrupted success, along with filtered non-zero direction reversals vs. cumulative difficulty variation. |
| `outputs/phase_3/plots/calibration_predicted_clean_success.png` (and mirrored in `outputs/audit/phase_3/`) | PNG | Reliability diagram comparing predicted clean task-success probability against empirical clean outcomes across 10 probability bins. |

---

## 4. Source Code & Configuration Manifest

| Bundle Relative Path | Role | Description |
| :--- | :---: | :--- |
| `src/grow21_simulation_benchmark/outcome_model.py` | New Source | Canonical item-response probability model, non-linear difficulty curve, composite mastery blending, and bounded bisection solver. |
| `src/grow21_simulation_benchmark/closed_loop.py` | Modified Source | Implements `ModelAlignedTargetAdaptiveController`, preserves `LegacySurrogateTargetAdaptiveController`, and provides 4-way closed-loop simulation comparison with full diagnostics. |
| `src/grow21_simulation_benchmark/simulator.py` | Modified Source | Refactored to delegate clean task-success probability and composite mastery directly to `outcome_model.py`. |
| `src/grow21_simulation_benchmark/tasks.py` | Modified Source | Refactored `calculate_difficulty_threshold` to delegate to `outcome_model.compute_difficulty_threshold`. |
| `src/grow21_simulation_benchmark/paths.py` | Modified Source | Centralized Phase 3 output directory definitions (`PHASE_3_OUTPUTS_DIR`, `PHASE_3_CLOSED_LOOP_DIR`, `PHASE_3_AUDIT_DIR`, etc.). |
| `src/grow21_simulation_benchmark/run_closed_loop.py` | Modified Source | CLI script updated to default to Phase 3 output directory to prevent overwriting Phase 2 evidence. |
| `src/grow21_simulation_benchmark/run_phase_3_experiments.py` | New Source | Complete multi-scenario, multi-seed experimental runner and plotting pipeline for Phase 3. |
| `configs/default.yaml` | Modified Config | Configures active `target_adaptive` controller with population priors and cleanly deprecates `gamma`/`nu_D` under `legacy_surrogate_target_adaptive`. |
| `configs/scenarios.yaml` | Config | 5 canonical simulation scenarios. |
| `tests/test_outcome_model_alignment.py` | New Tests | 10 deterministic tests verifying formula parity, non-linear curve, guess/lapse mechanics, bisection reachability, no-leakage, and bound contrast. |
| `tests/test_controller_constraints.py` | Tests | Closed-loop controller constraint and safety tests. |
| `tests/test_audit_regressions.py` | Tests | Phase 2 regression test suite verifying oscillations, calibration bins, fairness, and safety separation. |
| `tests/test_documentation_integrity.py` | Tests | Strict documentation integrity suite ensuring zero literal tabs and valid LaTeX syntax. |
