# Grow21 Benchmark — Evidence Bundle Index (Phase 2)

This index provides a comprehensive manifest of all files, logs, configs, source code, regression tests, and verified empirical summaries originally packaged within the Grow21 Phase 2 Evidence Bundle (`Grow21_Phase_2_Audit_Evidence.zip`).

> [!NOTE]
> Under the project evidence exclusion policy, `Grow21_Phase_2_Audit_Evidence.zip` (located in `evidence/phase_2/`) is a local-only / release archive excluded from public Git tracking by `.gitignore`. All canonical experiment files, source code, configs, and reports listed here exist directly uncompressed in the repository structure (`outputs/experiments/phase_2/`, `src/`, `configs/`, and `docs/research/phase_2/`).

---

## 1. Primary Reports & Audit Documentation

| Canonical Project & Bundle Path | Type | Description |
| :--- | :---: | :--- |
| `outputs/audit/phase_2/PHASE_2_AUDIT_REPORT.md` | Markdown | Comprehensive Phase 2 technical audit report detailing mathematical investigations, bound saturation findings, metric reconciliations, calibration audits, and statistical claims. |
| `outputs/audit/phase_2/BUNDLE_INDEX.md` | Markdown | This bundle index and artifact inventory. |
| `outputs/audit/phase_2/REPRODUCE.md` | Markdown | Step-by-step reproduction guide with environment specifications, exact terminal commands, and verification scripts. |
| `outputs/audit/phase_2/CHANGELOG_PHASE_2.md` | Markdown | Chronological changelog documenting all fixes, architectural reorganizations, and regression test additions. |

---

## 2. Verified Empirical Evidence Artifacts

| Canonical Project & Bundle Path | Type | Format / Records | Key Empirical Content |
| :--- | :---: | :---: | :--- |
| `outputs/audit/phase_2/calibration_bins.csv` | CSV | 450 rows | 10-bin calibration data for all 5 scenarios, 3 seeds, and 3 models on the canonical 500-learner cohort ($N = 4,500$ test events per slice). Bin counts sum exactly to 4,500. |
| `outputs/audit/phase_2/reliability_sweep_summary.json` | JSON | 4 conditions | Controlled 4-level sensor corruption sweep metrics with 95% learner-cluster bootstrap confidence intervals. |
| `outputs/audit/phase_2/paired_loss_evidence.csv` | CSV | 216 rows | Per-learner held-out test log losses across all 4 reliability sweep conditions, enabling independent recalculation of bootstrap confidence intervals. |
| `outputs/closed_loop/learner_mastery_summary.csv` (and mirrored in `outputs/audit/phase_2/`) | CSV | 1,080 rows | Granular per-learner, per-skill initial mastery, final mastery, delta mastery, primary practice counts, and secondary transfer counts. |
| `outputs/audit/phase_2/predictor_vs_simulator_comparison.csv` | CSV | 108 rows | Empirical diagnostic comparing the controller surrogate model predicted probability against the simulator IRT clean success probability across skills, difficulties, and mastery levels. |
| `outputs/benchmark/metrics.csv` | CSV | 45 rows | Offline benchmark results across 5 scenario families and 3 random seeds (Log Loss, ROC-AUC, Brier Score, Accuracy, ECE, test event counts). |
| `outputs/benchmark/summary.json` | JSON | 5 scenarios | Multi-seed benchmark aggregated means and standard deviations. |
| `outputs/closed_loop/closed_loop_events.csv` | CSV | 10,800 rows | Trial-by-trial closed-loop interaction records (60 learners $\times$ 60 trials $\times$ 3 controller policies) recording difficulty, time gaps, observed/clean outcomes, and bound flags. |
| `outputs/closed_loop/closed_loop_summary.json` | JSON | 3 policies | Performance metrics for Fixed, Streak, and Target controllers including immediate reversals (`immediate_direction_reversals_mean/std`), filtered non-zero reversals (`filtered_direction_reversals_mean/std`), legacy alias (`direction_reversals_mean/std`), total variation, and bound saturation rates. |

---

## 3. Visualization Artifacts

| Canonical Project & Bundle Path | Type | Description |
| :--- | :---: | :--- |
| `outputs/closed_loop/plots/difficulty_trajectories.png` | PNG | Sample trial-by-trial difficulty paths illustrating the target-adaptive controller's immediate clamp and 93.4% flatline saturation at $D_{\min} = 0.08$. |
| `outputs/closed_loop/plots/closed_loop_comparison.png` | PNG | Bar chart comparing mastery gains on primary practiced skills and controller difficulty dynamics (reversals vs. total variation). |
| `outputs/benchmark/plots/scenario_comparison.png` | PNG | Benchmark comparison of Log Loss and ROC-AUC across all 5 scenario families for Logistic Regression, Vanilla DKT, and Reliability-Gated DKT. |
| `outputs/benchmark/plots/calibration_base_heterogeneous.png` | PNG | Calibration reliability curve comparing predicted probabilities to empirical accuracy in standard heterogeneous learners. |
| `outputs/benchmark/plots/calibration_reliability_stress.png` | PNG | Calibration reliability curve under elevated sensor noise and missingness. |
| `outputs/benchmark/plots/calibration_simple_dynamics.png` | PNG | Calibration reliability curve in stationary, low-progression environments. |
| `outputs/benchmark/plots/calibration_temporal_dynamics.png` | PNG | Calibration reliability curve under fast skill learning and session-gap forgetting. |
| `outputs/benchmark/plots/calibration_distribution_shift.png` | PNG | Calibration reliability curve under severe population distribution shift, showing upper probability sparsity. |

---

## 4. Source Code & Configuration Manifest

| Directory Path | Description |
| :--- | :--- |
| `configs/default.yaml` | Task registry, game parameterizations, skill mappings, simulation hyperparameters, and controller configurations. |
| `configs/scenarios.yaml` | Scenario definitions for base, stress, stationary, temporal, and distribution shift cohorts. |
| `src/grow21_simulation_benchmark/` | Complete package source code: |
| `├── __init__.py` | Package version and exports. |
| `├── paths.py` | Centralized canonical path definitions for all output categories. |
| `├── tasks.py` | TaskRegistry, GameDefinition, ModalityDefinition, and YAML loading. |
| `├── simulator.py` | Virtual learner profile generation, IRT response logic, forgetting, and transfer updates. |
| `├── reliability.py` | Product-rule sensor uncertainty engine, noise sampling, and missingness logic. |
| `├── features.py` | Strictly causal feature extraction for Logistic Regression baseline. |
| `├── dataset.py` | Learner-disjoint sequence generators and PyTorch tensor datasets. |
| `├── train.py` | Training loops for Logistic Regression, Vanilla DKT, and Reliability-Gated DKT. |
| `├── evaluate.py` | Metric calculations, calibration table generation, paired difference testing, and bootstrap CIs. |
| `├── plots.py` | Calibration curve and cross-scenario plotting utilities. |
| `├── run_benchmark.py` | CLI runner for multi-scenario, multi-seed offline benchmarking. |
| `├── run_closed_loop.py` | CLI runner for paired closed-loop adaptive simulation experiments. |
| `├── audit_calibration_bins.py` | CLI runner for calibration exports, 4-level reliability sweep, and paired loss evidence. |
| `└── models/` | Model architectures: |
| `    ├── logistic_baseline.py` | L2-regularized logistic regression baseline with standardized features. |
| `    ├── dkt.py` | GRU-based recurrent Deep Knowledge Tracing with causal timing. |
| `    └── reliability_gated_dkt.py` | Grow21 Reliability-Gated DKT applying observation uncertainty gating. |
| `tests/` | Complete deterministic test suite: |
| `├── test_simulator.py` | Latent mastery and forgetting unit tests. |
| `├── test_reliability.py` | Uncertainty product rule and sensor dropout unit tests. |
| `├── test_no_leakage.py` | Strict causal timing and information isolation unit tests. |
| `├── test_model_shapes.py` | PyTorch model tensor shape and boundary behavior tests. |
| `├── test_controller_constraints.py` | Frustration safety drop and session break limit tests. |
| `├── test_audit_regressions.py` | 13 targeted regression tests for Phase 2 audit and reconciliation findings. |
| `└── test_documentation_integrity.py` | Deterministic verification of zero invalid ASCII control characters across all text files. |

---

## 5. Verification & Integrity Checklist

- [x] All 32 unit and regression tests pass deterministically (`python -m pytest -v tests`).
- [x] Evidence ZIP contains zero Python cache directories (`__pycache__`), virtual environments (`.venv`), or git history.
- [x] Evidence ZIP contains zero nested self-references.
- [x] All summary metrics and reported numbers match underlying CSV and JSON records.
- [x] Recomputed ECE from `calibration_bins.csv` matches `metrics.csv`.
- [x] Paired bootstrap confidence intervals independently verified from `paired_loss_evidence.csv`.
- [x] Archive manifest with verified SHA-256 digests preserved in `outputs/archive/pre_correction_pass/manifest.json` (payloads preserved in workspace; manifest bundled in ZIP).
