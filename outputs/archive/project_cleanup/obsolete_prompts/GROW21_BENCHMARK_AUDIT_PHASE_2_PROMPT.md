# Grow21 Benchmark Audit — Phase 2

## Role and constraints

Audit the existing standalone Grow21 synthetic simulation and benchmarking project. Do not rewrite the project, redesign the simulator, or change simulator assumptions merely to improve DKT results. Preserve the current implementation unless a specific, evidence-backed defect is found.

Do **not** integrate this module into the Grow21 frontend. This remains a standalone simulation and benchmarking project.

The current artifacts include benchmark metrics and plots, plus a closed-loop summary with a potential inconsistency: the README claims target adaptation reduces difficulty oscillations by 40%, while `closed_loop_summary.json` reports 3.117 oscillations for `target_adaptive` versus 2.348 for `streak_adaptive`. It also reports positive mean mastery gain for fixed difficulty (+0.0041) but negative gains for `streak_adaptive` (-0.0077) and `target_adaptive` (-0.0092). Treat these as issues to investigate, not conclusions to assume are correct.

## 1. Audit closed-loop results

Investigate why fixed difficulty has positive mean mastery gain while both adaptive controllers have negative mean mastery gain.

- Verify the mastery-gain calculation and its precise definition.
- Report initial and final latent mastery distributions, per-skill and per-learner changes, and the aggregation method behind `mastery_gain_mean`.
- Check whether policies receive comparable initial virtual learners, skill profiles, task opportunities, scenario settings, and random conditions. Prefer paired seeds/common random numbers where the existing design supports them; document limitations where it does not.
- Check whether the controller uses the intended predicted-success equation:
  `p_hat = sigmoid(logit(theta_k) - gamma[g,k] * D**nu_D)`
- Verify difficulty bounds, step-size limits, deadband, frustration constraints, and any break-limit rules against the documented equations/configuration.
- Verify the definition and calculation of difficulty changes/oscillations, including whether direction reversals, all changes, or some other event count is used.
- Determine whether the observed negative gains are a genuine consequence of the simulated policy/environment, a metric or implementation bug, or an unfair comparison. Do not assume in advance which explanation is correct.

## 2. Resolve report inconsistencies and regenerate results

- Reconcile the README's claim that target adaptation reduces oscillations by 40% with `closed_loop_summary.json` and the underlying event-level results.
- Treat summary JSON and existing reports as artifacts to audit, not automatically correct sources of truth.
- Recompute closed-loop summary statistics from verified event-level outputs.
- Regenerate summary tables and plots from the verified outputs. Ensure all plots and report statements use the same regenerated data.
- If the existing event-level outputs are insufficient to verify a claim, rerun the relevant experiment and clearly state what was rerun, with which seeds/configuration, and why.
- Update stale or contradictory README/report statements. Preserve old results or record a clear change log rather than silently overwriting evidence.

## 3. Audit calibration and offline predictive evaluation

- Report the observation count in every calibration bin for every model/scenario, along with mean predicted probability and observed positive rate.
- Verify all models use identical test examples, targets, and consistent bin definitions.
- Investigate high-probability bins under `distribution_shift`, including whether the apparent jumps are driven by small bin counts.
- Report log loss, Brier score, ROC-AUC, accuracy/balanced accuracy where appropriate, ECE, and uncertainty across random seeds.
- Include per-seed metrics and paired model differences on the same held-out learners. Do not claim statistical significance unless an appropriate paired analysis is performed.
- Make calibration plots interpretable: state the binning method, show counts (or provide a companion table), and use consistent bins for model comparisons.
- If ECE implementation or binning is questionable, audit and correct it, documenting the change and recomputing affected outputs.

## 4. Audit reliability gating and causal timing

- Verify the reliability gate uses only information available before the outcome being predicted.
- Confirm event `t`'s outcome cannot affect its own prediction. Inspect the actual prediction/update order and add or strengthen a test for this no-leakage invariant.
- Confirm missing modalities are handled according to the documented convention; unavailable modality estimates must not be treated as perfect confidence or as a fabricated zero uncertainty.
- Check for leakage from simulator-only clean outcomes, true mastery, true corruption probabilities, future events, post-outcome latency, or post-outcome retries.
- Compare vanilla DKT and reliability-gated DKT under controlled sensor-corruption levels, including:
  1. accurately estimated reliability,
  2. noisy reliability estimates,
  3. systematically miscalibrated reliability estimates,
  4. missing modality information.
- Keep model architecture, data splits, training budget, stopping protocol, test examples, and random seeds comparable. Report where exact comparability is not possible.
- Do not change the simulator to make DKT or reliability-gated DKT win. A negative or negligible result is valid.

## 5. Tests, changes, and scientific integrity

- Run the existing test suite.
- Add regression tests for any confirmed defects, especially mastery-gain calculation, oscillation calculation, fairness/common-random-number assumptions, calibration bin counts, missing-modality handling, and prediction leakage where relevant.
- Document every code/config/report change and the evidence motivating it.
- Distinguish confirmed findings, plausible explanations, unresolved questions, and recommendations.
- Clearly state that the benchmark uses synthetic virtual learners and is not clinical/developmental validation, is not evidence of efficacy for children with Trisomy 21, and is not trained or validated on real Grow21 child data.
- Do not tune the simulator merely to improve model rankings.

## 6. Required deliverables

### A. Audit report

Create a readable Markdown report at:

`outputs/audit/PHASE_2_AUDIT_REPORT.md`

The report must include:

1. Executive summary with the most important confirmed findings.
2. Exact commands and configurations used, seed list, runtime, environment/dependency versions where available.
3. Closed-loop audit: mastery-gain definitions and recalculated results; initial/final mastery summaries; policy comparability; controller equation/constraint checks; oscillation definition and corrected counts.
4. Reconciliation of the README versus JSON/results, with corrected claims.
5. Calibration audit: per-bin counts, per-seed metrics, ECE method, distribution-shift investigation, and uncertainty/paired differences where implemented.
6. Reliability-gating audit: causal timing, missing-modality behavior, corruption/reliability conditions, tests, and results.
7. Tests passed/failed and any test that could not be run.
8. A change log listing every modified/added file and why.
9. Confirmed findings, unresolved questions, limitations, and recommended next experiments.
10. A clear statement that this is synthetic-only research tooling, not clinical or developmental validation.

Do not fabricate results. If an audit cannot be completed because an output or code path is missing, say exactly what is missing and what remains unverified.

### B. Evidence bundle ZIP for the next review

Create:

`outputs/audit/Grow21_Phase_2_Audit_Evidence.zip`

This ZIP is required. It must contain the **smallest useful set of source files and artifacts needed for an independent reviewer to understand and verify the audit and to plan the next phase without requesting the entire repository**.

Include these items where they exist:

1. `PHASE_2_AUDIT_REPORT.md`.
2. Relevant source files needed to inspect the audited logic, especially:
   - closed-loop controller/simulation implementation (e.g. `closed_loop.py`);
   - virtual learner state-transition and outcome-generation logic (e.g. `simulator.py`);
   - prediction/evaluation/calibration logic (e.g. `evaluate.py`, `plots.py`);
   - reliability calculation and reliability-gated DKT update (e.g. `reliability.py`, `models/reliability_gated_dkt.py`, `models/dkt.py`);
   - logistic baseline/features only if needed to verify shared test sets or metric comparisons;
   - config files controlling scenarios, policies, difficulty limits, frustration/break constraints, seeds, and training/evaluation settings;
   - the specific tests relevant to the audited behavior.
3. Verified, compact data artifacts sufficient to reproduce reported summaries:
   - event-level closed-loop logs or a compact per-learner/per-session export containing policy, seed, learner ID, initial/final mastery or enough information to recompute it, task/difficulty history, outcomes, constraint flags, and values needed to calculate oscillations;
   - per-seed and per-scenario model metrics;
   - calibration-bin table with bin edges, counts, mean predictions, and observed rates;
   - any paired-difference or bootstrap results cited in the report.
4. Regenerated summary JSON/CSV files and the exact plots used in the report.
5. A short `BUNDLE_INDEX.md` explaining each included file, why it is needed, how it supports the report, and which important files were intentionally omitted.
6. A `REPRODUCE.md` with exact commands to regenerate the audited summaries from the included files/configs. Include dependency/setup notes. If full regeneration requires large data or omitted files, state that plainly and include the smallest relevant reproduction command or script.
7. A `CHANGELOG_PHASE_2.md` listing changed files, changes, reasons, and whether each change affects code, configuration, metrics, plots, or prose.

Do not include secrets, credentials, caches, virtual environments, model checkpoints, or huge raw datasets unless essential. Prefer compact CSV/JSON summaries and source files. Do not place unrelated application/frontend files in the ZIP. Avoid including duplicate copies of the same large output. Preserve paths inside the ZIP so file relationships remain clear.

Before finishing, validate the ZIP:
- list its contents;
- confirm the report, bundle index, reproduction instructions, change log, relevant source/config/test files, and verified outputs are present;
- ensure the ZIP opens successfully;
- report its size and file count.
If a required item does not exist or cannot be included, list it as missing and explain why.

### C. Final response to the user

At completion, provide:
- a concise audit outcome;
- the path/link to the audit report;
- the path/link to `Grow21_Phase_2_Audit_Evidence.zip`;
- test results, exact commands used, and runtime;
- a short list of confirmed fixes and unresolved questions;
- any deviations from this prompt.

Do not claim the audit or ZIP is complete unless the files actually exist and the ZIP has been checked.

## Execution order

1. Inspect the current repository, configs, outputs, and tests. Briefly state the audit plan and likely files to inspect.
2. Trace closed-loop metric calculations and controller implementation before editing report prose.
3. Recompute or rerun closed-loop outputs where needed; resolve the mastery-gain and oscillation discrepancies.
4. Audit offline metrics/calibration and reliability-gating causal timing.
5. Add regression tests, run the existing and new tests, and record failures honestly.
6. Regenerate summaries, calibration tables, plots, README/report claims, and the audit report from verified results.
7. Build and validate the evidence ZIP as specified above.
8. Provide the final response with actual paths, results, and unresolved issues.

Do not integrate the module into the Grow21 frontend.
