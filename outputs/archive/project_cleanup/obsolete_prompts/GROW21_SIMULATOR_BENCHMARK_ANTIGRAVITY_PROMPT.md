# Grow21 Synthetic Learner Simulator + Knowledge-Tracing Benchmark

## Role and execution contract

You are the implementation agent. Build a standalone Python simulation and benchmarking module for the Grow21 college project. The reasoning, experiment choices, and interpretation will be reviewed separately. Do not integrate with or modify the frontend/app. First inspect the existing workspace and preserve unrelated files. If a suitable project structure already exists, work within it; otherwise create a dedicated `grow21_simulation_benchmark/` directory.

Do not fabricate results, claim DKT must beat logistic regression, or describe synthetic evaluation as validation on real children. Build reproducible code, run it, report actual outputs, and make failures explicit.

## Research-derived principles to follow

1. Pagonis et al. (2024), *Analysis of Knowledge Tracing Performance on Synthesised Student Data*: fitting a marginal score distribution alone does not preserve temporal learning dependencies. Their sequence-reuse/noise approach preserved some sequence dependence better than independent grade sampling, but synthetic training did not reliably improve downstream performance. Since Grow21 has no real interaction paths to resample, implement a latent-state agent simulator that explicitly generates sequential learning, forgetting, task difficulty, and observation noise. Do not use independent random correctness labels as the main generator.
   - https://arxiv.org/abs/2401.16832
2. Zhang & MacLellan (2021), *Investigating Knowledge Tracing Models using Simulated Students*: simulated learner agents can support controlled comparisons and expose model/implementation bugs before human studies; their study compared BKT and a streak strategy inside a tutoring loop. Use simulation for both offline prediction tests and a later closed-loop policy test, and add strong correctness tests.
   - https://chrismaclellan.com/media/publications/Investigating-Knowledge-Tracing-Models-using-Simulated-Students.pdf
3. Gervet et al. (2020), *When Is Deep Learning the Best Approach to Knowledge Tracing?*: a strong logistic-regression model may win on moderate/smaller datasets, while DKT can benefit when data is larger and temporal sequence information is useful. Calibration matters. Therefore compare a competent logistic baseline, DKT, and reliability-gated DKT; test several scenario families and training-data sizes instead of selecting one favourable split.
   - https://jedm.educationaldatamining.org/index.php/JEDM/article/view/451
4. Pavlik, Cen & Koedinger (2009), *Performance Factors Analysis*: a feature-based logistic approach that tracks successes and failures is a relevant precedent for a strong interpretable baseline.
   - https://doi.org/10.3233/978-1-60750-028-5-531

These sources inform the experiment design; they do not specify clinically correct parameter values for Grow21 or prove that a synthetic learner distribution matches children with Trisomy 21.

## Project objective

Create a reproducible simulator and fair benchmark to test:
- Logistic regression using well-designed history features.
- Vanilla sequential DKT.
- Reliability-gated DKT using Grow21's update principle.

Core Grow21 reliability-gated hidden-state update:

`h_t = h_(t-1) + rho_t * (hbar_t - h_(t-1))`

where `hbar_t` is the candidate recurrent update and `rho_t` is estimated from only the uncertainty sources active for that event. For example, `rho_t = product_m(1 - p_hat_t_m)`. Missing modalities must not be imputed as perfect confidence or as a zero-valued uncertainty estimate. If no modeled uncertainty source applies, document the configured convention for the reliability value.

The benchmark must be designed so DKT is allowed to lose. No generator equation may simply reuse Grow21's current success predictor `sigmoid(logit(theta) - gamma * D**nu)` as the only mechanism that generates outcome labels. Use a distinct data-generating process and report the assumptions.

## Simulation design

### Virtual learners and skills

Implement reproducible virtual learner agents with heterogeneous parameters:
- Initial mastery by learner and skill, bounded away from exactly 0 and 1.
- Different learning rates by learner and skill.
- Skill-specific forgetting/retention parameters and variable time gaps between sessions.
- Variation in response noise, execution reliability, and engagement/fatigue patterns.
- At least 9 game IDs reflecting Grow21: `trace_glow`, `bubble_pop_brigade`, `block_stacker`, `echo_safari`, `picture_sentence_strips`, `audio_search`, `category_sorting`, `visual_flip_match`, `pattern_train`.
- At least one task/skill mapping configuration file so skill assignment can later be changed. Do not hardwire the general Grow21 design as a permanent one-game-equals-one-skill rule; cross-game/shared skill relationships should be configurable.

Keep the first iteration small and inspectable. Use a configuration file, not constants buried in code. Suggested initial run: 800–1,200 virtual learners total, roughly 60–100 events per learner, 3 random seeds for a smoke benchmark; expand to 5 or more seeds only once the pipeline is reliable and runtime is acceptable.

### Task scheduling and difficulty

For the primary offline prediction benchmark, generate the task/game/difficulty sequence independently of the evaluated model (e.g., a seeded stratified/random curriculum with some repeats and gaps). All models must see the same test events and task metadata. Record abstract difficulty `D in [0,1]` and use a configurable game-specific mapping table for game parameters, even if visual parameters are not rendered.

Use a different success-generation equation from Grow21's predictor. A suitable proposed simulator is a bounded, item-response-style mechanism with a guessing floor and lapse/slip ceiling, e.g.:

`p_clean = guess_gk + (1 - guess_gk - lapse_gk) * sigmoid(discrimination_gk * (mastery_ik - difficulty_threshold_gk(D)))`

This is a simulator proposal, not a claim that this exact form is empirically correct for Grow21. Allow game/skill-specific discrimination and difficulty thresholds. You may use a non-linear difficulty threshold, but do not copy the controller's exact `gamma * D**nu` predictor into the generator.

Generate the clean latent outcome from `p_clean`. Then apply observation/execution corruption using active task modalities. Motor and ASR corruption must be game-dependent; Echo Safari's current design accepts detected vocalization and must not automatically be treated as an exact-speech-recognition task. For each uncertainty source, maintain:
- Its simulated true corruption probability (hidden diagnostic only).
- Its noisy estimated probability available to Grow21 (`p_hat`), which may be well calibrated, noisy, or biased in stress scenarios.
- Whether the modality is available for that event.

Compute estimated reliability from the available estimated uncertainty sources using the configured Grow21 product rule. Never expose simulator-only true mastery, clean outcome, or true corruption probabilities as model input features. Keep them only for diagnostic evaluation.

### Sequential state transition

After an attempt, evolve hidden skill mastery with learner-specific learning and time-dependent forgetting. The update should have a clear documented assumption and noise, for example practice increments that shrink as mastery approaches 1, plus a forgetting reduction based on the simulated inter-session gap. Successful and unsuccessful practice may have different learning effects. Support a scenario with little or no learning so the model cannot exploit a guaranteed monotonic pattern.

Do not claim the transition equation reflects actual children; it defines controllable virtual-agent assumptions. Use at least these configurable scenario families:
1. `base_heterogeneous`: varied prior mastery, learning rates, gaps, skill difficulty and mild noise.
2. `reliability_stress`: higher execution corruption, missing modality values, and uncertainty estimates with calibration error.
3. `simple_dynamics`: more stationary/linear relationships where logistic regression may be competitive.
4. `temporal_dynamics`: strong order/session-gap/learning-history effects that may favour sequential modeling.
5. `distribution_shift`: train on one range of learner parameters and evaluate on a deliberately shifted range; report this separately from in-distribution test metrics.

Do not tune simulator parameters to force DKT to win. Record every scenario's actual parameters and random seed.

## Models to implement

### A. Logistic regression baseline

Use scikit-learn logistic regression with properly scaled numerical features and categorical encodings fitted on training data only. Make it a competent historical-feature baseline, not a straw man. Include available pre-prediction task features and history features computed strictly from the learner's preceding events, e.g.:
- Game/skill/task identifiers and current difficulty.
- Prior attempts, successes, failures, and smoothed success rate for the relevant skill.
- Recent outcomes/windowed success rate and last outcome when available.
- Time since prior practice/session gap.
- Current known modality/availability indicators and pre-outcome estimated risk features if available before the response.
- Missingness indicators where appropriate.

Never include the target response, post-response latency, post-response retries, or any features calculated using future events for the prediction of that response.

### B. Vanilla DKT

Implement a modest PyTorch recurrent model (GRU is acceptable) that consumes sequential event representations and predicts the next response. Condition each prediction on the next task's known pre-response metadata (at minimum skill, game and difficulty) plus the current history state. Keep model size small enough for CPU execution. Train with binary cross-entropy and validation-based early stopping.

### C. Reliability-gated DKT

Use the same data splits, inputs, architecture capacity, optimization method, and stopping protocol as vanilla DKT wherever possible. Apply the Grow21 reliability gate to the proposed state update after observing the event. Use only `rho_hat` computed from available/noisy estimated uncertainties. To keep timing causal, predict response `y_t` from the history through `t-1` and the pre-response task metadata for event `t`; only after the prediction/loss is formed should event `t` and its `rho_hat` update the state for the next event.

Do not accidentally allow the current outcome to affect its own prediction. Write an explicit unit test for this no-leakage invariant.

The three-model comparison is required. The vanilla DKT ablation is needed to determine whether reliability gating adds value beyond sequence learning itself.

## Prediction target and fair comparison

Primary task: predict the *observed* next interaction outcome (success/failure) from information available before that outcome occurs. It is available as a label in a simulated log and is the realistic primary prediction target. Keep simulator-only clean outcomes and true mastery hidden from model inputs.

Secondary diagnostic: use simulator-only truth to inspect whether reliability gating is associated with improved state/prediction behaviour under noise. Clearly label any metric that uses `y_clean`, true mastery, or true reliability as an oracle diagnostic, not a deployable training target. Do not silently train one model on hidden truth while other models are trained on observed labels.

Use learner-disjoint train/validation/test splits. No events from a test learner may occur in the training or validation sets. Fit feature scaling and model selection on training/validation only. Test all models on the same events, with the same targets. Keep offline predictive evaluation distinct from closed-loop adaptation evaluation.

## Metrics and analyses

Report at least:
- Log loss (primary probabilistic metric).
- Brier score.
- ROC-AUC where both classes are present.
- Accuracy and balanced accuracy as secondary metrics.
- Calibration table/plot (predicted-probability bins versus observed frequency).
- Per-scenario breakdown, including noise/missing-modality conditions.
- Results across multiple random seeds: mean and standard deviation, and paired model differences on the same test split.
- Training data-size sensitivity (optional after the first reliable run, e.g. 25%, 50%, 100% of training learners).
- Model runtime and parameter count as context, not as an accuracy substitute.

Do not claim statistical significance unless an appropriate paired analysis is implemented. Prefer paired bootstrap confidence intervals over test learners or per-learner aggregate losses if feasible. Avoid treating individual rows from one learner as independent samples when calculating uncertainty.

## Closed-loop simulation (second phase, after the offline benchmark is correct)

Add a separate experiment in which each controller strategy chooses task/difficulty using the same initial virtual-learner profiles and comparable random streams. The environment then generates the next outcome and updates latent mastery. Compare:
- Learning/mastery progression according to the simulator's own latent state.
- Observed success and latent clean success.
- Number and magnitude of difficulty changes / oscillations.
- Frustration constraint violations and break-limit violations (should be zero if correctly implemented).
- Task exposures and performance for each scenario family.

Make clear that the simulator's own state transition defines the measured learning effect; it is not evidence of real-world learning efficacy. Use paired seeds/common random numbers as practical to reduce Monte Carlo noise. This closed-loop experiment must not replace the simpler fixed-task offline benchmark.

## Suggested project structure

Adapt to the existing repository, but aim for clear separation like:

```text
grow21_simulation_benchmark/
  README.md
  requirements.txt or pyproject.toml
  configs/
    default.yaml
    scenarios.yaml
  src/grow21_simulation_benchmark/
    simulator.py
    tasks.py
    reliability.py
    dataset.py
    features.py
    models/logistic_baseline.py
    models/dkt.py
    models/reliability_gated_dkt.py
    train.py
    evaluate.py
    closed_loop.py
    plots.py
  tests/
    test_simulator.py
    test_reliability.py
    test_no_leakage.py
    test_model_shapes.py
    test_controller_constraints.py
  outputs/
    metrics.csv
    summary.json
    plots/
```

Don't commit generated multi-megabyte datasets unless appropriate; make them reproducible from saved configs and seeds. Store model checkpoints and generated output outside source modules.

## Acceptance criteria

1. A single documented command generates synthetic data with a fixed seed.
2. A single documented command runs the offline benchmark end-to-end on a small CPU-friendly config.
3. Repeating a run with the same seed reproduces the dataset and materially identical metrics within deterministic framework settings.
4. Tests verify latent mastery and probabilities remain in range, difficulty remains in `[0,1]`, active/missing modality logic is correct, reliability is bounded, and frustration/fatigue constraints are respected if those components are part of the first scope.
5. A leakage test proves event `t`'s outcome cannot influence its own prediction.
6. All three models train and evaluate against identical held-out events with learner-disjoint splits.
7. The final report includes actual metrics, plots, scenario configuration, seeds, known limitations, and whether DKT truly beat logistic regression. If it does not, report that honestly.
8. The README clearly states this is synthetic simulation, not clinical/developmental validation and not trained on real Grow21 children.
9. Do not modify the application/frontend or claim that this is integrated into the app.

## Execution order

1. Inspect workspace and report the planned file changes briefly.
2. Implement the simulator, schema/configs, and unit tests first.
3. Run tests and a small smoke-data generation.
4. Implement all three models and enforce causal prediction timing.
5. Run the small benchmark, debug issues, then run the configured multi-seed benchmark if runtime is acceptable.
6. Produce metrics tables, plots, and a short research-grounded report.
7. Only then begin the optional closed-loop experiment.

At completion, provide: changed-file summary, exact commands used, tests passed/failed, runtime, links/paths to plots and reports, actual benchmark results, deviations from this brief, and any unresolved issues. Do not proceed to app integration.
