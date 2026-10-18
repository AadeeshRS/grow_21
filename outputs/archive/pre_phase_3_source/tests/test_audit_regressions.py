"""Regression tests covering defects and invariants identified in the Phase 2 audit."""

import math
from pathlib import Path
import numpy as np
import pytest

from grow21_simulation_benchmark.closed_loop import (
    FixedDifficultyController,
    Grow21ModelBasedAdaptiveController,
    StreakAdaptiveController,
    calculate_direction_reversals,
    calculate_immediate_reversals,
    calculate_filtered_direction_reversals,
    run_closed_loop_comparison,
)
from grow21_simulation_benchmark.evaluate import compute_calibration_table
from grow21_simulation_benchmark.models import ReliabilityGatedDKT, VanillaDKT
from grow21_simulation_benchmark.tasks import TaskRegistry
import torch


@pytest.fixture
def registry():
    cfg_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    return TaskRegistry.from_yaml(cfg_path)


def test_regression_oscillation_vs_variation():
    """Confirms direction reversals (oscillations) and total variation are computed correctly using production helper."""
    # Sequence with 4 monotonic steps up: 0.1, 0.2, 0.3, 0.4
    diffs_monotonic = [0.1, 0.2, 0.3, 0.4]
    reversals = calculate_direction_reversals(diffs_monotonic)
    variation = sum(abs(diffs_monotonic[i] - diffs_monotonic[i - 1]) for i in range(1, len(diffs_monotonic)))

    assert reversals == 0, "Monotonic changes must have 0 direction reversals"
    assert pytest.approx(variation, abs=1e-5) == 0.3

    # Sequence with 2 reversals: 0.2 -> 0.4 -> 0.2 -> 0.4
    diffs_oscillating = [0.2, 0.4, 0.2, 0.4]
    reversals_osc = calculate_direction_reversals(diffs_oscillating)
    variation_osc = sum(abs(diffs_oscillating[i] - diffs_oscillating[i - 1]) for i in range(1, len(diffs_oscillating)))

    assert reversals_osc == 2, "Expected 2 direction reversals"
    assert pytest.approx(variation_osc, abs=1e-5) == 0.6


def test_production_direction_reversals_edge_cases():
    """Directly tests production functions calculate_immediate_reversals and calculate_filtered_direction_reversals on critical edge cases."""
    # 1. Strictly monotonic sequence
    assert calculate_immediate_reversals([0.1, 0.2, 0.3, 0.4, 0.5]) == 0
    assert calculate_immediate_reversals([0.5, 0.4, 0.3, 0.2, 0.1]) == 0
    assert calculate_filtered_direction_reversals([0.1, 0.2, 0.3, 0.4, 0.5]) == 0
    assert calculate_filtered_direction_reversals([0.5, 0.4, 0.3, 0.2, 0.1]) == 0

    # 2. Immediate reversals (zig-zag oscillation without flat steps)
    assert calculate_immediate_reversals([0.2, 0.4, 0.2, 0.4, 0.2]) == 3
    assert calculate_filtered_direction_reversals([0.2, 0.4, 0.2, 0.4, 0.2]) == 3
    assert calculate_direction_reversals([0.2, 0.4, 0.2, 0.4, 0.2]) == 3

    # 3. Handling zero steps (flat transitions):
    # [0.2 -> 0.4 (up), 0.4 -> 0.4 (zero), 0.4 -> 0.4 (zero), 0.4 -> 0.2 (down)]
    # - Immediate reversals: zero delta produces 0.0 * delta = 0, resetting immediate state -> 0 immediate reversals
    # - Filtered reversals: zero transitions are filtered out first, leaving [+0.2, -0.2] -> 1 filtered reversal
    assert calculate_immediate_reversals([0.2, 0.4, 0.4, 0.4, 0.2]) == 0
    assert calculate_filtered_direction_reversals([0.2, 0.4, 0.4, 0.4, 0.2]) == 1
    assert calculate_direction_reversals([0.2, 0.4, 0.4, 0.4, 0.2]) == 1

    # 4. Flat-bound sequence (clamped at boundary)
    assert calculate_immediate_reversals([0.08, 0.08, 0.08, 0.08, 0.08]) == 0
    assert calculate_filtered_direction_reversals([0.08, 0.08, 0.08, 0.08, 0.08]) == 0


def test_regression_calibration_bin_sum_equals_total_events():
    """Confirms that calibration bin counts sum exactly to the number of test observations."""
    rng = np.random.RandomState(42)
    N = 1450
    y_true = rng.randint(0, 2, size=N)
    y_prob = rng.uniform(0.0, 1.0, size=N)

    bins, ece = compute_calibration_table(y_true, y_prob, num_bins=10)
    total_binned = sum(b["count"] for b in bins)
    assert total_binned == N, f"Bin count sum {total_binned} != total samples {N}"
    assert 0.0 <= ece <= 1.0


def test_regression_fairness_common_random_numbers(registry):
    """Verifies that all three controllers receive identical virtual learner profiles under paired seeds."""
    scenario_cfg = {
        "description": "audit_fairness_test",
        "initial_mastery_mean": 0.45,
        "initial_mastery_std": 0.15,
        "learning_rate_mean": 0.08,
        "learning_rate_std": 0.02,
        "forgetting_rate_mean": 0.02,
        "forgetting_rate_std": 0.01,
        "corruption_multiplier": 1.0,
    }

    results, _, _ = run_closed_loop_comparison(
        task_registry=registry,
        scenario_cfg=scenario_cfg,
        num_learners=5,
        trials_per_learner=20,
        seed=42,
    )

    fixed_inits = [r.initial_mastery_mean for r in results["fixed_difficulty"]]
    streak_inits = [r.initial_mastery_mean for r in results["streak_adaptive"]]
    target_inits = [r.initial_mastery_mean for r in results["target_adaptive"]]

    # Invariant: Every learner's initial mastery must match across all three policies
    assert fixed_inits == streak_inits == target_inits, (
        "Initial learner profiles must be strictly identical across controllers"
    )


def test_target_success_reachability_and_bound_saturation(registry):
    """Verifies target reachability conditions (unreachable-high, unreachable-low, reachable) and bound saturation."""
    controller = Grow21ModelBasedAdaptiveController(
        task_registry=registry,
        target_success_prob=0.75,
        initial_difficulty=0.45,
        gamma=2.4,
        nu_D=1.0,
        min_difficulty=0.08,
        max_difficulty=0.95,
        deadband=0.04,
        max_step_change=0.08,
    )

    # 1. Unreachable-high target: learner with low mastery theta = 0.45, target = 0.75
    # For theta=0.45, logit(0.45) = -0.2007.
    # At D_min=0.08: logit_diff = -0.2007 - 2.4*(0.08) = -0.3927 -> p_hat = 0.403 < 0.75.
    # Therefore p* = 0.75 is unreachable-high; controller must step down toward D_min.
    controller.estimated_mastery["fine_motor"] = 0.45
    reachable, p_low, p_high = controller.check_reachability("fine_motor")
    assert not reachable, "Target 0.75 must be unreachable for theta=0.45"
    assert p_high < 0.75

    # Step down until it saturates at D_min (takes ~5 steps with max_step_change=0.08)
    for s in range(6):
        gid, diff, gap = controller.select_next_task(s)
    assert controller.last_diagnostic["reachability_applicable"] is True
    assert controller.last_diagnostic["target_reachable"] is False
    assert controller.last_diagnostic["reachability_status"] == "unreachable_high"
    assert controller.last_diagnostic["at_min"] is True
    assert diff == 0.08

    # 2. Reachable target: learner with high mastery theta = 0.90, target = 0.75
    # logit(0.90) = 2.1972.
    # p_low (D=0.95) = sigmoid(2.1972 - 2.28) = sigmoid(-0.0828) = 0.479.
    # p_high (D=0.08) = sigmoid(2.1972 - 0.192) = sigmoid(2.0052) = 0.881.
    # Since 0.479 <= 0.75 <= 0.881, target is reachable!
    controller.estimated_mastery["fine_motor"] = 0.90
    reachable, p_low, p_high = controller.check_reachability("fine_motor")
    assert reachable, "Target 0.75 must be reachable for theta=0.90"
    assert p_low <= 0.75 <= p_high

    gid, diff, gap = controller.select_next_task(10)
    assert controller.last_diagnostic["target_reachable"] is True
    assert controller.last_diagnostic["reachability_status"] == "reachable"

    # 3. Unreachable-low target: high mastery theta = 0.95, low target p* = 0.20
    # At D_max=0.95: logit_diff = logit(0.95) - 2.4*(0.95) = 2.9444 - 2.28 = 0.6644 -> p_low = 0.660 > 0.20.
    # At D_min=0.08: p_high = sigmoid(2.9444 - 0.192) = 0.940 > 0.20.
    # Even at hardest difficulty D_max, learner success prob exceeds 0.20.
    # The policy MUST choose D_max as the most achievable difficulty (step UP toward D_max, NOT down to D_min).
    controller_low = Grow21ModelBasedAdaptiveController(
        task_registry=registry,
        target_success_prob=0.20,
        initial_difficulty=0.45,
        gamma=2.4,
        nu_D=1.0,
        min_difficulty=0.08,
        max_difficulty=0.95,
        max_step_change=0.08,
    )
    for s in registry.skill_ids:
        controller_low.estimated_mastery[s] = 0.95

    reachable_low, p_low_l, p_high_l = controller_low.check_reachability("fine_motor")
    assert not reachable_low, "Target 0.20 must be unreachable-low for theta=0.95"
    assert 0.20 < p_low_l

    # Initial selection must step upward towards D_max
    gid, d_step1, gap = controller_low.select_next_task(0)
    assert d_step1 > 0.45, f"Unreachable-low target must step upward towards D_max, got {d_step1}"
    assert pytest.approx(d_step1, abs=1e-3) == 0.53
    assert controller_low.last_diagnostic["reachability_status"] == "unreachable_low"

    # Multiple steps must saturate at D_max (0.95)
    for s in range(1, 10):
        gid, d_step, gap = controller_low.select_next_task(s)
    assert d_step == 0.95
    assert controller_low.last_diagnostic["at_max"] is True
    assert controller_low.last_diagnostic["target_reachable"] is False


def test_controller_reachability_applicability_and_safety_separation(registry):
    """Verifies that non-target controllers report reachability as N/A and safety interventions are decoupled."""
    # 1. FixedDifficultyController
    fixed = FixedDifficultyController(registry, fixed_difficulty=0.50)
    fixed.select_next_task(0)
    assert fixed.last_diagnostic["reachability_applicable"] is False
    assert fixed.last_diagnostic["target_reachable"] is None
    assert fixed.last_diagnostic["safety_intervention"] is False

    # 2. StreakAdaptiveController
    streak = StreakAdaptiveController(registry, initial_difficulty=0.45)
    streak.select_next_task(0)
    assert streak.last_diagnostic["reachability_applicable"] is False
    assert streak.last_diagnostic["target_reachable"] is None
    assert streak.last_diagnostic["safety_intervention"] is False

    # 3. Grow21ModelBasedAdaptiveController safety intervention decoupling
    target_ctrl = Grow21ModelBasedAdaptiveController(
        task_registry=registry,
        target_success_prob=0.75,
        initial_difficulty=0.45,
    )
    # Set mastery high across all skills so target 0.75 is mathematically reachable regardless of task cycling
    for s in registry.skill_ids:
        target_ctrl.estimated_mastery[s] = 0.90

    is_reach, _, _ = target_ctrl.check_reachability("fine_motor")
    assert is_reach is True

    # Simulate 3 consecutive failures to trigger safety cooldown
    target_ctrl.observe_outcome("trace_glow", 0, 0.45)
    target_ctrl.observe_outcome("trace_glow", 0, 0.45)
    target_ctrl.observe_outcome("trace_glow", 0, 0.45)

    gid, diff, gap = target_ctrl.select_next_task(3)
    # Safety intervention occurred
    assert target_ctrl.last_diagnostic["safety_intervention"] is True
    assert target_ctrl.last_diagnostic["reachability_applicable"] is True
    # Mathematical reachability should NOT be forced to False just because safety cooldown triggered!
    assert target_ctrl.last_diagnostic["target_reachable"] is True
    assert target_ctrl.last_diagnostic["reachability_status"] == "reachable"


def test_bound_stuck_zero_reversals_not_active_stability():
    """Confirms that a sequence stuck flat at the minimum bound produces zero reversals via production helper."""
    diffs = [0.08] * 10
    reversals = calculate_direction_reversals(diffs)
    variation = sum(abs(diffs[i] - diffs[i - 1]) for i in range(1, len(diffs)))

    assert reversals == 0, "Flat bound-stuck sequence has 0 direction reversals"
    assert variation == 0.0, "Flat bound-stuck sequence has 0 total variation"


def test_mastery_change_and_secondary_skill_transfer(registry):
    """Verifies that secondary skills receive transfer learning and are not unpracticed."""
    from grow21_simulation_benchmark.reliability import ReliabilityEngine
    from grow21_simulation_benchmark.simulator import VirtualLearnerProfile, VirtualLearnerSimulator

    profile = VirtualLearnerProfile(
        learner_id=1,
        initial_mastery={s: 0.40 for s in registry.skill_ids},
        learning_rates={s: 0.10 for s in registry.skill_ids},
        forgetting_rates={s: 0.02 for s in registry.skill_ids},
        tremor_propensity=0.0,
        fatigue_sensitivity=0.0,
    )
    rel_engine = ReliabilityEngine(registry)
    sim = VirtualLearnerSimulator(profile, registry, rel_engine)

    scenario_params = {"corruption_multiplier": 0.0}
    rng = np.random.RandomState(42)

    # trace_glow has primary fine_motor and secondary visual_discrimination
    ev = sim.step(
        game_id="trace_glow",
        difficulty=0.20,
        time_gap_hours=0.01,
        rng=rng,
        scenario_params=scenario_params,
    )

    # Force outcome to 1 (success) to test transfer mechanics
    # If successful, primary increases by eta * (0.95 - theta)
    # secondary increases by 0.30 * eta * (0.95 - theta)
    # unrelated categorization only decays
    m_fine = sim.current_mastery["fine_motor"]
    m_vis = sim.current_mastery["visual_discrimination"]
    m_cat = sim.current_mastery["categorization"]

    assert m_cat <= 0.40, "Unpracticed unrelated skill must not gain mastery"
    # Even with small forgetting, if outcome was success, secondary gained transfer:
    if ev.clean_outcome == 1:
        assert m_fine > 0.40, "Primary practiced skill must gain mastery"
        assert m_vis > 0.40, "Secondary transfer skill must gain mastery from transfer"
        assert m_vis < m_fine, "Secondary transfer gain must be smaller than primary direct gain"


def test_break_count_and_elapsed_gap_logic(registry):
    """Confirms exactly 3 scheduled breaks (36 hours) occur in a 60-trial run with 15-trial limit."""
    controller = FixedDifficultyController(registry, fixed_difficulty=0.5, max_session_trials=15)
    break_steps = []
    total_break_hours = 0.0

    for step in range(60):
        gid, diff, gap = controller.select_next_task(step)
        if gap >= 10.0:
            break_steps.append(step)
            total_break_hours += gap
        controller.observe_outcome(gid, 1, diff)

    # In a 60-trial sequence (steps 0 to 59), breaks occur at step 15, 30, and 45.
    # That is exactly 3 breaks (36 hours), NOT 4 breaks / 48 hours (trial 60 is end of run).
    assert break_steps == [15, 30, 45], f"Expected breaks at [15, 30, 45], got {break_steps}"
    assert len(break_steps) == 3
    assert total_break_hours == 36.0


def test_calibration_bin_ece_consistency():
    """Confirms bin counts conserve total observations and recomputed ECE matches exactly."""
    rng = np.random.RandomState(123)
    N = 4500
    y_true = rng.randint(0, 2, size=N)
    y_prob = rng.uniform(0.0, 1.0, size=N)

    bins, ece = compute_calibration_table(y_true, y_prob, num_bins=10)
    assert sum(b["count"] for b in bins) == N

    # Manually recompute ECE from non-empty bins (matching within rounding precision)
    manual_ece = sum((b["count"] / N) * abs(b["mean_pred_prob"] - b["empirical_acc"]) for b in bins if b["count"] > 0)
    assert pytest.approx(manual_ece, abs=1e-4) == ece
    assert round(manual_ece, 4) == round(ece, 4)


def test_reliability_bias_direction(registry):
    """Verifies that positive uncertainty_bias overestimates corruption noise (underestimates reliability)."""
    from grow21_simulation_benchmark.reliability import ReliabilityEngine
    engine = ReliabilityEngine(registry)

    rng_unbiased = np.random.RandomState(42)
    rng_biased = np.random.RandomState(42)

    # Sample without bias
    r_unbiased = engine.sample_trial_uncertainties(
        game_id="trace_glow",
        rng=rng_unbiased,
        uncertainty_bias=0.0,
        uncertainty_noise_scale=0.01,
    )

    # Sample with positive bias (+0.12)
    r_biased = engine.sample_trial_uncertainties(
        game_id="trace_glow",
        rng=rng_biased,
        uncertainty_bias=0.12,
        uncertainty_noise_scale=0.01,
    )

    # Positive uncertainty_bias increases estimated corruption prob p_hat
    # and therefore DECREASES estimated reliability rho_hat = prod(1 - p_hat)
    assert r_biased.estimated_reliability < r_unbiased.estimated_reliability, (
        "Positive bias must reduce estimated reliability (overestimating corruption noise)"
    )


def test_persisted_calibration_bins_and_ece_consistency():
    """Validates that persisted calibration_bins.csv has exactly 4500 samples per slice and recomputed ECE matches metrics.csv."""
    import pandas as pd
    root_dir = Path(__file__).parent.parent
    bins_path = root_dir / "outputs" / "audit" / "phase_2" / "calibration_bins.csv"
    metrics_path = root_dir / "outputs" / "benchmark" / "metrics.csv"

    assert bins_path.exists(), f"Persisted calibration bins file not found at {bins_path}"
    assert metrics_path.exists(), f"Persisted metrics file not found at {metrics_path}"

    bins_df = pd.read_csv(bins_path)
    metrics_df = pd.read_csv(metrics_path)

    groups = bins_df.groupby(["scenario", "seed", "model"])
    # 5 scenarios * 3 seeds * 3 models = 45 slices
    assert len(groups) == 45, f"Expected 45 scenario/seed/model slices, got {len(groups)}"

    for (sc, sd, md), g in groups:
        total_samples = g["count"].sum()
        assert total_samples == 4500, (
            f"Slice ({sc}, {sd}, {md}) has {total_samples} samples, expected exactly 4500"
        )

        recomputed_ece = (g["count"] * (g["empirical_acc"] - g["mean_pred_prob"]).abs()).sum() / total_samples

        matching_rows = metrics_df[
            (metrics_df["scenario"] == sc) & (metrics_df["seed"] == sd) & (metrics_df["model"] == md)
        ]
        assert len(matching_rows) == 1, f"Expected 1 matching row in metrics.csv for ({sc}, {sd}, {md})"
        metric_ece = matching_rows["ece"].values[0]

        diff = abs(recomputed_ece - metric_ece)
        assert diff <= 1e-4, (
            f"ECE mismatch for ({sc}, {sd}, {md}): recomputed={recomputed_ece:.6f}, metric={metric_ece:.6f}, diff={diff:.6e}"
        )


def test_closed_loop_summary_reversals_and_safety_persistence(registry):
    """Validates that closed_loop_summary.json persists both immediate and filtered reversals,
    and that its metrics match independent recalculation from closed_loop_events.csv."""
    import json
    import pandas as pd
    root_dir = Path(__file__).parent.parent
    summary_path = root_dir / "outputs" / "closed_loop" / "closed_loop_summary.json"
    events_path = root_dir / "outputs" / "closed_loop" / "closed_loop_events.csv"

    assert summary_path.exists(), f"Summary JSON not found: {summary_path}"
    assert events_path.exists(), f"Events CSV not found: {events_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    events_df = pd.read_csv(events_path)

    expected_policies = ["fixed_difficulty", "streak_adaptive", "target_adaptive"]
    for policy in expected_policies:
        assert policy in summary, f"Policy {policy} missing from summary JSON"
        p_metrics = summary[policy]

        # Requirement: Fail if either documented metric is omitted
        assert "immediate_direction_reversals_mean" in p_metrics, f"Missing immediate_direction_reversals_mean in {policy}"
        assert "immediate_direction_reversals_std" in p_metrics, f"Missing immediate_direction_reversals_std in {policy}"
        assert "filtered_direction_reversals_mean" in p_metrics, f"Missing filtered_direction_reversals_mean in {policy}"
        assert "filtered_direction_reversals_std" in p_metrics, f"Missing filtered_direction_reversals_std in {policy}"
        assert "direction_reversals_mean" in p_metrics, f"Missing direction_reversals_mean in {policy}"
        assert "total_safety_interventions" in p_metrics, f"Missing total_safety_interventions in {policy}"

        # Recompute independently from closed_loop_events.csv
        p_events = events_df[events_df["controller"] == policy]
        imms, filts = [], []
        for _, lgroup in p_events.groupby("learner_id"):
            diffs = lgroup.sort_values("step_idx")["difficulty"].tolist()
            imms.append(calculate_immediate_reversals(diffs))
            filts.append(calculate_filtered_direction_reversals(diffs))

        assert abs(np.mean(imms) - p_metrics["immediate_direction_reversals_mean"]) < 1e-3
        assert abs(np.std(imms) - p_metrics["immediate_direction_reversals_std"]) < 1e-3
        assert abs(np.mean(filts) - p_metrics["filtered_direction_reversals_mean"]) < 1e-3
        assert abs(np.std(filts) - p_metrics["filtered_direction_reversals_std"]) < 1e-3
        assert abs(p_metrics["filtered_direction_reversals_mean"] - p_metrics["direction_reversals_mean"]) < 1e-3

        # Safety intervention counts match
        total_safety = int(p_events["safety_intervention"].sum())
        assert total_safety == p_metrics["total_safety_interventions"], (
            f"Safety count mismatch for {policy}: {total_safety} in CSV vs {p_metrics['total_safety_interventions']} in JSON"
        )

    # Small deterministic simulation to verify ClosedLoopRunResult fields
    scenario_cfg = {
        "description": "deterministic_test",
        "initial_mastery_mean": 0.45,
        "initial_mastery_std": 0.15,
        "learning_rate_mean": 0.08,
        "learning_rate_std": 0.02,
        "forgetting_rate_mean": 0.02,
        "forgetting_rate_std": 0.01,
        "corruption_multiplier": 1.0,
    }
    results, _, _ = run_closed_loop_comparison(
        task_registry=registry,
        scenario_cfg=scenario_cfg,
        num_learners=2,
        trials_per_learner=10,
        seed=42,
    )
    for c_name, runs in results.items():
        for r in runs:
            assert hasattr(r, "immediate_direction_reversals")
            assert hasattr(r, "filtered_direction_reversals")
            assert r.direction_reversals == r.filtered_direction_reversals


