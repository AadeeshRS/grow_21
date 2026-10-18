"""Deterministic unit and regression tests for Phase 3 Outcome Model Alignment.

Verifies:
1. Exact mathematical parity between canonical outcome model and simulator across all 9 games and state grids.
2. Non-linear difficulty curve (0.25*D + 0.75*D**1.35) preservation and rejection of linear approximations.
3. Guessing floor, lapse ceiling, and session fatigue lapse mechanics.
4. Target reachability logic, bounded bisection solver accuracy, and clipping handling.
5. Strict decoupling of step limiting and deadband constraints from target reachability.
6. Session gap projection and counter reset consistency with simulator ordering.
7. Population prior belief decay and strict absence of true simulator state leakage.
8. Non-target policy diagnostics neutrality and safety cooldown separation.
9. Closed-loop 4-policy execution and event diagnostic schema completeness.
10. Legacy surrogate vs. model-aligned bound saturation behavioral contrast.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

from grow21_simulation_benchmark.closed_loop import (
    FixedDifficultyController,
    LegacySurrogateTargetAdaptiveController,
    ModelAlignedTargetAdaptiveController,
    StreakAdaptiveController,
    run_closed_loop_comparison,
)
from grow21_simulation_benchmark.outcome_model import (
    compute_clean_task_success_prob,
    compute_composite_effective_mastery,
    compute_difficulty_curve,
    compute_difficulty_threshold,
    compute_game_clean_success_prob,
    solve_target_difficulty,
)
from grow21_simulation_benchmark.reliability import ReliabilityEngine
from grow21_simulation_benchmark.simulator import (
    VirtualLearnerProfile,
    VirtualLearnerSimulator,
    generate_learner_profile,
)
from grow21_simulation_benchmark.tasks import TaskRegistry


@pytest.fixture
def registry() -> TaskRegistry:
    cfg_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    return TaskRegistry.from_yaml(cfg_path)


def test_shared_clean_success_parity_all_nine_games(registry: TaskRegistry):
    """1. Shared simulator/controller clean-success function parity for all nine games.

    Tests interior and boundary values of effective mastery, difficulty, session counters, and fatigue sensitivity.
    """
    mastery_grid = [0.05, 0.20, 0.50, 0.80, 0.95]
    diff_grid = [0.0, 0.08, 0.35, 0.70, 0.95, 1.0]
    session_counters = [1, 5, 10, 11, 15, 20]
    fatigue_sensitivities = [0.005, 0.02, 0.035]

    for gid in registry.game_ids:
        game = registry.games[gid]
        for m in mastery_grid:
            for d in diff_grid:
                for sc in session_counters:
                    for fs in fatigue_sensitivities:
                        # Shared canonical calculation
                        p_canonical = compute_game_clean_success_prob(
                            game=game,
                            effective_mastery=m,
                            difficulty=d,
                            session_trial_counter=sc,
                            fatigue_sensitivity=fs,
                        )

                        # Manual direct evaluation of simulator canonical formula
                        curve = 0.25 * d + 0.75 * (d ** 1.35)
                        threshold = game.difficulty_offset + game.difficulty_scale * curve
                        logit_val = game.discrimination * (m - threshold)
                        sig = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit_val))))

                        excess = max(0, sc - 10)
                        fatigue_lapse = min(0.15, max(0.0, fs * excess))
                        total_lapse = min(0.35, game.lapse_ceiling + fatigue_lapse)
                        total_guess = game.guess_floor
                        p_expected = float(np.clip(total_guess + (1.0 - total_guess - total_lapse) * sig, 0.01, 0.99))

                        assert abs(p_canonical - p_expected) < 1e-12, (
                            f"Parity mismatch in game {gid} at m={m}, d={d}, sc={sc}, fs={fs}"
                        )


def test_difficulty_curve_nonlinearity_and_no_linear_substitute(registry: TaskRegistry):
    """2. Difficulty-curve parity: non-linear 0.25*D + 0.75*D**1.35 is strictly used; linear substitute rejected."""
    # Boundary checks
    assert pytest.approx(compute_difficulty_curve(0.0), abs=1e-9) == 0.0
    assert pytest.approx(compute_difficulty_curve(1.0), abs=1e-9) == 1.0

    # Convexity check: second derivative of 0.25*D + 0.75*D**1.35 is 0.75 * 1.35 * 0.35 * D**(-0.65) > 0 for D > 0
    d_vals = np.linspace(0.05, 0.95, 20)
    curves = [compute_difficulty_curve(d) for d in d_vals]
    first_diffs = np.diff(curves)
    second_diffs = np.diff(first_diffs)
    assert np.all(second_diffs > -1e-6), "Difficulty curve must have strictly positive second derivative (convex inflection)"

    for gid in registry.game_ids:
        game = registry.games[gid]
        for d in [0.08, 0.30, 0.50, 0.75, 0.95]:
            reg_threshold = registry.calculate_difficulty_threshold(gid, d)
            canonical_threshold = compute_difficulty_threshold(game.difficulty_offset, game.difficulty_scale, d)
            assert pytest.approx(reg_threshold, abs=1e-9) == canonical_threshold

            # Verify that a flawed linear approximation (offset + scale * D) produces a distinct, erroneous value
            flawed_linear_threshold = game.difficulty_offset + game.difficulty_scale * d
            assert abs(reg_threshold - flawed_linear_threshold) > 1e-3, (
                f"Registry threshold must not match naive linear curve for game {gid} at D={d}"
            )


def test_guess_lapse_and_fatigue_consistency(registry: TaskRegistry):
    """3. Guessing floor, lapse ceiling, and session fatigue calculations are consistent."""
    game = registry.games["trace_glow"]

    # Before fatigue onset (session count <= 10): fatigue lapse is 0
    p_sc10 = compute_game_clean_success_prob(game, effective_mastery=0.5, difficulty=0.5, session_trial_counter=10, fatigue_sensitivity=0.03)
    p_sc1 = compute_game_clean_success_prob(game, effective_mastery=0.5, difficulty=0.5, session_trial_counter=1, fatigue_sensitivity=0.03)
    assert pytest.approx(p_sc10, abs=1e-9) == p_sc1

    # After fatigue onset (session count 15): 5 excess trials
    p_sc15 = compute_game_clean_success_prob(game, effective_mastery=0.5, difficulty=0.5, session_trial_counter=15, fatigue_sensitivity=0.02)
    # 5 * 0.02 = 0.10 lapse increase -> clean success prob must decrease
    assert p_sc15 < p_sc10

    # Extreme trials: fatigue lapse capped at 0.15, total lapse capped at 0.35
    # For trace_glow (lapse_ceiling=0.08), total_lapse = min(0.35, 0.08 + 0.15) = 0.23.
    # Maximum clean success prob is total_guess + (1 - total_guess - total_lapse) = 0.05 + 0.72 = 0.77.
    p_sc100 = compute_game_clean_success_prob(game, effective_mastery=0.95, difficulty=0.08, session_trial_counter=100, fatigue_sensitivity=0.05)
    assert p_sc100 <= 0.77 + 1e-4

    # High lapse game testing total_lapse capped at 0.35:
    p_capped = compute_clean_task_success_prob(
        discrimination=3.0,
        difficulty_offset=0.0,
        difficulty_scale=1.0,
        guess_floor=0.10,
        lapse_ceiling=0.25,
        effective_mastery=0.95,
        difficulty=0.08,
        session_trial_counter=50,
        fatigue_sensitivity=0.05,
    )
    # 0.25 + 0.15 = 0.40 -> capped at 0.35. Max success = 0.10 + (1 - 0.10 - 0.35) = 0.65.
    assert p_capped <= 0.65 + 1e-4


def test_target_reachability_and_bisection_solver(registry: TaskRegistry):
    """4. Target reachability classification and bounded numerical bisection accuracy."""
    for gid in registry.game_ids:
        game = registry.games[gid]
        eff_m = 0.60

        # Evaluate bounds
        p_high = compute_game_clean_success_prob(game, eff_m, 0.08, session_trial_counter=1, fatigue_sensitivity=0.02)
        p_low = compute_game_clean_success_prob(game, eff_m, 0.95, session_trial_counter=1, fatigue_sensitivity=0.02)
        assert p_high > p_low, f"p_clean must be strictly decreasing in difficulty for game {gid}"

        # A. Reachable target within interior
        target_mid = round((p_high + p_low) / 2.0, 3)
        d_solved, status, high_eval, low_eval = solve_target_difficulty(
            game=game,
            effective_mastery=eff_m,
            target_prob=target_mid,
            session_trial_counter=1,
            fatigue_sensitivity=0.02,
            min_difficulty=0.08,
            max_difficulty=0.95,
        )
        assert status == "reachable"
        assert 0.08 <= d_solved <= 0.95
        p_at_solved = compute_game_clean_success_prob(game, eff_m, d_solved, 1, 0.02)
        assert abs(p_at_solved - target_mid) < 2e-3, f"Bisection error exceeded tolerance: {abs(p_at_solved - target_mid)}"

        # B. Unreachable-high target
        d_high, status_high, _, _ = solve_target_difficulty(
            game=game,
            effective_mastery=eff_m,
            target_prob=p_high + 0.05,
            session_trial_counter=1,
            fatigue_sensitivity=0.02,
        )
        assert status_high == "unreachable_high"
        assert d_high == 0.08, "Unreachable high target must aim at D_min"

        # C. Unreachable-low target
        d_low, status_low, _, _ = solve_target_difficulty(
            game=game,
            effective_mastery=eff_m,
            target_prob=p_low - 0.05,
            session_trial_counter=1,
            fatigue_sensitivity=0.02,
        )
        assert status_low == "unreachable_low"
        assert d_low == 0.95, "Unreachable low target must aim at D_max"


def test_step_limiting_and_deadband_constraints(registry: TaskRegistry):
    """5. Step-limiting and deadband apply after target-solving and respect difficulty bounds."""
    controller = ModelAlignedTargetAdaptiveController(
        task_registry=registry,
        target_success_prob=0.75,
        initial_difficulty=0.45,
        deadband=0.04,
        max_step_change=0.08,
        min_difficulty=0.08,
        max_difficulty=0.95,
    )

    # 1. Step-limit test: Force a target that demands a big jump
    # Set high mastery so target difficulty D* is high (e.g. > 0.65)
    for s in registry.skill_ids:
        controller.estimated_mastery[s] = 0.92

    gid, diff_next, gap = controller.select_next_task(0)
    diag = controller.last_diagnostic
    assert diag["step_limited"] == 1
    # Step change from initial 0.45 should be capped at +0.08 -> 0.53
    assert pytest.approx(diff_next, abs=1e-3) == 0.53

    # 2. Deadband test: When predicted success at current difficulty is within deadband
    next_gid = registry.game_ids[1]
    next_game = registry.games[next_gid]
    for s in registry.skill_ids:
        controller.estimated_mastery[s] = 0.60
    eff_m = controller.get_effective_mastery(next_gid)
    d_target, _, _, _ = solve_target_difficulty(
        game=next_game,
        effective_mastery=eff_m,
        target_prob=0.75,
        session_trial_counter=2,
        fatigue_sensitivity=0.02,
    )
    controller.current_difficulty = d_target

    gid_db, diff_db, gap_db = controller.select_next_task(1)
    diag_db = controller.last_diagnostic
    assert diag_db["deadband_held"] == 1
    assert diag_db["determination_rule"] == "deadband"
    assert pytest.approx(diff_db, abs=1e-3) == round(d_target, 3)


def test_session_gap_projection_ordering(registry: TaskRegistry):
    """6. Session-gap projection resets and increments the counter in the same order as the simulator."""
    controller = ModelAlignedTargetAdaptiveController(registry, max_session_trials=15)
    rel_engine = ReliabilityEngine(registry)

    rng = np.random.RandomState(42)
    profile = generate_learner_profile(0, registry.skill_ids, rng, {})
    sim = VirtualLearnerSimulator(profile, registry, rel_engine)

    # Run 30 steps and check that trial counter tracking is synchronized
    for step in range(30):
        gid, diff, gap = controller.select_next_task(step)
        ev = sim.step(gid, diff, gap, rng, {})
        controller.observe_outcome(gid, ev.observed_outcome, diff)

        # Simulator's session_trial_counter must match the controller's post-step session_trial_count
        assert sim.session_trial_counter == controller.session_trial_count, (
            f"Step {step}: sim counter {sim.session_trial_counter} != controller count {controller.session_trial_count}"
        )


def test_controller_population_priors_and_no_leakage(registry: TaskRegistry):
    """7 & 8. Controller uses population priors and has ZERO access to true learner profile or simulator truth."""
    controller = ModelAlignedTargetAdaptiveController(
        task_registry=registry,
        fatigue_sensitivity_prior=0.02,
        forgetting_rate_prior=0.02,
    )

    # Invariant: Controller does NOT store learner profile, true forgetting rates, or true fatigue
    assert not hasattr(controller, "profile")
    assert not hasattr(controller, "true_fatigue_sensitivity")
    assert not hasattr(controller, "true_forgetting_rates")
    assert not hasattr(controller, "p_clean")
    assert not hasattr(controller, "clean_outcome")
    assert not hasattr(controller, "corruption_occurred")

    # Invariant: Forgetting belief decay uses configured prior
    initial_beliefs = dict(controller.estimated_mastery)
    # Simulate a 12-hour session break
    controller.session_trial_count = 15
    gid, diff, gap = controller.select_next_task(15)
    assert gap == 12.0
    for s in registry.skill_ids:
        expected_decayed = 0.05 + (initial_beliefs[s] - 0.05) * math.exp(-0.02 * 12.0)
        assert pytest.approx(controller.estimated_mastery[s], abs=1e-4) == expected_decayed


def test_non_target_diagnostics_and_safety_separation(registry: TaskRegistry):
    """9. Non-target policy diagnostics remain N/A; safety overrides do not falsify target feasibility."""
    fixed = FixedDifficultyController(registry)
    streak = StreakAdaptiveController(registry)

    _, _, _ = fixed.select_next_task(0)
    assert fixed.last_diagnostic["reachability_applicable"] is False
    assert fixed.last_diagnostic["target_reachable"] is None
    assert fixed.last_diagnostic["reachability_status"] == "N/A"

    _, _, _ = streak.select_next_task(0)
    assert streak.last_diagnostic["reachability_applicable"] is False
    assert streak.last_diagnostic["target_reachable"] is None
    assert streak.last_diagnostic["reachability_status"] == "N/A"

    # Aligned controller under 3 consecutive failures: triggers safety cooldown
    aligned = ModelAlignedTargetAdaptiveController(registry, target_success_prob=0.75)
    for _ in range(3):
        aligned.observe_outcome("trace_glow", 0, 0.45)
    assert aligned.consecutive_failures == 3

    gid, diff, gap = aligned.select_next_task(3)
    diag = aligned.last_diagnostic
    assert diag["safety_intervention"] is True
    assert diag["determination_rule"] == "safety_cooldown"
    # Reachability must remain mathematically evaluated and not overwritten by safety
    assert diag["reachability_applicable"] is True
    assert diag["reachability_status"] in ["reachable", "unreachable_high", "unreachable_low"]


def test_closed_loop_four_policy_comparison_and_diagnostics_consistency(registry: TaskRegistry):
    """10. Closed-loop 4-policy simulation run verifies schema, event counts, and reachability diagnostics."""
    scenario_cfg = {
        "description": "test_phase_3_four_way",
        "initial_mastery_mean": 0.45,
        "initial_mastery_std": 0.15,
        "learning_rate_mean": 0.08,
        "learning_rate_std": 0.02,
        "forgetting_rate_mean": 0.02,
        "forgetting_rate_std": 0.01,
        "corruption_multiplier": 1.0,
    }

    results, events_df, mastery_df = run_closed_loop_comparison(
        task_registry=registry,
        scenario_cfg=scenario_cfg,
        num_learners=5,
        trials_per_learner=15,
        seed=42,
        include_legacy_surrogate=True,
    )

    expected_policies = {
        "fixed_difficulty",
        "streak_adaptive",
        "legacy_surrogate_target_adaptive",
        "model_aligned_target_adaptive",
    }
    assert set(results.keys()) == expected_policies
    assert len(events_df) == 5 * 15 * 4  # 5 learners * 15 trials * 4 policies = 300 events

    # Verify diagnostic schema in event DataFrame
    required_cols = [
        "controller",
        "learner_id",
        "step_idx",
        "game_id",
        "difficulty",
        "time_gap_hours",
        "observed_outcome",
        "clean_outcome",
        "simulator_p_clean",
        "predicted_clean_prob",
        "prediction_gap_clean",
        "target_prob",
        "target_error",
        "target_difficulty_solved",
        "applied_difficulty",
        "reachability_applicable",
        "target_reachable",
        "reachability_status",
        "p_high",
        "p_low",
        "deadband_held",
        "step_limited",
        "at_min_bound",
        "at_max_bound",
        "safety_intervention",
        "determination_rule",
    ]
    for col in required_cols:
        assert col in events_df.columns, f"Missing required diagnostic column: {col}"

    # Verify that prediction_gap_clean is computed only for target policies
    target_events = events_df[events_df["controller"].str.contains("target")]
    assert target_events["predicted_clean_prob"].notna().all()
    assert target_events["prediction_gap_clean"].notna().all()

    non_target_events = events_df[~events_df["controller"].str.contains("target")]
    assert non_target_events["predicted_clean_prob"].isna().all()
    assert non_target_events["prediction_gap_clean"].isna().all()


def test_legacy_vs_aligned_bound_saturation_contrast(registry: TaskRegistry):
    """Demonstrates that legacy surrogate underpredicts success at low mastery causing saturation, while aligned avoids it."""
    # Low mastery learner (mastery = 0.35)
    legacy = LegacySurrogateTargetAdaptiveController(registry, target_success_prob=0.75, initial_difficulty=0.45)
    aligned = ModelAlignedTargetAdaptiveController(registry, target_success_prob=0.75, initial_difficulty=0.45)

    for s in registry.skill_ids:
        legacy.estimated_mastery[s] = 0.35
        aligned.estimated_mastery[s] = 0.35

    # Legacy: logit(0.35) = -0.619. At D_min=0.08: logit - 2.4*0.08 = -0.811 -> p_hat = 0.308 << 0.75
    # Target 0.75 is unreachable-high for legacy controller!
    _, legacy_p_low, legacy_p_high = legacy.check_reachability("fine_motor")
    assert legacy_p_high < 0.40, f"Legacy surrogate severely underpredicts success at D_min: {legacy_p_high}"

    # Step legacy 5 times -> saturates at D_min (0.08)
    for step in range(5):
        _, diff_legacy, _ = legacy.select_next_task(step)
    assert diff_legacy == 0.08
    assert legacy.last_diagnostic["at_min"] is True

    # Aligned controller: computes true clean success prob with guessing floor and game thresholds
    # For many games (e.g. balloon_pop with guess_floor=0.10, offset=0.15), p_high is much higher (> 0.70)
    aligned_game = registry.games[registry.game_ids[0]]
    aligned_p_high = compute_game_clean_success_prob(aligned_game, 0.35, 0.08, 1, 0.02)
    assert aligned_p_high > legacy_p_high, (
        f"Aligned model clean success at D_min ({aligned_p_high}) should exceed legacy surrogate ({legacy_p_high})"
    )
