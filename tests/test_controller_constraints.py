"""Unit tests verifying closed-loop controllers adhere strictly to safety, Grow21 equations, and constraints."""

from pathlib import Path
import math
import numpy as np
import pytest

from grow21_simulation_benchmark.closed_loop import (
    FixedDifficultyController,
    Grow21ModelBasedAdaptiveController,
    StreakAdaptiveController,
    run_closed_loop_comparison,
)
from grow21_simulation_benchmark.tasks import TaskRegistry


@pytest.fixture
def registry():
    cfg_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    return TaskRegistry.from_yaml(cfg_path)


def test_streak_controller_frustration_safety(registry):
    controller = StreakAdaptiveController(registry, initial_difficulty=0.7)

    # Simulate 3 consecutive failures
    controller.observe_outcome("trace_glow", 0, 0.7)
    controller.observe_outcome("trace_glow", 0, 0.6)
    controller.observe_outcome("trace_glow", 0, 0.5)

    # Next task selection must drop difficulty drastically and trigger cooldown
    gid, new_diff, gap = controller.select_next_task(3)
    assert new_diff <= 0.45, f"Expected difficulty drop after 3 failures, got {new_diff}"
    assert gap >= 2.0, f"Expected cooldown gap after frustration event, got {gap}"
    assert controller.frustration_violations == 0


def test_grow21_model_based_controller_equations_and_deadband(registry):
    controller = Grow21ModelBasedAdaptiveController(
        task_registry=registry,
        target_success_prob=0.75,
        initial_difficulty=0.45,
        gamma=2.4,
        nu_D=1.0,
        deadband=0.04,
        max_step_change=0.08,
    )

    # 1. Test predicted success equation
    p_hat = controller.predict_success_prob("fine_motor", 0.45)
    # logit(0.45) = ln(0.45 / 0.55) = -0.20067
    # gamma * D^nu = 2.4 * 0.45 = 1.08
    # logit_diff = -1.28067 -> sigmoid(-1.28067) = 0.2174
    expected_p = 1.0 / (1.0 + math.exp(1.28067))
    assert pytest.approx(p_hat, abs=1e-3) == expected_p

    # 2. Test difficulty optimization towards target 0.75:
    # When theta is 0.94: logit(0.94) = 2.7515
    # gamma * D* = 2.7515 - logit(0.75) = 2.7515 - 1.0986 = 1.6529 -> D* = 0.6887
    # Since initial D = 0.45 and D* = 0.6887, controller must step up towards D*
    # bounded by max_step_change (+0.08) -> d_next = 0.45 + 0.08 = 0.53
    controller.estimated_mastery["fine_motor"] = 0.94
    gid, d_next, gap = controller.select_next_task(0)
    assert pytest.approx(d_next, abs=1e-3) == 0.53
    assert d_next > 0.45, "Should increase difficulty towards D* = 0.688"

    # 3. Test deadband: when current difficulty already satisfies |p_hat - target| <= deadband
    # Set mastery so that at D=0.53, p_hat is within deadband of 0.75
    # logit(p_target) + gamma * D = 1.0986 + 2.4 * 0.53 = 2.3706 -> theta ~ 0.914
    controller.estimated_mastery["fine_motor"] = 0.914
    gid, d_deadband, gap = controller.select_next_task(1)
    # Deadband should keep difficulty steady at 0.53
    assert pytest.approx(d_deadband, abs=0.02) == 0.53


def test_session_break_limit_constraint(registry):
    controller = FixedDifficultyController(registry, fixed_difficulty=0.5)

    # Simulate 15 trials (the maximum allowable continuous session length)
    for step in range(15):
        gid, diff, gap = controller.select_next_task(step)
        controller.observe_outcome(gid, 1, diff)

    # The 16th trial MUST mandate an inter-session rest gap
    gid, diff, break_gap = controller.select_next_task(15)
    assert break_gap >= 10.0, f"Expected rest break after 15 trials, got {break_gap} hours"
    assert controller.break_limit_violations == 0


def test_closed_loop_simulation_run(registry):
    scenario_cfg = {
        "description": "test_scenario",
        "initial_mastery_mean": 0.45,
        "initial_mastery_std": 0.15,
        "learning_rate_mean": 0.08,
        "learning_rate_std": 0.02,
        "forgetting_rate_mean": 0.02,
        "forgetting_rate_std": 0.01,
        "corruption_multiplier": 1.0,
    }

    results, events_df, _ = run_closed_loop_comparison(
        task_registry=registry,
        scenario_cfg=scenario_cfg,
        num_learners=10,
        trials_per_learner=30,
        seed=42,
    )

    assert len(events_df) == 10 * 30 * 3  # 10 learners * 30 trials * 3 controllers
    assert set(results.keys()) == {"fixed_difficulty", "streak_adaptive", "target_adaptive"}

    for c_name, runs in results.items():
        assert len(runs) == 10
        for r in runs:
            assert r.frustration_violations == 0, f"Frustration violation in {c_name}"
            assert r.break_limit_violations == 0, f"Break limit violation in {c_name}"
            assert 0.0 <= r.observed_success_rate <= 1.0
            assert 0.0 <= r.clean_success_rate <= 1.0
            assert r.direction_reversals >= 0
            assert r.total_difficulty_variation >= 0.0
