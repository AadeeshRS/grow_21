"""Regression tests and mathematical safeguards for the Grow21 Robust Adaptive Estimator.

Validates:
1. Bilateral transition symmetry (no ratcheting drift on failure).
2. Session gap uncertainty diffusion (expanding variance, avoiding overconfidence).
3. Observable reliability-weighted residual tracking and gated fallback.
4. Modulated secondary skill transfer bounds.
5. Strict causal timing and zero simulator truth leakage.
6. Controller stability safeguards and probability bounds.
7. Exact deterministic reproducibility.
"""

from __future__ import annotations

import copy
import math
import numpy as np
import pytest
import yaml

from grow21_simulation_benchmark.closed_loop import (
    ModelAlignedTargetAdaptiveController,
    RobustAdaptiveFilterController,
    _diffuse_1d_grid,
)
from grow21_simulation_benchmark.initializers import (
    MasteryPriorInitializer,
    SkillPriorEstimate,
)
from grow21_simulation_benchmark.paths import DEFAULT_CONFIG_PATH, SCENARIOS_CONFIG_PATH
from grow21_simulation_benchmark.reliability import ReliabilityEngine
from grow21_simulation_benchmark.simulator import (
    VirtualLearnerSimulator,
    generate_learner_profile,
)
from grow21_simulation_benchmark.tasks import TaskRegistry


@pytest.fixture
def registry():
    return TaskRegistry.from_yaml(DEFAULT_CONFIG_PATH)


@pytest.fixture
def default_cfg():
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_bilateral_transition_moves_downward_on_clean_failure(registry):
    """Test that observed failure under high sensor reliability moves belief downward.

    In Phase 5, the approximate Bayesian filter had a unilateral transition update
    cand_delta = lr * (0.95 - grid) * cand_feedback * cand_engagement which ALWAYS
    added positive practice gain even on clean failure, ratcheting beliefs upward.
    The robust adaptive filter must correct this via bilateral practice targets.
    """
    ctrl = RobustAdaptiveFilterController(
        task_registry=registry,
        learning_rate_prior=0.08,
    )

    skill = "fine_motor"
    initial_m = 0.50
    # Center grid weights around 0.50
    grid = ctrl._bayesian_filter_state[skill]["grid"]
    weights = np.exp(-((grid - initial_m) ** 2) / (2.0 * (0.10 ** 2)))
    weights /= weights.sum()
    ctrl._bayesian_filter_state[skill]["weights"] = weights
    ctrl.estimated_mastery[skill] = float(np.sum(weights * grid))

    # High reliability sensor observation of outcome = 0 (clean failure)
    ctrl.observe_outcome(
        game_id="trace_glow",
        outcome=0,
        difficulty=0.30,
        estimated_reliability=0.99,
    )

    new_m = ctrl.estimated_mastery[skill]
    assert new_m < initial_m, (
        f"Robust adaptive filter should decrease mastery belief after high-reliability failure, "
        f"got new_m={new_m:.4f} >= initial_m={initial_m:.4f}"
    )


def test_session_gap_uncertainty_diffusion_expands_variance(registry):
    """Test that elapsed time gaps expand belief variance via diffusion kernel."""
    ctrl = RobustAdaptiveFilterController(
        task_registry=registry,
        forgetting_rate_prior=0.02,
    )

    skill = "working_memory"
    grid = ctrl._bayesian_filter_state[skill]["grid"]
    # Start with a sharp distribution
    sharp_w = np.exp(-((grid - 0.60) ** 2) / (2.0 * (0.04 ** 2)))
    sharp_w /= sharp_w.sum()
    ctrl._bayesian_filter_state[skill]["weights"] = sharp_w

    var_before = float(np.sum(sharp_w * (grid - np.sum(sharp_w * grid)) ** 2))

    # Apply 12-hour session gap decay
    ctrl.session_trial_count = ctrl.max_session_trials  # forces 12-hour gap
    ctrl.select_next_task(0)

    diffused_w = ctrl._bayesian_filter_state[skill]["weights"]
    mean_after = float(np.sum(diffused_w * grid))
    var_after = float(np.sum(diffused_w * (grid - mean_after) ** 2))

    assert var_after > var_before, (
        f"Session gap diffusion must expand uncertainty variance across 12-hour gap: "
        f"var_after={var_after:.6f} <= var_before={var_before:.6f}"
    )


def test_reliability_weighted_residual_gating_behavior(registry):
    """Test that reliable residual gating only triggers when sensor reliability is high."""
    ctrl = RobustAdaptiveFilterController(
        task_registry=registry,
        learning_rate_prior=0.08,
    )

    # Initial state
    assert ctrl.rel_residual_ema == 0.0

    # Case A: Low reliability (rho = 0.1), repeated failures
    for _ in range(5):
        ctrl._last_predicted_clean_prob = 0.75
        ctrl.observe_outcome("trace_glow", outcome=0, difficulty=0.50, estimated_reliability=0.10)

    # Low reliability downweights the residual update
    assert ctrl.rel_residual_ema > -0.05, (
        f"Unreliable observations should not trigger strong negative residual: {ctrl.rel_residual_ema}"
    )

    # Case B: High reliability (rho = 0.95), repeated failures (true distribution shift)
    ctrl_shift = RobustAdaptiveFilterController(
        task_registry=registry,
        learning_rate_prior=0.08,
    )
    for _ in range(5):
        ctrl_shift._last_predicted_clean_prob = 0.75
        ctrl_shift.observe_outcome("trace_glow", outcome=0, difficulty=0.50, estimated_reliability=0.95)

    assert ctrl_shift.rel_residual_ema < ctrl_shift.fallback_threshold, (
        f"Persistent high-reliability failures should breach fallback threshold: "
        f"{ctrl_shift.rel_residual_ema} >= {ctrl_shift.fallback_threshold}"
    )


def test_secondary_transfer_modulation_on_failure(registry):
    """Test that secondary transfer is throttled when primary trial clean success expectation is low."""
    ctrl = RobustAdaptiveFilterController(
        task_registry=registry,
        learning_rate_prior=0.08,
        secondary_transfer_ratio=0.30,
    )

    # Game with secondary skill: trace_glow has primary=fine_motor, secondary=visual_discrimination
    game = registry.games["trace_glow"]
    assert game.secondary_skill is not None
    sec_skill = game.secondary_skill

    sec_m_start = 0.40
    ctrl.estimated_mastery[sec_skill] = sec_m_start

    # Simulate observed failure with high reliability (clean failure: q_cand -> 0)
    ctrl.observe_outcome(
        game_id="trace_glow",
        outcome=0,
        difficulty=0.80,
        estimated_reliability=0.99,
    )

    sec_m_after = ctrl.estimated_mastery[sec_skill]
    # Under modulated transfer, secondary target is q_mean * 0.95 + (1 - q_mean) * sec_m_start.
    # When q_mean -> 0, secondary target -> sec_m_start, so secondary gain should be near zero!
    sec_gain = sec_m_after - sec_m_start
    assert sec_gain < 0.005, (
        f"Clean failure must not generate large secondary skill practice gain: got gain={sec_gain:.5f}"
    )


def test_strict_zero_data_leakage(registry):
    """Test that controller decisions never access latent simulator ground truth or future information."""
    with open(SCENARIOS_CONFIG_PATH, "r", encoding="utf-8") as f:
        scenarios_cfg = yaml.safe_load(f)["scenarios"]

    sim_params = scenarios_cfg["distribution_shift"]
    rng = np.random.RandomState(42)
    rel_engine = ReliabilityEngine(registry)

    profile = generate_learner_profile(0, registry.skill_ids, rng, sim_params)
    sim = VirtualLearnerSimulator(profile, registry, rel_engine)
    ctrl = RobustAdaptiveFilterController(registry)

    for step in range(30):
        # 1. Controller selects task: inspect attributes to ensure no latent simulator state
        gid, diff, gap = ctrl.select_next_task(step)
        assert 0.08 <= diff <= 0.95
        assert gap >= 0.0

        # Verify controller does NOT have access to sim
        assert not hasattr(ctrl, "simulator")
        assert not hasattr(ctrl, "profile")
        assert not hasattr(ctrl, "latent_mastery")

        # 2. Simulator steps using private parameters
        ev = sim.step(gid, diff, gap, np.random.RandomState(100 + step), sim_params)

        # 3. Controller only receives observable outputs
        ctrl.observe_outcome(
            game_id=gid,
            outcome=ev.observed_outcome,
            difficulty=diff,
            estimated_reliability=ev.estimated_reliability,
        )

        # Beliefs must remain strictly within probability bounds
        for s in registry.skill_ids:
            assert 0.05 <= ctrl.estimated_mastery[s] <= 0.95


def test_deterministic_reproducibility(registry):
    """Test that two separate controller instances with identical inputs produce bitwise identical trajectories."""
    ctrl1 = RobustAdaptiveFilterController(registry)
    ctrl2 = RobustAdaptiveFilterController(registry)

    simulated_inputs = [
        ("trace_glow", 1, 0.45, 0.85),
        ("bubble_pop_brigade", 0, 0.50, 0.90),
        ("block_stacker", 1, 0.48, 0.70),
        ("echo_safari", 0, 0.55, 0.35),
        ("picture_sentence_strips", 1, 0.42, 0.95),
    ]

    for step, (gid, outcome, diff, rel) in enumerate(simulated_inputs):
        g1, d1, t1 = ctrl1.select_next_task(step)
        g2, d2, t2 = ctrl2.select_next_task(step)
        assert g1 == g2
        assert d1 == d2
        assert t1 == t2

        ctrl1.observe_outcome(gid, outcome, diff, estimated_reliability=rel)
        ctrl2.observe_outcome(gid, outcome, diff, estimated_reliability=rel)

        for s in registry.skill_ids:
            assert ctrl1.estimated_mastery[s] == ctrl2.estimated_mastery[s]
            np.testing.assert_array_equal(
                ctrl1._bayesian_filter_state[s]["weights"],
                ctrl2._bayesian_filter_state[s]["weights"],
            )


def test_prior_initializer_support(registry):
    """Test that RobustAdaptiveFilterController integrates properly with MasteryPriorInitializer."""
    custom_priors = {
        "fine_motor": SkillPriorEstimate(0.30, 0.08, "parent_report"),
        "sequential_reasoning": SkillPriorEstimate(0.65, 0.08, "parent_report"),
    }
    initializer = MasteryPriorInitializer.from_provided_priors(custom_priors, fallback_default=0.45)
    ctrl = RobustAdaptiveFilterController(registry, prior_initializer=initializer)

    # Initial mastery beliefs must reflect initializer
    assert abs(ctrl.estimated_mastery["fine_motor"] - 0.30) < 1e-4
    assert abs(ctrl.estimated_mastery["sequential_reasoning"] - 0.65) < 1e-4
    assert abs(ctrl.estimated_mastery["working_memory"] - 0.45) < 1e-4

    # Grid weights mean must match initial mastery
    for s in ["fine_motor", "sequential_reasoning", "working_memory"]:
        grid = ctrl._bayesian_filter_state[s]["grid"]
        w = ctrl._bayesian_filter_state[s]["weights"]
        grid_mean = float(np.sum(w * grid))
        # Discretized Gaussian on bounded [0.05, 0.95] has small boundary truncation shift
        assert abs(grid_mean - ctrl.estimated_mastery[s]) < 0.04
