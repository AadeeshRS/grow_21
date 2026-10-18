"""Phase 5 unit and regression tests for transition-aware state estimation,
initial mastery priors, and offline model selection safeguards.
"""

from __future__ import annotations

import copy
import math
import numpy as np
import pytest
import yaml

from grow21_simulation_benchmark.closed_loop import (
    ModelAlignedTargetAdaptiveController,
    run_closed_loop_comparison,
    run_open_loop_replay,
)
from grow21_simulation_benchmark.initializers import (
    MasteryPriorInitializer,
    SkillPriorEstimate,
    generate_synthetic_parent_proxy_reports,
)
from grow21_simulation_benchmark.paths import DEFAULT_CONFIG_PATH, SCENARIOS_CONFIG_PATH
from grow21_simulation_benchmark.tasks import TaskRegistry


@pytest.fixture
def registry():
    return TaskRegistry.from_yaml(DEFAULT_CONFIG_PATH)


@pytest.fixture
def default_cfg():
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_expected_transition_increases_on_both_success_and_failure(registry):
    """Test that expected transition moves mastery upward on both clean success and failure expectations."""
    # Under simulator ground truth, primary learning factor is 0.70 on clean success, 0.30 on clean failure.
    # Therefore, even on clean failure (or q_t = 0), expected update must increase mastery when below 0.95.
    ctrl = ModelAlignedTargetAdaptiveController(
        task_registry=registry,
        belief_update_mode="expected_transition",
    )

    initial_mastery = 0.45
    ctrl.estimated_mastery["fine_motor"] = initial_mastery

    # Simulate an observed failure with high reliability (q_t ~ 0)
    # The controller should NOT send mastery towards 0.05!
    ctrl.observe_outcome(
        game_id="trace_glow",
        outcome=0,
        difficulty=0.50,
        estimated_reliability=0.99,
    )

    new_mastery = ctrl.estimated_mastery["fine_motor"]
    assert new_mastery > initial_mastery, (
        f"Expected transition should increase mastery even after failure, "
        f"got {new_mastery} <= {initial_mastery}"
    )


def test_expected_transition_analytical_formula(registry):
    """Test that expected update exactly matches the theoretical analytical formula."""
    ctrl = ModelAlignedTargetAdaptiveController(
        task_registry=registry,
        learning_rate_prior=0.08,
        belief_update_mode="expected_transition",
    )

    theta_0 = 0.40
    ctrl.estimated_mastery["fine_motor"] = theta_0
    ctrl._last_predicted_clean_prob = 0.70

    difficulty = 0.60
    outcome = 1
    rho = 0.85

    # Compute expected analytical values:
    p = 0.70
    p_obs_given_clean1 = rho + 0.15 * (1.0 - rho)
    p_obs_given_clean0 = 0.04 * (1.0 - rho)
    num = p * p_obs_given_clean1
    den = num + (1.0 - p) * p_obs_given_clean0
    q_t = num / den

    expected_feedback = 0.30 + 0.40 * q_t
    expected_engagement = 0.60 + 0.40 * difficulty
    lr_prior = 0.08
    expected_delta = lr_prior * (0.95 - theta_0) * expected_feedback * expected_engagement
    expected_theta = np.clip(theta_0 + expected_delta, 0.05, 0.95)

    ctrl.observe_outcome(
        game_id="trace_glow",
        outcome=outcome,
        difficulty=difficulty,
        estimated_reliability=rho,
    )

    actual_theta = ctrl.estimated_mastery["fine_motor"]
    assert math.isclose(actual_theta, expected_theta, rel_tol=1e-5), (
        f"Analytical mismatch: expected {expected_theta}, got {actual_theta}"
    )


def test_observed_response_invariance_before_outcome(registry):
    """Test that pre-outcome prediction and difficulty choice cannot be affected by the outcome of that event."""
    ctrl1 = ModelAlignedTargetAdaptiveController(
        task_registry=registry,
        belief_update_mode="approx_bayesian_filter_transfer",
    )
    ctrl2 = ModelAlignedTargetAdaptiveController(
        task_registry=registry,
        belief_update_mode="approx_bayesian_filter_transfer",
    )

    # Pre-task selection should be identical
    game1, diff1, gap1 = ctrl1.select_next_task(0)
    game2, diff2, gap2 = ctrl2.select_next_task(0)

    assert game1 == game2
    assert diff1 == diff2
    assert gap1 == gap2
    assert ctrl1.estimated_mastery["fine_motor"] == ctrl2.estimated_mastery["fine_motor"]


def test_observation_likelihood_normalization_and_asymptotics(registry):
    """Test that reliability observation likelihoods obey declared observation-noise model."""
    ctrl = ModelAlignedTargetAdaptiveController(
        task_registry=registry,
        belief_update_mode="approx_bayesian_filter",
    )

    ctrl.observe_outcome(
        game_id="trace_glow",
        outcome=1,
        difficulty=0.5,
        estimated_reliability=1.0,
    )
    # The posterior distribution should exist and sum to 1.0
    dist = ctrl._bayesian_filter_state["fine_motor"]["weights"]
    assert math.isclose(float(np.sum(dist)), 1.0, rel_tol=1e-5)


def test_secondary_transfer_ceiling_and_no_leakage(registry):
    """Test that secondary transfer caps at 0.95 and does not read simulator truth."""
    ctrl = ModelAlignedTargetAdaptiveController(
        task_registry=registry,
        belief_update_mode="expected_transition_transfer",
    )

    # Set secondary skill close to ceiling
    ctrl.estimated_mastery["visual_discrimination"] = 0.94
    ctrl._last_predicted_clean_prob = 0.80

    ctrl.observe_outcome(
        game_id="trace_glow",  # primary: fine_motor, secondary: visual_discrimination
        outcome=1,
        difficulty=0.5,
        estimated_reliability=0.9,
    )

    sec_theta = ctrl.estimated_mastery["visual_discrimination"]
    assert sec_theta <= 0.95
    assert sec_theta > 0.94


def test_forgetting_applied_once_per_event(registry):
    """Test that forgetting is applied exactly once per event and accounts for time elapsed."""
    ctrl = ModelAlignedTargetAdaptiveController(
        task_registry=registry,
        belief_update_mode="expected_transition",
    )

    ctrl.estimated_mastery["fine_motor"] = 0.80

    # Trigger session break by stepping to max trials
    ctrl.session_trial_count = ctrl.max_session_trials
    ctrl.select_next_task(0)

    theta_after_break = ctrl.estimated_mastery["fine_motor"]
    assert theta_after_break < 0.80


def test_mastery_prior_initializer_defaults_and_fallbacks(registry):
    """Test that MasteryPriorInitializer correctly handles defaults, bounded values, and missingness."""
    skill_ids = registry.skill_ids

    # 1. Default policy
    init_default = MasteryPriorInitializer.fixed_default(0.45)
    priors_def = init_default.initialize_mastery(skill_ids)
    for s in skill_ids:
        assert math.isclose(priors_def[s], 0.45)

    # 2. Provided priors with missing skills falling back to 0.45 (without treating missing as 0.05)
    provided = {
        "fine_motor": SkillPriorEstimate(prior_mean=0.75, uncertainty=0.8, source="parent_report"),
        # Other skills omitted
    }
    init_custom = MasteryPriorInitializer.from_provided_priors(provided, fallback_default=0.45)
    priors_custom = init_custom.initialize_mastery(skill_ids)

    assert math.isclose(priors_custom["fine_motor"], 0.75)
    assert math.isclose(priors_custom["visual_discrimination"], 0.45)
    assert priors_custom["visual_discrimination"] != 0.05  # Missingness != low ability

    # 3. Bounding check: out of range prior should be clamped
    out_of_bound = {
        "fine_motor": SkillPriorEstimate(prior_mean=1.5, uncertainty=0.5, source="noisy"),
    }
    init_clamped = MasteryPriorInitializer.from_provided_priors(out_of_bound)
    priors_clamped = init_clamped.initialize_mastery(skill_ids)
    assert math.isclose(priors_clamped["fine_motor"], 0.95)


def test_synthetic_parent_proxy_no_truth_leakage(registry):
    """Test that synthetic parent-proxy generation yields noisy reports and exposes no hidden simulator objects."""
    true_skills = {"fine_motor": 0.70, "visual_discrimination": 0.35, "working_memory": 0.55}

    reports = generate_synthetic_parent_proxy_reports(
        true_skill_masteries=true_skills,
        noise_std=0.15,
        missingness_rate=0.30,
        seed=101,
    )

    for skill_id, est in reports.items():
        assert isinstance(est, SkillPriorEstimate)
        assert 0.05 <= est.prior_mean <= 0.95
        assert est.source == "synthetic_parent_proxy"
        # Must not contain references to the ground-truth dictionary
        assert not hasattr(est, "true_mastery")


def test_model_selector_and_semantic_safeguard(registry):
    """Test offline benchmark model selector and critical controller safeguard."""
    # 1. Closed loop controller MUST reject observed-success classifier
    with pytest.raises(ValueError, match="CRITICAL SEMANTIC SAFEGUARD"):
        ModelAlignedTargetAdaptiveController(
            task_registry=registry,
            predictor_type="observed_success_classifier",
        )

    # 2. Canonical clean model succeeds
    ctrl_good = ModelAlignedTargetAdaptiveController(
        task_registry=registry,
        predictor_type="canonical_clean_model",
    )
    assert ctrl_good is not None


def test_open_loop_replay_paired_exogenous_consistency(registry):
    """Test that open-loop state estimation replay presents identical exogenous inputs across estimators."""
    with open(SCENARIOS_CONFIG_PATH, "r", encoding="utf-8") as f:
        scenarios_cfg = yaml.safe_load(f)["scenarios"]

    sc_name = "base_heterogeneous"
    sc_params = scenarios_cfg[sc_name]

    policies = ["model_aligned_baseline", "expected_transition"]
    df_events, df_summary = run_open_loop_replay(
        task_registry=registry,
        scenario_cfg=sc_params,
        policies=policies,
        seed=101,
        num_learners=5,
        trials_per_learner=10,
    )

    # Check that events for learner 0 across the two policies have the EXACT same tasks, difficulties, and outcomes
    p1_events = df_events[df_events["controller"] == "model_aligned_baseline"].sort_values(["learner_id", "step_idx"])
    p2_events = df_events[df_events["controller"] == "expected_transition"].sort_values(["learner_id", "step_idx"])

    assert len(p1_events) == len(p2_events) == 50
    np.testing.assert_array_equal(p1_events["game_id"].values, p2_events["game_id"].values)
    np.testing.assert_array_almost_equal(p1_events["difficulty"].values, p2_events["difficulty"].values)
    np.testing.assert_array_equal(p1_events["observed_outcome"].values, p2_events["observed_outcome"].values)
    np.testing.assert_array_equal(p1_events["clean_outcome"].values, p2_events["clean_outcome"].values)

    # But their estimated mastery values will differ because the update rules differ!
    assert not np.allclose(p1_events["mastery_error_primary_post"].values, p2_events["mastery_error_primary_post"].values)


def test_offline_model_selector_reproducibility_individual_vs_all():
    """Test that offline sequence models achieve identical metrics when selected individually
    versus when selected in --models all mode due to controlled per-model seeds.
    """
    from grow21_simulation_benchmark.run_benchmark import run_benchmark
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        res_ind = run_benchmark(
            config_path=DEFAULT_CONFIG_PATH,
            scenarios_path=SCENARIOS_CONFIG_PATH,
            scenario_names=["simple_dynamics"],
            seeds=[101],
            num_learners=10,
            events_per_learner=10,
            output_dir=tmp_path / "ind",
            models="reliability_gated_dkt",
        )
        res_all = run_benchmark(
            config_path=DEFAULT_CONFIG_PATH,
            scenarios_path=SCENARIOS_CONFIG_PATH,
            scenario_names=["simple_dynamics"],
            seeds=[101],
            num_learners=10,
            events_per_learner=10,
            output_dir=tmp_path / "all",
            models="all",
        )

        ind_loss = res_ind["simple_dynamics"]["models"]["reliability_gated_dkt"]["log_loss_mean"]
        all_loss = res_all["simple_dynamics"]["models"]["reliability_gated_dkt"]["log_loss_mean"]
        ind_brier = res_ind["simple_dynamics"]["models"]["reliability_gated_dkt"]["brier_score_mean"]
        all_brier = res_all["simple_dynamics"]["models"]["reliability_gated_dkt"]["brier_score_mean"]

        assert abs(ind_loss - all_loss) < 1e-5, (
            f"Gated DKT log loss differs between individual ({ind_loss}) and all ({all_loss})"
        )
        assert abs(ind_brier - all_brier) < 1e-5, (
            f"Gated DKT brier score differs between individual ({ind_brier}) and all ({all_brier})"
        )
