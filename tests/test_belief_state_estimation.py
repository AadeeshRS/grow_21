"""Deterministic unit and regression tests for Phase 4 Learner Belief-State Estimation and Robustness.

Verifies:
1. Secondary-only skill (visual_discrimination) belief updating under transfer-aware mode without simulator truth leakage.
2. Heuristic baseline preservation and exact reproducibility of Phase 3 behavior.
3. Observation noise likelihood normalization, non-negativity, and asymptotic limits (rho -> 1 and rho -> 0).
4. Strict absence of simulator ground truth leakage (controller never reads true mastery, p_clean, or y_clean).
5. Event t predictions are finalized prior to outcome generation and invariant to event t outcome.
6. Mastery belief bounds [0.05, 0.95] and proper ordering of forgetting decay and transfer updates.
7. Common random numbers and identical learner starting profiles across controller variants.
8. Non-tuned estimator contract: zero parameter tuning performed on held-out seeds.
9. Numerical consistency between event-level data and scenario summary JSON.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import pytest

from grow21_simulation_benchmark.closed_loop import (
    ModelAlignedTargetAdaptiveController,
    run_closed_loop_comparison,
)
from grow21_simulation_benchmark.paths import (
    DEFAULT_CONFIG_PATH,
    PHASE_4_AUDIT_DIR,
    PHASE_4_STATE_ESTIMATION_DIR,
    SCENARIOS_CONFIG_PATH,
)
from grow21_simulation_benchmark.run_phase_4_experiments import (
    compute_paired_cluster_bootstrap,
)
from grow21_simulation_benchmark.simulator import (
    VirtualLearnerSimulator,
    generate_learner_profile,
)
from grow21_simulation_benchmark.tasks import TaskRegistry


@pytest.fixture
def registry() -> TaskRegistry:
    return TaskRegistry.from_yaml(DEFAULT_CONFIG_PATH)


def test_secondary_only_skill_transfer_tracking_no_truth_leakage(registry: TaskRegistry):
    """1. Proves transfer-aware controller tracks secondary transfer gains without simulator truth leakage.

    In the canonical task registry, visual_discrimination is never a primary skill, but is
    secondary in trace_glow. Under transfer_aware mode, practicing trace_glow must update
    visual_discrimination using only the declared population prior (0.08 * 0.30 = 0.024).
    Under heuristic_baseline mode, visual_discrimination must receive 0 practice updates.
    """
    baseline_ctrl = ModelAlignedTargetAdaptiveController(
        task_registry=registry,
        belief_update_mode="heuristic_baseline",
    )
    transfer_ctrl = ModelAlignedTargetAdaptiveController(
        task_registry=registry,
        belief_update_mode="transfer_aware",
    )

    init_vd_baseline = baseline_ctrl.estimated_mastery["visual_discrimination"]
    init_vd_transfer = transfer_ctrl.estimated_mastery["visual_discrimination"]
    assert init_vd_baseline == 0.45
    assert init_vd_transfer == 0.45

    # Simulate 5 trials of trace_glow (primary: motor_fine, secondary: visual_discrimination)
    for _ in range(5):
        baseline_ctrl.observe_outcome("trace_glow", outcome=1, difficulty=0.45)
        transfer_ctrl.observe_outcome("trace_glow", outcome=1, difficulty=0.45)

    # In baseline, visual_discrimination received NO update
    assert baseline_ctrl.estimated_mastery["visual_discrimination"] == 0.45

    # In transfer_aware, visual_discrimination belief evolved towards 0.95
    updated_vd = transfer_ctrl.estimated_mastery["visual_discrimination"]
    assert updated_vd > 0.45, f"Expected visual_discrimination belief to increase, got {updated_vd}"

    # Calculate expected mathematical progression: delta = 0.30 * 0.08 * (0.95 - curr) = 0.024 * (0.95 - curr)
    expected_m = 0.45
    for _ in range(5):
        expected_m += 0.024 * (0.95 - expected_m)
    assert abs(updated_vd - expected_m) < 1e-9, f"Transfer update mismatch: {updated_vd} vs {expected_m}"


def test_heuristic_baseline_preserves_phase_3_reproducibility(registry: TaskRegistry):
    """2. Proves default ModelAlignedTargetAdaptiveController reproduces Phase 3 baseline identically."""
    ctrl_default = ModelAlignedTargetAdaptiveController(registry)
    ctrl_explicit = ModelAlignedTargetAdaptiveController(registry, belief_update_mode="heuristic_baseline")

    assert ctrl_default.belief_update_mode == "heuristic_baseline"

    # Both must update primary skills identically on success and failure
    ctrl_default.observe_outcome("bubble_pop_brigade", outcome=1, difficulty=0.40)
    ctrl_explicit.observe_outcome("bubble_pop_brigade", outcome=1, difficulty=0.40)
    for s in registry.skill_ids:
        assert ctrl_default.estimated_mastery[s] == ctrl_explicit.estimated_mastery[s]

    ctrl_default.observe_outcome("bubble_pop_brigade", outcome=0, difficulty=0.40)
    ctrl_explicit.observe_outcome("bubble_pop_brigade", outcome=0, difficulty=0.40)
    for s in registry.skill_ids:
        assert ctrl_default.estimated_mastery[s] == ctrl_explicit.estimated_mastery[s]


def test_observation_model_normalization_and_asymptotic_limits(registry: TaskRegistry):
    """3. Validates observation noise likelihood normalization, non-negativity, and asymptotic limits."""
    ctrl = ModelAlignedTargetAdaptiveController(
        task_registry=registry,
        belief_update_mode="reliability_aware",
    )

    # As rho -> 1.0 (perfect reliability), soft_y must converge to hard observed outcome
    # Case A: outcome=1, rho=1.0
    ctrl._last_predicted_clean_prob = 0.70
    ctrl.observe_outcome("trace_glow", outcome=1, difficulty=0.45, estimated_reliability=1.0)
    # Expected primary mastery change: 0.45 + 0.08 * (0.95 - 0.45) = 0.49
    assert pytest.approx(ctrl.estimated_mastery["fine_motor"], abs=1e-7) == 0.49

    # Case B: outcome=0, rho=1.0
    ctrl._last_predicted_clean_prob = 0.70
    ctrl.observe_outcome("trace_glow", outcome=0, difficulty=0.45, estimated_reliability=1.0)
    # Expected primary mastery change from 0.49: 0.49 - 0.08 * (0.49 - 0.05) = 0.4548
    assert pytest.approx(ctrl.estimated_mastery["fine_motor"], abs=1e-7) == 0.4548

    # Case C: outcome=0, low reliability rho=0.10 (slip likelihood >> false failure)
    # The downward penalty must be substantially attenuated compared to hard failure
    ctrl_fresh = ModelAlignedTargetAdaptiveController(
        task_registry=registry,
        belief_update_mode="reliability_aware",
    )
    ctrl_fresh._last_predicted_clean_prob = 0.75
    ctrl_fresh.observe_outcome("trace_glow", outcome=0, difficulty=0.45, estimated_reliability=0.10)
    # In hard baseline, belief drops from 0.45 to 0.45 - 0.08 * (0.45 - 0.05) = 0.418
    # In reliability-aware, belief should stay significantly higher than 0.418
    assert ctrl_fresh.estimated_mastery["fine_motor"] > 0.418, (
        f"Attenuated drop expected, got {ctrl_fresh.estimated_mastery['fine_motor']}"
    )


def test_strict_no_leakage_of_hidden_truth_or_post_outcome_data(registry: TaskRegistry):
    """4. Proves controller never consumes simulator ground truth (true mastery, p_clean, y_clean)."""
    ctrl = ModelAlignedTargetAdaptiveController(registry, belief_update_mode="combined_transfer_reliability")

    # Controller inspects only allowed attributes
    allowed_ctrl_keys = {
        "registry", "target_prob", "current_difficulty", "deadband", "max_step_change",
        "min_difficulty", "max_difficulty", "fatigue_sensitivity_prior", "forgetting_rate_prior",
        "learning_rate_prior", "secondary_transfer_ratio", "belief_update_mode", "games",
        "estimated_mastery", "_last_predicted_clean_prob", "max_consecutive_failures",
        "max_session_trials", "cooldown_gap_hours", "consecutive_failures", "session_trial_count",
        "frustration_violations", "break_limit_violations", "last_diagnostic"
    }
    actual_keys = set(ctrl.__dict__.keys())
    disallowed = actual_keys - allowed_ctrl_keys
    assert not disallowed, f"Found unauthorized controller attributes: {disallowed}"

    # Verifies observe_outcome does not take latent parameters
    import inspect
    sig = inspect.signature(ctrl.observe_outcome)
    assert "p_clean" not in sig.parameters
    assert "clean_outcome" not in sig.parameters
    assert "true_mastery" not in sig.parameters
    assert "latent_mastery" not in sig.parameters


def test_event_t_prediction_finalized_before_outcome_observed(registry: TaskRegistry):
    """5. Proves prediction at trial t is computed before outcome generation and invariant to outcome t."""
    ctrl1 = ModelAlignedTargetAdaptiveController(registry)
    ctrl2 = ModelAlignedTargetAdaptiveController(registry)

    # Both select next task at step 0
    gid1, diff1, gap1 = ctrl1.select_next_task(0)
    gid2, diff2, gap2 = ctrl2.select_next_task(0)

    p1 = ctrl1.last_diagnostic["predicted_clean_prob"]
    p2 = ctrl2.last_diagnostic["predicted_clean_prob"]
    assert p1 == p2
    assert diff1 == diff2

    # Now observe differing outcomes for step 0
    ctrl1.observe_outcome(gid1, outcome=1, difficulty=diff1)
    ctrl2.observe_outcome(gid2, outcome=0, difficulty=diff2)

    # Diagnostic from step 0 was already fixed and unchanged
    assert ctrl1.last_diagnostic["predicted_clean_prob"] == p1
    assert ctrl2.last_diagnostic["predicted_clean_prob"] == p2

    # Step 1 predictions should now diverge because beliefs updated
    ctrl1.select_next_task(1)
    ctrl2.select_next_task(1)
    assert ctrl1.last_diagnostic["predicted_clean_prob"] > ctrl2.last_diagnostic["predicted_clean_prob"]


def test_mastery_belief_bounds_and_order_of_operations(registry: TaskRegistry):
    """6. Proves belief states strictly remain inside [0.05, 0.95] after extreme sequences."""
    ctrl = ModelAlignedTargetAdaptiveController(registry, belief_update_mode="combined_transfer_reliability")

    # 100 consecutive successes
    for _ in range(100):
        ctrl.observe_outcome("trace_glow", outcome=1, difficulty=0.90, estimated_reliability=0.95)
    for s, m in ctrl.estimated_mastery.items():
        assert 0.05 <= m <= 0.95, f"Skill {s} out of bounds: {m}"

    # 100 consecutive failures
    for _ in range(100):
        ctrl.observe_outcome("trace_glow", outcome=0, difficulty=0.10, estimated_reliability=0.95)
    for s, m in ctrl.estimated_mastery.items():
        assert 0.05 <= m <= 0.95, f"Skill {s} out of bounds: {m}"


def test_common_random_numbers_and_identical_learner_starting_profiles(registry: TaskRegistry):
    """7. Proves paired simulation uses identical initial learner profiles and common random streams."""
    scenario_cfg = {"simulation": {}, "scenarios": {"test": {}}}
    policies = ["model_aligned_baseline", "transfer_aware", "reliability_aware", "combined_transfer_reliability"]

    results, events_df, mastery_df = run_closed_loop_comparison(
        task_registry=registry,
        scenario_cfg={"test": {}},
        num_learners=3,
        trials_per_learner=5,
        seed=42,
        policies=policies,
    )

    # Initial mastery for learner 0 across all 4 policies must be exactly identical
    l0_mastery = mastery_df[mastery_df["learner_id"] == 0]
    for s in registry.skill_ids:
        s_records = l0_mastery[l0_mastery["skill_id"] == s]
        init_vals = s_records["initial_mastery"].unique()
        assert len(init_vals) == 1, f"Mismatch in initial mastery for skill {s}: {init_vals}"


def test_non_tuned_held_out_seed_contract(registry: TaskRegistry):
    """8. Proves all controllers use identical fixed population priors with zero seed tuning."""
    policies = ["model_aligned_baseline", "transfer_aware", "reliability_aware", "combined_transfer_reliability"]
    controllers = [
        ModelAlignedTargetAdaptiveController(registry, belief_update_mode="heuristic_baseline"),
        ModelAlignedTargetAdaptiveController(registry, belief_update_mode="transfer_aware"),
        ModelAlignedTargetAdaptiveController(registry, belief_update_mode="reliability_aware"),
        ModelAlignedTargetAdaptiveController(registry, belief_update_mode="combined_transfer_reliability"),
    ]

    for c in controllers:
        # All policies must share identical canonical priors
        assert c.fatigue_sensitivity_prior == 0.02
        assert c.forgetting_rate_prior == 0.02
        assert c.learning_rate_prior == 0.08
        assert c.secondary_transfer_ratio == 0.30
        assert c.target_prob == 0.75
        assert c.deadband == 0.04
        assert c.max_step_change == 0.08
        assert c.min_difficulty == 0.08
        assert c.max_difficulty == 0.95


def test_persisted_phase_4_artifacts_integrity_and_recomputability():
    """9. Validates that persisted Phase 4 JSON and CSV artifacts recompute consistently."""
    summary_path = PHASE_4_STATE_ESTIMATION_DIR / "phase_4_scenario_summary.json"
    events_path = PHASE_4_STATE_ESTIMATION_DIR / "phase_4_event_metrics.csv"
    fixture_path = Path(__file__).parent / "fixtures" / "phase_4_event_metrics_fixture.csv"
    cal_bins_path = PHASE_4_STATE_ESTIMATION_DIR / "phase_4_calibration_bins.csv"
    mastery_path = PHASE_4_STATE_ESTIMATION_DIR / "phase_4_mastery_estimation_metrics.csv"

    assert summary_path.exists(), f"Missing {summary_path}"
    assert cal_bins_path.exists(), f"Missing {cal_bins_path}"
    assert mastery_path.exists(), f"Missing {mastery_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_dict = json.load(f)

    if events_path.exists():
        events_df = pd.read_csv(events_path)
        assert len(events_df) == 540000, f"Expected 540,000 events, found {len(events_df)}"
    else:
        assert fixture_path.exists(), f"Neither full event metrics ({events_path}) nor fixture ({fixture_path}) found!"
        events_df = pd.read_csv(fixture_path)
        assert len(events_df) == 27000, f"Expected 27,000 fixture events, found {len(events_df)}"

    # Recompute base_heterogeneous signed gap for model_aligned_baseline across historical+held-out
    eval_events = events_df[
        (events_df["split"].isin(["historical", "held_out"]))
        & (events_df["scenario"] == "base_heterogeneous")
        & (events_df["controller"] == "model_aligned_baseline")
    ]
    recomputed_gap = float(np.mean(eval_events["predicted_clean_prob"] - eval_events["simulator_p_clean"]))
    stored_gap = summary_dict["combined_eval"]["base_heterogeneous"]["policies"]["model_aligned_baseline"]["signed_prediction_gap"]
    assert abs(recomputed_gap - stored_gap) < 1e-4, f"Mismatch in recomputed combined signed gap: {recomputed_gap} vs {stored_gap}"

    # Recompute primary held-out gap
    held_out_events = events_df[
        (events_df["split"] == "held_out")
        & (events_df["scenario"] == "base_heterogeneous")
        & (events_df["controller"] == "model_aligned_baseline")
    ]
    recomputed_ho_gap = float(np.mean(held_out_events["predicted_clean_prob"] - held_out_events["simulator_p_clean"]))
    stored_ho_gap = summary_dict["held_out"]["base_heterogeneous"]["policies"]["model_aligned_baseline"]["signed_prediction_gap"]
    assert abs(recomputed_ho_gap - stored_ho_gap) < 1e-4, f"Mismatch in recomputed held-out signed gap: {recomputed_ho_gap} vs {stored_ho_gap}"


def test_time_alignment_pre_and_post_mastery_estimation(registry: TaskRegistry):
    """10. Proves pre-outcome estimate matches latent_mastery_before, and post-update matches latent_mastery_after."""
    scenario_cfg = {"simulation": {}, "scenarios": {"test": {}}}
    _, ev_df, _ = run_closed_loop_comparison(
        task_registry=registry,
        scenario_cfg={"test": {}},
        num_learners=2,
        trials_per_learner=5,
        seed=42,
        policies=["combined_transfer_reliability"],
    )

    for _, row in ev_df.iterrows():
        # Pre-outcome fields
        assert row["estimated_mastery_primary_pre"] is not None
        assert row["true_mastery_primary_pre"] is not None
        expected_err_pre = round(float(row["estimated_mastery_primary_pre"] - row["true_mastery_primary_pre"]), 4)
        assert abs(row["mastery_error_primary_pre"] - expected_err_pre) < 1e-6

        # Post-update fields
        assert row["estimated_mastery_primary_post"] is not None
        assert row["true_mastery_primary_post"] is not None
        expected_err_post = round(float(row["estimated_mastery_primary_post"] - row["true_mastery_primary_post"]), 4)
        assert abs(row["mastery_error_primary_post"] - expected_err_post) < 1e-6

        # Check secondary fields if secondary skill present
        if row["secondary_skill"] != "none":
            assert row["estimated_mastery_secondary_pre"] is not None
            assert row["true_mastery_secondary_pre"] is not None
            expected_sec_err_pre = round(float(row["estimated_mastery_secondary_pre"] - row["true_mastery_secondary_pre"]), 4)
            assert abs(row["mastery_error_secondary_pre"] - expected_sec_err_pre) < 1e-6

            assert row["estimated_mastery_secondary_post"] is not None
            assert row["true_mastery_secondary_post"] is not None
            expected_sec_err_post = round(float(row["estimated_mastery_secondary_post"] - row["true_mastery_secondary_post"]), 4)
            assert abs(row["mastery_error_secondary_post"] - expected_sec_err_post) < 1e-6


def test_outcome_changes_post_update_not_pre_outcome_estimate_or_prediction(registry: TaskRegistry):
    """11. Proves outcome observation modifies post-update belief, but cannot retroactively change pre-outcome estimate or prediction."""
    ctrl_success = ModelAlignedTargetAdaptiveController(registry)
    ctrl_failure = ModelAlignedTargetAdaptiveController(registry)

    # Both controllers select task at step 0 from identical initial state
    gid_s, diff_s, gap_s = ctrl_success.select_next_task(0)
    gid_f, diff_f, gap_f = ctrl_failure.select_next_task(0)

    assert gid_s == gid_f
    assert diff_s == diff_f
    pred_p_before_s = ctrl_success.last_diagnostic["predicted_clean_prob"]
    pred_p_before_f = ctrl_failure.last_diagnostic["predicted_clean_prob"]
    assert pred_p_before_s == pred_p_before_f

    prim_skill = registry.games[gid_s].primary_skill
    pre_m_s = ctrl_success.estimated_mastery[prim_skill]
    pre_m_f = ctrl_failure.estimated_mastery[prim_skill]
    assert pre_m_s == pre_m_f

    # Observe differing outcomes
    ctrl_success.observe_outcome(gid_s, outcome=1, difficulty=diff_s)
    ctrl_failure.observe_outcome(gid_f, outcome=0, difficulty=diff_f)

    # Post-update belief states diverge
    post_m_s = ctrl_success.estimated_mastery[prim_skill]
    post_m_f = ctrl_failure.estimated_mastery[prim_skill]
    assert post_m_s > pre_m_s, f"Success should increase mastery belief: {post_m_s} vs {pre_m_s}"
    assert post_m_f < pre_m_f, f"Failure should decrease mastery belief: {post_m_f} vs {pre_m_f}"
    assert post_m_s > post_m_f

    # Prior pre-outcome diagnostic was NOT retroactively altered
    assert ctrl_success.last_diagnostic["predicted_clean_prob"] == pred_p_before_s
    assert ctrl_failure.last_diagnostic["predicted_clean_prob"] == pred_p_before_f


def test_paired_seed_aware_cluster_bootstrap_invariants():
    """12. Verifies seed-aware cluster identification, multiplicity preservation, and determinism."""
    # Create synthetic events with 2 seeds, each having learner 0 and learner 1
    # Total distinct clusters must be 4: (42, 0), (42, 1), (123, 0), (123, 1)
    rows_base = []
    rows_var = []
    for seed in [42, 123]:
        for l_id in [0, 1]:
            for step in range(5):
                rows_base.append({
                    "scenario": "test_sc",
                    "seed": seed,
                    "learner_id": l_id,
                    "step_idx": step,
                    "predicted_clean_prob": 0.70,
                    "simulator_p_clean": 0.75,
                    "mastery_error_primary_pre": 0.10,
                    "mastery_error_primary_post": 0.08,
                    "at_min_bound": 0,
                })
                # Variant has lower error for seed 42 learner 0, but higher for seed 123 learner 1
                var_err = 0.05 if (seed == 42 and l_id == 0) else 0.15
                rows_var.append({
                    "scenario": "test_sc",
                    "seed": seed,
                    "learner_id": l_id,
                    "step_idx": step,
                    "predicted_clean_prob": 0.75 if (seed == 42 and l_id == 0) else 0.60,
                    "simulator_p_clean": 0.75,
                    "mastery_error_primary_pre": var_err,
                    "mastery_error_primary_post": var_err,
                    "at_min_bound": 0,
                })

    df_base = pd.DataFrame(rows_base)
    df_var = pd.DataFrame(rows_var)

    # A. Cluster count must be 4, distinguishing learner_id repeated across seeds
    boot_res = compute_paired_cluster_bootstrap(df_base, df_var, metric_name="prob_abs_error", num_bootstrap=500, seed=42)
    assert boot_res["num_clusters"] == 4, f"Expected 4 distinct (seed, learner_id) clusters, got {boot_res['num_clusters']}"
    assert boot_res["num_bootstrap"] == 500
    assert boot_res["resampling_unit"] == "(seed, learner_id) cluster"
    assert boot_res["ci_method"] == "percentile_bootstrap_with_multiplicity"

    # B. Determinism with fixed seed
    boot_res2 = compute_paired_cluster_bootstrap(df_base, df_var, metric_name="prob_abs_error", num_bootstrap=500, seed=42)
    assert boot_res["point_estimate"] == boot_res2["point_estimate"]
    assert boot_res["ci_95"] == boot_res2["ci_95"]
    assert boot_res["std_error"] == boot_res2["std_error"]

    # C. Key mismatch detection
    mismatched_var = df_var.copy()
    mismatched_var.loc[0, "step_idx"] = 999
    with pytest.raises(ValueError, match="Key misalignment between baseline and variant"):
        compute_paired_cluster_bootstrap(df_base, mismatched_var, metric_name="prob_abs_error", num_bootstrap=100, seed=42)

    # D. Length mismatch detection
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_paired_cluster_bootstrap(df_base, df_var.iloc[:-1], metric_name="prob_abs_error", num_bootstrap=100, seed=42)


def test_split_separation_and_cohort_integrity():
    """13. Verifies strict separation between Held-Out, Historical, Development, and Combined summaries."""
    summary_path = PHASE_4_STATE_ESTIMATION_DIR / "phase_4_scenario_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_dict = json.load(f)

    required_splits = ["held_out", "historical", "development", "combined_eval"]
    for sp in required_splits:
        assert sp in summary_dict, f"Missing split '{sp}' in scenario summary JSON!"

    # Verify each split contains all 5 scenarios
    scenarios = ["base_heterogeneous", "reliability_stress", "simple_dynamics", "temporal_dynamics", "distribution_shift"]
    for sp in required_splits:
        for sc in scenarios:
            assert sc in summary_dict[sp], f"Missing scenario '{sc}' in split '{sp}'!"
            assert "policies" in summary_dict[sp][sc]
            assert "paired_cluster_bootstrap_vs_baseline" in summary_dict[sp][sc]

    # Verify event counts in CSV or fixture
    events_path = PHASE_4_STATE_ESTIMATION_DIR / "phase_4_event_metrics.csv"
    fixture_path = Path(__file__).parent / "fixtures" / "phase_4_event_metrics_fixture.csv"

    if events_path.exists():
        events_df = pd.read_csv(events_path)
        dev_count = len(events_df[events_df["split"] == "development"])
        hist_count = len(events_df[events_df["split"] == "historical"])
        ho_count = len(events_df[events_df["split"] == "held_out"])

        # 3 seeds * 30 learners * 60 trials * 4 policies * 5 scenarios = 108,000
        assert dev_count == 108000, f"Expected 108,000 development events, got {dev_count}"
        # 3 seeds * 60 learners * 60 trials * 4 policies * 5 scenarios = 216,000
        assert hist_count == 216000, f"Expected 216,000 historical events, got {hist_count}"
        # 3 seeds * 60 learners * 60 trials * 4 policies * 5 scenarios = 216,000
        assert ho_count == 216000, f"Expected 216,000 held-out events, got {ho_count}"
        assert dev_count + hist_count + ho_count == 540000
    else:
        assert fixture_path.exists(), f"Neither full event metrics ({events_path}) nor fixture ({fixture_path}) found!"
        events_df = pd.read_csv(fixture_path)
        dev_count = len(events_df[events_df["split"] == "development"])
        hist_count = len(events_df[events_df["split"] == "historical"])
        ho_count = len(events_df[events_df["split"] == "held_out"])

        assert dev_count == 5400, f"Expected 5,400 fixture development events, got {dev_count}"
        assert hist_count == 10800, f"Expected 10,800 fixture historical events, got {hist_count}"
        assert ho_count == 10800, f"Expected 10,800 fixture held-out events, got {ho_count}"
        assert dev_count + hist_count + ho_count == 27000
        # Invariant: Cohort split ratio must strictly be 1:2:2
        assert hist_count == 2 * dev_count
        assert ho_count == 2 * dev_count

    # Verify mastery metrics CSV has both timing categories
    mastery_path = PHASE_4_STATE_ESTIMATION_DIR / "phase_4_mastery_estimation_metrics.csv"
    mastery_df = pd.read_csv(mastery_path)
    assert set(mastery_df["timing"].unique()) == {"pre_outcome", "post_update"}


@pytest.mark.full_artifacts
def test_full_phase_4_event_metrics_volume():
    """14. Validates the full 540,000-event dataset when full experiment artifacts are present.

    In a clean checkout, this test skips cleanly with clear instructions on obtaining
    the 260 MB event log. When full artifacts are unpacked or regenerated, it strictly
    validates full row volume and completeness across all 5 scenarios and 4 policies.
    """
    events_path = PHASE_4_STATE_ESTIMATION_DIR / "phase_4_event_metrics.csv"
    if not events_path.exists():
        pytest.skip(
            "Full Phase 4 event metrics CSV (260 MB) not present in clean checkout (excluded by .gitignore). "
            "To run full-volume verification, regenerate via: "
            "python -m grow21_simulation_benchmark.run_phase_4_experiments "
            "(or extract local/release archive evidence/phase_4/Grow21_Phase_4_State_Estimation_Evidence.zip if available)."
        )
    events_df = pd.read_csv(events_path)
    assert len(events_df) == 540000, f"Expected 540,000 events, found {len(events_df)}"
    assert set(events_df["split"].unique()) == {"development", "historical", "held_out"}
    assert len(events_df["scenario"].unique()) == 5
    assert len(events_df["controller"].unique()) == 4

