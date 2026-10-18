"""Grow21 Phase 5 Comprehensive State Estimation, Robustness, and Model Selector Runner.

Executes:
  Stage 1: Open-Loop State Estimation Replay (isolates belief tracking from policy actions).
  Stage 2: Closed-Loop Multi-Cohort Adaptive Evaluation (Dev, Historical, Phase 4 Eval, Final Held-Out).
  Stage 3: Initial Mastery Prior & Synthetic Parent-Proxy Sensitivity Experiment.
  Stage 4: Testing Model Selector Verification & Predictive Benchmarking.

Policies compared:
  1. model_aligned_baseline (Phase 3 heuristic updater)
  2. transfer_aware (adds secondary skill transfer belief update)
  3. reliability_aware (adds Bayesian soft clean expectation observation update)
  4. combined_transfer_reliability (combines transfer and reliability updates)
  5. expected_transition (Phase 5 expected transition under ground-truth primary learning dynamics)
  6. expected_transition_transfer (Phase 5 expected transition + expected secondary transfer)
  7. approx_bayesian_filter (Phase 5 approximate Bayesian posterior update + state transition)
  8. approx_bayesian_filter_transfer (Phase 5 approximate Bayesian filter + expected secondary transfer)

Strict Sign Convention:
  Differences are computed as (variant - baseline).
  For error metrics (MAE, RMSE, Brier score), negative differences indicate error reduction (improvement).
  For bound saturation (min_bound), negative differences indicate reduced saturation.
  Fields are explicitly named `diff_variant_minus_baseline` and `prob_abs_error_diff_variant_minus_baseline`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
import yaml

from .closed_loop import (
    ModelAlignedTargetAdaptiveController,
    run_closed_loop_comparison,
    run_open_loop_replay,
)
from .evaluate import compute_calibration_table
from .initializers import (
    MasteryPriorInitializer,
    SkillPriorEstimate,
    generate_synthetic_parent_proxy_reports,
)
from .paths import (
    DEFAULT_CONFIG_PATH,
    PHASE_5_AUDIT_DIR,
    PHASE_5_CLOSED_LOOP_DIR,
    PHASE_5_INITIALIZERS_DIR,
    PHASE_5_OPEN_LOOP_DIR,
    PHASE_5_OUTPUTS_DIR,
    PHASE_5_PLOTS_DIR,
    SCENARIOS_CONFIG_PATH,
    ensure_output_directories,
)
from .run_benchmark import run_benchmark
from .tasks import TaskRegistry


PHASE_5_POLICIES = [
    "model_aligned_baseline",
    "transfer_aware",
    "reliability_aware",
    "combined_transfer_reliability",
    "expected_transition",
    "expected_transition_transfer",
    "approx_bayesian_filter",
    "approx_bayesian_filter_transfer",
]

POLICY_LABELS = {
    "model_aligned_baseline": "Baseline (Phase 3)",
    "transfer_aware": "Transfer-Aware",
    "reliability_aware": "Reliability-Aware",
    "combined_transfer_reliability": "Combined (Transfer + Rel)",
    "expected_transition": "Expected Transition",
    "expected_transition_transfer": "Expected Transition + Transfer",
    "approx_bayesian_filter": "Approx Bayesian Filter",
    "approx_bayesian_filter_transfer": "Approx Bayesian Filter + Transfer",
}

POLICY_COLORS = {
    "model_aligned_baseline": "#4C72B0",
    "transfer_aware": "#55A868",
    "reliability_aware": "#DD8452",
    "combined_transfer_reliability": "#8172B2",
    "expected_transition": "#CCB974",
    "expected_transition_transfer": "#64B5CD",
    "approx_bayesian_filter": "#C44E52",
    "approx_bayesian_filter_transfer": "#8C564B",
}

DEV_SEEDS = [101, 202, 303]
HISTORICAL_SEEDS = [42, 123, 999]
PHASE_4_EVAL_SEEDS = [777, 888, 555]
FINAL_HELD_OUT_SEEDS = [707, 808, 909]


def compute_metrics_for_events(events_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes performance, probability calibration, and time-aligned mastery metrics."""
    clean_p = events_df["simulator_p_clean"].to_numpy(dtype=float)
    pred_p = events_df["predicted_clean_prob"].to_numpy(dtype=float)
    clean_y = events_df["clean_outcome"].to_numpy(dtype=int)
    obs_y = events_df["observed_outcome"].to_numpy(dtype=int)

    signed_gap = float(np.mean(pred_p - clean_p))
    mae_prob = float(np.mean(np.abs(pred_p - clean_p)))
    rmse_prob = float(np.sqrt(np.mean((pred_p - clean_p) ** 2)))

    eps = 1e-6
    pred_p_clipped = np.clip(pred_p, eps, 1.0 - eps)
    brier = float(brier_score_loss(clean_y, pred_p_clipped))
    ll = float(log_loss(clean_y, pred_p_clipped))

    # Pre-outcome mastery error
    prim_err_pre = events_df["mastery_error_primary_pre"].dropna().to_numpy(dtype=float)
    prim_mae_pre = float(np.mean(np.abs(prim_err_pre))) if len(prim_err_pre) > 0 else None
    prim_rmse_pre = float(np.sqrt(np.mean(prim_err_pre ** 2))) if len(prim_err_pre) > 0 else None
    prim_bias_pre = float(np.mean(prim_err_pre)) if len(prim_err_pre) > 0 else None

    sec_err_pre = events_df["mastery_error_secondary_pre"].dropna().to_numpy(dtype=float)
    sec_mae_pre = float(np.mean(np.abs(sec_err_pre))) if len(sec_err_pre) > 0 else None
    sec_rmse_pre = float(np.sqrt(np.mean(sec_err_pre ** 2))) if len(sec_err_pre) > 0 else None
    sec_bias_pre = float(np.mean(sec_err_pre)) if len(sec_err_pre) > 0 else None

    # Post-update mastery error
    prim_err_post = events_df["mastery_error_primary_post"].dropna().to_numpy(dtype=float)
    prim_mae_post = float(np.mean(np.abs(prim_err_post))) if len(prim_err_post) > 0 else None
    prim_rmse_post = float(np.sqrt(np.mean(prim_err_post ** 2))) if len(prim_err_post) > 0 else None
    prim_bias_post = float(np.mean(prim_err_post)) if len(prim_err_post) > 0 else None

    sec_err_post = events_df["mastery_error_secondary_post"].dropna().to_numpy(dtype=float)
    sec_mae_post = float(np.mean(np.abs(sec_err_post))) if len(sec_err_post) > 0 else None
    sec_rmse_post = float(np.sqrt(np.mean(sec_err_post ** 2))) if len(sec_err_post) > 0 else None
    sec_bias_post = float(np.mean(sec_err_post)) if len(sec_err_post) > 0 else None

    return {
        "event_count": len(events_df),
        "mean_simulator_p_clean": float(np.mean(clean_p)),
        "mean_predicted_clean_prob": float(np.mean(pred_p)),
        "clean_success_rate": float(np.mean(clean_y)),
        "observed_success_rate": float(np.mean(obs_y)),
        "prob_signed_gap": signed_gap,
        "prob_abs_error": mae_prob,
        "prob_rmse": rmse_prob,
        "clean_brier_score": brier,
        "clean_log_loss": ll,
        # Pre-outcome metrics
        "primary_mastery_mae_pre": prim_mae_pre,
        "primary_mastery_rmse_pre": prim_rmse_pre,
        "primary_mastery_bias_pre": prim_bias_pre,
        "secondary_mastery_mae_pre": sec_mae_pre,
        "secondary_mastery_rmse_pre": sec_rmse_pre,
        "secondary_mastery_bias_pre": sec_bias_pre,
        # Post-update metrics
        "primary_mastery_mae_post": prim_mae_post,
        "primary_mastery_rmse_post": prim_rmse_post,
        "primary_mastery_bias_post": prim_bias_post,
        "secondary_mastery_mae_post": sec_mae_post,
        "secondary_mastery_rmse_post": sec_rmse_post,
        "secondary_mastery_bias_post": sec_bias_post,
        # Behavioral metrics
        "mean_difficulty": float(events_df["difficulty"].mean()) if "difficulty" in events_df.columns else None,
        "proportion_at_min_bound": float(events_df["at_min_bound"].mean()) if "at_min_bound" in events_df.columns else None,
        "proportion_at_max_bound": float(events_df["at_max_bound"].mean()) if "at_max_bound" in events_df.columns else None,
        "target_reachable_rate": float(events_df["target_reachable"].dropna().mean()) if "target_reachable" in events_df.columns and len(events_df["target_reachable"].dropna()) > 0 else None,
        "safety_intervention_count": int(events_df["safety_intervention"].sum()) if "safety_intervention" in events_df.columns else 0,
    }


def compute_paired_cluster_bootstrap(
    base_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    metric_name: str = "prob_abs_error",
    num_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes a paired, seed-aware learner cluster bootstrap with replacement.

    Clusters are defined by unique (seed, learner_id) tuples.
    Events within clusters are verified to match exactly on (scenario, seed, learner_id, step_idx).
    Resamples clusters with replacement, strictly preserving multiplicity of repeated draws.

    Convention: diff = variant - baseline. Negative values represent error reduction.
    """
    sort_cols = ["scenario", "seed", "learner_id", "step_idx"] if (
        "scenario" in base_df.columns and "scenario" in variant_df.columns
    ) else ["seed", "learner_id", "step_idx"]

    b_sorted = base_df.sort_values(sort_cols).reset_index(drop=True)
    v_sorted = variant_df.sort_values(sort_cols).reset_index(drop=True)

    if len(b_sorted) != len(v_sorted):
        raise ValueError(f"Length mismatch: baseline {len(b_sorted)} vs variant {len(v_sorted)}")

    b_keys = b_sorted[sort_cols].to_numpy()
    v_keys = v_sorted[sort_cols].to_numpy()
    if not np.array_equal(b_keys, v_keys):
        raise ValueError(f"Key misalignment between baseline and variant in paired bootstrap across {sort_cols}!")

    if metric_name == "prob_abs_error":
        b_val = np.abs(b_sorted["predicted_clean_prob"].to_numpy(dtype=float) - b_sorted["simulator_p_clean"].to_numpy(dtype=float))
        v_val = np.abs(v_sorted["predicted_clean_prob"].to_numpy(dtype=float) - v_sorted["simulator_p_clean"].to_numpy(dtype=float))
        diff_per_event = v_val - b_val
    elif metric_name == "primary_mastery_mae_pre":
        b_val = np.abs(b_sorted["mastery_error_primary_pre"].to_numpy(dtype=float))
        v_val = np.abs(v_sorted["mastery_error_primary_pre"].to_numpy(dtype=float))
        diff_per_event = v_val - b_val
    elif metric_name == "primary_mastery_mae_post":
        b_val = np.abs(b_sorted["mastery_error_primary_post"].to_numpy(dtype=float))
        v_val = np.abs(v_sorted["mastery_error_primary_post"].to_numpy(dtype=float))
        diff_per_event = v_val - b_val
    elif metric_name == "at_min_bound":
        b_val = b_sorted["at_min_bound"].to_numpy(dtype=float)
        v_val = v_sorted["at_min_bound"].to_numpy(dtype=float)
        diff_per_event = v_val - b_val
    else:
        raise ValueError(f"Unknown metric {metric_name}")

    b_sorted["_diff"] = diff_per_event
    grouped = b_sorted.groupby(["seed", "learner_id"])["_diff"]
    cluster_sums = grouped.sum().to_numpy()
    cluster_counts = grouped.count().to_numpy()
    n_clusters = len(cluster_sums)

    point_estimate = float(np.sum(cluster_sums) / np.sum(cluster_counts))

    rng = np.random.RandomState(seed)
    boot_indices = rng.randint(0, n_clusters, size=(num_bootstrap, n_clusters))
    boot_sums = np.sum(cluster_sums[boot_indices], axis=1)
    boot_counts = np.sum(cluster_counts[boot_indices], axis=1)
    boot_replicates = boot_sums / boot_counts

    ci_low = float(np.percentile(boot_replicates, 2.5))
    ci_high = float(np.percentile(boot_replicates, 97.5))
    boot_se = float(np.std(boot_replicates))

    # Interpretation
    if ci_high < 0.0:
        interp = "significant_reduction"
    elif ci_low > 0.0:
        interp = "significant_increase"
    else:
        interp = "not_significant_ci_crosses_zero"

    return {
        "diff_variant_minus_baseline": round(point_estimate, 4),
        "point_estimate": round(point_estimate, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "ci_lower": round(ci_low, 4),
        "ci_upper": round(ci_high, 4),
        "std_error": round(boot_se, 4),
        "num_clusters": int(n_clusters),
        "num_bootstrap": int(num_bootstrap),
        "seed": int(seed),
        "resampling_unit": "(seed, learner_id) cluster",
        "ci_method": "percentile_bootstrap_with_multiplicity",
        "sign_convention": "variant_minus_baseline (negative = error reduction)",
        "interpretation": interp,
    }


def compute_calibration_dataframe(events_df: pd.DataFrame, num_bins: int = 10) -> pd.DataFrame:
    """Computes calibration bins with split, scenario, and controller grouping."""
    rows = []
    bins = np.linspace(0.0, 1.0, num_bins + 1)

    for (sp, sc, pol), grp in events_df.groupby(["split", "scenario", "controller"]):
        y_prob = grp["predicted_clean_prob"].to_numpy(dtype=float)
        y_clean = grp["clean_outcome"].to_numpy(dtype=int)
        y_obs = grp["observed_outcome"].to_numpy(dtype=int)
        sim_p = grp["simulator_p_clean"].to_numpy(dtype=float)

        for b in range(num_bins):
            low, high = bins[b], bins[b + 1]
            if b == num_bins - 1:
                mask = (y_prob >= low) & (y_prob <= high)
            else:
                mask = (y_prob >= low) & (y_prob < high)

            count = int(np.sum(mask))
            if count > 0:
                mean_pred = float(np.mean(y_prob[mask]))
                emp_clean = float(np.mean(y_clean[mask]))
                emp_obs = float(np.mean(y_obs[mask]))
                mean_sim = float(np.mean(sim_p[mask]))
                cal_gap_clean = abs(mean_pred - emp_clean)
            else:
                mean_pred = (low + high) / 2.0
                emp_clean = 0.0
                emp_obs = 0.0
                mean_sim = (low + high) / 2.0
                cal_gap_clean = 0.0

            rows.append({
                "split": sp,
                "scenario": sc,
                "controller": pol,
                "bin_idx": b,
                "bin_low": round(low, 2),
                "bin_high": round(high, 2),
                "count": count,
                "mean_pred_prob": round(mean_pred, 4),
                "empirical_clean_acc": round(emp_clean, 4),
                "empirical_obs_acc": round(emp_obs, 4),
                "mean_sim_p_clean": round(mean_sim, 4),
                "calibration_gap_clean": round(cal_gap_clean, 4),
            })

    return pd.DataFrame(rows)


def run_open_loop_suite(
    registry: TaskRegistry,
    scenarios_cfg: Dict[str, Any],
    scenario_names: List[str],
    cohorts: Dict[str, List[int]],
    num_learners: int = 60,
    trials_per_learner: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Executes E1 Open-Loop Replay across cohorts and scenarios."""
    all_events: List[pd.DataFrame] = []
    all_summaries: List[pd.DataFrame] = []

    print("\n" + "=" * 80)
    print("STAGE 1: Open-Loop State-Estimation Replay (Exogenous Task Sequences)")
    print("=" * 80)

    for split_name, seed_list in cohorts.items():
        print(f"\n>>> Running Open-Loop Cohort: {split_name} (Seeds: {seed_list})")
        for sc_name in scenario_names:
            sc_cfg = scenarios_cfg[sc_name]
            for seed in seed_list:
                print(f"  [{split_name}] {sc_name} - Seed {seed} ...", end=" ", flush=True)
                t0 = time.time()
                ev_df, sum_df = run_open_loop_replay(
                    task_registry=registry,
                    scenario_cfg=sc_cfg,
                    num_learners=num_learners,
                    trials_per_learner=trials_per_learner,
                    seed=seed,
                    policies=PHASE_5_POLICIES,
                )
                ev_df["split"] = split_name
                ev_df["scenario"] = sc_name
                sum_df["split"] = split_name
                sum_df["scenario"] = sc_name
                all_events.append(ev_df)
                all_summaries.append(sum_df)
                print(f"done in {time.time()-t0:.2f}s ({len(ev_df)} events)")

    full_events_df = pd.concat(all_events, ignore_index=True)
    full_summary_df = pd.concat(all_summaries, ignore_index=True)

    # Compute paired bootstrap for open-loop
    baseline_pol = "model_aligned_baseline"
    bootstrap_rows = []
    for split_name in cohorts.keys():
        sp_df = full_events_df[full_events_df["split"] == split_name]
        for sc_name in scenario_names:
            sc_df = sp_df[sp_df["scenario"] == sc_name]
            base_df = sc_df[sc_df["controller"] == baseline_pol]
            for pol in PHASE_5_POLICIES:
                if pol == baseline_pol:
                    continue
                v_df = sc_df[sc_df["controller"] == pol]
                for metric in ["primary_mastery_mae_pre", "primary_mastery_mae_post"]:
                    res = compute_paired_cluster_bootstrap(base_df, v_df, metric_name=metric, seed=42)
                    bootstrap_rows.append({
                        "experiment": "open_loop_replay",
                        "split": split_name,
                        "scenario": sc_name,
                        "variant": pol,
                        "baseline": baseline_pol,
                        "metric": metric,
                        "prob_abs_error_diff_variant_minus_baseline": res["diff_variant_minus_baseline"],
                        "point_estimate": res["point_estimate"],
                        "ci_lower": res["ci_lower"],
                        "ci_upper": res["ci_upper"],
                        "std_error": res["std_error"],
                        "interpretation": res["interpretation"],
                        "num_clusters": res["num_clusters"],
                        "resampling_unit": res["resampling_unit"],
                        "sign_convention": res["sign_convention"],
                    })

    boot_df = pd.DataFrame(bootstrap_rows)
    return full_events_df, full_summary_df, boot_df


def run_closed_loop_suite(
    registry: TaskRegistry,
    scenarios_cfg: Dict[str, Any],
    default_cfg: Dict[str, Any],
    scenario_names: List[str],
    cohorts: Dict[str, List[int]],
    num_learners: int = 60,
    trials_per_learner: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Executes E2 Closed-Loop Adaptive Evaluation across all 8 policies."""
    all_events: List[pd.DataFrame] = []
    all_learners: List[pd.DataFrame] = []

    print("\n" + "=" * 80)
    print("STAGE 2: Closed-Loop Multi-Cohort Adaptive Evaluation (8 State Estimators)")
    print("=" * 80)

    for split_name, seed_list in cohorts.items():
        print(f"\n>>> Running Closed-Loop Cohort: {split_name} (Seeds: {seed_list})")
        for sc_name in scenario_names:
            sc_cfg = scenarios_cfg[sc_name]
            for seed in seed_list:
                print(f"  [{split_name}] {sc_name} - Seed {seed} ...", end=" ", flush=True)
                t0 = time.time()
                res_dict, ev_df, lm_df = run_closed_loop_comparison(
                    task_registry=registry,
                    scenario_cfg=sc_cfg,
                    num_learners=num_learners,
                    trials_per_learner=trials_per_learner,
                    seed=seed,
                    controller_configs=default_cfg.get("controllers", {}),
                    policies=PHASE_5_POLICIES,
                )
                ev_df["split"] = split_name
                ev_df["scenario"] = sc_name
                ev_df["seed"] = seed
                lm_df["split"] = split_name
                lm_df["scenario"] = sc_name
                lm_df["seed"] = seed
                all_events.append(ev_df)
                all_learners.append(lm_df)
                print(f"done in {time.time()-t0:.2f}s ({len(ev_df)} events)")

    full_events_df = pd.concat(all_events, ignore_index=True)
    full_learners_df = pd.concat(all_learners, ignore_index=True)

    # Compute scenario level metrics summary
    summary_rows = []
    for (sp, sc, pol), grp in full_events_df.groupby(["split", "scenario", "controller"]):
        m = compute_metrics_for_events(grp)
        m["split"] = sp
        m["scenario"] = sc
        m["controller"] = pol
        summary_rows.append(m)
    scenario_summary_df = pd.DataFrame(summary_rows)

    # Compute paired cluster bootstraps vs model_aligned_baseline
    baseline_pol = "model_aligned_baseline"
    boot_rows = []
    for split_name in cohorts.keys():
        sp_df = full_events_df[full_events_df["split"] == split_name]
        for sc_name in scenario_names:
            sc_df = sp_df[sp_df["scenario"] == sc_name]
            base_df = sc_df[sc_df["controller"] == baseline_pol]
            for pol in PHASE_5_POLICIES:
                if pol == baseline_pol:
                    continue
                v_df = sc_df[sc_df["controller"] == pol]
                for metric in ["prob_abs_error", "primary_mastery_mae_pre", "primary_mastery_mae_post", "at_min_bound"]:
                    res = compute_paired_cluster_bootstrap(base_df, v_df, metric_name=metric, seed=42)
                    boot_rows.append({
                        "experiment": "closed_loop_evaluation",
                        "split": split_name,
                        "scenario": sc_name,
                        "variant": pol,
                        "baseline": baseline_pol,
                        "metric": metric,
                        "prob_abs_error_diff_variant_minus_baseline": res["diff_variant_minus_baseline"],
                        "point_estimate": res["point_estimate"],
                        "ci_lower": res["ci_lower"],
                        "ci_upper": res["ci_upper"],
                        "std_error": res["std_error"],
                        "interpretation": res["interpretation"],
                        "num_clusters": res["num_clusters"],
                        "resampling_unit": res["resampling_unit"],
                        "sign_convention": res["sign_convention"],
                    })

    boot_df = pd.DataFrame(boot_rows)
    return full_events_df, full_learners_df, scenario_summary_df, boot_df


def run_initializer_sensitivity_suite(
    registry: TaskRegistry,
    scenarios_cfg: Dict[str, Any],
    default_cfg: Dict[str, Any],
    seed: int = 101,
    num_learners: int = 50,
    trials_per_learner: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executes E3 Synthetic Questionnaire Proxy Sensitivity Analysis."""
    print("\n" + "=" * 80)
    print("STAGE 3: Initial Mastery Prior & Synthetic Parent-Proxy Sensitivity Experiment")
    print("=" * 80)

    sc_cfg = scenarios_cfg["base_heterogeneous"]
    noise_levels = [0.05, 0.15, 0.30]
    missing_levels = [0.0, 0.30, 0.60]

    all_event_records = []
    summary_records = []

    # 1. Fixed default baseline (0.45)
    init_default = MasteryPriorInitializer.fixed_default(0.45)
    print("  --> Running Baseline: fixed_default (0.45) ...", end=" ", flush=True)
    _, ev_def, lm_def = run_closed_loop_comparison(
        task_registry=registry,
        scenario_cfg=sc_cfg,
        num_learners=num_learners,
        trials_per_learner=trials_per_learner,
        seed=seed,
        controller_configs=default_cfg.get("controllers", {}),
        policies=["expected_transition"],
        prior_initializer=init_default,
    )
    ev_def["condition"] = "fixed_default_0.45"
    ev_def["noise_std"] = 0.0
    ev_def["missingness_rate"] = 0.0
    ev_def["seed"] = seed
    all_event_records.append(ev_def)
    m_def = compute_metrics_for_events(ev_def)
    m_def["condition"] = "fixed_default_0.45"
    m_def["noise_std"] = 0.0
    m_def["missingness_rate"] = 0.0
    m_def["seed"] = seed
    summary_records.append(m_def)
    print("done.")

    # 2. Uninformative prior U(0.05, 0.95)
    rng_uninf = np.random.RandomState(seed + 1)
    uninf_reports = {
        s: SkillPriorEstimate(
            prior_mean=float(rng_uninf.uniform(0.05, 0.95)),
            uncertainty=0.25,
            source="synthetic_proxy_uninformative",
        )
        for s in registry.skill_ids
    }
    init_uninf = MasteryPriorInitializer.from_provided_priors(uninf_reports, fallback_default=0.45)
    print("  --> Running Condition: uninformative_uniform ...", end=" ", flush=True)
    _, ev_uninf, _ = run_closed_loop_comparison(
        task_registry=registry,
        scenario_cfg=sc_cfg,
        num_learners=num_learners,
        trials_per_learner=trials_per_learner,
        seed=seed,
        controller_configs=default_cfg.get("controllers", {}),
        policies=["expected_transition"],
        prior_initializer=init_uninf,
    )
    ev_uninf["condition"] = "uninformative_uniform"
    ev_uninf["noise_std"] = float(np.sqrt(1.0 / 12.0) * 0.90)
    ev_uninf["missingness_rate"] = 0.0
    ev_uninf["seed"] = seed
    all_event_records.append(ev_uninf)
    m_uninf = compute_metrics_for_events(ev_uninf)
    m_uninf["condition"] = "uninformative_uniform"
    m_uninf["noise_std"] = round(float(np.sqrt(1.0 / 12.0) * 0.90), 4)
    m_uninf["missingness_rate"] = 0.0
    m_uninf["seed"] = seed
    summary_records.append(m_uninf)
    print("done.")

    # 3. Grid of synthetic parent proxy noise and missingness
    # Generate population of profiles to extract ground-truth initial skills
    from .simulator import generate_learner_profile
    rng_prof = np.random.RandomState(seed)
    profiles = [
        generate_learner_profile(
            learner_id=i,
            skill_ids=registry.skill_ids,
            rng=rng_prof,
            scenario_params=sc_cfg,
        )
        for i in range(num_learners)
    ]

    for n_std in noise_levels:
        for m_rate in missing_levels:
            cond_name = f"proxy_noise_{n_std:.2f}_miss_{m_rate:.2f}"
            print(f"  --> Running Proxy Condition: {cond_name} ...", end=" ", flush=True)

            # Generate synthetic proxy reports per learner
            # Note: We simulate by creating a customized runner loop or passing aggregate prior
            # For controller initialization test, we compute the cohort average proxy report
            proxy_reports_by_learner = []
            for prof in profiles:
                rep = generate_synthetic_parent_proxy_reports(
                    true_initial_mastery=prof.initial_mastery,
                    skill_ids=registry.skill_ids,
                    noise_std=n_std,
                    missingness_rate=m_rate,
                    seed=seed + prof.learner_id,
                )
                proxy_reports_by_learner.append(rep)

            # We use the mean proxy report across learners as the cohort prior initializer
            avg_priors: Dict[str, SkillPriorEstimate] = {}
            for s in registry.skill_ids:
                vals = [r[s].prior_mean for r in proxy_reports_by_learner if s in r]
                if vals:
                    avg_priors[s] = SkillPriorEstimate(
                        prior_mean=float(np.mean(vals)),
                        uncertainty=float(n_std),
                        source="synthetic_parent_proxy_cohort",
                    )

            init_proxy = MasteryPriorInitializer.from_provided_priors(avg_priors, fallback_default=0.45)
            _, ev_p, _ = run_closed_loop_comparison(
                task_registry=registry,
                scenario_cfg=sc_cfg,
                num_learners=num_learners,
                trials_per_learner=trials_per_learner,
                seed=seed,
                controller_configs=default_cfg.get("controllers", {}),
                policies=["expected_transition"],
                prior_initializer=init_proxy,
            )
            ev_p["condition"] = cond_name
            ev_p["noise_std"] = n_std
            ev_p["missingness_rate"] = m_rate
            ev_p["seed"] = seed
            all_event_records.append(ev_p)

            m_p = compute_metrics_for_events(ev_p)
            m_p["condition"] = cond_name
            m_p["noise_std"] = n_std
            m_p["missingness_rate"] = m_rate
            m_p["seed"] = seed
            summary_records.append(m_p)
            print("done.")

    full_ev_df = pd.concat(all_event_records, ignore_index=True)
    summary_df = pd.DataFrame(summary_records)
    return full_ev_df, summary_df


def run_model_selector_benchmark_suite(
    output_dir: Path,
    seed: int = 101,
    num_learners: int = 60,
    events_per_learner: int = 30,
) -> pd.DataFrame:
    """Executes E4 Testing Model Selector Verification."""
    print("\n" + "=" * 80)
    print("STAGE 4: Offline Model Selector Verification & Benchmarking")
    print("=" * 80)

    models_to_test = ["logistic_baseline", "vanilla_dkt", "reliability_gated_dkt"]
    results = []

    for m in models_to_test:
        print(f"  --> Benchmarking Model Selector: '{m}' ...", end=" ", flush=True)
        t0 = time.time()
        res = run_benchmark(
            config_path=DEFAULT_CONFIG_PATH,
            scenarios_path=SCENARIOS_CONFIG_PATH,
            scenario_names=["simple_dynamics"],
            seeds=[seed],
            num_learners=num_learners,
            events_per_learner=events_per_learner,
            output_dir=output_dir / "selector_runs" / m,
            models=m,
        )
        runtime = time.time() - t0
        print(f"done in {runtime:.2f}s.")
        sc_res = res.get("summary", res)
        rec = {
            "selected_model": m,
            "target_semantics": "P(y_obs=1 | features)",
            "safe_for_clean_controller": False,
            "scenario": "simple_dynamics",
            "seed": seed,
            "runtime_sec": round(runtime, 2),
            "log_loss": sc_res["simple_dynamics"]["models"][m]["log_loss_mean"],
            "brier_score": sc_res["simple_dynamics"]["models"][m]["brier_score_mean"],
            "accuracy": sc_res["simple_dynamics"]["models"][m]["accuracy_mean"],
        }
        results.append(rec)

    # All models execution
    print("  --> Benchmarking Model Selector: 'all' ...", end=" ", flush=True)
    t0 = time.time()
    res_all = run_benchmark(
        config_path=DEFAULT_CONFIG_PATH,
        scenarios_path=SCENARIOS_CONFIG_PATH,
        scenario_names=["simple_dynamics"],
        seeds=[seed],
        num_learners=num_learners,
        events_per_learner=events_per_learner,
        output_dir=output_dir / "selector_runs" / "all",
        models="all",
    )
    runtime = time.time() - t0
    print(f"done in {runtime:.2f}s.")
    sc_res_all = res_all.get("summary", res_all)
    for m in models_to_test:
        rec = {
            "selected_model": f"all_mode_{m}",
            "target_semantics": "P(y_obs=1 | features)",
            "safe_for_clean_controller": False,
            "scenario": "simple_dynamics",
            "seed": seed,
            "runtime_sec": round(runtime, 2),
            "log_loss": sc_res_all["simple_dynamics"]["models"][m]["log_loss_mean"],
            "brier_score": sc_res_all["simple_dynamics"]["models"][m]["brier_score_mean"],
            "accuracy": sc_res_all["simple_dynamics"]["models"][m]["accuracy_mean"],
        }
        results.append(rec)

    return pd.DataFrame(results)


def generate_phase_5_plots(
    open_loop_summary: pd.DataFrame,
    closed_loop_summary: pd.DataFrame,
    calibration_df: pd.DataFrame,
    initializer_summary: pd.DataFrame,
    plots_dir: Path,
):
    """Generates all publication-quality figures for Phase 5."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Open-Loop Mastery Tracking MAE by Scenario (Post-Update)
    plt.figure(figsize=(12, 6))
    scenarios = open_loop_summary["scenario"].unique()
    x = np.arange(len(scenarios))
    width = 0.10

    # Filter to dev or held-out split
    split_focus = "held_out" if "held_out" in open_loop_summary["split"].values else open_loop_summary["split"].iloc[0]
    sub_ol = open_loop_summary[open_loop_summary["split"] == split_focus]

    for i, pol in enumerate(PHASE_5_POLICIES):
        pol_data = sub_ol[sub_ol["controller"] == pol]
        mae_col = "primary_mastery_mae_post" if "primary_mastery_mae_post" in pol_data.columns else "primary_mae_post"
        maes = [
            pol_data[pol_data["scenario"] == sc][mae_col].mean()
            if len(pol_data[pol_data["scenario"] == sc]) > 0 else 0.0
            for sc in scenarios
        ]
        plt.bar(x + i * width, maes, width, label=POLICY_LABELS.get(pol, pol), color=POLICY_COLORS.get(pol, "#888888"))

    plt.xlabel("Simulation Scenario", fontsize=11, fontweight="bold")
    plt.ylabel("Primary Mastery MAE (Post-Update)", fontsize=11, fontweight="bold")
    plt.title(f"E1 Open-Loop State Tracking Error across Scenarios ({split_focus} cohort)", fontsize=13, fontweight="bold")
    plt.xticks(x + width * 3.5, scenarios, rotation=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(plots_dir / "e1_open_loop_mastery_error_by_scenario.png", dpi=300)
    plt.close()

    # Plot 2: Closed-Loop Mastery MAE vs Clean Prob Calibration MAE
    plt.figure(figsize=(11, 6))
    sub_cl = closed_loop_summary[closed_loop_summary["split"] == split_focus]
    for pol in PHASE_5_POLICIES:
        pol_rows = sub_cl[sub_cl["controller"] == pol]
        plt.scatter(
            pol_rows["primary_mastery_mae_post"],
            pol_rows["prob_abs_error"],
            label=POLICY_LABELS.get(pol, pol),
            color=POLICY_COLORS.get(pol, "#888888"),
            s=80,
            alpha=0.85,
        )
    plt.xlabel("Primary Mastery MAE (Post-Update)", fontsize=11, fontweight="bold")
    plt.ylabel("Clean Probability Absolute Error", fontsize=11, fontweight="bold")
    plt.title(f"E2 Closed-Loop Mastery Tracking vs Outcome Calibration ({split_focus} cohort)", fontsize=13, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(plots_dir / "e2_closed_loop_mastery_error_comparison.png", dpi=300)
    plt.close()

    # Plot 3: Closed-Loop Bound Saturation vs Mastery Error
    plt.figure(figsize=(11, 6))
    for pol in PHASE_5_POLICIES:
        pol_rows = sub_cl[sub_cl["controller"] == pol]
        plt.scatter(
            pol_rows["proportion_at_min_bound"],
            pol_rows["primary_mastery_mae_post"],
            label=POLICY_LABELS.get(pol, pol),
            color=POLICY_COLORS.get(pol, "#888888"),
            s=80,
            alpha=0.85,
        )
    plt.xlabel("Proportion at Minimum Difficulty Bound", fontsize=11, fontweight="bold")
    plt.ylabel("Primary Mastery MAE (Post-Update)", fontsize=11, fontweight="bold")
    plt.title("E2 Minimum-Difficulty Saturation vs Mastery Error (Caveat: Lower Saturation != Lower Error)", fontsize=13, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(plots_dir / "e2_closed_loop_difficulty_and_saturation.png", dpi=300)
    plt.close()

    # Plot 4: Clean Probability Calibration Curves
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    cal_sub = calibration_df[calibration_df["split"] == split_focus]
    for pol in ["model_aligned_baseline", "expected_transition", "approx_bayesian_filter"]:
        p_cal = cal_sub[cal_sub["controller"] == pol]
        if len(p_cal) > 0:
            agg = p_cal.groupby("bin_idx").agg({"mean_pred_prob": "mean", "empirical_clean_acc": "mean"}).reset_index()
            plt.plot(agg["mean_pred_prob"], agg["empirical_clean_acc"], marker="o", label=POLICY_LABELS.get(pol, pol), color=POLICY_COLORS.get(pol, "#333"))
    plt.xlabel("Predicted Clean Probability", fontsize=11, fontweight="bold")
    plt.ylabel("Empirical Clean Success Accuracy", fontsize=11, fontweight="bold")
    plt.title(f"E2 Clean Probability Calibration Curves ({split_focus} cohort)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(plots_dir / "e2_clean_prob_calibration_curves.png", dpi=300)
    plt.close()

    # Plot 5: Initializer Noise Sensitivity
    plt.figure(figsize=(9, 5))
    proxy_data = initializer_summary[initializer_summary["condition"].str.startswith("proxy")]
    if len(proxy_data) > 0:
        for m_rate in proxy_data["missingness_rate"].unique():
            sub = proxy_data[proxy_data["missingness_rate"] == m_rate].sort_values("noise_std")
            plt.plot(sub["noise_std"], sub["primary_mastery_mae_post"], marker="s", label=f"Missingness = {int(m_rate*100)}%")
        # Baseline reference
        base_mae = initializer_summary[initializer_summary["condition"] == "fixed_default_0.45"]["primary_mastery_mae_post"].iloc[0]
        plt.axhline(base_mae, color="red", linestyle="--", label="Fixed Default 0.45 Baseline")
        plt.xlabel("Synthetic Proxy Noise Std (sigma)", fontsize=11, fontweight="bold")
        plt.ylabel("Primary Mastery MAE (Post-Update)", fontsize=11, fontweight="bold")
        plt.title("E3 Sensitivity to Synthetic Questionnaire Proxy Noise & Missingness", fontsize=13, fontweight="bold")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(plots_dir / "e3_parent_proxy_noise_sensitivity.png", dpi=300)
        plt.close()


def copy_tree_contents(src_dir: Path, dst_dir: Path):
    """Copies all files and subdirectories from src_dir to dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src_dir):
        rel_path = Path(root).relative_to(src_dir)
        target_root = dst_dir / rel_path
        target_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(Path(root) / f, target_root / f)


def save_configuration_snapshot(out_path: Path):
    """Saves frozen configuration state and checksums before running held-out evaluation."""
    import hashlib
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "Phase 5",
        "description": "Pre-evaluation frozen configuration snapshot",
        "policies": PHASE_5_POLICIES,
        "cohorts": {
            "dev_seeds": DEV_SEEDS,
            "historical_seeds": HISTORICAL_SEEDS,
            "phase_4_eval_seeds": PHASE_4_EVAL_SEEDS,
            "final_held_out_seeds": FINAL_HELD_OUT_SEEDS,
        },
        "file_hashes": {},
    }
    key_files = [
        "configs/default.yaml",
        "configs/scenarios.yaml",
        "src/grow21_simulation_benchmark/simulator.py",
        "src/grow21_simulation_benchmark/closed_loop.py",
        "src/grow21_simulation_benchmark/outcome_model.py",
        "src/grow21_simulation_benchmark/initializers.py",
        "src/grow21_simulation_benchmark/run_phase_5_experiments.py",
    ]
    for rel_f in key_files:
        p = Path(rel_f)
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            snapshot["file_hashes"][rel_f] = h

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f">>> Configuration snapshot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Grow21 Phase 5 Comprehensive Experiments Runner")
    parser.add_argument("--num-learners", type=int, default=60, help="Learners per seed per scenario")
    parser.add_argument("--trials-per-learner", type=int, default=60, help="Trials per learner")
    parser.add_argument("--cohorts", type=str, default="all", help="Cohorts: 'all', 'dev', 'historical', 'phase_4_eval', 'held_out', or comma-separated")
    parser.add_argument("--skip-open-loop", action="store_true", help="Skip Stage 1 open-loop replay")
    parser.add_argument("--skip-closed-loop", action="store_true", help="Skip Stage 2 closed-loop comparison")
    parser.add_argument("--skip-initializers", action="store_true", help="Skip Stage 3 initializer experiment")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip Stage 4 model selector benchmark")
    args = parser.parse_args()

    ensure_output_directories()
    registry = TaskRegistry.from_yaml(DEFAULT_CONFIG_PATH)
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        default_cfg = yaml.safe_load(f)
    with open(SCENARIOS_CONFIG_PATH, "r", encoding="utf-8") as f:
        scenarios_cfg = yaml.safe_load(f)["scenarios"]
    scenario_names = list(scenarios_cfg.keys())

    all_cohort_defs = {
        "dev": DEV_SEEDS,
        "historical": HISTORICAL_SEEDS,
        "phase_4_eval": PHASE_4_EVAL_SEEDS,
        "held_out": FINAL_HELD_OUT_SEEDS,
    }
    if args.cohorts.strip().lower() == "all":
        cohorts = all_cohort_defs
    else:
        selected = [c.strip() for c in args.cohorts.split(",") if c.strip()]
        cohorts = {c: all_cohort_defs[c] for c in selected if c in all_cohort_defs}

    # Save configuration snapshot before evaluation runs
    save_configuration_snapshot(PHASE_5_AUDIT_DIR / "CONFIG_SNAPSHOT.json")
    save_configuration_snapshot(PHASE_5_OUTPUTS_DIR / "CONFIG_SNAPSHOT.json")

    t_start = time.time()
    print("=" * 80)
    print("GROW21 PHASE 5 EXPERIMENTAL CAMPAIGN STARTING")
    print(f"Cohorts: {list(cohorts.keys())} | Learners: {args.num_learners} | Trials: {args.trials_per_learner}")
    print("=" * 80)

    # Stage 1: Open-Loop Replay
    if not args.skip_open_loop:
        ol_events, ol_summary, ol_boot = run_open_loop_suite(
            registry=registry,
            scenarios_cfg=scenarios_cfg,
            scenario_names=scenario_names,
            cohorts=cohorts,
            num_learners=args.num_learners,
            trials_per_learner=args.trials_per_learner,
        )
        ol_events.to_csv(PHASE_5_OPEN_LOOP_DIR / "open_loop_events.csv", index=False)
        ol_summary.to_csv(PHASE_5_OPEN_LOOP_DIR / "open_loop_summary.csv", index=False)
        ol_boot.to_csv(PHASE_5_OPEN_LOOP_DIR / "open_loop_paired_bootstrap.csv", index=False)
    else:
        ol_events = pd.read_csv(PHASE_5_OPEN_LOOP_DIR / "open_loop_events.csv")
        ol_summary = pd.read_csv(PHASE_5_OPEN_LOOP_DIR / "open_loop_summary.csv")

    # Stage 2: Closed-Loop Multi-Cohort Evaluation
    if not args.skip_closed_loop:
        cl_events, cl_learners, cl_summary, cl_boot = run_closed_loop_suite(
            registry=registry,
            scenarios_cfg=scenarios_cfg,
            default_cfg=default_cfg,
            scenario_names=scenario_names,
            cohorts=cohorts,
            num_learners=args.num_learners,
            trials_per_learner=args.trials_per_learner,
        )
        cl_events.to_csv(PHASE_5_CLOSED_LOOP_DIR / "phase_5_event_metrics.csv", index=False)
        cl_learners.to_csv(PHASE_5_CLOSED_LOOP_DIR / "phase_5_learner_summary.csv", index=False)
        cl_summary.to_csv(PHASE_5_CLOSED_LOOP_DIR / "phase_5_scenario_summary.csv", index=False)
        cl_boot.to_csv(PHASE_5_CLOSED_LOOP_DIR / "phase_5_paired_bootstrap.csv", index=False)

        cal_df = compute_calibration_dataframe(cl_events)
        cal_df.to_csv(PHASE_5_CLOSED_LOOP_DIR / "phase_5_calibration_table.csv", index=False)
    else:
        cl_events = pd.read_csv(PHASE_5_CLOSED_LOOP_DIR / "phase_5_event_metrics.csv")
        cl_summary = pd.read_csv(PHASE_5_CLOSED_LOOP_DIR / "phase_5_scenario_summary.csv")
        cal_df = pd.read_csv(PHASE_5_CLOSED_LOOP_DIR / "phase_5_calibration_table.csv")

    # Stage 3: Initial Mastery Prior Sensitivity Experiment
    if not args.skip_initializers:
        init_events, init_summary = run_initializer_sensitivity_suite(
            registry=registry,
            scenarios_cfg=scenarios_cfg,
            default_cfg=default_cfg,
            seed=DEV_SEEDS[0],
            num_learners=args.num_learners,
            trials_per_learner=args.trials_per_learner,
        )
        init_events.to_csv(PHASE_5_INITIALIZERS_DIR / "parent_proxy_events.csv", index=False)
        init_summary.to_csv(PHASE_5_INITIALIZERS_DIR / "parent_proxy_sensitivity_metrics.csv", index=False)
    else:
        init_summary = pd.read_csv(PHASE_5_INITIALIZERS_DIR / "parent_proxy_sensitivity_metrics.csv")

    # Stage 4: Testing Model Selector Benchmark
    if not args.skip_benchmark:
        bm_df = run_model_selector_benchmark_suite(output_dir=PHASE_5_OUTPUTS_DIR / "benchmark")
        (PHASE_5_OUTPUTS_DIR / "benchmark").mkdir(parents=True, exist_ok=True)
        bm_df.to_csv(PHASE_5_OUTPUTS_DIR / "benchmark" / "model_selector_benchmark_metrics.csv", index=False)

    # Generate figures
    print("\n>>> Generating publication figures...")
    generate_phase_5_plots(
        open_loop_summary=ol_summary,
        closed_loop_summary=cl_summary,
        calibration_df=cal_df,
        initializer_summary=init_summary,
        plots_dir=PHASE_5_PLOTS_DIR,
    )

    # Mirror artifacts to audit/phase_5/
    print(f"\n>>> Mirroring all outputs from {PHASE_5_OUTPUTS_DIR} to {PHASE_5_AUDIT_DIR} ...")
    copy_tree_contents(PHASE_5_OUTPUTS_DIR, PHASE_5_AUDIT_DIR)

    total_time = time.time() - t_start
    print("=" * 80)
    print(f"GROW21 PHASE 5 EXPERIMENTAL CAMPAIGN COMPLETED in {total_time:.1f}s ({total_time/60.0:.2f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
