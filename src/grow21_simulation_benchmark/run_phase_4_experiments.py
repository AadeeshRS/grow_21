"""Grow21 Phase 4 Comprehensive State Estimation and Robustness Runner (Audited & Corrected).

Executes:
  Stage 1: Development Seeds [101, 202, 303] for offline diagnostic exploration.
  Stage 2: Historical Baseline Seeds [42, 123, 999] for direct comparison with Phase 3 baseline.
  Stage 3: Held-Out Evaluation Seeds [777, 888, 555] as primary confirmatory out-of-sample evidence.

Policies compared:
  1. model_aligned_baseline (Phase 3 heuristic updater)
  2. transfer_aware (adds secondary skill transfer belief update)
  3. reliability_aware (adds Bayesian soft clean expectation observation update)
  4. combined_transfer_reliability (combines transfer and reliability updates)

Key Audited Improvements:
  - Temporal alignment: Separates pre-outcome belief accuracy (vs latent_mastery_before)
    and post-update belief accuracy (vs latent_mastery_after).
  - Seed-aware paired cluster bootstrap: Resamples (seed, learner_id) clusters with replacement,
    preserving draw multiplicity across 2,000 replicates.
  - Strict split separation: Held-out, historical, and development cohorts are exported
    and analyzed distinctly.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
)
from .evaluate import compute_calibration_table
from .paths import (
    DEFAULT_CONFIG_PATH,
    PHASE_4_AUDIT_DIR,
    PHASE_4_OUTPUTS_DIR,
    PHASE_4_PLOTS_DIR,
    PHASE_4_STATE_ESTIMATION_DIR,
    SCENARIOS_CONFIG_PATH,
    ensure_output_directories,
)
from .tasks import TaskRegistry


PHASE_4_POLICIES = [
    "model_aligned_baseline",
    "transfer_aware",
    "reliability_aware",
    "combined_transfer_reliability",
]

POLICY_LABELS = {
    "model_aligned_baseline": "Baseline (Phase 3)",
    "transfer_aware": "Transfer-Aware",
    "reliability_aware": "Reliability-Aware",
    "combined_transfer_reliability": "Combined (Transfer + Rel)",
}

POLICY_COLORS = {
    "model_aligned_baseline": "#4C72B0",
    "transfer_aware": "#55A868",
    "reliability_aware": "#DD8452",
    "combined_transfer_reliability": "#8172B2",
}


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

    # Pre-outcome mastery error (scored against latent_mastery_before)
    prim_err_pre = events_df["mastery_error_primary_pre"].dropna().to_numpy(dtype=float)
    prim_mae_pre = float(np.mean(np.abs(prim_err_pre))) if len(prim_err_pre) > 0 else None
    prim_rmse_pre = float(np.sqrt(np.mean(prim_err_pre ** 2))) if len(prim_err_pre) > 0 else None
    prim_bias_pre = float(np.mean(prim_err_pre)) if len(prim_err_pre) > 0 else None

    sec_err_pre = events_df["mastery_error_secondary_pre"].dropna().to_numpy(dtype=float)
    sec_mae_pre = float(np.mean(np.abs(sec_err_pre))) if len(sec_err_pre) > 0 else None
    sec_rmse_pre = float(np.sqrt(np.mean(sec_err_pre ** 2))) if len(sec_err_pre) > 0 else None
    sec_bias_pre = float(np.mean(sec_err_pre)) if len(sec_err_pre) > 0 else None

    # Post-update mastery error (scored against latent_mastery_after)
    prim_err_post = events_df["mastery_error_primary_post"].dropna().to_numpy(dtype=float)
    prim_mae_post = float(np.mean(np.abs(prim_err_post))) if len(prim_err_post) > 0 else None
    prim_rmse_post = float(np.sqrt(np.mean(prim_err_post ** 2))) if len(prim_err_post) > 0 else None
    prim_bias_post = float(np.mean(prim_err_post)) if len(prim_err_post) > 0 else None

    sec_err_post = events_df["mastery_error_secondary_post"].dropna().to_numpy(dtype=float)
    sec_mae_post = float(np.mean(np.abs(sec_err_post))) if len(sec_err_post) > 0 else None
    sec_rmse_post = float(np.sqrt(np.mean(sec_err_post ** 2))) if len(sec_err_post) > 0 else None
    sec_bias_post = float(np.mean(sec_err_post)) if len(sec_err_post) > 0 else None

    return {
        "n_events": len(events_df),
        "clean_success_rate": float(np.mean(clean_y)),
        "observed_success_rate": float(np.mean(obs_y)),
        "mean_sim_p_clean": float(np.mean(clean_p)),
        "mean_pred_clean_prob": float(np.mean(pred_p)),
        "signed_prediction_gap": signed_gap,
        "prob_mae": mae_prob,
        "prob_rmse": rmse_prob,
        "brier_score_clean": brier,
        "log_loss_clean": ll,
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
        # Unadorned aliases for compatibility
        "primary_mastery_mae": prim_mae_pre,
        "secondary_mastery_mae": sec_mae_pre,
        # Behavioral metrics
        "mean_difficulty": float(events_df["difficulty"].mean()),
        "proportion_at_min_bound": float(events_df["at_min_bound"].mean()),
        "proportion_at_max_bound": float(events_df["at_max_bound"].mean()),
        "target_reachable_rate": float(events_df["target_reachable"].dropna().mean()) if len(events_df["target_reachable"].dropna()) > 0 else None,
        "safety_intervention_count": int(events_df["safety_intervention"].sum()),
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

    return {
        "point_estimate": round(point_estimate, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "std_error": round(boot_se, 4),
        "num_clusters": int(n_clusters),
        "num_bootstrap": int(num_bootstrap),
        "seed": int(seed),
        "resampling_unit": "(seed, learner_id) cluster",
        "ci_method": "percentile_bootstrap_with_multiplicity",
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


def compute_mastery_estimation_breakdown(events_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates granular mastery MAE and RMSE grouped by split, scenario, controller, timing, and skill."""
    rows = []
    splits = events_df["split"].unique().tolist()
    if "held_out" in splits and "historical" in splits and "combined_eval" not in splits:
        splits.append("combined_eval")

    for sp in splits:
        if sp == "combined_eval":
            sp_df = events_df[events_df["split"].isin(["held_out", "historical"])]
        else:
            sp_df = events_df[events_df["split"] == sp]

        for (sc, pol), grp in sp_df.groupby(["scenario", "controller"]):
            for timing in ["pre_outcome", "post_update"]:
                suffix = "_pre" if timing == "pre_outcome" else "_post"
                prim_err_col = f"mastery_error_primary{suffix}"
                sec_err_col = f"mastery_error_secondary{suffix}"
                prim_est_col = f"estimated_mastery_primary{suffix}"
                sec_est_col = f"estimated_mastery_secondary{suffix}"
                prim_true_col = f"true_mastery_primary{suffix}"
                sec_true_col = f"true_mastery_secondary{suffix}"

                # Primary skills
                prim_sub = grp.dropna(subset=[prim_err_col])
                if len(prim_sub) > 0:
                    err = prim_sub[prim_err_col].to_numpy(dtype=float)
                    rows.append({
                        "split": sp,
                        "scenario": sc,
                        "controller": pol,
                        "timing": timing,
                        "skill_category": "primary_skills",
                        "skill_id": "all_primary",
                        "n_evaluations": len(err),
                        "mean_true_mastery": round(float(prim_sub[prim_true_col].mean()), 4),
                        "mean_estimated_mastery": round(float(prim_sub[prim_est_col].mean()), 4),
                        "signed_bias": round(float(np.mean(err)), 4),
                        "mae": round(float(np.mean(np.abs(err))), 4),
                        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
                    })

                # Secondary-only visual discrimination
                sec_sub = grp[grp["secondary_skill"] == "visual_discrimination"].dropna(subset=[sec_err_col])
                if len(sec_sub) > 0:
                    err = sec_sub[sec_err_col].to_numpy(dtype=float)
                    rows.append({
                        "split": sp,
                        "scenario": sc,
                        "controller": pol,
                        "timing": timing,
                        "skill_category": "secondary_only_skill",
                        "skill_id": "visual_discrimination",
                        "n_evaluations": len(err),
                        "mean_true_mastery": round(float(sec_sub[sec_true_col].mean()), 4),
                        "mean_estimated_mastery": round(float(sec_sub[sec_est_col].mean()), 4),
                        "signed_bias": round(float(np.mean(err)), 4),
                        "mae": round(float(np.mean(np.abs(err))), 4),
                        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
                    })

    return pd.DataFrame(rows)


def run_phase_4_suite(
    registry: TaskRegistry,
    scenarios_cfg: Dict[str, Any],
    historical_seeds: List[int] = [42, 123, 999],
    held_out_seeds: List[int] = [777, 888, 555],
    dev_seeds: List[int] = [101, 202, 303],
    num_learners: int = 60,
    trials_per_learner: int = 60,
) -> Dict[str, Any]:
    """Executes the complete Phase 4 simulation study across all stages and seeds."""
    ensure_output_directories()
    start_time = time.time()

    all_scenarios = [
        "base_heterogeneous",
        "reliability_stress",
        "simple_dynamics",
        "temporal_dynamics",
        "distribution_shift",
    ]

    print("=" * 80)
    print("STARTING GROW21 PHASE 4 STATE ESTIMATION & ROBUSTNESS STUDY (AUDITED)")
    print(f"Scenarios: {all_scenarios}")
    print(f"Policies:  {PHASE_4_POLICIES}")
    print(f"Historical Seeds: {historical_seeds} (N={num_learners} learners/seed)")
    print(f"Held-Out Seeds:   {held_out_seeds} (N={num_learners} learners/seed) [PRIMARY EVIDENCE]")
    print(f"Development Seeds: {dev_seeds} (N={max(15, num_learners // 2)} learners/seed)")
    print("=" * 80)

    all_events_list: List[pd.DataFrame] = []

    # Stage 1: Development Seeds
    print("\n[Stage 1/3] Running Development Seeds for Offline Diagnostics...")
    dev_learners = max(15, num_learners // 2)
    for sc_name in all_scenarios:
        sc_cfg = scenarios_cfg["scenarios"][sc_name]
        for seed in dev_seeds:
            _, ev_df, _ = run_closed_loop_comparison(
                task_registry=registry,
                scenario_cfg=sc_cfg,
                num_learners=dev_learners,
                trials_per_learner=trials_per_learner,
                seed=seed,
                policies=PHASE_4_POLICIES,
            )
            ev_df["scenario"] = sc_name
            ev_df["seed"] = seed
            ev_df["split"] = "development"
            all_events_list.append(ev_df)
            print(f"  Stage 1: {sc_name} | Seed {seed} | {len(ev_df)} events")

    # Stage 2: Historical Baseline Seeds
    print("\n[Stage 2/3] Running Historical Seeds [42, 123, 999] Closed-Loop Comparison...")
    for sc_name in all_scenarios:
        sc_cfg = scenarios_cfg["scenarios"][sc_name]
        for seed in historical_seeds:
            _, ev_df, _ = run_closed_loop_comparison(
                task_registry=registry,
                scenario_cfg=sc_cfg,
                num_learners=num_learners,
                trials_per_learner=trials_per_learner,
                seed=seed,
                policies=PHASE_4_POLICIES,
            )
            ev_df["scenario"] = sc_name
            ev_df["seed"] = seed
            ev_df["split"] = "historical"
            all_events_list.append(ev_df)
            print(f"  Stage 2: {sc_name} | Seed {seed} | {len(ev_df)} events")

    # Stage 3: Held-Out Evaluation Seeds (Primary Evidence)
    print("\n[Stage 3/3] Running Held-Out Evaluation Seeds [777, 888, 555]...")
    for sc_name in all_scenarios:
        sc_cfg = scenarios_cfg["scenarios"][sc_name]
        for seed in held_out_seeds:
            _, ev_df, _ = run_closed_loop_comparison(
                task_registry=registry,
                scenario_cfg=sc_cfg,
                num_learners=num_learners,
                trials_per_learner=trials_per_learner,
                seed=seed,
                policies=PHASE_4_POLICIES,
            )
            ev_df["scenario"] = sc_name
            ev_df["seed"] = seed
            ev_df["split"] = "held_out"
            all_events_list.append(ev_df)
            print(f"  Stage 3: {sc_name} | Seed {seed} | {len(ev_df)} events")

    total_events_df = pd.concat(all_events_list, ignore_index=True)
    print(f"\nAll simulations completed! Total events generated: {len(total_events_df)}")

    # Persist Datasets
    print("\nPersisting Phase 4 datasets...")
    event_metrics_path = PHASE_4_STATE_ESTIMATION_DIR / "phase_4_event_metrics.csv"
    total_events_df.to_csv(event_metrics_path, index=False)
    print(f"  -> Saved {event_metrics_path} ({len(total_events_df)} rows)")

    cal_bins_df = compute_calibration_dataframe(total_events_df)
    cal_bins_path = PHASE_4_STATE_ESTIMATION_DIR / "phase_4_calibration_bins.csv"
    cal_bins_df.to_csv(cal_bins_path, index=False)
    print(f"  -> Saved {cal_bins_path} ({len(cal_bins_df)} rows)")

    mastery_metrics_df = compute_mastery_estimation_breakdown(total_events_df)
    mastery_metrics_path = PHASE_4_STATE_ESTIMATION_DIR / "phase_4_mastery_estimation_metrics.csv"
    mastery_metrics_df.to_csv(mastery_metrics_path, index=False)
    print(f"  -> Saved {mastery_metrics_path} ({len(mastery_metrics_df)} rows)")

    # Build Multi-Split Hierarchical Summary JSON
    print("\nComputing multi-split hierarchical summary JSON with seed-aware paired cluster bootstrap...")
    splits_to_evaluate = {
        "held_out": total_events_df[total_events_df["split"] == "held_out"],
        "historical": total_events_df[total_events_df["split"] == "historical"],
        "development": total_events_df[total_events_df["split"] == "development"],
        "combined_eval": total_events_df[total_events_df["split"].isin(["historical", "held_out"])],
    }

    summary_dict: Dict[str, Any] = {}

    for split_key, sp_df in splits_to_evaluate.items():
        summary_dict[split_key] = {}
        for sc_name in all_scenarios:
            sc_df = sp_df[sp_df["scenario"] == sc_name]
            pol_metrics: Dict[str, Dict[str, Any]] = {}
            for pol in PHASE_4_POLICIES:
                p_df = sc_df[sc_df["controller"] == pol]
                m = compute_metrics_for_events(p_df)

                # Seed-level variance
                seed_gaps = []
                seed_clean_succ = []
                seed_obs_succ = []
                seed_briers = []
                seed_mastery_maes_pre = []
                seed_mastery_maes_post = []
                for s_id, s_grp in p_df.groupby("seed"):
                    seed_clean_succ.append(float(s_grp["clean_outcome"].mean()))
                    seed_obs_succ.append(float(s_grp["observed_outcome"].mean()))
                    seed_gaps.append(float(np.mean(s_grp["predicted_clean_prob"] - s_grp["simulator_p_clean"])))
                    seed_briers.append(float(brier_score_loss(s_grp["clean_outcome"], np.clip(s_grp["predicted_clean_prob"], 1e-6, 1 - 1e-6))))
                    p_err_pre = s_grp["mastery_error_primary_pre"].dropna()
                    if len(p_err_pre) > 0:
                        seed_mastery_maes_pre.append(float(np.mean(np.abs(p_err_pre))))
                    p_err_post = s_grp["mastery_error_primary_post"].dropna()
                    if len(p_err_post) > 0:
                        seed_mastery_maes_post.append(float(np.mean(np.abs(p_err_post))))

                m["clean_success_rate_seed_std"] = round(float(np.std(seed_clean_succ)), 4)
                m["observed_success_rate_seed_std"] = round(float(np.std(seed_obs_succ)), 4)
                m["signed_prediction_gap_seed_std"] = round(float(np.std(seed_gaps)), 4)
                m["brier_score_seed_std"] = round(float(np.std(seed_briers)), 4)
                m["primary_mastery_mae_pre_seed_std"] = round(float(np.std(seed_mastery_maes_pre)), 4) if seed_mastery_maes_pre else None
                m["primary_mastery_mae_post_seed_std"] = round(float(np.std(seed_mastery_maes_post)), 4) if seed_mastery_maes_post else None
                pol_metrics[pol] = m

            # Paired seed-aware cluster bootstrap vs baseline
            base_df = sc_df[sc_df["controller"] == "model_aligned_baseline"]
            paired_cluster_boot: Dict[str, Any] = {}
            for pol in ["transfer_aware", "reliability_aware", "combined_transfer_reliability"]:
                pol_df = sc_df[sc_df["controller"] == pol]

                prob_boot = compute_paired_cluster_bootstrap(base_df, pol_df, metric_name="prob_abs_error", num_bootstrap=2000, seed=42)
                prim_pre_boot = compute_paired_cluster_bootstrap(base_df, pol_df, metric_name="primary_mastery_mae_pre", num_bootstrap=2000, seed=42)
                prim_post_boot = compute_paired_cluster_bootstrap(base_df, pol_df, metric_name="primary_mastery_mae_post", num_bootstrap=2000, seed=42)
                sat_boot = compute_paired_cluster_bootstrap(base_df, pol_df, metric_name="at_min_bound", num_bootstrap=2000, seed=42)

                paired_cluster_boot[pol] = {
                    "prob_abs_error_diff_variant_minus_baseline": prob_boot,
                    "prob_abs_error_reduction": prob_boot,  # Retained as backwards-compatibility alias
                    "primary_mastery_mae_pre_diff": prim_pre_boot,
                    "primary_mastery_mae_post_diff": prim_post_boot,
                    "at_min_bound_diff": sat_boot,
                }

            summary_dict[split_key][sc_name] = {
                "policies": pol_metrics,
                "paired_cluster_bootstrap_vs_baseline": paired_cluster_boot,
            }

    summary_json_path = PHASE_4_STATE_ESTIMATION_DIR / "phase_4_scenario_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"  -> Saved {summary_json_path}")

    # Mirror to audit directory
    with open(PHASE_4_AUDIT_DIR / "phase_4_scenario_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    cal_bins_df.to_csv(PHASE_4_AUDIT_DIR / "phase_4_calibration_bins.csv", index=False)
    mastery_metrics_df.to_csv(PHASE_4_AUDIT_DIR / "phase_4_mastery_estimation_metrics.csv", index=False)

    # Generate Plots on Primary Held-Out Cohort
    print("\nGenerating Phase 4 comparative evaluation plots on Held-Out Cohort [777, 888, 555]...")
    held_out_events = total_events_df[total_events_df["split"] == "held_out"]
    generate_all_plots(held_out_events, summary_dict["held_out"])

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"PHASE 4 SIMULATION AND EVALUATION COMPLETE IN {elapsed}s")
    print("=" * 80)

    # Print executive tables for Held-Out Evaluation Cohort
    print("\nPRIMARY HEADLINE EVIDENCE: HELD-OUT COHORT [Seeds 777, 888, 555] (`base_heterogeneous`)")
    print("-" * 125)
    print(f"{'Policy':<30} | {'Clean Succ':<11} | {'Sim p':<10} | {'Pred p':<10} | {'Signed Gap':<12} | {'Pre-MAE':<10} | {'Post-MAE':<10} | {'At D_min':<10}")
    print("-" * 125)
    for pol in PHASE_4_POLICIES:
        p_stats = summary_dict["held_out"]["base_heterogeneous"]["policies"][pol]
        cs = f"{p_stats['clean_success_rate']*100:.2f}%"
        sim_p = f"{p_stats['mean_sim_p_clean']*100:.2f}%"
        pred_p = f"{p_stats['mean_pred_clean_prob']*100:.2f}%"
        gap = f"{p_stats['signed_prediction_gap']*100:+.2f} pp"
        pre_m = f"{p_stats['primary_mastery_mae_pre']:.4f}" if p_stats['primary_mastery_mae_pre'] else "N/A"
        post_m = f"{p_stats['primary_mastery_mae_post']:.4f}" if p_stats['primary_mastery_mae_post'] else "N/A"
        min_b = f"{p_stats['proportion_at_min_bound']*100:.1f}%"
        print(f"{pol:<30} | {cs:<11} | {sim_p:<10} | {pred_p:<10} | {gap:<12} | {pre_m:<10} | {post_m:<10} | {min_b:<10}")

    print("\nPRIMARY HEADLINE EVIDENCE: HELD-OUT COHORT [Seeds 777, 888, 555] (`distribution_shift`)")
    print("-" * 125)
    print(f"{'Policy':<30} | {'Clean Succ':<11} | {'Sim p':<10} | {'Pred p':<10} | {'Signed Gap':<12} | {'Pre-MAE':<10} | {'Post-MAE':<10} | {'At D_min':<10}")
    print("-" * 125)
    for pol in PHASE_4_POLICIES:
        p_stats = summary_dict["held_out"]["distribution_shift"]["policies"][pol]
        cs = f"{p_stats['clean_success_rate']*100:.2f}%"
        sim_p = f"{p_stats['mean_sim_p_clean']*100:.2f}%"
        pred_p = f"{p_stats['mean_pred_clean_prob']*100:.2f}%"
        gap = f"{p_stats['signed_prediction_gap']*100:+.2f} pp"
        pre_m = f"{p_stats['primary_mastery_mae_pre']:.4f}" if p_stats['primary_mastery_mae_pre'] else "N/A"
        post_m = f"{p_stats['primary_mastery_mae_post']:.4f}" if p_stats['primary_mastery_mae_post'] else "N/A"
        min_b = f"{p_stats['proportion_at_min_bound']*100:.1f}%"
        print(f"{pol:<30} | {cs:<11} | {sim_p:<10} | {pred_p:<10} | {gap:<12} | {pre_m:<10} | {post_m:<10} | {min_b:<10}")
    print("=" * 125)

    return {
        "elapsed_seconds": elapsed,
        "total_events": len(total_events_df),
        "summary": summary_dict,
    }


def generate_all_plots(events_df: pd.DataFrame, scenario_summary: Dict[str, Any]):
    """Generates all 5 required publication-grade plots on the evaluated cohort."""
    scenarios = ["base_heterogeneous", "reliability_stress", "simple_dynamics", "temporal_dynamics", "distribution_shift"]
    titles = ["Base Heterogeneous", "Reliability Stress", "Simple Dynamics", "Temporal Dynamics", "Distribution Shift"]
    width = 0.20
    x = np.arange(len(scenarios))

    # 1. Calibration Curves
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)
    for ax, sc, title in zip(axes, scenarios, titles):
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calib")
        for pol in PHASE_4_POLICIES:
            p_df = events_df[(events_df["scenario"] == sc) & (events_df["controller"] == pol)]
            cal, _ = compute_calibration_table(p_df["clean_outcome"].to_numpy(), np.clip(p_df["predicted_clean_prob"].to_numpy(), 1e-6, 1 - 1e-6))
            preds = [b["mean_pred_prob"] for b in cal if b["count"] > 0]
            accs = [b["empirical_acc"] for b in cal if b["count"] > 0]
            ax.plot(preds, accs, marker="o", markersize=4, label=POLICY_LABELS[pol], color=POLICY_COLORS[pol])
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].set_ylabel("Empirical Clean Success Rate", fontsize=10)
    axes[-1].legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    cal_plot_path = PHASE_4_PLOTS_DIR / "phase_4_calibration_curves.png"
    plt.savefig(cal_plot_path, dpi=300)
    plt.close()

    # 2. Signed Prediction Gaps by Scenario
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, pol in enumerate(PHASE_4_POLICIES):
        gaps = [scenario_summary[sc]["policies"][pol]["signed_prediction_gap"] * 100 for sc in scenarios]
        errs = [scenario_summary[sc]["policies"][pol]["signed_prediction_gap_seed_std"] * 100 for sc in scenarios]
        ax.bar(x + (i - 1.5) * width, gaps, width, yerr=errs, capsize=3, label=POLICY_LABELS[pol], color=POLICY_COLORS[pol], alpha=0.85)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(titles, fontsize=10, fontweight="bold")
    ax.set_ylabel("Signed Gap (Pred p - Sim p_clean) [pp]", fontsize=11)
    ax.set_title("Held-Out Signed Clean Prediction Gap Across Controller Variants", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.tight_layout()
    gap_scen_plot = PHASE_4_PLOTS_DIR / "phase_4_prediction_gaps_by_scenario.png"
    plt.savefig(gap_scen_plot, dpi=300)
    plt.close()

    # 3. Prediction Gaps by Game
    fig, ax = plt.subplots(figsize=(14, 5.5))
    games = sorted(events_df["game_id"].unique())
    x_g = np.arange(len(games))
    w_g = 0.20
    base_events = events_df[events_df["scenario"] == "base_heterogeneous"]
    for i, pol in enumerate(PHASE_4_POLICIES):
        game_gaps = []
        for gid in games:
            g_df = base_events[(base_events["game_id"] == gid) & (base_events["controller"] == pol)]
            gap = float(np.mean(g_df["predicted_clean_prob"] - g_df["simulator_p_clean"])) * 100
            game_gaps.append(gap)
        ax.bar(x_g + (i - 1.5) * w_g, game_gaps, w_g, label=POLICY_LABELS[pol], color=POLICY_COLORS[pol], alpha=0.85)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x_g)
    ax.set_xticklabels([f"{gid}\n({'VD sec' if gid in ['trace_glow', 'bubble_pop_brigade', 'category_sorting', 'visual_flip_match', 'pattern_train'] else 'no VD'})" for gid in games], fontsize=8)
    ax.set_ylabel("Signed Gap [pp]", fontsize=11)
    ax.set_title("Held-Out Base Prediction Gaps by Game (Secondary Skill Transfer)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.tight_layout()
    gap_game_plot = PHASE_4_PLOTS_DIR / "phase_4_prediction_gaps_by_game.png"
    plt.savefig(gap_game_plot, dpi=300)
    plt.close()

    # 4. Mastery Estimation MAE: Pre-Outcome vs Post-Update Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    for i, pol in enumerate(PHASE_4_POLICIES):
        pre_maes = [scenario_summary[sc]["policies"][pol]["primary_mastery_mae_pre"] for sc in scenarios]
        post_maes = [scenario_summary[sc]["policies"][pol]["primary_mastery_mae_post"] for sc in scenarios]
        ax1.bar(x + (i - 1.5) * width, pre_maes, width, label=POLICY_LABELS[pol], color=POLICY_COLORS[pol], alpha=0.85)
        ax2.bar(x + (i - 1.5) * width, post_maes, width, label=POLICY_LABELS[pol], color=POLICY_COLORS[pol], alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(titles, fontsize=9, fontweight="bold")
    ax1.set_ylabel("Primary Mastery MAE (Pre-Outcome)", fontsize=10)
    ax1.set_title("Pre-Outcome Primary Mastery MAE (Estimate vs Latent Before)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")

    ax2.set_xticks(x)
    ax2.set_xticklabels(titles, fontsize=9, fontweight="bold")
    ax2.set_ylabel("Primary Mastery MAE (Post-Update)", fontsize=10)
    ax2.set_title("Post-Update Primary Mastery MAE (Estimate vs Latent After)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.tight_layout()
    mastery_err_plot = PHASE_4_PLOTS_DIR / "phase_4_mastery_estimation_errors.png"
    plt.savefig(mastery_err_plot, dpi=300)
    plt.close()

    # 5. Difficulty Trajectories Sample
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    sample_df = events_df[(events_df["scenario"] == "base_heterogeneous") & (events_df["seed"] == events_df["seed"].unique()[0])]
    learners = [0, 1, 2]
    for ax, pol in zip(axes, PHASE_4_POLICIES):
        p_df = sample_df[sample_df["controller"] == pol]
        for l_id in learners:
            l_df = p_df[p_df["learner_id"] == l_id].sort_values("step_idx")
            ax.plot(l_df["step_idx"], l_df["difficulty"], marker="o", markersize=2.5, linewidth=1.2, alpha=0.8, label=f"Learner {l_id}")
        ax.axhline(0.08, color="red", linestyle=":", linewidth=1.2, label="D_min (0.08)")
        ax.axhline(0.95, color="gray", linestyle=":", linewidth=1.2, label="D_max (0.95)")
        ax.set_title(POLICY_LABELS[pol], fontsize=10, fontweight="bold")
        ax.set_xlabel("Trial Step", fontsize=10)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.4)
    axes[0].set_ylabel("Difficulty D", fontsize=10)
    axes[-1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    traj_plot = PHASE_4_PLOTS_DIR / "phase_4_difficulty_trajectories.png"
    plt.savefig(traj_plot, dpi=300)
    plt.close()

    # Mirror plots to audit directory
    for p_file in PHASE_4_PLOTS_DIR.glob("*.png"):
        (PHASE_4_AUDIT_DIR / p_file.name).write_bytes(p_file.read_bytes())
    print("  -> All 5 comparison plots generated and mirrored to outputs/audit/phase_4/")


def main():
    parser = argparse.ArgumentParser(description="Run Grow21 Phase 4 Simulation & State Estimation Study.")
    parser.add_argument("--num-learners", type=int, default=60, help="Cohort size per scenario/seed")
    parser.add_argument("--trials-per-learner", type=int, default=60, help="Trial horizon per learner")
    parser.add_argument("--historical-seeds", nargs="+", type=int, default=[42, 123, 999])
    parser.add_argument("--held-out-seeds", nargs="+", type=int, default=[777, 888, 555])
    parser.add_argument("--dev-seeds", nargs="+", type=int, default=[101, 202, 303])
    args = parser.parse_args()

    registry = TaskRegistry.from_yaml(DEFAULT_CONFIG_PATH)
    with open(SCENARIOS_CONFIG_PATH, "r", encoding="utf-8") as f:
        scenarios_cfg = yaml.safe_load(f)

    run_phase_4_suite(
        registry=registry,
        scenarios_cfg=scenarios_cfg,
        historical_seeds=args.historical_seeds,
        held_out_seeds=args.held_out_seeds,
        dev_seeds=args.dev_seeds,
        num_learners=args.num_learners,
        trials_per_learner=args.trials_per_learner,
    )


if __name__ == "__main__":
    main()
