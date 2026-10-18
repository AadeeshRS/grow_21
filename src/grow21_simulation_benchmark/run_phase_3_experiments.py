"""Phase 3 Comprehensive Simulation, Evaluation, and Diagnostic Runner.

Executes closed-loop simulations across all 5 canonical scenarios and seeds [42, 123, 999],
comparing:
  1. fixed_difficulty
  2. streak_adaptive
  3. legacy_surrogate_target_adaptive
  4. model_aligned_target_adaptive

Computes formula-parity verification, calibration, reachability, bound saturation,
and difficulty oscillation metrics, saving all artifacts under outputs/phase_3/ and outputs/audit/phase_3/.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .closed_loop import run_closed_loop_comparison
from .outcome_model import (
    compute_clean_task_success_prob,
    compute_composite_effective_mastery,
    compute_difficulty_curve,
    compute_difficulty_threshold,
    compute_game_clean_success_prob,
    solve_target_difficulty,
)
from .paths import (
    DEFAULT_CONFIG_PATH,
    PHASE_3_AUDIT_DIR,
    PHASE_3_CLOSED_LOOP_DIR,
    PHASE_3_DIAGNOSTICS_DIR,
    PHASE_3_PLOTS_DIR,
    SCENARIOS_CONFIG_PATH,
    ensure_output_directories,
)
from .tasks import TaskRegistry


def run_formula_parity_audit(registry: TaskRegistry) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluates numerical parity between canonical outcome model and simulator across all 9 games."""
    records: List[Dict[str, Any]] = []
    mastery_grid = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]
    diff_grid = [0.0, 0.08, 0.25, 0.50, 0.75, 0.95, 1.0]
    session_counters = [1, 5, 10, 11, 15, 20]
    fatigue_sensitivities = [0.005, 0.02, 0.035]

    max_abs_err = 0.0

    for gid in registry.game_ids:
        game = registry.games[gid]
        for m in mastery_grid:
            for d in diff_grid:
                for sc in session_counters:
                    for fs in fatigue_sensitivities:
                        # 1. Canonical function
                        p_canonical = compute_game_clean_success_prob(
                            game=game,
                            effective_mastery=m,
                            difficulty=d,
                            session_trial_counter=sc,
                            fatigue_sensitivity=fs,
                        )

                        # 2. Simulator ground truth formula
                        curve = 0.25 * d + 0.75 * (d ** 1.35)
                        threshold = game.difficulty_offset + game.difficulty_scale * curve
                        logit_val = game.discrimination * (m - threshold)
                        sig = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit_val))))

                        excess = max(0, sc - 10)
                        fatigue_lapse = min(0.15, max(0.0, fs * excess))
                        total_lapse = min(0.35, game.lapse_ceiling + fatigue_lapse)
                        total_guess = game.guess_floor
                        p_sim = float(np.clip(total_guess + (1.0 - total_guess - total_lapse) * sig, 0.01, 0.99))

                        abs_err = abs(p_canonical - p_sim)
                        if abs_err > max_abs_err:
                            max_abs_err = abs_err

                        records.append({
                            "game_id": gid,
                            "effective_mastery": m,
                            "difficulty": d,
                            "session_trial_counter": sc,
                            "fatigue_sensitivity": fs,
                            "p_canonical": round(p_canonical, 6),
                            "p_simulator": round(p_sim, 6),
                            "abs_error": abs_err,
                        })

    df = pd.DataFrame(records)
    summary = {
        "total_evaluations": len(df),
        "total_games": len(registry.game_ids),
        "max_absolute_error": max_abs_err,
        "mean_absolute_error": float(df["abs_error"].mean()),
        "parity_verified": bool(max_abs_err < 1e-10),
    }
    return df, summary


def plot_difficulty_trajectories_sample(events_df: pd.DataFrame, output_path: Path):
    """Generates trajectory plots comparing sample learner difficulty curves across 4 policies."""
    policies = [
        "fixed_difficulty",
        "streak_adaptive",
        "legacy_surrogate_target_adaptive",
        "model_aligned_target_adaptive",
    ]
    titles = [
        "Fixed Difficulty (D=0.50)",
        "Streak Adaptive (2-up/1-down)",
        "Phase 2 Legacy Surrogate (gamma=2.4)",
        "Phase 3 Model-Aligned Target Adaptive",
    ]
    colors = ["#4C72B0", "#DD8452", "#C44E52", "#55A868"]

    # Filter to base_heterogeneous seed 42
    sample_df = events_df[
        (events_df["scenario"] == "base_heterogeneous") & (events_df["seed"] == 42)
    ]
    learners = [0, 1, 2]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, p_name, title, col in zip(axes, policies, titles, colors):
        p_df = sample_df[sample_df["controller"] == p_name]
        for l_id in learners:
            l_df = p_df[p_df["learner_id"] == l_id].sort_values("step_idx")
            ax.plot(
                l_df["step_idx"],
                l_df["difficulty"],
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                alpha=0.8,
                label=f"Learner {l_id}",
            )
        ax.axhline(0.08, color="red", linestyle=":", linewidth=1.2, label="D_min (0.08)")
        ax.axhline(0.95, color="gray", linestyle=":", linewidth=1.2, label="D_max (0.95)")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Trial Step", fontsize=10)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.5)
        if ax == axes[0]:
            ax.set_ylabel("Abstract Difficulty D", fontsize=10)
            ax.legend(loc="upper right", fontsize=7.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_saturation_and_reachability(summary_df: pd.DataFrame, output_path: Path):
    """Bar chart comparing bound saturation (% at D_min) and target reachability across policies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

    base_df = summary_df[summary_df["scenario"] == "base_heterogeneous"]
    policies = [
        "fixed_difficulty",
        "streak_adaptive",
        "legacy_surrogate_target_adaptive",
        "model_aligned_target_adaptive",
    ]
    labels = ["Fixed (0.50)", "Streak Heuristic", "Legacy Surrogate", "Model Aligned"]
    colors = ["#4C72B0", "#DD8452", "#C44E52", "#55A868"]

    min_bounds = []
    reachables = []
    for p in policies:
        sub = base_df[base_df["policy"] == p]
        min_bounds.append(sub["proportion_at_min_bound_mean"].mean() * 100)
        r_val = sub["target_reachable_proportion_mean"].mean()
        reachables.append((r_val * 100) if pd.notna(r_val) else 0.0)

    x = np.arange(len(policies))
    ax1.bar(labels, min_bounds, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Trials at Minimum Bound D_min = 0.08 (%)", fontsize=11)
    ax1.set_title("Lower-Bound Saturation (% at D_min)", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(min_bounds):
        ax1.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9.5, fontweight="bold")

    # Reachability only applies to target policies
    target_labels = ["Legacy Surrogate", "Model Aligned"]
    target_reach = [reachables[2], reachables[3]]
    t_colors = ["#C44E52", "#55A868"]
    ax2.bar(target_labels, target_reach, color=t_colors, alpha=0.85, edgecolor="black", linewidth=0.5, width=0.5)
    ax2.set_ylabel("Feasible Target Fraction (%)", fontsize=11)
    ax2.set_title("Target Reachability Rate (p* = 0.75)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(target_reach):
        ax2.text(i, v + 2.0, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_policy_comparisons(summary_df: pd.DataFrame, output_path: Path):
    """Bar chart comparing clean success rates and difficulty dynamics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))
    base_df = summary_df[summary_df["scenario"] == "base_heterogeneous"]
    policies = [
        "fixed_difficulty",
        "streak_adaptive",
        "legacy_surrogate_target_adaptive",
        "model_aligned_target_adaptive",
    ]
    labels = ["Fixed", "Streak", "Legacy Target", "Aligned Target"]

    clean_rates = [base_df[base_df["policy"] == p]["clean_success_rate_mean"].mean() * 100 for p in policies]
    obs_rates = [base_df[base_df["policy"] == p]["observed_success_rate_mean"].mean() * 100 for p in policies]

    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w/2, clean_rates, width=w, label="Clean Success Rate (Latent)", color="#2b5c8f", alpha=0.85)
    ax1.bar(x + w/2, obs_rates, width=w, label="Observed Success Rate", color="#e8853b", alpha=0.85)
    ax1.axhline(75.0, color="green", linestyle="--", linewidth=1.5, label="Target p* = 75%")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Success Rate (%)", fontsize=11)
    ax1.set_title("Success Rates: Clean Task vs. Observed Labels", fontsize=12, fontweight="bold")
    ax1.set_ylim(40, 90)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8.5)

    reversals = [base_df[base_df["policy"] == p]["filtered_direction_reversals_mean"].mean() for p in policies]
    variations = [base_df[base_df["policy"] == p]["total_difficulty_variation_mean"].mean() for p in policies]

    ax2.bar(x - w/2, reversals, width=w, label="Filtered Reversals (Non-Zero)", color="#488f31", alpha=0.85)
    ax2.bar(x + w/2, variations, width=w, label="Total Difficulty Variation (Σ|ΔD|)", color="#de425b", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Count / Variation Magnitude", fontsize=11)
    ax2.set_title("Controller Dynamics & Direction Reversals", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_calibration_curves(events_df: pd.DataFrame, output_path: Path):
    """Reliability diagram comparing predicted clean probability against actual clean outcome."""
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    target_policies = ["legacy_surrogate_target_adaptive", "model_aligned_target_adaptive"]
    policy_labels = ["Legacy Surrogate (gamma=2.4)", "Model-Aligned Controller"]
    colors = ["#C44E52", "#55A868"]

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

    for p_name, label, col in zip(target_policies, policy_labels, colors):
        sub = events_df[events_df["controller"] == p_name].dropna(subset=["predicted_clean_prob", "clean_outcome"])
        if len(sub) == 0:
            continue
        preds = sub["predicted_clean_prob"].values
        actuals = sub["clean_outcome"].values

        # 10 bins
        bins = np.linspace(0.0, 1.0, 11)
        bin_preds = []
        bin_actuals = []
        for i in range(len(bins) - 1):
            mask = (preds >= bins[i]) & (preds < bins[i + 1]) if i < len(bins) - 2 else (preds >= bins[i]) & (preds <= bins[i + 1])
            if np.sum(mask) > 10:
                bin_preds.append(np.mean(preds[mask]))
                bin_actuals.append(np.mean(actuals[mask]))

        ax.plot(bin_preds, bin_actuals, marker="o", linewidth=2.0, label=label, color=col)

    ax.set_xlabel("Predicted Clean Success Probability", fontsize=11)
    ax.set_ylabel("Empirical Clean Success Rate", fontsize=11)
    ax.set_title("Calibration: Predicted vs. Empirical Clean Success", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_all_experiments(
    num_learners: int = 60,
    trials_per_learner: int = 60,
    seeds: List[int] = [42, 123, 999],
) -> Dict[str, Any]:
    """Executes the full Phase 3 experimental campaign across all scenarios, seeds, and policies."""
    start_time = time.time()
    ensure_output_directories()

    registry = TaskRegistry.from_yaml(DEFAULT_CONFIG_PATH)
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        default_cfg = yaml.safe_load(f)
    with open(SCENARIOS_CONFIG_PATH, "r", encoding="utf-8") as f:
        scenarios_cfg = yaml.safe_load(f)["scenarios"]

    controller_cfgs = default_cfg.get("controllers", {})
    all_scenarios = list(scenarios_cfg.keys())

    print("=" * 80)
    print("STARTING GROW21 PHASE 3 EXPERIMENTAL CAMPAIGN")
    print(f"Scenarios ({len(all_scenarios)}): {all_scenarios}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Cohort Size: {num_learners} learners x {trials_per_learner} trials")
    print("Policies: fixed_difficulty, streak_adaptive, legacy_surrogate_target_adaptive, model_aligned_target_adaptive")
    print("=" * 80)

    # 1. Run Formula Parity Audit
    print("\n[Step 1/4] Running Shared Canonical Formula Parity Audit across all 9 games...")
    parity_df, parity_summary = run_formula_parity_audit(registry)
    parity_csv = PHASE_3_DIAGNOSTICS_DIR / "formula_parity_results.csv"
    parity_df.to_csv(parity_csv, index=False)
    parity_json = PHASE_3_DIAGNOSTICS_DIR / "formula_parity_summary.json"
    with open(parity_json, "w", encoding="utf-8") as f:
        json.dump(parity_summary, f, indent=2)
    # Mirror to audit dir
    with open(PHASE_3_AUDIT_DIR / "formula_parity_summary.json", "w", encoding="utf-8") as f:
        json.dump(parity_summary, f, indent=2)
    print(f"  -> Verified {parity_summary['total_evaluations']} grid evaluations. Max abs error = {parity_summary['max_absolute_error']:.2e}")

    # 2. Run Closed-Loop Simulations
    print("\n[Step 2/4] Executing multi-scenario closed-loop simulation runs...")
    all_events_list: List[pd.DataFrame] = []
    all_mastery_list: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, Any]] = []

    total_runs = len(all_scenarios) * len(seeds)
    run_idx = 0

    for sc_name in all_scenarios:
        sc_cfg = scenarios_cfg[sc_name]
        for seed in seeds:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] Running {sc_name} (seed={seed})...")
            results, ev_df, m_df = run_closed_loop_comparison(
                task_registry=registry,
                scenario_cfg=sc_cfg,
                num_learners=num_learners,
                trials_per_learner=trials_per_learner,
                seed=seed,
                controller_configs=controller_cfgs,
                include_legacy_surrogate=True,
            )

            ev_df["scenario"] = sc_name
            ev_df["seed"] = seed
            m_df["scenario"] = sc_name
            all_events_list.append(ev_df)
            all_mastery_list.append(m_df)

            # Record summary rows
            for c_name, runs in results.items():
                gains_all = [r.mastery_gain_all_skills for r in runs]
                gains_prim = [r.mastery_gain_primary_skills for r in runs]
                gains_sec = [r.mastery_gain_secondary_skills for r in runs]
                obs_rates = [r.observed_success_rate for r in runs]
                clean_rates = [r.clean_success_rate for r in runs]
                imm_reversals = [r.immediate_direction_reversals for r in runs]
                filt_reversals = [r.filtered_direction_reversals for r in runs]
                variations = [r.total_difficulty_variation for r in runs]
                mean_diffs = [r.mean_difficulty for r in runs]
                at_mins = [r.proportion_at_min_bound for r in runs]
                at_maxs = [r.proportion_at_max_bound for r in runs]
                reachables = [r.target_reachable_proportion for r in runs if r.target_reachable_proportion is not None]
                pred_clean_probs = [r.mean_predicted_clean_prob for r in runs if r.mean_predicted_clean_prob is not None]
                target_errors = [r.mean_target_error for r in runs if r.mean_target_error is not None]
                pred_gaps = [r.mean_prediction_gap for r in runs if r.mean_prediction_gap is not None]
                briers = [r.brier_score_clean for r in runs if r.brier_score_clean is not None]

                metric_rows.append({
                    "scenario": sc_name,
                    "seed": seed,
                    "policy": c_name,
                    "num_learners": num_learners,
                    "trials_per_learner": trials_per_learner,
                    "total_trials": len(runs) * trials_per_learner,
                    "mastery_gain_all_skills_mean": round(float(np.mean(gains_all)), 4),
                    "mastery_gain_all_skills_std": round(float(np.std(gains_all)), 4),
                    "mastery_gain_primary_skills_mean": round(float(np.mean(gains_prim)), 4),
                    "mastery_gain_primary_skills_std": round(float(np.std(gains_prim)), 4),
                    "mastery_gain_secondary_skills_mean": round(float(np.mean(gains_sec)), 4),
                    "mastery_gain_secondary_skills_std": round(float(np.std(gains_sec)), 4),
                    "observed_success_rate_mean": round(float(np.mean(obs_rates)), 4),
                    "clean_success_rate_mean": round(float(np.mean(clean_rates)), 4),
                    "immediate_direction_reversals_mean": round(float(np.mean(imm_reversals)), 3),
                    "immediate_direction_reversals_std": round(float(np.std(imm_reversals)), 3),
                    "filtered_direction_reversals_mean": round(float(np.mean(filt_reversals)), 3),
                    "filtered_direction_reversals_std": round(float(np.std(filt_reversals)), 3),
                    "total_difficulty_variation_mean": round(float(np.mean(variations)), 3),
                    "total_difficulty_variation_std": round(float(np.std(variations)), 3),
                    "mean_difficulty": round(float(np.mean(mean_diffs)), 3),
                    "proportion_at_min_bound_mean": round(float(np.mean(at_mins)), 4),
                    "proportion_at_max_bound_mean": round(float(np.mean(at_maxs)), 4),
                    "target_reachable_proportion_mean": round(float(np.mean(reachables)), 4) if reachables else None,
                    "mean_predicted_clean_prob": round(float(np.mean(pred_clean_probs)), 4) if pred_clean_probs else None,
                    "mean_target_error": round(float(np.mean(target_errors)), 4) if target_errors else None,
                    "mean_prediction_gap": round(float(np.mean(pred_gaps)), 4) if pred_gaps else None,
                    "brier_score_clean": round(float(np.mean(briers)), 4) if briers else None,
                    "total_frustration_violations": sum(r.frustration_violations for r in runs),
                    "total_break_limit_violations": sum(r.break_limit_violations for r in runs),
                    "total_safety_interventions": sum(r.safety_interventions for r in runs),
                })

    all_events_df = pd.concat(all_events_list, ignore_index=True)
    all_mastery_df = pd.concat(all_mastery_list, ignore_index=True)
    summary_df = pd.DataFrame(metric_rows)

    # 3. Save datasets
    print("\n[Step 3/4] Persisting Phase 3 datasets and metrics...")
    events_csv = PHASE_3_CLOSED_LOOP_DIR / "phase_3_closed_loop_events.csv"
    all_events_df.to_csv(events_csv, index=False)
    print(f"  -> Saved {len(all_events_df)} interaction events to {events_csv}")

    mastery_csv = PHASE_3_CLOSED_LOOP_DIR / "phase_3_learner_mastery_summary.csv"
    all_mastery_df.to_csv(mastery_csv, index=False)

    metrics_csv = PHASE_3_CLOSED_LOOP_DIR / "phase_3_closed_loop_metrics.csv"
    summary_df.to_csv(metrics_csv, index=False)
    # Mirror metrics to audit dir
    summary_df.to_csv(PHASE_3_AUDIT_DIR / "phase_3_closed_loop_metrics.csv", index=False)

    # Per-game summary table
    game_rows = []
    for (sc, pol, gid), grp in all_events_df.groupby(["scenario", "controller", "game_id"]):
        pred_p_col = grp["predicted_clean_prob"].dropna()
        target_err_col = grp["target_error"].dropna()
        gap_col = grp["prediction_gap_clean"].dropna()
        reach_col = grp["target_reachable"].dropna()
        game_rows.append({
            "scenario": sc,
            "policy": pol,
            "game_id": gid,
            "total_trials": len(grp),
            "clean_success_rate": round(float(grp["clean_outcome"].mean()), 4),
            "observed_success_rate": round(float(grp["observed_outcome"].mean()), 4),
            "mean_difficulty": round(float(grp["difficulty"].mean()), 3),
            "proportion_at_min_bound": round(float(grp["at_min_bound"].mean()), 4),
            "proportion_at_max_bound": round(float(grp["at_max_bound"].mean()), 4),
            "target_reachable_rate": round(float(reach_col.mean()), 4) if len(reach_col) > 0 else None,
            "mean_predicted_clean_prob": round(float(pred_p_col.mean()), 4) if len(pred_p_col) > 0 else None,
            "mean_target_error": round(float(target_err_col.mean()), 4) if len(target_err_col) > 0 else None,
            "mean_prediction_gap": round(float(gap_col.mean()), 4) if len(gap_col) > 0 else None,
        })
    game_summary_df = pd.DataFrame(game_rows)
    game_summary_csv = PHASE_3_CLOSED_LOOP_DIR / "phase_3_per_game_summary.csv"
    game_summary_df.to_csv(game_summary_csv, index=False)
    game_summary_df.to_csv(PHASE_3_AUDIT_DIR / "phase_3_per_game_summary.csv", index=False)

    # Hierarchical JSON summary aggregating across seeds (Mean +- Std)
    summary_dict: Dict[str, Any] = {}
    for sc_name in all_scenarios:
        summary_dict[sc_name] = {}
        sc_sub = summary_df[summary_df["scenario"] == sc_name]
        for pol in ["fixed_difficulty", "streak_adaptive", "legacy_surrogate_target_adaptive", "model_aligned_target_adaptive"]:
            pol_sub = sc_sub[sc_sub["policy"] == pol]
            summary_dict[sc_name][pol] = {
                "clean_success_rate_mean": round(float(pol_sub["clean_success_rate_mean"].mean()), 4),
                "clean_success_rate_seed_std": round(float(pol_sub["clean_success_rate_mean"].std()), 4),
                "observed_success_rate_mean": round(float(pol_sub["observed_success_rate_mean"].mean()), 4),
                "observed_success_rate_seed_std": round(float(pol_sub["observed_success_rate_mean"].std()), 4),
                "mean_difficulty": round(float(pol_sub["mean_difficulty"].mean()), 3),
                "proportion_at_min_bound_mean": round(float(pol_sub["proportion_at_min_bound_mean"].mean()), 4),
                "proportion_at_max_bound_mean": round(float(pol_sub["proportion_at_max_bound_mean"].mean()), 4),
                "target_reachable_proportion_mean": round(float(pol_sub["target_reachable_proportion_mean"].mean()), 4) if pol_sub["target_reachable_proportion_mean"].notna().any() else None,
                "filtered_direction_reversals_mean": round(float(pol_sub["filtered_direction_reversals_mean"].mean()), 3),
                "immediate_direction_reversals_mean": round(float(pol_sub["immediate_direction_reversals_mean"].mean()), 3),
                "total_difficulty_variation_mean": round(float(pol_sub["total_difficulty_variation_mean"].mean()), 3),
                "mastery_gain_all_skills_mean": round(float(pol_sub["mastery_gain_all_skills_mean"].mean()), 4),
                "mastery_gain_primary_skills_mean": round(float(pol_sub["mastery_gain_primary_skills_mean"].mean()), 4),
                "mean_predicted_clean_prob": round(float(pol_sub["mean_predicted_clean_prob"].mean()), 4) if pol_sub["mean_predicted_clean_prob"].notna().any() else None,
                "mean_target_error": round(float(pol_sub["mean_target_error"].mean()), 4) if pol_sub["mean_target_error"].notna().any() else None,
                "mean_prediction_gap": round(float(pol_sub["mean_prediction_gap"].mean()), 4) if pol_sub["mean_prediction_gap"].notna().any() else None,
                "brier_score_clean": round(float(pol_sub["brier_score_clean"].mean()), 4) if pol_sub["brier_score_clean"].notna().any() else None,
                "total_safety_interventions": int(pol_sub["total_safety_interventions"].sum()),
                "total_frustration_violations": int(pol_sub["total_frustration_violations"].sum()),
                "total_break_limit_violations": int(pol_sub["total_break_limit_violations"].sum()),
            }

    summary_json_path = PHASE_3_CLOSED_LOOP_DIR / "phase_3_closed_loop_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    # Mirror to audit dir
    with open(PHASE_3_AUDIT_DIR / "phase_3_closed_loop_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    # 4. Generate plots
    print("\n[Step 4/4] Generating publication-grade comparison plots...")
    plot_difficulty_trajectories_sample(all_events_df, PHASE_3_PLOTS_DIR / "difficulty_trajectories_comparison.png")
    plot_saturation_and_reachability(summary_df, PHASE_3_PLOTS_DIR / "target_reachability_and_saturation.png")
    plot_policy_comparisons(summary_df, PHASE_3_PLOTS_DIR / "policy_comparison_metrics.png")
    plot_calibration_curves(all_events_df, PHASE_3_PLOTS_DIR / "calibration_predicted_clean_success.png")

    # Mirror plots to audit dir
    for p_file in PHASE_3_PLOTS_DIR.glob("*.png"):
        (PHASE_3_AUDIT_DIR / p_file.name).write_bytes(p_file.read_bytes())

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"PHASE 3 EXPERIMENTAL CAMPAIGN COMPLETED IN {elapsed}s")
    print("=" * 80)

    # Print executive summary table for base_heterogeneous scenario
    base_summary = summary_dict["base_heterogeneous"]
    print(f"\nEXECUTIVE SUMMARY: SCENARIO `base_heterogeneous` (Averaged across Seeds [42, 123, 999])")
    print("-" * 115)
    print(f"{'Policy':<34} | {'Clean Succ':<11} | {'Obs Succ':<10} | {'Mean D':<8} | {'At D_min':<10} | {'Reachable':<10} | {'Reversals':<10}")
    print("-" * 115)
    for pol, stats in base_summary.items():
        cs = f"{stats['clean_success_rate_mean']*100:.2f}%"
        os = f"{stats['observed_success_rate_mean']*100:.2f}%"
        md = f"{stats['mean_difficulty']:.3f}"
        at_min = f"{stats['proportion_at_min_bound_mean']*100:.1f}%"
        reach = f"{stats['target_reachable_proportion_mean']*100:.1f}%" if stats['target_reachable_proportion_mean'] is not None else "N/A"
        rev = f"{stats['filtered_direction_reversals_mean']:.2f}"
        print(f"{pol:<34} | {cs:<11} | {os:<10} | {md:<8} | {at_min:<10} | {reach:<10} | {rev:<10}")
    print("=" * 115)

    return {
        "elapsed_seconds": elapsed,
        "total_events": len(all_events_df),
        "total_metrics_rows": len(summary_df),
        "formula_parity_summary": parity_summary,
        "summary": summary_dict,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Grow21 Phase 3 Full Simulation Campaign.")
    parser.add_argument("--num-learners", type=int, default=60, help="Cohort size per scenario/seed")
    parser.add_argument("--trials-per-learner", type=int, default=60, help="Trials per learner")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999], help="Random seeds")
    args = parser.parse_args()

    run_all_experiments(
        num_learners=args.num_learners,
        trials_per_learner=args.trials_per_learner,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
