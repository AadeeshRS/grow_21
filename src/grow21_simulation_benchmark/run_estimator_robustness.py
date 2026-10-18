"""Grow21 Estimator Robustness Campaign: Diagnostics, Dev Benchmark, and Final Held-Out Validation.

Evaluates the Phase 5 Robust Adaptive Bayesian Filter against the Phase 3 Baseline
and the Phase 5 Approximate Bayesian Filter + Transfer.

Outputs are written to: outputs/experiments/phase_5/estimator_robustness/
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from grow21_simulation_benchmark.closed_loop import (
    ModelAlignedTargetAdaptiveController,
    RobustAdaptiveFilterController,
    run_closed_loop_comparison,
)
from grow21_simulation_benchmark.paths import (
    DEFAULT_CONFIG_PATH,
    OUTPUTS_DIR,
    SCENARIOS_CONFIG_PATH,
)
from grow21_simulation_benchmark.reliability import ReliabilityEngine
from grow21_simulation_benchmark.run_phase_5_experiments import (
    compute_calibration_dataframe,
    compute_paired_cluster_bootstrap,
)
from grow21_simulation_benchmark.tasks import TaskRegistry

ROBUSTNESS_OUTPUT_DIR = OUTPUTS_DIR / "experiments" / "phase_5" / "estimator_robustness"
DEV_SEEDS = [101, 202, 303]
FRESH_HELD_OUT_SEEDS = [1103, 1207, 1301, 1409, 1511]

POLICIES_TO_EVALUATE = [
    "model_aligned_baseline",
    "approx_bayesian_filter_transfer",
    "robust_adaptive_filter",
]

POLICY_DISPLAY_NAMES = {
    "model_aligned_baseline": "Baseline (Phase 3)",
    "approx_bayesian_filter_transfer": "Approx Bayesian Filter + Transfer",
    "robust_adaptive_filter": "Robust Adaptive Filter (Candidate C)",
}

POLICY_COLORS = {
    "model_aligned_baseline": "#4C72B0",
    "approx_bayesian_filter_transfer": "#8C564B",
    "robust_adaptive_filter": "#2CA02C",
}

SCENARIO_DISPLAY_NAMES = {
    "base_heterogeneous": "Base / Heterogeneous",
    "reliability_stress": "Reliability Stress",
    "simple_dynamics": "Simple Dynamics",
    "temporal_dynamics": "Temporal Dynamics",
    "distribution_shift": "Distribution Shift",
}


def run_cohort_evaluation(
    registry: TaskRegistry,
    scenarios_cfg: Dict[str, Any],
    default_cfg: Dict[str, Any],
    scenario_names: List[str],
    seeds: List[int],
    cohort_name: str,
    num_learners: int = 20,
    trials_per_learner: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs closed-loop simulation across seeds and scenarios for selected policies.

    Returns:
        (events_df, scenario_summary_df, paired_bootstrap_df)
    """
    all_events_list: List[Dict[str, Any]] = []

    print(f"\n=======================================================")
    print(f"RUNNING COHORT: {cohort_name.upper()} (Seeds: {seeds})")
    print(f"Learners/seed: {num_learners}, Trials/learner: {trials_per_learner}")
    print(f"=======================================================")

    for sc_name in scenario_names:
        sc_cfg = scenarios_cfg[sc_name]
        print(f"\n--- Scenario: {sc_name} ({SCENARIO_DISPLAY_NAMES.get(sc_name, sc_name)}) ---")

        for seed in seeds:
            t0 = time.time()
            _, ev_df, _ = run_closed_loop_comparison(
                task_registry=registry,
                scenario_cfg=sc_cfg,
                num_learners=num_learners,
                trials_per_learner=trials_per_learner,
                seed=seed,
                policies=POLICIES_TO_EVALUATE,
                controller_configs=default_cfg.get("controllers"),
            )
            ev_df["scenario"] = sc_name
            ev_df["seed"] = seed
            ev_df["split"] = cohort_name
            all_events_list.extend(ev_df.to_dict(orient="records"))
            elapsed = time.time() - t0
            print(f"  Seed {seed} completed in {elapsed:.1f}s ({len(ev_df)} events)")

    events_df = pd.DataFrame(all_events_list)

    # Compute scenario-level and overall summaries
    summary_rows: List[Dict[str, Any]] = []

    for sc_name in scenario_names:
        sc_events = events_df[events_df["scenario"] == sc_name]
        for pol in POLICIES_TO_EVALUATE:
            pol_events = sc_events[sc_events["controller"] == pol]
            if len(pol_events) == 0:
                continue

            p_err_pre = pol_events["mastery_error_primary_pre"].dropna().to_numpy(dtype=float)
            p_err_post = pol_events["mastery_error_primary_post"].dropna().to_numpy(dtype=float)
            prob_diff = np.abs(
                pol_events["predicted_clean_prob"].to_numpy(dtype=float)
                - pol_events["simulator_p_clean"].to_numpy(dtype=float)
            )
            prob_gap = pol_events["prediction_gap_clean"].dropna().to_numpy(dtype=float)
            target_err = pol_events["target_error"].dropna().to_numpy(dtype=float)
            diffs = pol_events["difficulty"].to_numpy(dtype=float)

            summary_rows.append({
                "split": cohort_name,
                "scenario": sc_name,
                "controller": pol,
                "display_name": POLICY_DISPLAY_NAMES.get(pol, pol),
                "n_learners": pol_events["learner_id"].nunique() * len(seeds),
                "n_events": len(pol_events),
                "primary_mastery_mae_pre": round(float(np.mean(np.abs(p_err_pre))), 4),
                "primary_mastery_mae_post": round(float(np.mean(np.abs(p_err_post))), 4),
                "primary_mastery_rmse_post": round(float(np.sqrt(np.mean(p_err_post ** 2))), 4),
                "primary_bias_post": round(float(np.mean(p_err_post)), 4),
                "primary_bias_pre": round(float(np.mean(p_err_pre)), 4),
                "clean_prob_mae": round(float(np.mean(prob_diff)), 4),
                "clean_prob_rmse": round(float(np.sqrt(np.mean(prob_gap ** 2))), 4),
                "signed_prediction_gap": round(float(np.mean(prob_gap)), 4),
                "mean_target_error": round(float(np.mean(target_err)), 4) if len(target_err) > 0 else None,
                "target_reachable_proportion": round(float(pol_events["target_reachable"].mean()), 4),
                "mean_difficulty": round(float(np.mean(diffs)), 4),
                "proportion_at_min_bound": round(float(pol_events["at_min_bound"].mean()), 4),
                "proportion_at_max_bound": round(float(pol_events["at_max_bound"].mean()), 4),
                "safety_interventions_total": int(pol_events["safety_intervention"].sum()),
                "observed_success_rate": round(float(pol_events["observed_outcome"].mean()), 4),
                "clean_success_rate": round(float(pol_events["clean_outcome"].mean()), 4),
            })

    # Add macro average across all scenarios
    for pol in POLICIES_TO_EVALUATE:
        pol_events = events_df[events_df["controller"] == pol]
        p_err_pre = pol_events["mastery_error_primary_pre"].dropna().to_numpy(dtype=float)
        p_err_post = pol_events["mastery_error_primary_post"].dropna().to_numpy(dtype=float)
        prob_diff = np.abs(
            pol_events["predicted_clean_prob"].to_numpy(dtype=float)
            - pol_events["simulator_p_clean"].to_numpy(dtype=float)
        )
        prob_gap = pol_events["prediction_gap_clean"].dropna().to_numpy(dtype=float)
        target_err = pol_events["target_error"].dropna().to_numpy(dtype=float)
        diffs = pol_events["difficulty"].to_numpy(dtype=float)

        summary_rows.append({
            "split": cohort_name,
            "scenario": "overall_macro_mean",
            "controller": pol,
            "display_name": POLICY_DISPLAY_NAMES.get(pol, pol),
            "n_learners": pol_events["learner_id"].nunique() * len(seeds),
            "n_events": len(pol_events),
            "primary_mastery_mae_pre": round(float(np.mean(np.abs(p_err_pre))), 4),
            "primary_mastery_mae_post": round(float(np.mean(np.abs(p_err_post))), 4),
            "primary_mastery_rmse_post": round(float(np.sqrt(np.mean(p_err_post ** 2))), 4),
            "primary_bias_post": round(float(np.mean(p_err_post)), 4),
            "primary_bias_pre": round(float(np.mean(p_err_pre)), 4),
            "clean_prob_mae": round(float(np.mean(prob_diff)), 4),
            "clean_prob_rmse": round(float(np.sqrt(np.mean(prob_gap ** 2))), 4),
            "signed_prediction_gap": round(float(np.mean(prob_gap)), 4),
            "mean_target_error": round(float(np.mean(target_err)), 4) if len(target_err) > 0 else None,
            "target_reachable_proportion": round(float(pol_events["target_reachable"].mean()), 4),
            "mean_difficulty": round(float(np.mean(diffs)), 4),
            "proportion_at_min_bound": round(float(pol_events["at_min_bound"].mean()), 4),
            "proportion_at_max_bound": round(float(pol_events["at_max_bound"].mean()), 4),
            "safety_interventions_total": int(pol_events["safety_intervention"].sum()),
            "observed_success_rate": round(float(pol_events["observed_outcome"].mean()), 4),
            "clean_success_rate": round(float(pol_events["clean_outcome"].mean()), 4),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Compute paired 2000-replicate cluster bootstraps
    print(f"\nComputing paired cluster bootstraps (2,000 replicates, cluster=(seed, learner_id))...")
    bootstrap_rows: List[Dict[str, Any]] = []
    base_policy = "model_aligned_baseline"
    filter_policy = "approx_bayesian_filter_transfer"
    candidate_policy = "robust_adaptive_filter"

    metrics_to_bootstrap = ["primary_mastery_mae_post", "primary_mastery_mae_pre", "prob_abs_error"]

    scenarios_for_boot = scenario_names + ["overall_macro_mean"]

    for sc in scenarios_for_boot:
        if sc == "overall_macro_mean":
            sub_events = events_df
        else:
            sub_events = events_df[events_df["scenario"] == sc]

        cand_df = sub_events[sub_events["controller"] == candidate_policy]
        base_df = sub_events[sub_events["controller"] == base_policy]
        filt_df = sub_events[sub_events["controller"] == filter_policy]

        for metric in metrics_to_bootstrap:
            # 1. Candidate vs Baseline
            res_base = compute_paired_cluster_bootstrap(
                base_df=base_df,
                variant_df=cand_df,
                metric_name=metric,
                num_bootstrap=2000,
                seed=42,
            )
            bootstrap_rows.append({
                "split": cohort_name,
                "scenario": sc,
                "contrast": "robust_adaptive_vs_baseline",
                "variant": candidate_policy,
                "baseline": base_policy,
                "metric": metric,
                **res_base,
            })

            # 2. Candidate vs Filter+Transfer
            res_filt = compute_paired_cluster_bootstrap(
                base_df=filt_df,
                variant_df=cand_df,
                metric_name=metric,
                num_bootstrap=2000,
                seed=42,
            )
            bootstrap_rows.append({
                "split": cohort_name,
                "scenario": sc,
                "contrast": "robust_adaptive_vs_approx_filter",
                "variant": candidate_policy,
                "baseline": filter_policy,
                "metric": metric,
                **res_filt,
            })

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    return events_df, summary_df, bootstrap_df


def generate_robustness_plots(
    held_out_summary: pd.DataFrame,
    held_out_bootstrap: pd.DataFrame,
    output_dir: Path,
):
    """Generates publication-quality comparison charts and forest plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns_colors = ["#4C72B0", "#8C564B", "#2CA02C"]

    # 1. Bar Chart: Post-Update Mastery MAE by Scenario
    plt.figure(figsize=(12, 6))
    scenarios = [
        "base_heterogeneous",
        "reliability_stress",
        "simple_dynamics",
        "temporal_dynamics",
        "distribution_shift",
        "overall_macro_mean",
    ]
    sc_labels = [
        "Base\nHeterogeneous",
        "Reliability\nStress",
        "Simple\nDynamics",
        "Temporal\nDynamics",
        "Distribution\nShift",
        "Overall\n(Macro)",
    ]

    x = np.arange(len(scenarios))
    width = 0.25

    for idx, pol in enumerate(POLICIES_TO_EVALUATE):
        pol_vals = []
        for sc in scenarios:
            row = held_out_summary[
                (held_out_summary["scenario"] == sc) & (held_out_summary["controller"] == pol)
            ]
            pol_vals.append(row["primary_mastery_mae_post"].values[0] if len(row) > 0 else 0.0)

        plt.bar(
            x + (idx - 1) * width,
            pol_vals,
            width,
            label=POLICY_DISPLAY_NAMES.get(pol, pol),
            color=POLICY_COLORS.get(pol, sns_colors[idx]),
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
        )

        for i, val in enumerate(pol_vals):
            plt.text(
                x[i] + (idx - 1) * width,
                val + 0.004,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    plt.title(
        "Grow21 Fresh Held-Out Final Evaluation: Post-Update Mastery MAE by Scenario\n(Seeds: 1103, 1207, 1301, 1409, 1511 | 100 Learners / Scenario)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Simulation Scenario Family", fontsize=11, fontweight="bold")
    plt.ylabel("Post-Update Mastery MAE (Lower is Better)", fontsize=11, fontweight="bold")
    plt.xticks(x, sc_labels, fontsize=10)
    plt.ylim(0, 0.28)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    plt.tight_layout()
    plot_path_mae = output_dir / "held_out_mae_by_scenario.png"
    plt.savefig(plot_path_mae, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path_mae}")

    # 2. Forest Plot: Paired Differences (Variant - Baseline) with 95% Bootstrap CIs
    plt.figure(figsize=(10, 7))
    boot_mae = held_out_bootstrap[
        (held_out_bootstrap["metric"] == "primary_mastery_mae_post")
        & (held_out_bootstrap["scenario"] != "overall_macro_mean")
    ].copy()

    y_pos = np.arange(len(boot_mae))
    plt.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)

    y_labels = []
    for idx, (_, row) in enumerate(boot_mae.iterrows()):
        pt = row["diff_variant_minus_baseline"]
        ci_lo, ci_hi = row["ci_lower"], row["ci_upper"]
        color = "#2CA02C" if ci_hi < 0.0 else ("#D62728" if ci_lo > 0.0 else "#7F7F7F")

        contrast_short = (
            "vs Baseline" if "baseline" in row["contrast"] else "vs Phase 5 Filter"
        )
        sc_short = row["scenario"].replace("_", " ").title()
        y_labels.append(f"{sc_short} ({contrast_short})")

        plt.errorbar(
            pt,
            idx,
            xerr=[[pt - ci_lo], [ci_hi - pt]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.2,
            capsize=4,
            capthick=1.5,
            markersize=6,
        )
        plt.text(
            ci_hi + 0.002,
            idx,
            f"{pt:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]",
            va="center",
            fontsize=8.5,
            color=color,
            fontweight="bold",
        )

    plt.yticks(y_pos, y_labels, fontsize=9.5)
    plt.xlabel("Paired Difference in Mastery MAE (Variant - Baseline)\nNegative = Improvement / Error Reduction", fontsize=11, fontweight="bold")
    plt.title(
        "Grow21 Fresh Held-Out Evaluation: Paired 2,000-Replicate Cluster Bootstrap (95% CI)\nRobust Adaptive Filter vs Baseline and Phase 5 Filter",
        fontsize=11.5,
        fontweight="bold",
        pad=15,
    )
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path_forest = output_dir / "held_out_bootstrap_forest.png"
    plt.savefig(plot_path_forest, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path_forest}")


def main():
    parser = argparse.ArgumentParser(description="Grow21 Estimator Robustness Campaign Runner")
    parser.add_argument("--num-learners", type=int, default=20, help="Learners per seed (default 20)")
    parser.add_argument("--trials-per-learner", type=int, default=60, help="Trials per learner (default 60)")
    parser.add_argument("--dev-only", action="store_true", help="Run only on development seeds")
    parser.add_argument("--held-out-only", action="store_true", help="Run only on fresh held-out seeds")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    t_start = time.time()
    ROBUSTNESS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        default_cfg = yaml.safe_load(f)

    with open(SCENARIOS_CONFIG_PATH, "r", encoding="utf-8") as f:
        scenarios_cfg = yaml.safe_load(f)["scenarios"]

    registry = TaskRegistry.from_yaml(DEFAULT_CONFIG_PATH)

    scenario_names = [
        "base_heterogeneous",
        "reliability_stress",
        "simple_dynamics",
        "temporal_dynamics",
        "distribution_shift",
    ]

    # 1. Dev Cohort Evaluation
    if not args.held_out_only:
        dev_events, dev_summary, dev_boot = run_cohort_evaluation(
            registry=registry,
            scenarios_cfg=scenarios_cfg,
            default_cfg=default_cfg,
            scenario_names=scenario_names,
            seeds=DEV_SEEDS,
            cohort_name="development",
            num_learners=args.num_learners,
            trials_per_learner=args.trials_per_learner,
        )
        dev_summary.to_csv(ROBUSTNESS_OUTPUT_DIR / "dev_summary.csv", index=False)
        dev_boot.to_csv(ROBUSTNESS_OUTPUT_DIR / "dev_paired_bootstrap.csv", index=False)
        print(f"\nDev summary saved to {ROBUSTNESS_OUTPUT_DIR / 'dev_summary.csv'}")

    # 2. Fresh Held-Out Final Evaluation
    if not args.dev_only:
        print("\n" + "#" * 80)
        print("EVALUATING ON FRESH HELD-OUT SEEDS: [1103, 1207, 1301, 1409, 1511]")
        print("Candidate configuration is strictly frozen.")
        print("#" * 80)

        ho_events, ho_summary, ho_boot = run_cohort_evaluation(
            registry=registry,
            scenarios_cfg=scenarios_cfg,
            default_cfg=default_cfg,
            scenario_names=scenario_names,
            seeds=FRESH_HELD_OUT_SEEDS,
            cohort_name="held_out_fresh",
            num_learners=args.num_learners,
            trials_per_learner=args.trials_per_learner,
        )
        ho_summary.to_csv(ROBUSTNESS_OUTPUT_DIR / "held_out_summary.csv", index=False)
        ho_boot.to_csv(ROBUSTNESS_OUTPUT_DIR / "held_out_paired_bootstrap.csv", index=False)

        # Build comparison table for the report
        comp_rows = []
        for sc in scenario_names + ["overall_macro_mean"]:
            b_row = ho_summary[(ho_summary["scenario"] == sc) & (ho_summary["controller"] == "model_aligned_baseline")]
            f_row = ho_summary[(ho_summary["scenario"] == sc) & (ho_summary["controller"] == "approx_bayesian_filter_transfer")]
            c_row = ho_summary[(ho_summary["scenario"] == sc) & (ho_summary["controller"] == "robust_adaptive_filter")]

            b_mae = b_row["primary_mastery_mae_post"].values[0]
            f_mae = f_row["primary_mastery_mae_post"].values[0]
            c_mae = c_row["primary_mastery_mae_post"].values[0]

            b_bias = b_row["primary_bias_post"].values[0]
            f_bias = f_row["primary_bias_post"].values[0]
            c_bias = c_row["primary_bias_post"].values[0]

            # Relative reductions
            pct_vs_filter = ((c_mae - f_mae) / f_mae) * 100.0
            pct_vs_base = ((c_mae - b_mae) / b_mae) * 100.0

            comp_rows.append({
                "scenario": sc,
                "display_name": SCENARIO_DISPLAY_NAMES.get(sc, sc),
                "baseline_mae": b_mae,
                "phase_5_filter_mae": f_mae,
                "robust_adaptive_mae": c_mae,
                "diff_vs_filter_abs": round(c_mae - f_mae, 4),
                "diff_vs_filter_pct": round(pct_vs_filter, 2),
                "diff_vs_baseline_abs": round(c_mae - b_mae, 4),
                "diff_vs_baseline_pct": round(pct_vs_base, 2),
                "baseline_bias": b_bias,
                "phase_5_filter_bias": f_bias,
                "robust_adaptive_bias": c_bias,
            })

        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv(ROBUSTNESS_OUTPUT_DIR / "held_out_scenario_comparison.csv", index=False)

        print("\n" + "=" * 105)
        print("FRESH HELD-OUT FINAL ESTIMATOR COMPARISON TABLE (Seeds: [1103, 1207, 1301, 1409, 1511])")
        print("=" * 105)
        for _, r in comp_df.iterrows():
            print(
                f"{r['display_name']:25s} | Baseline: {r['baseline_mae']:.4f} | Filter: {r['phase_5_filter_mae']:.4f} "
                f"| Robust: {r['robust_adaptive_mae']:.4f} | vs Filter: {r['diff_vs_filter_pct']:+6.2f}% | vs Base: {r['diff_vs_baseline_pct']:+6.2f}%"
            )
        print("=" * 105)

        if not args.skip_plots:
            generate_robustness_plots(
                held_out_summary=ho_summary,
                held_out_bootstrap=ho_boot,
                output_dir=ROBUSTNESS_OUTPUT_DIR,
            )

        # Save metrics manifest JSON
        manifest = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dev_seeds": DEV_SEEDS,
            "fresh_held_out_seeds": FRESH_HELD_OUT_SEEDS,
            "num_learners_per_seed": args.num_learners,
            "trials_per_learner": args.trials_per_learner,
            "policies": POLICIES_TO_EVALUATE,
            "scenarios": scenario_names,
            "held_out_comparison": comp_rows,
        }
        with open(ROBUSTNESS_OUTPUT_DIR / "estimator_robustness_metrics.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nESTIMATOR ROBUSTNESS PIPELINE COMPLETED in {total_time:.1f}s ({total_time/60.0:.2f} min)")


if __name__ == "__main__":
    main()
