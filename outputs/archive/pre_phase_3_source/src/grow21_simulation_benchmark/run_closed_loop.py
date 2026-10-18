"""CLI script for running the closed-loop adaptive simulation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .closed_loop import run_closed_loop_comparison
from .paths import (
    AUDIT_OUTPUTS_DIR,
    CLOSED_LOOP_OUTPUTS_DIR,
    DEFAULT_CONFIG_PATH,
    SCENARIOS_CONFIG_PATH,
    ensure_output_directories,
)
from .tasks import TaskRegistry


def plot_difficulty_trajectories(events_df: pd.DataFrame, output_path: str | Path, num_sample_learners: int = 3):
    """Plots trial-by-trial difficulty trajectories for sample learners to visualize saturation and oscillations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    controllers = ["fixed_difficulty", "streak_adaptive", "target_adaptive"]
    titles = ["Fixed Difficulty (D=0.50)", "Streak Adaptive (2-up/1-down)", "Grow21 Model-Based Target Adaptive"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    learners_to_plot = sorted(events_df["learner_id"].unique())[:num_sample_learners]

    for ax, c_name, title, color in zip(axes, controllers, titles, colors):
        c_df = events_df[events_df["controller"] == c_name]
        for l_id in learners_to_plot:
            l_df = c_df[c_df["learner_id"] == l_id].sort_values("step_idx")
            ax.plot(
                l_df["step_idx"],
                l_df["difficulty"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                alpha=0.7,
                label=f"Learner {l_id}",
            )
        ax.axhline(0.08, color="gray", linestyle=":", label="D_min (0.08)")
        ax.axhline(0.95, color="gray", linestyle=":", label="D_max (0.95)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Trial Step", fontsize=10)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax == axes[0]:
            ax.set_ylabel("Abstract Difficulty D", fontsize=10)
            ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Grow21 Closed-Loop Simulation Experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Default config YAML",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=str(SCENARIOS_CONFIG_PATH),
        help="Scenarios YAML",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="base_heterogeneous",
        help="Scenario for closed-loop evaluation",
    )
    parser.add_argument(
        "--num-learners",
        type=int,
        default=60,
        help="Number of virtual learners",
    )
    parser.add_argument(
        "--trials-per-learner",
        type=int,
        default=60,
        help="Trials per learner",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CLOSED_LOOP_OUTPUTS_DIR),
        help="Output directory",
    )
    args = parser.parse_args()

    ensure_output_directories()
    registry = TaskRegistry.from_yaml(args.config)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.scenarios, "r", encoding="utf-8") as f:
        sc_cfg = yaml.safe_load(f)["scenarios"][args.scenario]

    controller_cfgs = cfg.get("controllers", {})

    print(f"Running Closed-Loop Experiment on scenario: {args.scenario} ({args.num_learners} learners, {args.trials_per_learner} trials, seed={args.seed})")
    results, events_df, mastery_df = run_closed_loop_comparison(
        task_registry=registry,
        scenario_cfg=sc_cfg,
        num_learners=args.num_learners,
        trials_per_learner=args.trials_per_learner,
        seed=args.seed,
        controller_configs=controller_cfgs,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save event-level CSV
    events_csv_path = out_dir / "closed_loop_events.csv"
    events_df.to_csv(events_csv_path, index=False)
    print(f"Saved {len(events_df)} event-level closed-loop trials to {events_csv_path}")

    # 2. Save per-learner per-skill mastery CSV
    mastery_csv_path = out_dir / "learner_mastery_summary.csv"
    mastery_df.to_csv(mastery_csv_path, index=False)
    print(f"Saved {len(mastery_df)} per-skill mastery records to {mastery_csv_path}")

    # Also mirror to audit directory for the bundle
    audit_mastery_path = AUDIT_OUTPUTS_DIR / "learner_mastery_summary.csv"
    AUDIT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    mastery_df.to_csv(audit_mastery_path, index=False)

    summary = {}
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
        target_reach_mean = round(float(np.mean(reachables)), 4) if reachables else None
        frust_viols = sum(r.frustration_violations for r in runs)
        break_viols = sum(r.break_limit_violations for r in runs)
        safety_intervs = sum(r.safety_interventions for r in runs)

        summary[c_name] = {
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
            # Legacy alias fields for backward compatibility (matches filtered reversals)
            "direction_reversals_mean": round(float(np.mean(filt_reversals)), 3),
            "direction_reversals_std": round(float(np.std(filt_reversals)), 3),
            "total_difficulty_variation_mean": round(float(np.mean(variations)), 3),
            "total_difficulty_variation_std": round(float(np.std(variations)), 3),
            "mean_difficulty": round(float(np.mean(mean_diffs)), 3),
            "proportion_at_min_bound_mean": round(float(np.mean(at_mins)), 4),
            "proportion_at_max_bound_mean": round(float(np.mean(at_maxs)), 4),
            "target_reachable_proportion_mean": target_reach_mean,
            "total_frustration_violations": frust_viols,
            "total_break_limit_violations": break_viols,
            "total_safety_interventions": safety_intervs,
        }

    summary_file = out_dir / "closed_loop_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Closed loop summary saved to {summary_file}")

    # 3. Plots
    # A. Bar comparison of gains & stability
    controllers = list(summary.keys())
    ctrl_labels = [c.replace("_", " ").title() for c in controllers]
    gains_p = [summary[c]["mastery_gain_primary_skills_mean"] for c in controllers]
    gain_p_stds = [summary[c]["mastery_gain_primary_skills_std"] for c in controllers]
    reversals = [summary[c]["filtered_direction_reversals_mean"] for c in controllers]
    variations = [summary[c]["total_difficulty_variation_mean"] for c in controllers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    ax1.bar(ctrl_labels, gains_p, yerr=gain_p_stds, capsize=4, color=colors, alpha=0.85)
    ax1.set_ylabel("Primary Skill Mastery Gain (Δθ)", fontsize=11)
    ax1.set_title("Mastery Growth on Primary Practiced Skills", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    x = np.arange(len(controllers))
    w = 0.35
    ax2.bar(x - w/2, reversals, width=w, label="Filtered Reversals (Non-Zero Transitions)", color="#3274a1", alpha=0.85)
    ax2.bar(x + w/2, variations, width=w, label="Total Difficulty Variation (Σ|ΔD|)", color="#e1812c", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ctrl_labels)
    ax2.set_ylabel("Count / Cumulative Variation", fontsize=11)
    ax2.set_title("Controller Difficulty Dynamics & Stability", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_file = plots_dir / "closed_loop_comparison.png"
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"Closed-loop comparison plot saved to {plot_file}")

    # B. Trial-by-trial difficulty trajectories
    traj_file = plots_dir / "difficulty_trajectories.png"
    plot_difficulty_trajectories(events_df, traj_file, num_sample_learners=3)
    print(f"Difficulty trajectories plot saved to {traj_file}")

    print("\n" + "=" * 105)
    print(f"{'Controller':<20} | {'Obs Succ':<9} | {'Mean D':<7} | {'Reversals':<11} | {'Variation':<11} | {'At D_min':<9} | {'Reachable':<9}")
    print("-" * 105)
    for c, s in summary.items():
        obs_s = f"{s['observed_success_rate_mean']*100:.2f}%"
        d_str = f"{s['mean_difficulty']:.3f}"
        rev_str = f"{s['direction_reversals_mean']:.3f}"
        var_str = f"{s['total_difficulty_variation_mean']:.3f}"
        min_p = f"{s['proportion_at_min_bound_mean']*100:.1f}%"
        reach_p = (
            f"{s['target_reachable_proportion_mean']*100:.2f}%"
            if s["target_reachable_proportion_mean"] is not None
            else "N/A"
        )
        print(f"{c:<20} | {obs_s:<9} | {d_str:<7} | {rev_str:<11} | {var_str:<11} | {min_p:<9} | {reach_p:<9}")
    print("=" * 105)


if __name__ == "__main__":
    main()
