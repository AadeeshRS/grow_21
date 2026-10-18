"""Plotting and visualization routines for Grow21 simulation benchmark.

Generates calibration curves, scenario metric comparisons, and sample learning trajectories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_calibration_curves(
    eval_results: Dict[str, Any],
    output_path: str | Path,
    title_suffix: str = "",
):
    """Plots calibration curves comparing predicted probabilities to empirical accuracy."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # Reference diagonal
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

    colors = {
        "logistic_baseline": "#1f77b4",
        "vanilla_dkt": "#ff7f0e",
        "reliability_gated_dkt": "#2ca02c",
    }
    labels = {
        "logistic_baseline": "Logistic Baseline",
        "vanilla_dkt": "Vanilla DKT",
        "reliability_gated_dkt": "Reliability-Gated DKT",
    }

    for model_key in ["logistic_baseline", "vanilla_dkt", "reliability_gated_dkt"]:
        if "models" not in eval_results or model_key not in eval_results["models"]:
            continue
        m_info = eval_results["models"][model_key]
        bins = m_info["calibration_bins"]
        x_pts = []
        y_pts = []
        for b in bins:
            if b["count"] > 0:
                x_pts.append(b["mean_pred_prob"])
                y_pts.append(b["empirical_acc"])

        ece_val = m_info.get("ece", 0.0)
        ax.plot(
            x_pts,
            y_pts,
            marker="o",
            linewidth=2,
            color=colors.get(model_key, "blue"),
            label=f"{labels.get(model_key, model_key)} (ECE={ece_val:.3f})",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Empirical Accuracy", fontsize=11)
    ax.set_title(f"Reliability & Calibration Curve {title_suffix}".strip(), fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=200)
    plt.close()


def plot_scenario_comparison(
    summary_by_scenario: Dict[str, Dict[str, Any]],
    output_path: str | Path,
):
    """Plots bar charts comparing Log Loss and ROC-AUC across scenario families."""
    scenarios = list(summary_by_scenario.keys())
    models = ["logistic_baseline", "vanilla_dkt", "reliability_gated_dkt"]
    model_labels = ["Logistic Baseline", "Vanilla DKT", "Reliability-Gated DKT"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    active_indices = [i for i, m in enumerate(models) if any(m in summary_by_scenario[s].get("models", {}) for s in scenarios)]
    if not active_indices:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(scenarios))
    n_act = len(active_indices)
    width = 0.8 / max(1, n_act)

    # 1. Log Loss comparison
    for plot_idx, i in enumerate(active_indices):
        m_key = models[i]
        losses = [summary_by_scenario[s]["models"].get(m_key, {}).get("log_loss_mean", 0.0) for s in scenarios]
        errs = [summary_by_scenario[s]["models"].get(m_key, {}).get("log_loss_std", 0.0) for s in scenarios]
        offset = (plot_idx - (n_act - 1) / 2.0) * width
        ax1.bar(
            x + offset,
            losses,
            width,
            yerr=errs,
            capsize=3,
            label=model_labels[i],
            color=colors[i],
            alpha=0.9,
        )

    ax1.set_ylabel("Log Loss (Lower is better)", fontsize=11)
    ax1.set_title("Test Log Loss by Scenario Family", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=25, ha="right")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax1.legend()

    # 2. ROC-AUC comparison
    for plot_idx, i in enumerate(active_indices):
        m_key = models[i]
        aucs = [summary_by_scenario[s]["models"].get(m_key, {}).get("roc_auc_mean", 0.0) for s in scenarios]
        errs = [summary_by_scenario[s]["models"].get(m_key, {}).get("roc_auc_std", 0.0) for s in scenarios]
        offset = (plot_idx - (n_act - 1) / 2.0) * width
        ax2.bar(
            x + offset,
            aucs,
            width,
            yerr=errs,
            capsize=3,
            label=model_labels[i],
            color=colors[i],
            alpha=0.9,
        )

    ax2.set_ylabel("ROC-AUC (Higher is better)", fontsize=11)
    ax2.set_title("Test ROC-AUC by Scenario Family", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=25, ha="right")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=200)
    plt.close()
