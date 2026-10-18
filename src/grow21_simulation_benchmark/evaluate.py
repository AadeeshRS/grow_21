"""Evaluation and metric computation module for Grow21 knowledge-tracing benchmark.

Computes Log Loss, Brier Score, ROC-AUC, Accuracy, Balanced Accuracy,
Expected Calibration Error (ECE), calibration tables, paired differences,
and learner-level bootstrap confidence intervals.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
import torch
from torch.utils.data import DataLoader

from .dataset import DKTSequenceDataset
from .models import LogisticBaselineModel, ReliabilityGatedDKT, VanillaDKT
from .simulator import SimulatorEvent
from .tasks import TaskRegistry


def compute_calibration_table(
    y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10
) -> Tuple[List[Dict[str, Any]], float]:
    """Computes calibration bins and Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_records = []
    total_samples = len(y_true)
    ece = 0.0

    for b in range(num_bins):
        low, high = bins[b], bins[b + 1]
        if b == num_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(y_prob[mask]))
            empirical_acc = float(np.mean(y_true[mask]))
            gap = abs(mean_pred - empirical_acc)
            ece += (count / total_samples) * gap
        else:
            mean_pred = (low + high) / 2.0
            empirical_acc = 0.0
            gap = 0.0

        bin_records.append(
            {
                "bin_idx": b,
                "bin_low": round(low, 2),
                "bin_high": round(high, 2),
                "count": count,
                "mean_pred_prob": round(mean_pred, 4),
                "empirical_acc": round(empirical_acc, 4),
                "calibration_gap": round(gap, 4),
            }
        )

    return bin_records, float(ece)


def calculate_metrics_dict(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_clean: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculates comprehensive classification and probabilistic calibration metrics."""
    # Ensure numerical stability for log loss
    eps = 1e-7
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)

    ll = float(log_loss(y_true, y_prob_clipped))
    brier = float(brier_score_loss(y_true, y_prob_clipped))

    # ROC-AUC requires both positive and negative classes
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        auc = float(roc_auc_score(y_true, y_prob_clipped))
    else:
        auc = 0.5

    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    cal_bins, ece = compute_calibration_table(y_true, y_prob_clipped)

    metrics = {
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "roc_auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "ece": round(ece, 4),
        "calibration_bins": cal_bins,
    }

    # Secondary oracle diagnostic (if y_clean provided)
    if y_clean is not None:
        metrics["oracle_clean_log_loss"] = round(float(log_loss(y_clean, y_prob_clipped)), 4)
        if len(np.unique(y_clean)) > 1:
            metrics["oracle_clean_roc_auc"] = round(float(roc_auc_score(y_clean, y_prob_clipped)), 4)

    return metrics


def compute_paired_bootstrap_ci(
    loss_model: np.ndarray,
    loss_base: np.ndarray,
    learner_ids: np.ndarray,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes paired bootstrap confidence interval over learner clusters.

    Returns:
        (point_diff, ci_lower, ci_upper)
        where negative point_diff means model achieved lower loss than baseline.
    """
    rng = np.random.RandomState(seed)
    unique_learners = np.unique(learner_ids)
    n_learners = len(unique_learners)

    learner_diffs = []
    for l_id in unique_learners:
        mask = learner_ids == l_id
        diff_l = np.mean(loss_model[mask]) - np.mean(loss_base[mask])
        learner_diffs.append(diff_l)
    learner_diffs = np.array(learner_diffs)
    point_diff = float(np.mean(learner_diffs))

    bootstrap_means = []
    for _ in range(num_bootstrap):
        sample_indices = rng.choice(n_learners, size=n_learners, replace=True)
        boot_mean = float(np.mean(learner_diffs[sample_indices]))
        bootstrap_means.append(boot_mean)

    ci_lower = float(np.percentile(bootstrap_means, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1.0 - alpha / 2.0)))

    return round(point_diff, 4), round(ci_lower, 4), round(ci_upper, 4)


def evaluate_all_models_on_test(
    models_dict: Dict[str, Any],
    test_seqs: List[List[SimulatorEvent]],
    task_registry: TaskRegistry,
) -> Dict[str, Any]:
    """Runs fair, identical held-out evaluation across all models present in models_dict."""
    # Extract ground truth diagnostics directly from test sequences
    diagnostics_list = []
    for seq in test_seqs:
        for ev in seq:
            diagnostics_list.append({
                "learner_id": ev.learner_id,
                "y_true": ev.observed_outcome,
                "clean_outcome": ev.clean_outcome,
                "true_reliability": ev.true_reliability,
                "estimated_reliability": ev.estimated_reliability,
            })

    y_true_arr = np.array([d["y_true"] for d in diagnostics_list], dtype=int)
    learner_ids = np.array([d["learner_id"] for d in diagnostics_list], dtype=int)
    clean_outcomes = np.array([d["clean_outcome"] for d in diagnostics_list], dtype=int)
    true_reliabilities = np.array([d["true_reliability"] for d in diagnostics_list], dtype=float)
    est_reliabilities = np.array([d["estimated_reliability"] for d in diagnostics_list], dtype=float)

    model_probs: Dict[str, np.ndarray] = {}
    model_metrics: Dict[str, Any] = {}

    # 1. Predictions from Logistic Baseline if present
    if "logistic_baseline" in models_dict:
        log_model: LogisticBaselineModel = models_dict["logistic_baseline"]
        log_probs, _, _ = log_model.predict_proba(test_seqs)
        model_probs["logistic_baseline"] = np.array(log_probs, dtype=np.float32)
        model_metrics["logistic_baseline"] = calculate_metrics_dict(y_true_arr, model_probs["logistic_baseline"], clean_outcomes)

    # 2. Predictions from PyTorch DKT models if present
    needs_dkt = any(m in models_dict for m in ("vanilla_dkt", "reliability_gated_dkt"))
    if needs_dkt:
        test_dataset = DKTSequenceDataset(test_seqs, task_registry)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        vanilla_model = models_dict.get("vanilla_dkt")
        gated_model = models_dict.get("reliability_gated_dkt")
        if vanilla_model:
            vanilla_model.eval()
        if gated_model:
            gated_model.eval()

        vanilla_preds_list: List[float] = []
        gated_preds_list: List[float] = []

        with torch.no_grad():
            for batch in test_loader:
                meta = batch["meta_features"]
                outcomes = batch["outcomes"]
                reliabilities = batch["reliabilities"]
                mask = batch["mask"]

                v_preds = vanilla_model(meta, outcomes, mask=mask)[0] if vanilla_model else None
                g_preds = gated_model(meta, outcomes, reliabilities, mask=mask)[0] if gated_model else None

                batch_mask = mask.bool().cpu().numpy()
                if v_preds is not None:
                    v_np = v_preds.cpu().numpy()
                if g_preds is not None:
                    g_np = g_preds.cpu().numpy()

                for b in range(meta.shape[0]):
                    m_b = batch_mask[b]
                    if v_preds is not None:
                        vanilla_preds_list.extend(v_np[b, m_b].tolist())
                    if g_preds is not None:
                        gated_preds_list.extend(g_np[b, m_b].tolist())

        if "vanilla_dkt" in models_dict:
            model_probs["vanilla_dkt"] = np.array(vanilla_preds_list, dtype=np.float32)
            model_metrics["vanilla_dkt"] = calculate_metrics_dict(y_true_arr, model_probs["vanilla_dkt"], clean_outcomes)

        if "reliability_gated_dkt" in models_dict:
            model_probs["reliability_gated_dkt"] = np.array(gated_preds_list, dtype=np.float32)
            model_metrics["reliability_gated_dkt"] = calculate_metrics_dict(y_true_arr, model_probs["reliability_gated_dkt"], clean_outcomes)

    # 3. Paired loss differences and Learner-level Bootstrap Confidence Intervals
    paired_comparisons: Dict[str, Any] = {}
    eps = 1e-7

    def _bce(probs):
        return -(y_true_arr * np.log(np.clip(probs, eps, 1 - eps)) + (1 - y_true_arr) * np.log(np.clip(1 - probs, eps, 1 - eps)))

    if "vanilla_dkt" in model_probs and "logistic_baseline" in model_probs:
        diff_v_point, diff_v_low, diff_v_high = compute_paired_bootstrap_ci(
            _bce(model_probs["vanilla_dkt"]), _bce(model_probs["logistic_baseline"]), learner_ids
        )
        paired_comparisons["vanilla_minus_logistic"] = {
            "mean_log_loss_diff": diff_v_point,
            "bootstrap_95_ci": [diff_v_low, diff_v_high],
            "favors": "vanilla_dkt" if diff_v_point < 0 else "logistic_baseline",
        }

    if "reliability_gated_dkt" in model_probs and "logistic_baseline" in model_probs:
        diff_g_point, diff_g_low, diff_g_high = compute_paired_bootstrap_ci(
            _bce(model_probs["reliability_gated_dkt"]), _bce(model_probs["logistic_baseline"]), learner_ids
        )
        paired_comparisons["reliability_gated_minus_logistic"] = {
            "mean_log_loss_diff": diff_g_point,
            "bootstrap_95_ci": [diff_g_low, diff_g_high],
            "favors": "reliability_gated_dkt" if diff_g_point < 0 else "logistic_baseline",
        }

    if "reliability_gated_dkt" in model_probs and "vanilla_dkt" in model_probs:
        diff_gv_point, diff_gv_low, diff_gv_high = compute_paired_bootstrap_ci(
            _bce(model_probs["reliability_gated_dkt"]), _bce(model_probs["vanilla_dkt"]), learner_ids
        )
        paired_comparisons["reliability_gated_minus_vanilla"] = {
            "mean_log_loss_diff": diff_gv_point,
            "bootstrap_95_ci": [diff_gv_low, diff_gv_high],
            "favors": "reliability_gated_dkt" if diff_gv_point < 0 else "vanilla_dkt",
        }

    return {
        "test_events_count": len(y_true_arr),
        "test_learners_count": len(np.unique(learner_ids)),
        "observed_positive_rate": round(float(np.mean(y_true_arr)), 4),
        "clean_positive_rate": round(float(np.mean(clean_outcomes)), 4),
        "models": model_metrics,
        "paired_comparisons": paired_comparisons,
        "diagnostics": {
            "learner_id": learner_ids,
            "y_true": y_true_arr,
            "y_clean": clean_outcomes,
            "true_reliability": true_reliabilities,
            "estimated_reliability": est_reliabilities,
            "pred_probs": model_probs,
        },
    }

