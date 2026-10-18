"""Audit script computing full calibration bin tables and the 4-level reliability gating sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import yaml

from .dataset import SimulationDatasetGenerator
from .evaluate import calculate_metrics_dict, compute_paired_bootstrap_ci, evaluate_all_models_on_test
from .paths import AUDIT_OUTPUTS_DIR, DEFAULT_CONFIG_PATH, SCENARIOS_CONFIG_PATH
from .tasks import TaskRegistry
from .train import train_benchmark_models


def export_full_calibration_bins(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    scenarios_path: str | Path = SCENARIOS_CONFIG_PATH,
    output_csv_path: str | Path = AUDIT_OUTPUTS_DIR / "calibration_bins.csv",
    num_learners: int = 500,
    events_per_learner: int = 60,
    seeds: List[int] = [42, 123, 999],
) -> pd.DataFrame:
    registry = TaskRegistry.from_yaml(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(scenarios_path, "r", encoding="utf-8") as f:
        scenarios_cfg = yaml.safe_load(f)["scenarios"]

    generator = SimulationDatasetGenerator(registry, cfg)
    bin_rows: List[Dict[str, Any]] = []

    for sc_name, sc_params in scenarios_cfg.items():
        print(f"Auditing calibration bins for scenario: {sc_name}")
        for seed in seeds:
            train_seqs, val_seqs, test_seqs = generator.generate_scenario_dataset(
                scenario_name=sc_name,
                scenario_cfg=sc_params,
                seed=seed,
                num_learners=num_learners,
                events_per_learner=events_per_learner,
            )
            train_res = train_benchmark_models(
                train_seqs, val_seqs, registry, cfg.get("models", {}), seed=seed
            )
            eval_out = evaluate_all_models_on_test(train_res["models"], test_seqs, registry)

            cohort_id = f"main_benchmark_{num_learners}_learners_{eval_out['test_events_count']}_test_events"
            for m_name in ["logistic_baseline", "vanilla_dkt", "reliability_gated_dkt"]:
                bins = eval_out["models"][m_name]["calibration_bins"]
                for b in bins:
                    bin_rows.append({
                        "scenario": sc_name,
                        "seed": seed,
                        "cohort_id": cohort_id,
                        "model": m_name,
                        "bin_idx": b["bin_idx"],
                        "bin_low": b["bin_low"],
                        "bin_high": b["bin_high"],
                        "count": b["count"],
                        "mean_pred_prob": b["mean_pred_prob"],
                        "observed_positive_rate": b["empirical_acc"],
                        "empirical_acc": b["empirical_acc"],
                        "calibration_gap": b["calibration_gap"],
                    })

    df = pd.DataFrame(bin_rows)
    out_p = Path(output_csv_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    print(f"Saved {len(df)} calibration bin records to {out_p}")
    return df


def run_reliability_gating_controlled_sweep(
    config_path: str | Path,
    output_json_path: str | Path,
    num_learners: int = 400,
    events_per_learner: int = 60,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluates Vanilla DKT vs Reliability-Gated DKT across 4 controlled sensor corruption levels.

    Levels:
      1. Accurately estimated reliability (low noise, zero bias)
      2. Noisy reliability estimates (high noise scale, zero bias)
      3. Systematically miscalibrated reliability estimates (high bias)
      4. Missing modality information (high sensor dropout rate)
    """
    registry = TaskRegistry.from_yaml(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    conditions = {
        "level_1_accurate": {
            "description": "Composite stress Level 1: Accurately estimated reliability (low noise, zero bias, no sensor dropouts; corruption_multiplier=1.2)",
            "initial_mastery_mean": 0.45,
            "initial_mastery_std": 0.15,
            "learning_rate_mean": 0.08,
            "learning_rate_std": 0.02,
            "forgetting_rate_mean": 0.02,
            "forgetting_rate_std": 0.01,
            "corruption_multiplier": 1.2,
            "modality_dropout_rate": 0.0,
            "uncertainty_bias": 0.0,
            "uncertainty_noise_scale": 0.01,
            "session_gap_prob": 0.15,
            "max_gap_hours": 48.0,
        },
        "level_2_noisy": {
            "description": "Composite stress Level 2: Noisy reliability estimates (zero bias, high estimation noise sigma=0.15; corruption_multiplier=1.5)",
            "initial_mastery_mean": 0.45,
            "initial_mastery_std": 0.15,
            "learning_rate_mean": 0.08,
            "learning_rate_std": 0.02,
            "forgetting_rate_mean": 0.02,
            "forgetting_rate_std": 0.01,
            "corruption_multiplier": 1.5,
            "modality_dropout_rate": 0.0,
            "uncertainty_bias": 0.0,
            "uncertainty_noise_scale": 0.15,
            "session_gap_prob": 0.15,
            "max_gap_hours": 48.0,
        },
        "level_3_miscalibrated": {
            "description": "Composite stress Level 3: Systematically miscalibrated reliability (positive corruption bias=+0.12, over-estimating corruption noise / under-estimating reliability; corruption_multiplier=1.8)",
            "initial_mastery_mean": 0.45,
            "initial_mastery_std": 0.15,
            "learning_rate_mean": 0.08,
            "learning_rate_std": 0.02,
            "forgetting_rate_mean": 0.02,
            "forgetting_rate_std": 0.01,
            "corruption_multiplier": 1.8,
            "modality_dropout_rate": 0.0,
            "uncertainty_bias": 0.12,
            "uncertainty_noise_scale": 0.05,
            "session_gap_prob": 0.15,
            "max_gap_hours": 48.0,
        },
        "level_4_missing_modalities": {
            "description": "Composite stress Level 4: High missing modality rate (35% sensor dropouts requiring conservative uncertainty penalty; corruption_multiplier=1.8)",
            "initial_mastery_mean": 0.45,
            "initial_mastery_std": 0.15,
            "learning_rate_mean": 0.08,
            "learning_rate_std": 0.02,
            "forgetting_rate_mean": 0.02,
            "forgetting_rate_std": 0.01,
            "corruption_multiplier": 1.8,
            "modality_dropout_rate": 0.35,
            "uncertainty_bias": 0.02,
            "uncertainty_noise_scale": 0.05,
            "session_gap_prob": 0.15,
            "max_gap_hours": 48.0,
        },
    }

    generator = SimulationDatasetGenerator(registry, cfg)
    sweep_summary: Dict[str, Any] = {}
    paired_loss_records: List[Dict[str, Any]] = []

    for lvl_name, lvl_cfg in conditions.items():
        print(f"\n--- Running Reliability Sweep: {lvl_name} ---")
        train_seqs, val_seqs, test_seqs = generator.generate_scenario_dataset(
            scenario_name=lvl_name,
            scenario_cfg=lvl_cfg,
            seed=seed,
            num_learners=num_learners,
            events_per_learner=events_per_learner,
        )
        train_res = train_benchmark_models(
            train_seqs, val_seqs, registry, cfg.get("models", {}), seed=seed
        )
        eval_out = evaluate_all_models_on_test(train_res["models"], test_seqs, registry)

        v_m = eval_out["models"]["vanilla_dkt"]
        g_m = eval_out["models"]["reliability_gated_dkt"]
        l_m = eval_out["models"]["logistic_baseline"]
        paired = eval_out["paired_comparisons"]["reliability_gated_minus_vanilla"]

        # Compute per-learner losses for the paired loss artifact
        diag = eval_out["diagnostics"]
        y_true = diag["y_true"]
        l_ids = diag["learner_id"]
        p_v = diag["pred_probs"]["vanilla_dkt"]
        p_g = diag["pred_probs"]["reliability_gated_dkt"]
        p_l = diag["pred_probs"]["logistic_baseline"]

        eps = 1e-7
        bce_v = -(y_true * np.log(np.clip(p_v, eps, 1 - eps)) + (1 - y_true) * np.log(np.clip(1 - p_v, eps, 1 - eps)))
        bce_g = -(y_true * np.log(np.clip(p_g, eps, 1 - eps)) + (1 - y_true) * np.log(np.clip(1 - p_g, eps, 1 - eps)))
        bce_l = -(y_true * np.log(np.clip(p_l, eps, 1 - eps)) + (1 - y_true) * np.log(np.clip(1 - p_l, eps, 1 - eps)))

        unique_learners = np.unique(l_ids)
        for uid in unique_learners:
            mask = (l_ids == uid)
            loss_v = float(np.mean(bce_v[mask]))
            loss_g = float(np.mean(bce_g[mask]))
            loss_l = float(np.mean(bce_l[mask]))
            paired_loss_records.append({
                "condition": lvl_name,
                "learner_id": int(uid),
                "test_events": int(np.sum(mask)),
                "vanilla_log_loss": round(loss_v, 5),
                "reliability_gated_log_loss": round(loss_g, 5),
                "logistic_log_loss": round(loss_l, 5),
                "gated_minus_vanilla_loss_diff": round(loss_g - loss_v, 5),
            })

        sweep_summary[lvl_name] = {
            "description": lvl_cfg["description"],
            "vanilla_dkt": {
                "log_loss": v_m["log_loss"],
                "brier_score": v_m["brier_score"],
                "roc_auc": v_m["roc_auc"],
                "ece": v_m["ece"],
            },
            "reliability_gated_dkt": {
                "log_loss": g_m["log_loss"],
                "brier_score": g_m["brier_score"],
                "roc_auc": g_m["roc_auc"],
                "ece": g_m["ece"],
            },
            "logistic_baseline": {
                "log_loss": l_m["log_loss"],
                "brier_score": l_m["brier_score"],
                "roc_auc": l_m["roc_auc"],
                "ece": l_m["ece"],
            },
            "paired_difference_gated_minus_vanilla": {
                "mean_diff": paired["mean_log_loss_diff"],
                "ci_95": paired["bootstrap_95_ci"],
                "favors": paired["favors"],
                "statistically_significant_at_alpha_0_05": bool(
                    paired["bootstrap_95_ci"][0] > 0 or paired["bootstrap_95_ci"][1] < 0
                ),
            },
            "bootstrap_settings": {
                "resampling_unit": "learner_clusters",
                "num_replicates": 1000,
                "confidence_level": 0.95,
                "random_seed": 42,
            },
        }

    out_p = Path(output_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(sweep_summary, f, indent=2)
    print(f"\nSaved reliability sweep summary to {out_p}")

    # Save paired loss evidence CSV
    paired_csv_path = out_p.parent / "paired_loss_evidence.csv"
    pd.DataFrame(paired_loss_records).to_csv(paired_csv_path, index=False)
    print(f"Saved {len(paired_loss_records)} paired learner loss records to {paired_csv_path}")

    return sweep_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["calibration_bins", "reliability_sweep", "all"], default="all")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--scenarios", default=str(SCENARIOS_CONFIG_PATH))
    parser.add_argument("--bins-output", default=str(AUDIT_OUTPUTS_DIR / "calibration_bins.csv"))
    parser.add_argument("--sweep-output", default=str(AUDIT_OUTPUTS_DIR / "reliability_sweep_summary.json"))
    args = parser.parse_args()

    if args.action in ["calibration_bins", "all"]:
        export_full_calibration_bins(
            config_path=args.config,
            scenarios_path=args.scenarios,
            output_csv_path=args.bins_output,
            num_learners=500,
            events_per_learner=60,
            seeds=[42, 123, 999],
        )

    if args.action in ["reliability_sweep", "all"]:
        run_reliability_gating_controlled_sweep(
            config_path=args.config,
            output_json_path=args.sweep_output,
            num_learners=350,
            events_per_learner=60,
            seed=42,
        )
