"""End-to-end CLI runner for Grow21 knowledge-tracing benchmark.

Runs offline comparison across Logistic Regression, Vanilla DKT, and Reliability-Gated DKT
over specified scenario families and multiple random seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import yaml

from .dataset import SimulationDatasetGenerator
from .evaluate import evaluate_all_models_on_test
from .paths import DEFAULT_CONFIG_PATH, SCENARIOS_CONFIG_PATH, OUTPUTS_DIR
from .plots import plot_calibration_curves, plot_scenario_comparison
from .tasks import TaskRegistry
from .train import train_benchmark_models


def run_benchmark(
    config_path: str | Path,
    scenarios_path: str | Path,
    scenario_names: List[str],
    seeds: List[int],
    num_learners: int,
    events_per_learner: int,
    output_dir: str | Path,
    models: str | List[str] | None = None,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    registry = TaskRegistry.from_yaml(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(scenarios_path, "r", encoding="utf-8") as f:
        scenarios_cfg = yaml.safe_load(f)["scenarios"]

    generator = SimulationDatasetGenerator(registry, cfg)
    all_run_records: List[Dict[str, Any]] = []
    summary_by_scenario: Dict[str, Any] = {}

    valid_models = ["logistic_baseline", "vanilla_dkt", "reliability_gated_dkt"]
    if models is None:
        models = cfg.get("benchmark", {}).get("models", "all")

    if isinstance(models, str):
        if models.strip().lower() == "all":
            models_to_compare = list(valid_models)
        elif "," in models:
            models_to_compare = [m.strip() for m in models.split(",") if m.strip()]
        else:
            models_to_compare = [models.strip()]
    elif isinstance(models, (list, tuple)):
        models_to_compare = list(models)
    else:
        models_to_compare = list(valid_models)

    for m in models_to_compare:
        if m not in valid_models:
            raise ValueError(
                f"Invalid model '{m}'. Allowed choices are 'all' or subsets of {valid_models}."
            )

    print("=" * 80)
    print("Grow21 Knowledge-Tracing Benchmark Execution")
    print(f"Scenarios: {scenario_names}")
    print(f"Seeds: {seeds} | Learners: {num_learners} | Events/Learner: {events_per_learner}")
    print(f"Selected Models: {models_to_compare}")
    print("=" * 80)

    for sc_name in scenario_names:
        if sc_name not in scenarios_cfg:
            print(f"Skipping unknown scenario: {sc_name}")
            continue

        sc_params = scenarios_cfg[sc_name]
        print(f"\n>>> Running Scenario: {sc_name}")
        print(f"    Description: {sc_params.get('description', '')}")

        seed_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models_to_compare}
        last_eval_result = None

        for seed in seeds:
            print(f"  --> Seed {seed} ...", end=" ", flush=True)

            # 1. Generate data with strictly disjoint learners
            train_seqs, val_seqs, test_seqs = generator.generate_scenario_dataset(
                scenario_name=sc_name,
                scenario_cfg=sc_params,
                seed=seed,
                num_learners=num_learners,
                events_per_learner=events_per_learner,
            )

            # 2. Train selected benchmark models
            train_output = train_benchmark_models(
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                task_registry=registry,
                model_cfg=cfg.get("models", {}),
                seed=seed,
                models_to_train=models_to_compare,
            )

            # 3. Evaluate identically on held-out test learners
            eval_output = evaluate_all_models_on_test(
                models_dict=train_output["models"],
                test_seqs=test_seqs,
                task_registry=registry,
            )
            last_eval_result = eval_output

            # Log row per model
            for m in models_to_compare:
                m_metrics = eval_output["models"][m]
                rec = {
                    "scenario": sc_name,
                    "seed": seed,
                    "model": m,
                    "runtime_sec": train_output["runtimes"][m],
                    "param_count": train_output["param_counts"][m],
                    "log_loss": m_metrics["log_loss"],
                    "brier_score": m_metrics["brier_score"],
                    "roc_auc": m_metrics["roc_auc"],
                    "accuracy": m_metrics["accuracy"],
                    "balanced_accuracy": m_metrics["balanced_accuracy"],
                    "ece": m_metrics["ece"],
                    "test_events": eval_output["test_events_count"],
                    "test_learners": eval_output["test_learners_count"],
                }
                all_run_records.append(rec)
                seed_results[m].append(rec)

            paired = eval_output.get("paired_comparisons", {})
            log_strs = [f"{m}={eval_output['models'][m]['log_loss']:.4f}" for m in models_to_compare if m in eval_output.get("models", {})]
            paired_str = ""
            if "reliability_gated_minus_logistic" in paired:
                paired_str = f" (Diff G-Log: {paired['reliability_gated_minus_logistic']['mean_log_loss_diff']:.4f})"
            print(f"Done. LogLoss: {', '.join(log_strs)}{paired_str}")

        # Plot calibration curve for the last seed of this scenario
        if last_eval_result:
            plot_calibration_curves(
                last_eval_result,
                plots_dir / f"calibration_{sc_name}.png",
                title_suffix=f"({sc_name})",
            )

        # Aggregate across seeds
        sc_summary: Dict[str, Any] = {"models": {}}
        for m in models_to_compare:
            m_recs = seed_results[m]
            ll_vals = [r["log_loss"] for r in m_recs]
            brier_vals = [r["brier_score"] for r in m_recs]
            auc_vals = [r["roc_auc"] for r in m_recs]
            acc_vals = [r["accuracy"] for r in m_recs]
            runtimes = [r["runtime_sec"] for r in m_recs]

            sc_summary["models"][m] = {
                "log_loss_mean": round(float(np.mean(ll_vals)), 4),
                "log_loss_std": round(float(np.std(ll_vals)), 4),
                "brier_score_mean": round(float(np.mean(brier_vals)), 4),
                "brier_score_std": round(float(np.std(brier_vals)), 4),
                "roc_auc_mean": round(float(np.mean(auc_vals)), 4),
                "roc_auc_std": round(float(np.std(auc_vals)), 4),
                "accuracy_mean": round(float(np.mean(acc_vals)), 4),
                "accuracy_std": round(float(np.std(acc_vals)), 4),
                "runtime_sec_mean": round(float(np.mean(runtimes)), 2),
                "param_count": m_recs[0]["param_count"],
            }
        summary_by_scenario[sc_name] = sc_summary

    # Save detailed metrics CSV
    metrics_df = pd.DataFrame(all_run_records)
    csv_path = out_dir / "metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to {csv_path}")

    # Generate scenario comparison bar charts if multiple scenarios
    if len(summary_by_scenario) > 1:
        plot_scenario_comparison(summary_by_scenario, plots_dir / "scenario_comparison.png")

    # Save JSON summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_by_scenario, f, indent=2)
    print(f"Summary JSON saved to {summary_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Scenario':<22} | {'Model':<22} | {'Log Loss':<15} | {'ROC-AUC':<15} | {'Runtime'}")
    print("-" * 90)
    for sc, sc_data in summary_by_scenario.items():
        for m, m_data in sc_data["models"].items():
            ll_str = f"{m_data['log_loss_mean']:.4f} ± {m_data['log_loss_std']:.4f}"
            auc_str = f"{m_data['roc_auc_mean']:.4f} ± {m_data['roc_auc_std']:.4f}"
            rt_str = f"{m_data['runtime_sec_mean']:.1f}s"
            print(f"{sc:<22} | {m:<22} | {ll_str:<15} | {auc_str:<15} | {rt_str}")
    print("=" * 90)

    return summary_by_scenario


def main():
    parser = argparse.ArgumentParser(description="Run Grow21 KT Benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Default config path",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=str(SCENARIOS_CONFIG_PATH),
        help="Scenarios config path",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Comma-separated scenarios or 'all'",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,999",
        help="Comma-separated random seeds",
    )
    parser.add_argument(
        "--num-learners",
        type=int,
        default=600,
        help="Total virtual learners per scenario",
    )
    parser.add_argument(
        "--events-per-learner",
        type=int,
        default=60,
        help="Events per learner",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUTS_DIR),
        help="Output directory",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Offline predictive benchmark selector: 'all', 'logistic_baseline', 'vanilla_dkt', 'reliability_gated_dkt', or comma-separated list",
    )
    args = parser.parse_args()

    with open(args.scenarios, "r", encoding="utf-8") as f:
        sc_data = yaml.safe_load(f)["scenarios"]

    if args.scenario.strip().lower() == "all":
        scenario_list = list(sc_data.keys())
    else:
        scenario_list = [s.strip() for s in args.scenario.split(",") if s.strip()]

    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    run_benchmark(
        config_path=args.config,
        scenarios_path=args.scenarios,
        scenario_names=scenario_list,
        seeds=seed_list,
        num_learners=args.num_learners,
        events_per_learner=args.events_per_learner,
        output_dir=args.output_dir,
        models=args.models,
    )


if __name__ == "__main__":
    main()
