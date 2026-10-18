"""CLI command for generating synthetic learner datasets with deterministic seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from .dataset import SimulationDatasetGenerator, events_to_dataframe
from .paths import DEFAULT_CONFIG_PATH, SCENARIOS_CONFIG_PATH, OUTPUTS_DIR
from .tasks import TaskRegistry


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic learner interaction dataset for Grow21."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to default config YAML",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=str(SCENARIOS_CONFIG_PATH),
        help="Path to scenarios YAML",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="base_heterogeneous",
        help="Scenario family to simulate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-learners",
        type=int,
        default=50,
        help="Number of virtual learners to generate",
    )
    parser.add_argument(
        "--events-per-learner",
        type=int,
        default=60,
        help="Number of events per learner",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "synthetic_dataset.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    registry = TaskRegistry.from_yaml(args.config)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.scenarios, "r", encoding="utf-8") as f:
        scenarios_cfg = yaml.safe_load(f)["scenarios"]

    sc_params = scenarios_cfg[args.scenario]
    generator = SimulationDatasetGenerator(registry, cfg)

    print(f"Generating synthetic dataset: scenario={args.scenario}, learners={args.num_learners}, events={args.events_per_learner}, seed={args.seed}")
    train_seqs, val_seqs, test_seqs = generator.generate_scenario_dataset(
        scenario_name=args.scenario,
        scenario_cfg=sc_params,
        seed=args.seed,
        num_learners=args.num_learners,
        events_per_learner=args.events_per_learner,
    )

    all_seqs = train_seqs + val_seqs + test_seqs
    df = events_to_dataframe(all_seqs)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    print(f"Saved {len(df)} synthetic interaction rows to {out_p}")


if __name__ == "__main__":
    main()
