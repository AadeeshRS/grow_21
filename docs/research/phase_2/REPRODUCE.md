# Grow21 Benchmark — Reproduction Instructions

This guide provides the exact steps to reproduce all benchmark, closed-loop simulation, calibration, and reliability-gating experiments from a clean working environment.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10 to 3.13
- Git
- PowerShell (Windows) or Bash (Linux / macOS)

### Setup Steps
```bash
# 1. Clone repository and navigate to package root
cd grow21_simulation

# 2. Create isolated virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# 4. Install dependencies in editable mode
pip install --upgrade pip
pip install -e .
```

---

## 2. Test Suite Execution

Run the complete test suite (32 unit and targeted regression tests):
```bash
python -m pytest -v tests
```
*Expected runtime: ~4 seconds. Status: All 32 passed.*

---

## 3. Running Experiments

### Experiment 1: Multi-Scenario Knowledge-Tracing Benchmark
Trains Logistic Regression, Vanilla DKT, and Reliability-Gated DKT across 5 scenarios and 3 random seeds on 500 virtual learners (4,500 test events per model/seed):
```bash
python -m grow21_simulation_benchmark.run_benchmark \
    --config configs/default.yaml \
    --scenarios configs/scenarios.yaml \
    --scenario all \
    --seeds 42,123,999 \
    --num-learners 500 \
    --events-per-learner 60 \
    --output-dir outputs/benchmark
```
*Outputs generated in `outputs/benchmark/`:*
- `metrics.csv`: Per-seed, per-model evaluation metrics (Log Loss, ROC-AUC, Brier Score, Accuracy, ECE, test event count).
- `summary.json`: Mean and standard deviation across seeds.
- `plots/calibration_*.png`: Calibration reliability curves for each scenario family.
- `plots/scenario_comparison.png`: Cross-scenario performance bar charts.

### Experiment 2: Closed-Loop Adaptive Simulation
Executes paired simulation comparison across Fixed Difficulty, Streak Adaptive (2-up/1-down), and Grow21 Model-Based Target Adaptive controllers with 60 learners and 60 trials per learner:
```bash
python -m grow21_simulation_benchmark.run_closed_loop \
    --config configs/default.yaml \
    --scenarios configs/scenarios.yaml \
    --scenario base_heterogeneous \
    --num-learners 60 \
    --trials-per-learner 60 \
    --seed 42 \
    --output-dir outputs/closed_loop
```
*Outputs generated in `outputs/closed_loop/`:*
- `closed_loop_events.csv`: 10,800 event-level interaction records with trial difficulty, timing, outcomes, and bound flags.
- `closed_loop_summary.json`: Performance summary (observed/clean success, immediate/filtered direction reversals, total variation, bound saturation proportions).
- `learner_mastery_summary.csv`: Initial and final latent mastery across all skills with task practice and secondary transfer counts.
- `plots/closed_loop_comparison.png`: Bar comparison of practiced skill mastery gains and difficulty dynamics.
- `plots/difficulty_trajectories.png`: Trial-by-trial difficulty paths illustrating target controller bound saturation.

### Experiment 3: Controlled Reliability-Gating Sweep & Paired Loss Evidence
Evaluates Vanilla DKT vs. Reliability-Gated DKT across 4 controlled sensor corruption conditions and exports learner-level paired losses:
```bash
python -m grow21_simulation_benchmark.audit_calibration_bins --action reliability_sweep
```
*Outputs generated in `outputs/audit/phase_2/`:*
- `reliability_sweep_summary.json`: Classification and calibration metrics with 95% learner-cluster bootstrap confidence intervals.
- `paired_loss_evidence.csv`: 216 learner-level paired loss records enabling independent reproduction of confidence intervals.

### Experiment 4: Full Calibration Bins Export
Generates detailed 10-bin calibration tables for the canonical 500-learner benchmark cohort ($N=4,500$ test events per model/scenario/seed):
```bash
python -m grow21_simulation_benchmark.audit_calibration_bins --action calibration_bins
```
*Outputs generated in `outputs/audit/phase_2/`:*
- `calibration_bins.csv`: 450 calibration bin rows with counts summing to exactly 4,500 for every evaluation slice.

---

## 4. Independent Verification Snippets

### Verifying Paired Bootstrap Confidence Intervals from Saved Evidence
```python
import pandas as pd
import numpy as np

df = pd.read_csv("outputs/audit/phase_2/paired_loss_evidence.csv")
for condition, group in df.groupby("condition"):
    diffs = group["gated_minus_vanilla_loss_diff"].values
    mean_diff = np.mean(diffs)
    
    # Bootstrap over learner units (1000 replicates, seed 42)
    rng = np.random.RandomState(42)
    boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(1000)]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    
    sig = "Significant" if (ci_low > 0 or ci_high < 0) else "Not Significant"
    print(f"{condition:<26}: Diff={mean_diff:+.4f}, 95% CI=[{ci_low:+.4f}, {ci_high:+.4f}] -> {sig}")
```

### Verifying Direction Reversals vs. Total Variation
```python
import pandas as pd
import numpy as np

df = pd.read_csv("outputs/closed_loop/closed_loop_events.csv")
for ctrl, group in df.groupby("controller"):
    imm_revs, filtered_revs, total_vars = [], [], []
    for lid, lgroup in group.groupby("learner_id"):
        diffs = lgroup.sort_values("step_idx")["difficulty"].values
        
        # Immediate reversals
        deltas = np.diff(diffs)
        imm = sum(1 for i in range(1, len(deltas)) if deltas[i] * deltas[i-1] < -1e-5)
        imm_revs.append(imm)
        
        # Filtered reversals (nonzero deltas only)
        nonzeros = deltas[np.abs(deltas) > 1e-4]
        filt = sum(1 for i in range(1, len(nonzeros)) if nonzeros[i] * nonzeros[i-1] < 0)
        filtered_revs.append(filt)
        
        # Total variation
        total_vars.append(np.sum(np.abs(deltas)))
        
    print(f"{ctrl:<18}: Imm Reversals={np.mean(imm_revs):.3f} | Filt Reversals={np.mean(filtered_revs):.3f} | Total Var={np.mean(total_vars):.3f}")
```

### Verifying Persisted Calibration Bins and Recomputed ECE
```python
import pandas as pd
import numpy as np

bins = pd.read_csv("outputs/audit/phase_2/calibration_bins.csv")
metrics = pd.read_csv("outputs/benchmark/metrics.csv")

for (scenario, seed, model), g in bins.groupby(["scenario", "seed", "model"]):
    assert g["count"].sum() == 4500, f"Sample count mismatch: {g['count'].sum()}"
    ece = (g["count"] * (g["predicted_prob"] - g["empirical_accuracy"]).abs()).sum() / g["count"].sum()
    m_row = metrics[(metrics["scenario"] == scenario) & (metrics["seed"] == seed) & (metrics["model"] == model)]
    m_ece = m_row["ece"].values[0]
    assert abs(ece - m_ece) < 1e-4, f"ECE mismatch: {ece} vs {m_ece}"
print("All 45 calibration bin slices verified: N=4,500 each and ECE matches metrics.csv within 1e-4.")
```

### Verifying Predictor-Simulator Probability Mismatch
```python
import pandas as pd

df = pd.read_csv("outputs/audit/phase_2/predictor_vs_simulator_comparison.csv")
low_d = df[df["difficulty"] == 0.08]
print(f"Mean controller predicted prob at D=0.08: {low_d['p_hat_controller'].mean():.4f}")
print(f"Mean simulator clean prob at D=0.08:     {low_d['p_clean_simulator'].mean():.4f}")
print(f"Mean gap (clean - controller):           {low_d['diff_clean_minus_controller'].mean():.4f}")
```
