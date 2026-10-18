# Grow21 Phase 3 — Reproduction Guide

This document provides explicit, terminal-executable commands to deterministically reproduce all Phase 3 model-alignment experiments, numerical parity checks, regression tests, and figures.

---

## 1. Environment Requirements

- **OS**: Windows (tested on Windows 11 Enterprise x86_64), Linux, or macOS
- **Python**: 3.10 to 3.13 (Active environment: Python 3.13.11)
- **Key Dependencies**:
  - `torch >= 2.0.0`
  - `numpy >= 1.24.0`
  - `pandas >= 2.0.0`
  - `matplotlib >= 3.7.0`
  - `pyyaml >= 6.0`
  - `pytest >= 8.0.0`

### Environment Activation
From the repository root:

```bash
# Windows (PowerShell)
& ".\.venv\Scripts\Activate.ps1"

# Windows (Command Prompt)
.\.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

---

## 2. Deterministic Test Suite Verification

Run the complete 42-test suite covering simulator physics, offline models, regression invariants, and the 10 new Phase 3 model alignment tests:

```bash
python -m pytest -v tests
```

Expected output:
```text
============================== 42 passed in ~8.7s ==============================
```

To run only the Phase 3 alignment test suite:

```bash
python -m pytest -v tests/test_outcome_model_alignment.py
```

Expected output:
```text
============================== 10 passed in ~0.8s ==============================
```

---

## 3. Re-running the Phase 3 Experimental Campaign

To re-run the full 5-scenario, 3-seed, 4-policy simulation campaign ($N = 216,000$ events) and generate all Phase 3 datasets and plots:

```bash
python -m grow21_simulation_benchmark.run_phase_3_experiments --num-learners 60 --trials-per-learner 60 --seeds 42 123 999
```

This automated pipeline:
1. Runs the 7,938-point canonical formula parity audit across all 9 games, writing to `outputs/phase_3/diagnostics/formula_parity_results.csv` and `formula_parity_summary.json`.
2. Simulates closed-loop interactions for 60 learners $\times$ 60 trials across all 5 scenarios and seeds [42, 123, 999] comparing Fixed, Streak, Legacy Surrogate, and Model-Aligned controllers.
3. Computes summary metrics and exports `phase_3_closed_loop_metrics.csv`, `phase_3_closed_loop_summary.json`, `phase_3_learner_mastery_summary.csv`, and `phase_3_per_game_summary.csv`.
4. Renders publication figures in `outputs/phase_3/plots/`.
5. Mirrors primary summary artifacts and plots into `outputs/audit/phase_3/`.

Typical execution time: **~30 seconds**.

---

## 4. Re-running a Single Scenario for Inspection

To run a single scenario (e.g. `base_heterogeneous`, seed 42) through the standard closed-loop CLI:

```bash
python -m grow21_simulation_benchmark.run_closed_loop --scenario base_heterogeneous --seed 42 --num-learners 60 --trials-per-learner 60 --output-dir outputs/phase_3/closed_loop
```

---

## 5. Optional: Verifying Evidence Bundle Integrity (Local Archive / Release Asset)

> [!NOTE]
> Evidence ZIPs are local-only / release assets excluded from public Git tracking by `.gitignore`. The verification below is applicable if the archive has been retained locally or acquired as a release artifact.

To verify the integrity and structure of the Phase 3 Evidence ZIP:

```python
import zipfile
from pathlib import Path

zip_path = Path("evidence/phase_3/Grow21_Phase_3_Model_Alignment_Evidence.zip")
if not zip_path.exists():
    print(f"Archive {zip_path} not found (excluded from Git). Re-run experiments via Section 3 to generate outputs directly.")
else:
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_file = zf.testzip()
        assert bad_file is None, f"Corrupted file found in ZIP: {bad_file}"
        files = zf.namelist()
        print(f"Verified {len(files)} files in {zip_path.name}. Compressed size: {zip_path.stat().st_size / 1024:.1f} KB.")
```
