# Grow21 Testing, Validation, and Reproducibility Methodology

**Date:** October 18, 2026  
**Audited Status:** Complete, Verified, and Deterministic  
**Full Suite Result:** **66 passed in 23.76s** across 10 test modules  
**Validated Environment:** Python 3.13.11, pytest 9.1.1, pluggy 1.7.0, Windows 11 (win32)  

---

> [!CAUTION]
> **MANDATORY CLINICAL & RESEARCH DISCLAIMER**  
> All automated tests, empirical simulations, and statistical evaluations described in this document use **entirely synthetic computational agents and simulated sensor noise models**. 
> - **NO real-child data collection, clinical trials, or developmental studies have been conducted.**
> - Passing 66 unit/regression tests is **software verification of code invariants**, NOT proof of educational effectiveness, cognitive validity, or clinical suitability for children with Down syndrome (Trisomy 21).
> - Do not cite unit test pass rates as predictive accuracy, and do not describe synthetic simulation campaigns as human user trials.

---

## 1. Hierarchy of Testing & Validation Levels

To maintain scientific integrity, the Grow21 project distinguishes between five distinct levels of verification:

```
┌────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: Deterministic Automated Tests (pytest full suite)             │
│ • 66 tests across 10 modules verifying software invariants, bounds,    │
│   causal no-leakage timing, and regression edge cases in ~24s.         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ LEVEL 2: Mathematical Parity & Analytical Consistency Checks           │
│ • Discrete grid evaluations (7,938 points, max error 0.00e+00 IRT)     │
│ • Numerical bisection convergence, deadband, and step-limiting proofs  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ LEVEL 3: Synthetic Empirical Benchmarks (Multi-Seed Campaigns)         │
│ • Multi-cohort closed-loop simulations (1,728,000 interaction events)   │
│ • Paired seed-aware cluster bootstrap resampling (B=2,000 replicates)  │
│ • Distribution-shift stress testing and observation noise sweeps       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ LEVEL 4: Package & Fresh-Extraction Reproducibility Checks             │
│ • Self-contained ZIP integrity verification (zipfile.testzip() == None)│
│ • Standalone scratch extraction and execution from clean environment   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ LEVEL 5: NOT PERFORMED (Explicit Non-Goals of this Repository)         │
│ ✗ Real-world child or caregiver data collection                        │
│ ✗ Clinical or educational intervention trials                          │
│ ✗ Production frontend / mobile game end-to-end integration testing    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Executing the Test Suite

### 2.1 Clean-Checkout Self-Contained Test Suite (Default)
In a fresh repository checkout (where multi-hundred-megabyte raw event logs are excluded via `.gitignore`), the test suite runs 100% self-contained using compact deterministic fixtures located in `tests/fixtures/`:

```bash
# Portable command (with activated virtual environment or editable install)
python -m pytest -v tests
```

On Windows PowerShell using the local repository environment:
```powershell
.\.venv\Scripts\python.exe -m pytest -v tests
```

> [!NOTE]
> `pyproject.toml` explicitly configures `pythonpath = ["src"]`. When running `pytest` from the repository root, `src` is automatically added to `sys.path`. Setting `PYTHONPATH=src` manually is therefore not required in a standard virtual environment.

### 2.2 Clean-Checkout Observed Test Output
In a clean extraction without ignored >25MB event logs, 66 tests pass deterministically and 1 test (`test_full_phase_4_event_metrics_volume`) is cleanly skipped with actionable instructions:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.11, pytest-9.1.1, pluggy-1.7.0 -- .\.venv\Scripts\python.exe
rootdir: grow21_simulation
configfile: pyproject.toml
collected 67 items

tests/test_audit_regressions.py (13 tests) .............                 [ 19%]
tests/test_belief_state_estimation.py (14 tests) .............s          [ 40%]
tests/test_controller_constraints.py (4 tests) ....                     [ 46%]
tests/test_documentation_integrity.py (4 tests) ....                    [ 52%]
tests/test_model_shapes.py (2 tests) ..                                  [ 55%]
tests/test_no_leakage.py (3 tests) ...                                   [ 60%]
tests/test_outcome_model_alignment.py (10 tests) ..........              [ 75%]
tests/test_phase_5_state_estimation.py (11 tests) ...........            [ 91%]
tests/test_reliability.py (3 tests) ...                                  [ 96%]
tests/test_simulator.py (3 tests) ...                                    [100%]

======================= 66 passed, 1 skipped in 10.89s ========================
```

### 2.3 Full-Artifact Tests and Data Acquisition
Tests marked `@pytest.mark.full_artifacts` validate exhaustive multi-hundred-thousand interaction logs (e.g., verifying the full 540,000-row volume across all 5 scenarios and 4 policies).

Because raw event interaction logs and evidence ZIP archives exceed standard Git file limits and are excluded via `.gitignore`, clean checkouts skip this test cleanly by default.

To acquire or regenerate the full experimental artifacts:
- **Primary Method (Regenerate from Source):**
  Run the Phase 4 experiment orchestrator:
  ```powershell
  python -m grow21_simulation_benchmark.run_phase_4_experiments
  ```
- **Optional Archive Extraction Route (If You Possess the Archive):**
  For users who independently retain or download the release archive `Grow21_Phase_4_State_Estimation_Evidence.zip`:
  ```powershell
  python -c "import zipfile; zipfile.ZipFile('evidence/phase_4/Grow21_Phase_4_State_Estimation_Evidence.zip').extractall('.')"
  ```

Once `outputs/experiments/phase_4/state_estimation/phase_4_event_metrics.csv` is present, running `pytest -v tests` executes all 67 tests with zero skips:
```text
============================= 67 passed in 34.22s =============================
```

To run only the full-artifact tests:
```bash
python -m pytest -v -m full_artifacts
```
To run strictly self-contained tests without full-artifact tests:
```bash
python -m pytest -v -m "not full_artifacts"
```


---

## 3. Test Modules & Invariants Verified

| Test Module | Tests | Primary Behaviors & Invariants Verified |
| :--- | :---: | :--- |
| [`tests/test_simulator.py`](../../tests/test_simulator.py) | 3 | Game metadata definitions, virtual learner profile parameters, latent mastery bounding in $[0.05, 0.95]$, and inter-session exponential forgetting dynamics. |
| [`tests/test_reliability.py`](../../tests/test_reliability.py) | 3 | Grow21 sensor product rule $\hat{\rho}_t = \prod (1 - \hat{p}_m)$, missing sensor modalities **never impute zero uncertainty**, and vocalization modality properties. |
| [`tests/test_no_leakage.py`](../../tests/test_no_leakage.py) | 3 | Strict causal timing: DKT predictions are finalized strictly before hidden-state update with outcome $y_t$; feature extraction contains no future-leakage. |
| [`tests/test_model_shapes.py`](../../tests/test_model_shapes.py) | 2 | PyTorch tensor shapes, batch dimension alignment, and extreme sensor gating conditions ($\hat{\rho} \to 0$ freezes state; $\hat{\rho} \to 1$ accepts update). |
| [`tests/test_controller_constraints.py`](../../tests/test_controller_constraints.py) | 4 | Streak controller frustration cooldown drops, model-based controller deadband ($0.02$), session break rules (15 trials $\to$ 12h gap), and closed-loop execution. |
| [`tests/test_audit_regressions.py`](../../tests/test_audit_regressions.py) | 13 | Phase 2 audit regressions: immediate vs filtered direction reversals, sample totals in calibration bins (4,500/slice), common random numbers fairness, target reachability edge cases (reachable, unreachable-high, unreachable-low), bound flatlining vs dynamic stability, secondary skill transfer accounting, calibration bin ECE consistency, and closed-loop summary persistence. |
| [`tests/test_outcome_model_alignment.py`](../../tests/test_outcome_model_alignment.py) | 10 | Phase 3 model alignment: canonical 3-parameter IRT clean success parity across all 9 games (7,938 grid points, max error strictly $0.00\text{e}+00$), bisection solver reachability, step-limiting ($|\Delta D| \le 0.08$), non-linear difficulty curves, and legacy vs aligned bound saturation contrast. |
| [`tests/test_belief_state_estimation.py`](../../tests/test_belief_state_estimation.py) | 14 | Phase 4 state estimation: time alignment separating pre-outcome belief estimation from post-outcome belief updating, secondary transfer tracking on unpracticed skills, strict no-leakage of hidden ground truth, observation likelihood normalization, paired seed-aware cluster bootstrap invariants, split separation across cohorts, self-contained deterministic fixture recomputation, and full-artifact volume verification. |
| [`tests/test_phase_5_state_estimation.py`](../../tests/test_phase_5_state_estimation.py) | 11 | Phase 5 state estimation: expected transitions increase on both success and failure, analytical formula verification, secondary transfer ceiling ($0.95$), single forgetting per event, mastery prior defaults/fallbacks, synthetic parent-proxy no-leakage, **critical semantic safeguard** (prohibits routing $P(y_{\text{obs}})$ into clean-success controller), open-loop replay consistency, and offline model selector reproducibility (individual vs all). |
| [`tests/test_documentation_integrity.py`](../../tests/test_documentation_integrity.py) | 4 | Documentation integrity: absence of non-printing ASCII control characters in text files, absence of literal tabs in Markdown, preservation of required LaTeX expressions and code identifiers, and absence of fractured multi-line LaTeX splits. |


---

## 4. Fresh-Extraction Reproducibility Verification (Optional Archive Route)

> [!NOTE]
> All four evidence ZIP archives are excluded from public Git tracking via `.gitignore`. The procedure below is intended for environments where the evidence archive has been obtained as an independent release asset or retained in local storage.

To guarantee that research evidence packages are self-contained and reproducible outside the original developer workspace:
1. **Archive Integrity Check**: Evaluates `zipfile.is_zipfile()` and `zipfile.testzip() == None` for every evidence bundle in `evidence/phase_N/`.
2. **Isolated Extraction**: Extracts archive contents into an isolated scratch directory without polluting the primary tree.
3. **Automated Verification**: Runs full pytest suite and verifies artifact checksums.
4. **Clean Teardown**: Removes temporary extraction scratch directories.

### Reproducible Fresh-Extraction Verification Script
```python
import zipfile, os, shutil, subprocess, sys

zip_path = "evidence/phase_5/Grow21_Phase_5_Evidence.zip"
scratch_dir = "scratch_verification_env"

# 1. Verify archive
with zipfile.ZipFile(zip_path, "r") as zf:
    assert zf.testzip() is None, "Corrupted archive entry detected!"
    zf.extractall(scratch_dir)

# 2. Run test suite inside extracted bundle
result = subprocess.run([sys.executable, "-m", "pytest", "-v", "tests"], cwd=scratch_dir)
assert result.returncode == 0, f"Tests failed in fresh extraction: code {result.returncode}"

# 3. Teardown
shutil.rmtree(scratch_dir)
print("Fresh-extraction verification: PASSED")
```


---

## 5. Metric Definitions & Interpretations

| Metric | Mathematical Definition | What Improvement Means | What Improvement Does NOT Mean |
| :--- | :--- | :--- | :--- |
| **Log Loss (Cross-Entropy)** | $-\frac{1}{N} \sum [y_i \ln \hat{p}_i + (1 - y_i) \ln(1 - \hat{p}_i)]$ | Lower value indicates better probabilistic calibration and uncertainty quantification on observable binary outcomes. | Does NOT mean the model captures latent cognitive mastery or is suitable for direct difficulty adaptation. |
| **Brier Score** | $\frac{1}{N} \sum (\hat{p}_i - y_i)^2$ | Lower value indicates lower mean squared error in predicted probabilities. | Does NOT indicate that the predictor can invert observation sensor noise. |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve | Higher value indicates superior ranking separation between success and failure instances. | Insensitive to probability calibration (a model can have high AUC while being miscalibrated). |
| **Mastery MAE (Pre-Outcome)** | $\frac{1}{K} \sum_k \|\hat{\theta}_k(t-) - \theta_{\text{true}, k}(t-)\|$ | Lower value means the state estimator's belief state prior to trial $t$ is closer to true latent ability. | Does NOT mean the controller will achieve optimal challenge if the target is unreachable. |
| **Mastery Bias** | $\frac{1}{K} \sum_k (\hat{\theta}_k - \theta_{\text{true}, k})$ | Negative bias indicates underestimation (pessimistic); positive bias indicates overestimation (inflation). | Zero bias does not imply low variance or accurate individual tracking. |
| **Clean Probability Error** | $\|\hat{p}_{\text{clean}} - p_{\text{clean, true}}\|$ | Lower error means the controller's internal estimate of clean task success matches true generative IRT probability. | Does NOT guarantee the learner will enjoy the task or show educational progress. |
| **Minimum-Bound Saturation** | $\frac{1}{N_{\text{trials}}} \sum \mathbb{I}(D_t = D_{\min})$ | Percentage of trials clamped at lowest difficulty ($D_{\min}=0.08$). | **Lower saturation is NOT inherently better.** (Under distribution shift, positive belief inflation reduced saturation while severely worsening task mismatch). |
| **Target Reachability Rate** | $\frac{1}{N_{\text{trials}}} \sum \mathbb{I}(\hat{p}(D_{\max}) \le p^* \le \hat{p}(D_{\min}))$ | Percentage of trials where the target clean success $p^* = 0.75$ is mathematically achievable within difficulty bounds $[0.08, 0.92]$. | Reachability depends on learner mastery; struggling learners cannot reach $75\%$ success without reducing difficulty. |
| **Expected Calibration Error (ECE)** | $\sum_{b=1}^B \frac{\|B_b\|}{N} \|\text{acc}(B_b) - \text{conf}(B_b)\|$ | Lower value indicates predicted probabilities match empirical observation frequencies across decile bins. | Does NOT imply that the model understands causal cognitive factors. |
| **Paired Seed-Aware Cluster Bootstrap CI** | Resampling $K=180$ `(seed, learner_id)` clusters ($B=2,000$ replicates) | Resamples learner trajectories as complete clusters, preserving intra-learner correlations and draw multiplicity. | Does NOT validate generalization to real human populations outside simulated scenario distributions. |

### Sign Convention for Paired Differences
All comparative differences follow the strict standard convention:
$$\Delta = \text{Metric}_{\text{variant}} - \text{Metric}_{\text{baseline}}$$
- **For error metrics** (MAE, Log Loss, Probability Error): A **negative difference ($\Delta < 0$)** represents an **error reduction** (improvement).
- A result is reported as statistically significant **if and only if** the 95% bootstrap confidence interval does not cross zero (`interpretation = "significant_reduction"` or `"significant_increase"`).

---

## 6. Critical Limitations & Threats to Validity

1. **Synthetic-Only Data & Dynamics**: All virtual learners follow parametric IRT curves, Gaussian noise processes, and single-parameter exponential forgetting. Human children exhibit non-stationary cognitive shifts, sensory overstimulation, executive function variability, and emotional dynamics not modeled here.
2. **Fixed Population Priors**: Both `expected_transition` and `approx_bayesian_filter` assume static population parameters ($\bar{\eta}=0.08$, $\bar{\lambda}=0.02$). Under `distribution_shift` and `temporal_dynamics`, learners deviate systematically from these priors, leading to estimation error.
3. **The "Saturation Trap" under Distribution Shift**: In `distribution_shift`, lower-bound saturation drops from 60.5% to 37.5%, but mastery MAE increases by +34.1%. Estimator variants must NEVER be selected solely because they reduce minimum-difficulty clamp frequency.
4. **Caregiver Questionnaire Uncertainty**: In Phase 5, the questionnaire uncertainty field is metadata-only. Parent-proxy experiments demonstrate that unvalidated, noisy parent reports ($\sigma \ge 0.30$, missingness $\ge 60\%$) degrade tracking accuracy relative to the fixed default prior ($0.45$).
5. **Observed vs. Clean Outcome Divergence**: Offline sequence models predict $P(y_{\text{obs}}=1 \mid \text{features})$. The difficulty controller requires $P(y_{\text{clean}}=1 \mid \hat{\theta}, D)$. Routing offline models into clean-success controllers violates semantic invariants and is strictly blocked by software safeguards.
