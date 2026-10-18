"""Explicit unit tests proving the no-leakage invariant across all three models.

Verifies that event t's observed outcome cannot influence its own prediction y_hat_t,
while legitimately updating the state for subsequent events t+1 onwards.
"""

from pathlib import Path
import copy
import numpy as np
import pytest
import torch

from grow21_simulation_benchmark.dataset import SimulationDatasetGenerator
from grow21_simulation_benchmark.features import FeatureExtractor
from grow21_simulation_benchmark.models import (
    LogisticBaselineModel,
    ReliabilityGatedDKT,
    VanillaDKT,
)
from grow21_simulation_benchmark.tasks import TaskRegistry


@pytest.fixture
def test_setup():
    cfg_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    registry = TaskRegistry.from_yaml(cfg_path)
    return cfg_path, registry


def test_dkt_no_leakage_invariant():
    """Proves VanillaDKT's prediction at step t is invariant to outcome t."""
    batch_size = 3
    seq_len = 5
    meta_dim = 20

    meta = torch.randn(batch_size, seq_len, meta_dim)
    # Case 0: outcome at t=2 is 0
    outcomes_case0 = torch.zeros(batch_size, seq_len)
    # Case 1: outcome at t=2 is 1
    outcomes_case1 = torch.zeros(batch_size, seq_len)
    outcomes_case1[:, 2] = 1.0

    model = VanillaDKT(meta_dim=meta_dim, hidden_dim=32, embedding_dim=16)
    model.eval()

    with torch.no_grad():
        preds_case0, _ = model(meta, outcomes_case0)
        preds_case1, _ = model(meta, outcomes_case1)

    # Invariant: Step t=2 prediction MUST BE EXACTLY IDENTICAL in both cases
    for b in range(batch_size):
        for step in range(3):  # steps 0, 1, and 2
            diff = abs(preds_case0[b, step].item() - preds_case1[b, step].item())
            assert diff == 0.0, (
                f"Leakage detected at step {step}! Outcome at t=2 leaked into prediction: diff={diff}"
            )

        # But subsequent step t=3 SHOULD differ because the state was updated by outcome t=2
        subsequent_diff = abs(preds_case0[b, 3].item() - preds_case1[b, 3].item())
        assert subsequent_diff > 1e-6, (
            f"Step 3 failed to update state from outcome 2: diff={subsequent_diff}"
        )


def test_reliability_gated_dkt_no_leakage_invariant():
    """Proves ReliabilityGatedDKT's prediction at step t is invariant to outcome t."""
    batch_size = 4
    seq_len = 6
    meta_dim = 20

    meta = torch.randn(batch_size, seq_len, meta_dim)
    reliabilities = torch.full((batch_size, seq_len), 0.75)

    outcomes_case0 = torch.zeros(batch_size, seq_len)
    outcomes_case1 = torch.zeros(batch_size, seq_len)
    target_step = 3
    outcomes_case1[:, target_step] = 1.0

    model = ReliabilityGatedDKT(meta_dim=meta_dim, hidden_dim=32, embedding_dim=16)
    model.eval()

    with torch.no_grad():
        preds_case0, _ = model(meta, outcomes_case0, reliabilities)
        preds_case1, _ = model(meta, outcomes_case1, reliabilities)

    for b in range(batch_size):
        for step in range(target_step + 1):  # up to target_step
            diff = abs(preds_case0[b, step].item() - preds_case1[b, step].item())
            assert diff == 0.0, (
                f"ReliabilityGatedDKT leakage at step {step}: diff={diff}"
            )

        subsequent_diff = abs(preds_case0[b, target_step + 1].item() - preds_case1[b, target_step + 1].item())
        assert subsequent_diff > 1e-6, (
            f"Subsequent step did not reflect updated state: diff={subsequent_diff}"
        )


def test_logistic_feature_extractor_no_leakage(test_setup):
    """Proves FeatureExtractor for Logistic Baseline uses no target outcome from event t."""
    _, registry = test_setup
    extractor = FeatureExtractor(registry)

    # Generate a sample sequence
    generator = SimulationDatasetGenerator(registry, {"simulation": {}})
    rng = np.random.RandomState(42)
    from grow21_simulation_benchmark.simulator import generate_learner_profile
    profile = generate_learner_profile(1, registry.skill_ids, rng, {})
    seq = generator.generate_events_for_learner(profile, 10, rng, {})

    # Extract features for original sequence
    X_orig, y_orig, _ = extractor.extract_sequence_features(seq)

    # Make a copy of the sequence and flip the outcome of event index 4
    seq_flipped = copy.deepcopy(seq)
    orig_y4 = seq_flipped[4].observed_outcome
    seq_flipped[4].observed_outcome = 1 - orig_y4

    X_flipped, y_flipped, _ = extractor.extract_sequence_features(seq_flipped)

    # Feature row 4 MUST be identical (no leakage of event 4's outcome into feature 4)
    np.testing.assert_array_equal(
        X_orig[4],
        X_flipped[4],
        err_msg="Feature row 4 changed when observed_outcome 4 was modified! Target leakage detected.",
    )

    # Features before step 4 must also be completely identical
    for t in range(4):
        np.testing.assert_array_equal(X_orig[t], X_flipped[t])

    # Feature row 5 SHOULD differ because event 4's outcome is part of causal history for step 5
    assert not np.array_equal(X_orig[5], X_flipped[5]), (
        "Feature row 5 should reflect historical outcome of event 4"
    )
