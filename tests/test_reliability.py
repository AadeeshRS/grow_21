"""Unit tests for Grow21 reliability calculation, product rule, and missing modality logic."""

from pathlib import Path
import numpy as np
import pytest

from grow21_simulation_benchmark.reliability import ReliabilityEngine
from grow21_simulation_benchmark.tasks import TaskRegistry


@pytest.fixture
def registry():
    cfg_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    return TaskRegistry.from_yaml(cfg_path)


def test_grow21_product_rule(registry):
    engine = ReliabilityEngine(
        task_registry=registry,
        default_reliability_convention=1.0,
        missing_modality_default_uncertainty=0.25,
    )
    rng = np.random.RandomState(42)

    # Sample uncertainties with no missing modalities
    rec = engine.sample_trial_uncertainties(
        game_id="trace_glow",
        rng=rng,
        corruption_multiplier=1.0,
        modality_dropout_rate=0.0,  # force available
        uncertainty_bias=0.0,
        uncertainty_noise_scale=0.01,
    )

    # Check active modalities for trace_glow
    assert set(rec.active_modalities) == {"motor", "attention"}
    motor_p_hat = rec.modality_states["motor"].estimated_uncertainty
    att_p_hat = rec.modality_states["attention"].estimated_uncertainty

    expected_rho_hat = (1.0 - motor_p_hat) * (1.0 - att_p_hat)
    assert pytest.approx(rec.estimated_reliability, abs=1e-3) == expected_rho_hat
    assert 0.0 <= rec.estimated_reliability <= 1.0


def test_missing_modality_never_imputes_zero_uncertainty(registry):
    engine = ReliabilityEngine(
        task_registry=registry,
        default_reliability_convention=1.0,
        missing_modality_default_uncertainty=0.30,
    )
    rng = np.random.RandomState(99)

    # Force 100% modality dropout (all active sensors missing)
    rec = engine.sample_trial_uncertainties(
        game_id="bubble_pop_brigade",
        rng=rng,
        corruption_multiplier=1.0,
        modality_dropout_rate=1.0,
    )

    assert rec.any_modality_missing is True
    # For every active modality, estimated uncertainty must be conservative (not 0.0)
    for mid in rec.active_modalities:
        state = rec.modality_states[mid]
        assert state.is_available is False
        assert state.estimated_uncertainty >= 0.20
        # Crucial invariant: missing modality must NOT give 1.0 confidence factor
        assert state.estimated_uncertainty > 0.0

    # Total estimated reliability must be penalized, not 1.0
    assert rec.estimated_reliability < 0.90


def test_echo_safari_vocalization_modality(registry):
    engine = ReliabilityEngine(registry)
    rng = np.random.RandomState(7)

    rec = engine.sample_trial_uncertainties(
        game_id="echo_safari",
        rng=rng,
        corruption_multiplier=1.0,
        modality_dropout_rate=0.0,
    )

    assert "vocalization" in rec.active_modalities
    assert "motor" not in rec.active_modalities
    assert rec.modality_states["vocalization"].is_active is True
