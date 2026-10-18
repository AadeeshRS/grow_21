"""Unit tests verifying model tensor shapes, parameter counts, and gating mathematics."""

import pytest
import torch

from grow21_simulation_benchmark.models import ReliabilityGatedDKT, VanillaDKT


def test_dkt_shapes_and_bounds():
    batch_size = 5
    seq_len = 12
    meta_dim = 26
    hidden_dim = 48
    embedding_dim = 24

    model = VanillaDKT(meta_dim=meta_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    meta = torch.randn(batch_size, seq_len, meta_dim)
    outcomes = torch.randint(0, 2, (batch_size, seq_len)).float()

    preds, final_h = model(meta, outcomes)

    assert preds.shape == (batch_size, seq_len)
    assert final_h.shape == (batch_size, hidden_dim)
    assert torch.all(preds >= 0.0) and torch.all(preds <= 1.0)
    assert model.count_parameters() < 50_000  # lightweight CPU model


def test_reliability_gated_dkt_extreme_gating_behavior():
    """Verifies h_t = h_{t-1} + rho_t * (hbar_t - h_{t-1}) at rho=0 and rho=1."""
    batch_size = 2
    seq_len = 1
    meta_dim = 15
    hidden_dim = 32
    embedding_dim = 16

    model = ReliabilityGatedDKT(meta_dim=meta_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    model.eval()

    meta = torch.randn(batch_size, seq_len, meta_dim)
    outcomes = torch.ones(batch_size, seq_len)
    initial_h = torch.randn(batch_size, hidden_dim)

    # Case A: rho = 0.0 (completely unreliable event)
    # The recurrent state must remain exactly equal to initial_h
    zero_rel = torch.zeros(batch_size, seq_len)
    with torch.no_grad():
        _, final_h_zero = model(meta, outcomes, zero_rel, initial_state=initial_h)

    assert torch.allclose(final_h_zero, initial_h, atol=1e-6), (
        "When reliability is 0, hidden state must not change from initial state."
    )

    # Case B: rho = 1.0 (fully reliable event)
    # The recurrent state must equal candidate hbar_t
    one_rel = torch.ones(batch_size, seq_len)
    with torch.no_grad():
        e_meta = model.meta_proj(meta[:, 0, :])
        event_in = torch.cat([e_meta, outcomes[:, 0].unsqueeze(-1)], dim=-1)
        expected_hbar = model.gru_cell(event_in, initial_h)
        _, final_h_one = model(meta, outcomes, one_rel, initial_state=initial_h)

    assert torch.allclose(final_h_one, expected_hbar, atol=1e-6), (
        "When reliability is 1, hidden state must equal candidate GRU update."
    )
