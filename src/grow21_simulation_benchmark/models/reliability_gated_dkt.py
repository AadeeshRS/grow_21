"""Grow21 Reliability-Gated Deep Knowledge Tracing (DKT) model.

Implements the Grow21 reliability-gated recurrent state transition:
  h_t = h_{t-1} + rho_hat_t * (hbar_t - h_{t-1})
where hbar_t is the candidate GRU update and rho_hat_t in [0, 1] is the
estimated event reliability from active uncertainty modalities.
Prediction y_hat_t is generated strictly before outcome y_t and rho_hat_t
update the recurrent state.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class ReliabilityGatedDKT(nn.Module):
    """Reliability-gated recurrent knowledge tracing model."""

    def __init__(
        self,
        meta_dim: int,
        hidden_dim: int = 48,
        embedding_dim: int = 24,
    ):
        super().__init__()
        self.meta_dim = meta_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Pre-response metadata projection (identical architecture capacity to Vanilla DKT)
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

        # Candidate recurrent update cell
        self.gru_cell = nn.GRUCell(
            input_size=embedding_dim + 1,
            hidden_size=hidden_dim,
        )

        # Prediction head conditioning on h_{t-1} and e_meta_t
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        meta_features: torch.Tensor,
        outcomes: torch.Tensor,
        reliabilities: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the sequence forward pass with reliability gating and causal timing.

        Args:
            meta_features: [batch_size, seq_len, meta_dim] pre-response task features.
            outcomes: [batch_size, seq_len] observed interaction outcomes y_t in {0, 1}.
            reliabilities: [batch_size, seq_len] estimated reliability rho_hat_t in [0, 1].
            mask: [batch_size, seq_len] sequence padding mask (1 = valid, 0 = pad).
            initial_state: [batch_size, hidden_dim] optional initial recurrent state.

        Returns:
            predictions: [batch_size, seq_len] predicted probabilities P(y_t = 1).
            final_hidden: [batch_size, hidden_dim] final recurrent state.
        """
        batch_size, seq_len, _ = meta_features.shape
        device = meta_features.device

        if mask is None:
            mask = torch.ones((batch_size, seq_len), device=device)

        if initial_state is None:
            h_prev = torch.zeros((batch_size, self.hidden_dim), device=device)
        else:
            h_prev = initial_state

        predictions_list = []

        for t in range(seq_len):
            meta_t = meta_features[:, t, :]
            e_meta_t = self.meta_proj(meta_t)

            # 1. PREDICT y_t CAUSALLY: depends solely on h_{t-1} and pre-response e_meta_t
            pred_input = torch.cat([h_prev, e_meta_t], dim=-1)
            y_hat_t = self.pred_head(pred_input).squeeze(-1)  # [batch_size]
            predictions_list.append(y_hat_t)

            # 2. CANDIDATE UPDATE hbar_t: formed after observing outcome y_t
            y_t = outcomes[:, t].unsqueeze(-1)  # [batch_size, 1]
            event_input = torch.cat([e_meta_t, y_t], dim=-1)
            h_bar_t = self.gru_cell(event_input, h_prev)

            # 3. GROW21 RELIABILITY GATE:
            # h_t = h_{t-1} + rho_hat_t * (hbar_t - h_{t-1})
            rho_t = reliabilities[:, t].unsqueeze(-1)  # [batch_size, 1]
            # Ensure rho_t is clamped in [0, 1]
            rho_t = torch.clamp(rho_t, 0.0, 1.0)
            h_gated = h_prev + rho_t * (h_bar_t - h_prev)

            # Masked update for variable length sequences
            mask_t = mask[:, t].unsqueeze(-1)
            h_prev = mask_t * h_gated + (1.0 - mask_t) * h_prev

        predictions = torch.stack(predictions_list, dim=1)  # [batch_size, seq_len]
        return predictions, h_prev

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
