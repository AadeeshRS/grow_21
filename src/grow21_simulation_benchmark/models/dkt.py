"""Vanilla Deep Knowledge Tracing (DKT) model for Grow21.

Sequential GRU-based knowledge-tracing model strictly enforcing causal prediction
timing: next interaction outcome is predicted using history through t-1 and
pre-response task metadata before the outcome updates the recurrent state.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class VanillaDKT(nn.Module):
    """Vanilla recurrent DKT model with pre-response metadata conditioning."""

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

        # Pre-response metadata projection
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

        # Recurrent state transition consumes [meta_emb, y_outcome]
        self.gru_cell = nn.GRUCell(
            input_size=embedding_dim + 1,
            hidden_size=hidden_dim,
        )

        # Causal prediction head conditioning on h_{t-1} and current task meta
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
        mask: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the sequence forward pass with strict causal timing.

        Args:
            meta_features: [batch_size, seq_len, meta_dim] pre-response features.
            outcomes: [batch_size, seq_len] observed interaction outcomes y_t in {0, 1}.
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

            # 1. PREDICT y_t CAUSALLY: strictly using h_{t-1} and pre-response e_meta_t
            pred_input = torch.cat([h_prev, e_meta_t], dim=-1)
            y_hat_t = self.pred_head(pred_input).squeeze(-1)  # [batch_size]
            predictions_list.append(y_hat_t)

            # 2. STATE UPDATE: outcome y_t is observed ONLY AFTER prediction is generated
            y_t = outcomes[:, t].unsqueeze(-1)  # [batch_size, 1]
            event_input = torch.cat([e_meta_t, y_t], dim=-1)
            h_next = self.gru_cell(event_input, h_prev)

            # Masked update for variable length sequences
            mask_t = mask[:, t].unsqueeze(-1)
            h_prev = mask_t * h_next + (1.0 - mask_t) * h_prev

        predictions = torch.stack(predictions_list, dim=1)  # [batch_size, seq_len]
        return predictions, h_prev

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
