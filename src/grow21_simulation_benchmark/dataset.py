"""Dataset generation, splitting, and PyTorch sequence collation for Grow21.

Ensures strictly learner-disjoint train/val/test splits, seeded reproducibility,
and aligned pre-response task features.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .reliability import ReliabilityEngine
from .simulator import (
    SimulatorEvent,
    VirtualLearnerProfile,
    VirtualLearnerSimulator,
    generate_learner_profile,
)
from .tasks import TaskRegistry


class SimulationDatasetGenerator:
    """Generates synthetic learner interaction sequences under specified scenarios."""

    def __init__(self, task_registry: TaskRegistry, config: Dict[str, Any]):
        self.registry = task_registry
        self.config = config
        self.reliability_engine = ReliabilityEngine(
            task_registry=task_registry,
            default_reliability_convention=float(
                config.get("simulation", {}).get("default_reliability_convention", 1.0)
            ),
            missing_modality_default_uncertainty=float(
                config.get("simulation", {}).get("missing_modality_default_uncertainty", 0.25)
            ),
        )

    def generate_events_for_learner(
        self,
        profile: VirtualLearnerProfile,
        num_events: int,
        rng: np.random.RandomState,
        scenario_params: Dict[str, Any],
    ) -> List[SimulatorEvent]:
        """Simulates an interaction history for a single learner with an independent curriculum."""
        sim = VirtualLearnerSimulator(
            profile=profile,
            task_registry=self.registry,
            reliability_engine=self.reliability_engine,
            mastery_floor=float(self.config.get("simulation", {}).get("mastery_min", 0.05)),
            mastery_ceiling=float(self.config.get("simulation", {}).get("mastery_max", 0.95)),
        )

        events: List[SimulatorEvent] = []
        games = self.registry.game_ids
        gap_prob = float(scenario_params.get("session_gap_prob", 0.15))
        max_gap = float(scenario_params.get("max_gap_hours", 48.0))

        # Recent games queue to balance curriculum variety and natural repeats
        for step in range(num_events):
            # Select game independently from model: balanced seeded curriculum
            # with 25% chance of repeating previous game and 75% uniform exploration
            if step > 0 and rng.uniform(0.0, 1.0) < 0.25:
                game_id = events[-1].game_id
            else:
                game_id = str(rng.choice(games))

            # Sample abstract difficulty D in [0, 1] using stratified Beta distribution
            difficulty = float(np.clip(rng.beta(2.0, 2.0), 0.05, 0.95))

            # Sample inter-event time gap
            if step == 0:
                time_gap = 0.0
            else:
                is_session_break = rng.uniform(0.0, 1.0) < gap_prob
                if is_session_break:
                    time_gap = float(rng.uniform(3.0, max_gap))
                else:
                    # Intra-session pause in hours (~20-120 seconds)
                    time_gap = float(rng.uniform(0.005, 0.035))

            ev = sim.step(
                game_id=game_id,
                difficulty=difficulty,
                time_gap_hours=time_gap,
                rng=rng,
                scenario_params=scenario_params,
            )
            events.append(ev)

        return events

    def generate_scenario_dataset(
        self,
        scenario_name: str,
        scenario_cfg: Dict[str, Any],
        seed: int,
        num_learners: Optional[int] = None,
        events_per_learner: Optional[int] = None,
    ) -> Tuple[List[List[SimulatorEvent]], List[List[SimulatorEvent]], List[List[SimulatorEvent]]]:
        """Generates learner-disjoint train, validation, and test datasets.

        Returns:
            (train_sequences, val_sequences, test_sequences)
            where each element is a list of SimulatorEvents for one learner.
        """
        rng = np.random.RandomState(seed)
        sim_cfg = self.config.get("simulation", {})
        total_learners = num_learners or int(sim_cfg.get("num_learners", 900))
        n_events = events_per_learner or int(sim_cfg.get("events_per_learner", 70))
        split_ratio = sim_cfg.get("split_ratio", [0.70, 0.15, 0.15])

        n_train = int(total_learners * split_ratio[0])
        n_val = int(total_learners * split_ratio[1])
        n_test = total_learners - n_train - n_val

        is_shift = scenario_cfg.get("evaluation_type") == "distribution_shift"

        if is_shift:
            train_params = scenario_cfg["train_params"]
            test_params = scenario_cfg["test_params"]
        else:
            train_params = scenario_cfg
            test_params = scenario_cfg

        train_seqs: List[List[SimulatorEvent]] = []
        for i in range(n_train):
            prof = generate_learner_profile(
                learner_id=i,
                skill_ids=self.registry.skill_ids,
                rng=rng,
                scenario_params=train_params,
            )
            seq = self.generate_events_for_learner(prof, n_events, rng, train_params)
            train_seqs.append(seq)

        val_seqs: List[List[SimulatorEvent]] = []
        for i in range(n_train, n_train + n_val):
            prof = generate_learner_profile(
                learner_id=i,
                skill_ids=self.registry.skill_ids,
                rng=rng,
                scenario_params=train_params,
            )
            seq = self.generate_events_for_learner(prof, n_events, rng, train_params)
            val_seqs.append(seq)

        test_seqs: List[List[SimulatorEvent]] = []
        for i in range(n_train + n_val, total_learners):
            prof = generate_learner_profile(
                learner_id=i,
                skill_ids=self.registry.skill_ids,
                rng=rng,
                scenario_params=test_params,
            )
            seq = self.generate_events_for_learner(prof, n_events, rng, test_params)
            test_seqs.append(seq)

        return train_seqs, val_seqs, test_seqs


def events_to_dataframe(sequences: List[List[SimulatorEvent]]) -> pd.DataFrame:
    """Flattens a list of event sequences into a tabular pandas DataFrame."""
    rows = []
    for seq in sequences:
        for ev in seq:
            r = {
                "learner_id": ev.learner_id,
                "step_idx": ev.step_idx,
                "session_idx": ev.session_idx,
                "time_hours": ev.time_hours,
                "session_gap_hours": ev.session_gap_hours,
                "game_id": ev.game_id,
                "primary_skill": ev.primary_skill,
                "secondary_skill": ev.secondary_skill or "none",
                "difficulty": ev.difficulty,
                "observed_outcome": ev.observed_outcome,
                "estimated_reliability": ev.estimated_reliability,
                # Hidden diagnostics
                "p_clean": ev.p_clean,
                "clean_outcome": ev.clean_outcome,
                "true_reliability": ev.true_reliability,
                "corruption_occurred": int(ev.corruption_occurred),
            }
            # Modality observable flags
            for mid, phat in ev.modality_p_hats.items():
                r[f"modality_p_hat_{mid}"] = phat
            for mid, is_av in ev.modality_available.items():
                r[f"modality_avail_{mid}"] = int(is_av)
            rows.append(r)
    return pd.DataFrame(rows)


class DKTSequenceDataset(Dataset):
    """PyTorch Dataset for sequence modeling (Vanilla DKT & Reliability-Gated DKT).

    Tensors returned for each learner sequence:
      meta_features: [seq_len, meta_dim] (pre-response task features strictly before response t)
      outcomes: [seq_len] (observed outcome y_t in {0, 1})
      reliabilities: [seq_len] (estimated reliability rho_hat_t in [0, 1])
      clean_outcomes: [seq_len] (oracle diagnostic only)
      true_reliabilities: [seq_len] (oracle diagnostic only)
      mask: [seq_len] (1 for valid steps, 0 for padded steps)
    """

    def __init__(
        self,
        sequences: List[List[SimulatorEvent]],
        task_registry: TaskRegistry,
        max_seq_len: Optional[int] = None,
    ):
        self.sequences = sequences
        self.registry = task_registry
        self.num_games = len(task_registry.game_ids)
        self.num_skills = len(task_registry.skill_ids)
        self.num_modalities = len(task_registry.modality_ids)

        self.max_len = max_seq_len or max(len(seq) for seq in sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        seq_len = len(seq)

        # Meta features dimensionality:
        # one-hot game (num_games) + one-hot primary skill (num_skills) +
        # difficulty (1) + session_gap_hours (1) + estimated_reliability (1) +
        # modality p_hats (num_modalities) + modality available flags (num_modalities)
        meta_dim = (
            self.num_games + self.num_skills + 3 + 2 * self.num_modalities
        )

        meta_tensor = torch.zeros((self.max_len, meta_dim), dtype=torch.float32)
        outcome_tensor = torch.zeros(self.max_len, dtype=torch.float32)
        rel_tensor = torch.zeros(self.max_len, dtype=torch.float32)
        clean_outcome_tensor = torch.zeros(self.max_len, dtype=torch.float32)
        true_rel_tensor = torch.zeros(self.max_len, dtype=torch.float32)
        mask_tensor = torch.zeros(self.max_len, dtype=torch.float32)

        for t, ev in enumerate(seq[: self.max_len]):
            # Pre-response features:
            g_idx = self.registry.game_to_idx[ev.game_id]
            s_idx = self.registry.skill_to_idx[ev.primary_skill]

            meta_vec = np.zeros(meta_dim, dtype=np.float32)
            meta_vec[g_idx] = 1.0
            meta_vec[self.num_games + s_idx] = 1.0

            cursor = self.num_games + self.num_skills
            meta_vec[cursor] = ev.difficulty
            meta_vec[cursor + 1] = min(1.0, ev.session_gap_hours / 48.0)  # normalized gap
            meta_vec[cursor + 2] = ev.estimated_reliability

            cursor += 3
            for m_i, mid in enumerate(self.registry.modality_ids):
                meta_vec[cursor + m_i] = ev.modality_p_hats.get(mid, 0.0)
                meta_vec[cursor + self.num_modalities + m_i] = float(
                    ev.modality_available.get(mid, False)
                )

            meta_tensor[t] = torch.from_numpy(meta_vec)
            outcome_tensor[t] = float(ev.observed_outcome)
            rel_tensor[t] = float(ev.estimated_reliability)

            # Oracle diagnostics
            clean_outcome_tensor[t] = float(ev.clean_outcome)
            true_rel_tensor[t] = float(ev.true_reliability)
            mask_tensor[t] = 1.0

        return {
            "meta_features": meta_tensor,
            "outcomes": outcome_tensor,
            "reliabilities": rel_tensor,
            "clean_outcomes": clean_outcome_tensor,
            "true_reliabilities": true_rel_tensor,
            "mask": mask_tensor,
            "seq_len": torch.tensor(min(seq_len, self.max_len), dtype=torch.long),
        }
