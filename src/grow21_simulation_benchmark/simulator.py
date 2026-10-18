"""Virtual Learner Simulator for Grow21.

Simulates heterogeneous learner agents with latent mastery dynamics, forgetting over
session gaps, clean response generation using an item-response formulation, and
sensor/execution corruption based on active modalities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from .outcome_model import (
    compute_composite_effective_mastery,
    compute_game_clean_success_prob,
)
from .reliability import ReliabilityEngine, TrialReliabilityRecord
from .tasks import TaskRegistry


@dataclass
class VirtualLearnerProfile:
    learner_id: int
    initial_mastery: Dict[str, float]
    learning_rates: Dict[str, float]
    forgetting_rates: Dict[str, float]
    tremor_propensity: float
    fatigue_sensitivity: float


@dataclass
class SimulatorEvent:
    learner_id: int
    step_idx: int
    session_idx: int
    time_hours: float
    session_gap_hours: float
    game_id: str
    primary_skill: str
    secondary_skill: Optional[str]
    difficulty: float
    game_params: Dict[str, Any]

    # Observable targets and signals (available to models)
    observed_outcome: int  # y_obs in {0, 1}
    estimated_reliability: float  # rho_hat in [0, 1]
    modality_p_hats: Dict[str, float]  # {modality_id: p_hat}
    modality_available: Dict[str, bool]  # {modality_id: is_avail}

    # Simulator-only diagnostics (NEVER given to models during training/testing)
    p_clean: float
    clean_outcome: int  # y_clean in {0, 1}
    true_reliability: float
    latent_mastery_before: Dict[str, float]
    latent_mastery_after: Dict[str, float]
    corruption_occurred: bool


class VirtualLearnerSimulator:
    """Simulates an individual virtual learner over sequential educational interactions."""

    def __init__(
        self,
        profile: VirtualLearnerProfile,
        task_registry: TaskRegistry,
        reliability_engine: ReliabilityEngine,
        mastery_floor: float = 0.05,
        mastery_ceiling: float = 0.95,
    ):
        self.profile = profile
        self.registry = task_registry
        self.reliability_engine = reliability_engine
        self.mastery_floor = mastery_floor
        self.mastery_ceiling = mastery_ceiling

        # Current latent mastery state per skill
        self.current_mastery: Dict[str, float] = {
            s: float(np.clip(v, self.mastery_floor, self.mastery_ceiling))
            for s, v in profile.initial_mastery.items()
        }
        self.step_counter = 0
        self.current_session_idx = 0
        self.session_trial_counter = 0
        self.current_time_hours = 0.0

    def apply_forgetting(self, time_gap_hours: float):
        """Applies exponential memory decay toward the mastery floor over time gaps."""
        if time_gap_hours <= 0.0:
            return
        for s in self.registry.skill_ids:
            rate = self.profile.forgetting_rates.get(s, 0.02)
            # Asymptotic decay toward base floor
            decayed = self.mastery_floor + (self.current_mastery[s] - self.mastery_floor) * math.exp(
                -rate * time_gap_hours
            )
            self.current_mastery[s] = float(np.clip(decayed, self.mastery_floor, self.mastery_ceiling))

    def step(
        self,
        game_id: str,
        difficulty: float,
        time_gap_hours: float,
        rng: np.random.RandomState,
        scenario_params: Dict[str, Any],
    ) -> SimulatorEvent:
        """Executes a single interaction step for this virtual learner.

        Args:
            game_id: The game selected for this interaction.
            difficulty: Abstract difficulty D in [0, 1].
            time_gap_hours: Time elapsed since the previous event.
            rng: Seeded random generator.
            scenario_params: Scenario-specific noise and transition parameters.
        """
        # 1. Update session tracking and apply temporal forgetting
        if time_gap_hours > 2.0:
            self.current_session_idx += 1
            self.session_trial_counter = 0
        self.session_trial_counter += 1
        self.current_time_hours += time_gap_hours

        self.apply_forgetting(time_gap_hours)

        mastery_snapshot_before = dict(self.current_mastery)

        # 2. Retrieve game and skill definitions
        game = self.registry.games[game_id]
        prim_skill = game.primary_skill
        sec_skill = game.secondary_skill

        # Composite effective mastery for this task
        prim_m = self.current_mastery[prim_skill]
        sec_m = self.current_mastery.get(sec_skill) if sec_skill else None
        effective_m = compute_composite_effective_mastery(prim_m, sec_m)

        # 3. Compute clean success probability via canonical outcome model
        p_clean = compute_game_clean_success_prob(
            game=game,
            effective_mastery=effective_m,
            difficulty=difficulty,
            session_trial_counter=self.session_trial_counter,
            fatigue_sensitivity=self.profile.fatigue_sensitivity,
        )

        # Sample clean latent outcome
        y_clean = int(rng.uniform(0.0, 1.0) < p_clean)

        # 4. Modality corruption and noisy reliability estimation
        rel_rec: TrialReliabilityRecord = self.reliability_engine.sample_trial_uncertainties(
            game_id=game_id,
            rng=rng,
            corruption_multiplier=float(scenario_params.get("corruption_multiplier", 1.0)),
            modality_dropout_rate=float(scenario_params.get("modality_dropout_rate", 0.05)),
            uncertainty_bias=float(scenario_params.get("uncertainty_bias", 0.0)),
            uncertainty_noise_scale=float(scenario_params.get("uncertainty_noise_scale", 0.04)),
            learner_tremor_propensity=self.profile.tremor_propensity,
        )

        # Determine if physical corruption occurred on any active sensor/modality
        physical_corruption = any(
            ms.is_corrupted for ms in rel_rec.modality_states.values() if ms.is_active
        )

        # Corrupt the outcome to form the observed interaction label y_obs
        if physical_corruption:
            if y_clean == 1:
                # Motor slip or audio cutoff turns success into observed failure
                slip_prob = 0.85
                y_obs = 0 if rng.uniform(0.0, 1.0) < slip_prob else 1
            else:
                # Accidental contact turns failure into spurious observed hit
                accidental_hit_prob = 0.04
                y_obs = 1 if rng.uniform(0.0, 1.0) < accidental_hit_prob else 0
        else:
            y_obs = y_clean

        # 5. Sequential latent mastery evolution (learning from practice)
        learn_rate = self.profile.learning_rates.get(prim_skill, 0.08)
        # Diminishing returns as mastery approaches ceiling; success reinforces slightly more than failure
        feedback_factor = 0.70 * y_clean + 0.30 * (1 - y_clean)
        # Practicing on appropriate/challenging difficulty prompts stronger learning
        difficulty_engagement = 0.60 + 0.40 * float(np.clip(difficulty, 0.0, 1.0))

        delta_m = (
            learn_rate
            * (self.mastery_ceiling - self.current_mastery[prim_skill])
            * feedback_factor
            * difficulty_engagement
        )
        # Add stochastic practice fluctuation
        noise_m = float(rng.normal(0.0, 0.005))
        updated_prim_m = self.current_mastery[prim_skill] + delta_m + noise_m
        self.current_mastery[prim_skill] = float(
            np.clip(updated_prim_m, self.mastery_floor, self.mastery_ceiling)
        )

        # Secondary skill gains partial transfer practice
        if sec_skill and sec_skill in self.current_mastery:
            sec_learn_rate = self.profile.learning_rates.get(sec_skill, 0.06)
            sec_delta = 0.30 * sec_learn_rate * (self.mastery_ceiling - self.current_mastery[sec_skill])
            self.current_mastery[sec_skill] = float(
                np.clip(self.current_mastery[sec_skill] + sec_delta, self.mastery_floor, self.mastery_ceiling)
            )

        mastery_snapshot_after = dict(self.current_mastery)

        # Map abstract difficulty to concrete game parameters
        game_params = self.registry.map_abstract_difficulty_to_params(game_id, difficulty)

        modality_p_hats = {
            mid: state.estimated_uncertainty
            for mid, state in rel_rec.modality_states.items()
            if state.is_active
        }
        modality_avail = {
            mid: state.is_available
            for mid, state in rel_rec.modality_states.items()
            if state.is_active
        }

        event = SimulatorEvent(
            learner_id=self.profile.learner_id,
            step_idx=self.step_counter,
            session_idx=self.current_session_idx,
            time_hours=round(self.current_time_hours, 3),
            session_gap_hours=round(time_gap_hours, 3),
            game_id=game_id,
            primary_skill=prim_skill,
            secondary_skill=sec_skill,
            difficulty=round(float(difficulty), 4),
            game_params=game_params,
            observed_outcome=y_obs,
            estimated_reliability=round(rel_rec.estimated_reliability, 4),
            modality_p_hats=modality_p_hats,
            modality_available=modality_avail,
            p_clean=round(p_clean, 4),
            clean_outcome=y_clean,
            true_reliability=round(rel_rec.true_reliability, 4),
            latent_mastery_before=mastery_snapshot_before,
            latent_mastery_after=mastery_snapshot_after,
            corruption_occurred=physical_corruption,
        )

        self.step_counter += 1
        return event


def generate_learner_profile(
    learner_id: int,
    skill_ids: List[str],
    rng: np.random.RandomState,
    scenario_params: Dict[str, Any],
    mastery_min: float = 0.05,
    mastery_max: float = 0.95,
) -> VirtualLearnerProfile:
    """Samples heterogeneous parameters for a single synthetic learner."""
    m_mean = float(scenario_params.get("initial_mastery_mean", 0.45))
    m_std = float(scenario_params.get("initial_mastery_std", 0.18))
    lr_mean = float(scenario_params.get("learning_rate_mean", 0.08))
    lr_std = float(scenario_params.get("learning_rate_std", 0.03))
    fr_mean = float(scenario_params.get("forgetting_rate_mean", 0.02))
    fr_std = float(scenario_params.get("forgetting_rate_std", 0.01))

    initial_mastery = {}
    learning_rates = {}
    forgetting_rates = {}

    for s in skill_ids:
        # Mastery bounded away from 0 and 1
        m_val = rng.normal(m_mean, m_std)
        initial_mastery[s] = float(np.clip(m_val, mastery_min, mastery_max))

        lr_val = max(0.005, rng.normal(lr_mean, lr_std))
        learning_rates[s] = float(np.clip(lr_val, 0.005, 0.35))

        fr_val = max(0.001, rng.normal(fr_mean, fr_std))
        forgetting_rates[s] = float(np.clip(fr_val, 0.001, 0.15))

    tremor = float(np.clip(rng.exponential(0.08), 0.0, 0.45))
    fatigue = float(np.clip(rng.uniform(0.005, 0.035), 0.005, 0.05))

    return VirtualLearnerProfile(
        learner_id=learner_id,
        initial_mastery=initial_mastery,
        learning_rates=learning_rates,
        forgetting_rates=forgetting_rates,
        tremor_propensity=tremor,
        fatigue_sensitivity=fatigue,
    )
