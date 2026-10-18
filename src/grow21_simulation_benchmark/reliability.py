"""Reliability and uncertainty estimation module for Grow21.

Implements the Grow21 product rule for combining active uncertainty sources,
handles sensor missingness without imputing zero uncertainty, and tracks
both true physical corruption probabilities (for diagnostics) and noisy
estimated uncertainties (available to models).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .tasks import TaskRegistry


@dataclass
class ModalityTrialState:
    modality_id: str
    is_active: bool
    is_available: bool
    true_corruption_prob: float  # Hidden simulator-only diagnostic
    estimated_uncertainty: float  # p_hat available to Grow21
    is_corrupted: bool  # Whether physical corruption actually occurred on this trial


@dataclass
class TrialReliabilityRecord:
    active_modalities: List[str]
    modality_states: Dict[str, ModalityTrialState]
    true_reliability: float  # Diagnostic only
    estimated_reliability: float  # rho_hat available to model (in [0, 1])
    any_modality_missing: bool
    total_physical_corruption_risk: float


class ReliabilityEngine:
    """Manages observation noise, sensor availability, and reliability estimation."""

    def __init__(
        self,
        task_registry: TaskRegistry,
        default_reliability_convention: float = 1.0,
        missing_modality_default_uncertainty: float = 0.25,
    ):
        self.registry = task_registry
        self.default_convention = default_reliability_convention
        self.missing_default_uncertainty = missing_modality_default_uncertainty

    def sample_trial_uncertainties(
        self,
        game_id: str,
        rng: np.random.RandomState,
        corruption_multiplier: float = 1.0,
        modality_dropout_rate: float = 0.05,
        uncertainty_bias: float = 0.0,
        uncertainty_noise_scale: float = 0.04,
        learner_tremor_propensity: float = 0.0,
    ) -> TrialReliabilityRecord:
        """Samples true corruption and noisy estimated uncertainties for an event.

        Args:
            game_id: Identifier for the current game.
            rng: Seeded numpy RandomState.
            corruption_multiplier: Scenario multiplier on base physical corruption.
            modality_dropout_rate: Probability an active sensor is missing.
            uncertainty_bias: Calibration error added to p_hat.
            uncertainty_noise_scale: Standard deviation of observation noise on p_hat.
            learner_tremor_propensity: Individual learner motor noise factor.
        """
        game = self.registry.games[game_id]
        active_mods = game.active_modalities

        modality_states: Dict[str, ModalityTrialState] = {}
        true_survival_factors: List[float] = []
        est_survival_factors: List[float] = []
        any_missing = False

        for mid in self.registry.modality_ids:
            is_active = mid in active_mods
            m_def = self.registry.modalities[mid]

            if not is_active:
                modality_states[mid] = ModalityTrialState(
                    modality_id=mid,
                    is_active=False,
                    is_available=False,
                    true_corruption_prob=0.0,
                    estimated_uncertainty=0.0,
                    is_corrupted=False,
                )
                continue

            # Active modality: determine sensor availability (missingness)
            dropout_prob = min(0.95, max(0.0, m_def.sensor_dropout_rate + modality_dropout_rate))
            is_avail = bool(rng.uniform(0.0, 1.0) >= dropout_prob)
            if not is_avail:
                any_missing = True

            # Calculate true corruption probability
            base_p = m_def.base_corruption_rate * corruption_multiplier
            if mid == "motor":
                # Learner tremor propensity modulates motor corruption
                base_p = base_p * (1.0 + learner_tremor_propensity * 2.0)
            elif mid == "vocalization":
                # Echo Safari accepts vocalization onset; environmental noise can mask
                base_p = min(0.65, base_p * 1.1)

            p_true = float(np.clip(base_p, 0.01, 0.85))
            true_corrupted = bool(rng.uniform(0.0, 1.0) < p_true)
            true_survival_factors.append(1.0 - p_true)

            # Compute estimated uncertainty p_hat available to Grow21
            if is_avail:
                noise = rng.normal(loc=uncertainty_bias, scale=uncertainty_noise_scale)
                p_hat = float(np.clip(p_true + noise, 0.01, 0.95))
            else:
                # Modality is missing: do NOT impute as 0.0 (perfect confidence).
                # Fall back to calibrated prior uncertainty with a conservative penalty.
                p_hat = float(np.clip(self.missing_default_uncertainty, 0.05, 0.95))

            est_survival_factors.append(1.0 - p_hat)

            modality_states[mid] = ModalityTrialState(
                modality_id=mid,
                is_active=True,
                is_available=is_avail,
                true_corruption_prob=p_true,
                estimated_uncertainty=p_hat,
                is_corrupted=true_corrupted,
            )

        if not active_mods:
            # If no uncertainty source applies, use configured convention
            rho_true = self.default_convention
            rho_hat = self.default_convention
            total_risk = 0.0
        else:
            rho_true = float(np.clip(np.prod(true_survival_factors), 0.0, 1.0))
            rho_hat = float(np.clip(np.prod(est_survival_factors), 0.0, 1.0))
            total_risk = 1.0 - rho_true

        return TrialReliabilityRecord(
            active_modalities=active_mods,
            modality_states=modality_states,
            true_reliability=rho_true,
            estimated_reliability=rho_hat,
            any_modality_missing=any_missing,
            total_physical_corruption_risk=total_risk,
        )
