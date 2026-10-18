"""Learner Mastery Prior Initializers and Synthetic Questionnaire Proxy Interfaces.

Provides modular interfaces for initializing skill mastery beliefs:
- `fixed_default`: Stable baseline prior (0.45 across all skills).
- `provided_skill_priors`: Accepts bounded per-skill prior estimates and uncertainty metadata,
  with explicit fallback to default for unobserved/missing skills (never interpreted as low ability).
- `generate_synthetic_parent_proxy_reports`: Synthetic evaluation utility for testing sensitivity
  to noisy and missing prior reports without exposing true latent parameters to controllers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class SkillPriorEstimate:
    """Internal data contract for skill-specific prior belief initialization."""
    prior_mean: float
    uncertainty: Optional[float] = None
    source: str = "fixed_default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.prior_mean = float(np.clip(self.prior_mean, 0.05, 0.95))
        if self.uncertainty is not None:
            self.uncertainty = max(0.0, float(self.uncertainty))


class MasteryPriorInitializer:
    """Configurable initializer for learner belief state vectors."""

    def __init__(
        self,
        mode: str = "fixed_default",
        provided_priors: Optional[Dict[str, Union[float, SkillPriorEstimate]]] = None,
        default_prior: float = 0.45,
        mastery_min: float = 0.05,
        mastery_max: float = 0.95,
    ):
        if mode not in ("fixed_default", "provided_skill_priors"):
            raise ValueError(f"Unknown prior initialization mode: '{mode}'. Must be 'fixed_default' or 'provided_skill_priors'.")
        self.mode = mode
        self.provided_priors = provided_priors or {}
        self.default_prior = float(np.clip(default_prior, mastery_min, mastery_max))
        self.mastery_min = mastery_min
        self.mastery_max = mastery_max

    @classmethod
    def fixed_default(cls, default_prior: float = 0.45) -> MasteryPriorInitializer:
        return cls(mode="fixed_default", default_prior=default_prior)

    @classmethod
    def from_provided_priors(
        cls,
        provided_priors: Dict[str, Union[float, SkillPriorEstimate]],
        fallback_default: float = 0.45,
    ) -> MasteryPriorInitializer:
        return cls(
            mode="provided_skill_priors",
            provided_priors=provided_priors,
            default_prior=fallback_default,
        )

    def initialize_mastery(self, skill_ids: List[str]) -> Dict[str, float]:
        """Convenience method returning only the initial belief dictionary."""
        beliefs, _ = self.initialize(skill_ids)
        return beliefs

    def initialize(self, skill_ids: List[str]) -> Tuple[Dict[str, float], Dict[str, SkillPriorEstimate]]:
        """Initializes mastery beliefs for specified skill IDs.

        Returns:
            Tuple of:
            - Dict[skill_id, float]: Bounded initial belief point values in [mastery_min, mastery_max].
            - Dict[skill_id, SkillPriorEstimate]: Rich metadata including source and uncertainty.
        """
        initial_beliefs: Dict[str, float] = {}
        metadata: Dict[str, SkillPriorEstimate] = {}

        for s in skill_ids:
            if self.mode == "fixed_default":
                initial_beliefs[s] = self.default_prior
                metadata[s] = SkillPriorEstimate(
                    prior_mean=self.default_prior,
                    uncertainty=None,
                    source="fixed_default",
                )
            elif self.mode == "provided_skill_priors":
                if s in self.provided_priors and self.provided_priors[s] is not None:
                    raw_val = self.provided_priors[s]
                    if isinstance(raw_val, SkillPriorEstimate):
                        m_val = float(np.clip(raw_val.prior_mean, self.mastery_min, self.mastery_max))
                        initial_beliefs[s] = m_val
                        metadata[s] = SkillPriorEstimate(
                            prior_mean=m_val,
                            uncertainty=raw_val.uncertainty,
                            source=raw_val.source,
                            metadata=dict(raw_val.metadata),
                        )
                    else:
                        m_val = float(np.clip(float(raw_val), self.mastery_min, self.mastery_max))
                        initial_beliefs[s] = m_val
                        metadata[s] = SkillPriorEstimate(
                            prior_mean=m_val,
                            uncertainty=None,
                            source="provided_value",
                        )
                else:
                    # Missing or unobserved skill: Fall back to default prior; DO NOT treat as low ability!
                    initial_beliefs[s] = self.default_prior
                    metadata[s] = SkillPriorEstimate(
                        prior_mean=self.default_prior,
                        uncertainty=None,
                        source="default_fallback_missing",
                    )

        return initial_beliefs, metadata


def generate_synthetic_parent_proxy_reports(
    true_initial_mastery: Optional[Dict[str, float]] = None,
    skill_ids: Optional[List[str]] = None,
    noise_std: float = 0.15,
    missingness_rate: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
    seed: Optional[int] = None,
    uninformative: bool = False,
    true_skill_masteries: Optional[Dict[str, float]] = None,
) -> Dict[str, SkillPriorEstimate]:
    """Generates synthetic questionnaire proxy reports for sensitivity experiments.

    DISCLAIMER: This function is strictly a computational proxy measurement model for sensitivity
    testing under synthetic noise and missingness. It does NOT represent real parent questionnaires,
    clinical assessments, or developmental validations.

    Args:
        true_initial_mastery: Hidden ground-truth initial mastery (consumed ONLY here in data generation).
        skill_ids: List of skill IDs to evaluate. If None, derived from true_initial_mastery.
        noise_std: Standard deviation of zero-mean Gaussian report error.
        missingness_rate: Probability [0, 1] of caregiver omitting a skill question.
        rng: Seeded random generator.
        seed: Seed to create RandomState if rng is None.
        uninformative: If True, generates random uninformative reports U(0.05, 0.95).
        true_skill_masteries: Alias for true_initial_mastery.

    Returns:
        Dict mapping skill_id to SkillPriorEstimate. Omitted skills are excluded from dict.
    """
    if true_initial_mastery is None and true_skill_masteries is not None:
        true_initial_mastery = true_skill_masteries
    if true_initial_mastery is None:
        true_initial_mastery = {}

    if skill_ids is None:
        skill_ids = list(true_initial_mastery.keys())

    if rng is None:
        rng = np.random.RandomState(seed if seed is not None else 42)

    proxy_reports: Dict[str, SkillPriorEstimate] = {}

    for s in skill_ids:
        # Check missingness
        if missingness_rate > 0.0 and rng.uniform(0.0, 1.0) < missingness_rate:
            # Skill skipped / omitted by caregiver
            continue

        if uninformative:
            # Completely uninformative report uniformly distributed on [0.05, 0.95]
            val = float(rng.uniform(0.05, 0.95))
            proxy_reports[s] = SkillPriorEstimate(
                prior_mean=val,
                uncertainty=float(np.sqrt(1.0 / 12.0) * (0.95 - 0.05)),
                source="synthetic_proxy_uninformative",
            )
        else:
            true_m = true_initial_mastery.get(s, 0.45)
            # Noisy measurement: true value + zero-mean Gaussian noise
            noise = float(rng.normal(0.0, max(1e-6, noise_std)))
            noisy_val = float(np.clip(true_m + noise, 0.05, 0.95))
            proxy_reports[s] = SkillPriorEstimate(
                prior_mean=noisy_val,
                uncertainty=float(noise_std),
                source="synthetic_parent_proxy",
            )

    return proxy_reports
