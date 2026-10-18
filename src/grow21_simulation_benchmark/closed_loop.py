"""Closed-loop adaptive simulation experiment for Grow21.

Compares controller policies (Fixed, Streak Heuristic, Legacy Surrogate, and Model-Aligned Target Adaptive)
acting on virtual learners, measuring latent mastery progression, difficulty oscillations,
bound saturation, target reachability, and enforcing strict frustration and session break-limit constraints.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .outcome_model import (
    compute_composite_effective_mastery,
    compute_game_clean_success_prob,
    solve_target_difficulty,
)
from .reliability import ReliabilityEngine
from .simulator import (
    SimulatorEvent,
    VirtualLearnerProfile,
    VirtualLearnerSimulator,
    generate_learner_profile,
)
from .initializers import MasteryPriorInitializer, SkillPriorEstimate
from .tasks import TaskRegistry


@dataclass
class ClosedLoopRunResult:
    controller_name: str
    scenario_name: str
    learner_id: int
    initial_mastery_mean: float
    final_mastery_mean: float
    mastery_gain_all_skills: float
    mastery_gain_primary_skills: float
    mastery_gain_secondary_skills: float
    per_skill_gains: Dict[str, float]
    total_trials: int
    observed_success_rate: float
    clean_success_rate: float
    immediate_direction_reversals: int  # Immediate consecutive trial reversals ((D_t - D_{t-1})(D_{t-1} - D_{t-2}) < 0)
    filtered_direction_reversals: int  # Sign reversals of consecutive non-zero difficulty changes
    direction_reversals: int  # Legacy field matching filtered_direction_reversals for backward compatibility
    total_difficulty_variation: float  # Sum of |D_t - D_{t-1}|
    mean_difficulty: float
    proportion_at_min_bound: float  # Fraction of trials where D == D_min
    proportion_at_max_bound: float  # Fraction of trials where D == D_max
    target_reachable_proportion: Optional[float]  # Fraction of trials where p* was feasible (None for fixed/streak)
    frustration_violations: int  # Must be 0 (never >3 failures)
    break_limit_violations: int  # Must be 0 (never >15 trials without rest)
    safety_interventions: int = 0  # Number of safety cooldown events triggered
    mean_predicted_clean_prob: Optional[float] = None  # Pre-outcome model prediction mean
    mean_target_error: Optional[float] = None  # Absolute deviation |p_clean_hat - target_p|
    mean_prediction_gap: Optional[float] = None  # (p_clean_hat - p_clean_simulator) mean
    brier_score_clean: Optional[float] = None  # Brier score against clean latent outcome


class ControllerPolicy:
    """Base interface for task and difficulty selection policies."""

    def __init__(
        self,
        task_registry: TaskRegistry,
        max_consecutive_failures: int = 3,
        max_session_trials: int = 15,
        cooldown_gap_hours: float = 2.0,
    ):
        self.registry = task_registry
        self.max_consecutive_failures = max_consecutive_failures
        self.max_session_trials = max_session_trials
        self.cooldown_gap_hours = cooldown_gap_hours
        self.consecutive_failures = 0
        self.session_trial_count = 0
        self.frustration_violations = 0
        self.break_limit_violations = 0
        self.last_diagnostic: Dict[str, Any] = {}

    def select_next_task(self, step_idx: int) -> Tuple[str, float, float]:
        """Returns (game_id, difficulty, time_gap_hours)."""
        raise NotImplementedError

    def observe_outcome(
        self,
        game_id: str,
        outcome: int,
        difficulty: float,
        estimated_reliability: Optional[float] = None,
    ):
        """Updates controller state after trial response."""
        self.session_trial_count += 1
        if outcome == 0:
            self.consecutive_failures += 1
            if self.consecutive_failures > self.max_consecutive_failures:
                self.frustration_violations += 1
        else:
            self.consecutive_failures = 0


class FixedDifficultyController(ControllerPolicy):
    """Control condition: maintains constant difficulty (e.g. D=0.50) with cyclical game curriculum."""

    def __init__(
        self,
        task_registry: TaskRegistry,
        fixed_difficulty: float = 0.50,
        max_consecutive_failures: int = 3,
        max_session_trials: int = 15,
        cooldown_gap_hours: float = 2.0,
    ):
        super().__init__(
            task_registry,
            max_consecutive_failures=max_consecutive_failures,
            max_session_trials=max_session_trials,
            cooldown_gap_hours=cooldown_gap_hours,
        )
        self.fixed_difficulty = fixed_difficulty
        self.games = task_registry.game_ids

    def select_next_task(self, step_idx: int) -> Tuple[str, float, float]:
        game_id = self.games[step_idx % len(self.games)]

        # Check session break constraint: breaks after trials 15, 30, 45
        if self.session_trial_count >= self.max_session_trials:
            time_gap = 12.0  # Mandatory inter-session rest
            self.session_trial_count = 0
        else:
            time_gap = 0.02

        # Check frustration safety rule
        safety_triggered = False
        if self.consecutive_failures >= self.max_consecutive_failures:
            diff = max(0.10, self.fixed_difficulty - 0.25)
            self.consecutive_failures = 0
            time_gap = max(time_gap, self.cooldown_gap_hours)
            safety_triggered = True
        else:
            diff = self.fixed_difficulty

        diff_rounded = round(diff, 3)
        self.last_diagnostic = {
            "reachability_applicable": False,
            "target_reachable": None,
            "reachability_status": "N/A",
            "p_high": None,
            "p_low": None,
            "target_difficulty_solved": None,
            "applied_difficulty": diff_rounded,
            "deadband_held": None,
            "step_limited": None,
            "at_min": False,
            "at_max": False,
            "safety_intervention": safety_triggered,
            "predicted_clean_prob": None,
            "target_prob": None,
            "target_error": None,
            "determination_rule": "safety_cooldown" if safety_triggered else "fixed_rule",
        }
        return game_id, diff_rounded, time_gap


class StreakAdaptiveController(ControllerPolicy):
    """Heuristic adaptive controller adjusting difficulty based on success/failure streaks."""

    def __init__(
        self,
        task_registry: TaskRegistry,
        initial_difficulty: float = 0.45,
        up_step: float = 0.06,
        down_step: float = 0.08,
        min_difficulty: float = 0.08,
        max_difficulty: float = 0.95,
        max_consecutive_failures: int = 3,
        max_session_trials: int = 15,
        cooldown_gap_hours: float = 2.0,
    ):
        super().__init__(
            task_registry,
            max_consecutive_failures=max_consecutive_failures,
            max_session_trials=max_session_trials,
            cooldown_gap_hours=cooldown_gap_hours,
        )
        self.current_difficulty = initial_difficulty
        self.up_step = up_step
        self.down_step = down_step
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.consecutive_successes = 0
        self.games = task_registry.game_ids

    def select_next_task(self, step_idx: int) -> Tuple[str, float, float]:
        game_id = self.games[step_idx % len(self.games)]

        if self.session_trial_count >= self.max_session_trials:
            time_gap = 12.0
            self.session_trial_count = 0
        else:
            time_gap = 0.02

        # Frustration safety intervention
        safety_triggered = False
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.current_difficulty = max(self.min_difficulty, self.current_difficulty - 0.25)
            self.consecutive_failures = 0
            time_gap = max(time_gap, self.cooldown_gap_hours)
            safety_triggered = True

        diff = round(self.current_difficulty, 3)
        self.last_diagnostic = {
            "reachability_applicable": False,
            "target_reachable": None,
            "reachability_status": "N/A",
            "p_high": None,
            "p_low": None,
            "target_difficulty_solved": None,
            "applied_difficulty": diff,
            "deadband_held": None,
            "step_limited": None,
            "at_min": diff <= self.min_difficulty + 1e-4,
            "at_max": diff >= self.max_difficulty - 1e-4,
            "safety_intervention": safety_triggered,
            "predicted_clean_prob": None,
            "target_prob": None,
            "target_error": None,
            "determination_rule": "safety_cooldown" if safety_triggered else "streak_rule",
        }
        return game_id, diff, time_gap

    def observe_outcome(
        self,
        game_id: str,
        outcome: int,
        difficulty: float,
        estimated_reliability: Optional[float] = None,
    ):
        super().observe_outcome(game_id, outcome, difficulty, estimated_reliability=estimated_reliability)
        if outcome == 1:
            self.consecutive_successes += 1
            if self.consecutive_successes >= 2:
                self.current_difficulty = min(self.max_difficulty, self.current_difficulty + self.up_step)
                self.consecutive_successes = 0
        else:
            self.consecutive_successes = 0
            self.current_difficulty = max(self.min_difficulty, self.current_difficulty - self.down_step)


class LegacySurrogateTargetAdaptiveController(ControllerPolicy):
    """Phase 2 Legacy Target-Adaptive Controller using the power surrogate model.

    Predicted success surrogate:
      p_hat = sigmoid(logit(theta_hat) - gamma * D**nu_D)
    with active Phase 2 defaults gamma=2.4, nu_D=1.0.

    Retained strictly for Phase 2 historical replication and comparison.
    """

    def __init__(
        self,
        task_registry: TaskRegistry,
        target_success_prob: float = 0.75,
        initial_difficulty: float = 0.45,
        gamma: float = 2.4,
        nu_D: float = 1.0,
        deadband: float = 0.04,
        max_step_change: float = 0.08,
        min_difficulty: float = 0.08,
        max_difficulty: float = 0.95,
        max_consecutive_failures: int = 3,
        max_session_trials: int = 15,
        cooldown_gap_hours: float = 2.0,
    ):
        super().__init__(
            task_registry,
            max_consecutive_failures=max_consecutive_failures,
            max_session_trials=max_session_trials,
            cooldown_gap_hours=cooldown_gap_hours,
        )
        self.target_prob = target_success_prob
        self.current_difficulty = initial_difficulty
        self.gamma = gamma
        self.nu_D = nu_D
        self.deadband = deadband
        self.max_step_change = max_step_change
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.games = task_registry.game_ids

        # Estimated belief mastery theta_hat per skill
        self.estimated_mastery: Dict[str, float] = {
            s: 0.45 for s in task_registry.skill_ids
        }

    def predict_success_prob(self, skill_id: str, difficulty: float) -> float:
        """Computes p_hat = sigmoid(logit(theta_k) - gamma * D**nu_D)."""
        theta = float(np.clip(self.estimated_mastery.get(skill_id, 0.45), 0.05, 0.95))
        logit_theta = math.log(theta / (1.0 - theta))
        d_val = float(np.clip(difficulty, 0.0, 1.0))
        power_term = self.gamma * (d_val ** self.nu_D)
        logit_diff = logit_theta - power_term
        return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit_diff))))

    def check_reachability(self, skill_id: str) -> Tuple[bool, float, float]:
        """Checks if target_prob is reachable: p_hat(D_max) <= target_prob <= p_hat(D_min)."""
        p_low = self.predict_success_prob(skill_id, self.max_difficulty)
        p_high = self.predict_success_prob(skill_id, self.min_difficulty)
        is_reachable = bool(p_low <= self.target_prob <= p_high)
        return is_reachable, p_low, p_high

    def select_next_task(self, step_idx: int) -> Tuple[str, float, float]:
        game_id = self.games[step_idx % len(self.games)]
        prim_skill = self.registry.games[game_id].primary_skill

        # Check session break constraint
        if self.session_trial_count >= self.max_session_trials:
            time_gap = 12.0
            self.session_trial_count = 0
        else:
            time_gap = 0.02

        # Frustration safety intervention
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.current_difficulty = max(self.min_difficulty, self.current_difficulty - 0.25)
            self.consecutive_failures = 0
            time_gap = max(time_gap, self.cooldown_gap_hours)
            diff = round(self.current_difficulty, 3)
            is_reachable, p_low, p_high = self.check_reachability(prim_skill)
            if self.target_prob > p_high:
                status = "unreachable_high"
            elif self.target_prob < p_low:
                status = "unreachable_low"
            else:
                status = "reachable"
            pred_p = self.predict_success_prob(prim_skill, diff)
            self.last_diagnostic = {
                "reachability_applicable": True,
                "target_reachable": is_reachable,
                "reachability_status": status,
                "p_high": round(p_high, 4),
                "p_low": round(p_low, 4),
                "target_difficulty_solved": None,
                "applied_difficulty": diff,
                "deadband_held": 0,
                "step_limited": 0,
                "at_min": diff <= self.min_difficulty + 1e-4,
                "at_max": diff >= self.max_difficulty - 1e-4,
                "safety_intervention": True,
                "predicted_clean_prob": round(pred_p, 4),
                "target_prob": self.target_prob,
                "target_error": round(abs(pred_p - self.target_prob), 4),
                "determination_rule": "safety_cooldown",
            }
            return game_id, diff, time_gap

        # Check reachability across difficulty bounds [D_min, D_max]
        is_reachable, p_low, p_high = self.check_reachability(prim_skill)

        if self.target_prob > p_high:
            target_D = self.min_difficulty
            status = "unreachable_high"
            rule = "unreachable_high_bound"
        elif self.target_prob < p_low:
            target_D = self.max_difficulty
            status = "unreachable_low"
            rule = "unreachable_low_bound"
        else:
            theta = float(np.clip(self.estimated_mastery.get(prim_skill, 0.45), 0.05, 0.95))
            logit_theta = math.log(theta / (1.0 - theta))
            logit_target = math.log(self.target_prob / (1.0 - self.target_prob))
            req_power = (logit_theta - logit_target) / self.gamma
            target_D = float(np.clip(req_power ** (1.0 / self.nu_D), self.min_difficulty, self.max_difficulty))
            status = "reachable"
            rule = "surrogate_analytical_inversion"

        solved_D = target_D

        # Check deadband: hold steady if current difficulty is within deadband of target
        current_pred_p = self.predict_success_prob(prim_skill, self.current_difficulty)
        deadband_held = False
        if abs(current_pred_p - self.target_prob) <= self.deadband:
            target_D = self.current_difficulty
            deadband_held = True
            rule = "deadband"

        # Apply maximum step change limit (smooth control)
        step_delta = float(np.clip(target_D - self.current_difficulty, -self.max_step_change, self.max_step_change))
        step_limited = abs(step_delta - (target_D - self.current_difficulty)) > 1e-5
        if step_limited and not deadband_held:
            rule = "step_limit"

        self.current_difficulty = float(np.clip(
            self.current_difficulty + step_delta, self.min_difficulty, self.max_difficulty
        ))

        diff = round(self.current_difficulty, 3)
        final_pred_p = self.predict_success_prob(prim_skill, diff)
        self.last_diagnostic = {
            "reachability_applicable": True,
            "target_reachable": is_reachable,
            "reachability_status": status,
            "p_high": round(p_high, 4),
            "p_low": round(p_low, 4),
            "target_difficulty_solved": round(solved_D, 4),
            "applied_difficulty": diff,
            "deadband_held": int(deadband_held),
            "step_limited": int(step_limited),
            "at_min": diff <= self.min_difficulty + 1e-4,
            "at_max": diff >= self.max_difficulty - 1e-4,
            "safety_intervention": False,
            "predicted_clean_prob": round(final_pred_p, 4),
            "target_prob": self.target_prob,
            "target_error": round(abs(final_pred_p - self.target_prob), 4),
            "determination_rule": rule,
        }
        return game_id, diff, time_gap

    def observe_outcome(
        self,
        game_id: str,
        outcome: int,
        difficulty: float,
        estimated_reliability: Optional[float] = None,
    ):
        super().observe_outcome(game_id, outcome, difficulty, estimated_reliability=estimated_reliability)
        prim_skill = self.registry.games[game_id].primary_skill

        # Online belief update on skill mastery (Bayesian smoothed knowledge update)
        curr_m = self.estimated_mastery.get(prim_skill, 0.45)
        if outcome == 1:
            new_m = curr_m + 0.08 * (0.95 - curr_m)
        else:
            new_m = curr_m - 0.08 * (curr_m - 0.05)
        self.estimated_mastery[prim_skill] = float(np.clip(new_m, 0.05, 0.95))


# Maintain Grow21ModelBasedAdaptiveController as legacy alias for backwards test compatibility
Grow21ModelBasedAdaptiveController = LegacySurrogateTargetAdaptiveController


def _project_distribution_onto_grid(
    source_points: np.ndarray,
    source_weights: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Consistently projects arbitrary 1D point masses onto a fixed 1D grid via linear interpolation."""
    new_weights = np.zeros_like(grid, dtype=float)
    g_min, g_max = float(grid[0]), float(grid[-1])
    n_bins = len(grid) - 1
    step = (g_max - g_min) / max(1, n_bins)

    for p, w in zip(source_points, source_weights):
        p_clamped = min(max(float(p), g_min), g_max)
        raw_idx = (p_clamped - g_min) / step
        idx = int(raw_idx)
        if idx >= n_bins:
            new_weights[-1] += float(w)
        else:
            frac = raw_idx - idx
            new_weights[idx] += float(w) * (1.0 - frac)
            new_weights[idx + 1] += float(w) * frac

    total = float(np.sum(new_weights))
    if total > 0.0:
        new_weights /= total
    return new_weights


def _diffuse_1d_grid(
    weights: np.ndarray,
    grid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Applies a continuous Gaussian convolution kernel across fixed 1D grid points to model diffusion / process noise."""
    if sigma <= 1e-5:
        return weights
    diff = grid[:, None] - grid[None, :]
    kernel = np.exp(-(diff ** 2) / (2.0 * (sigma ** 2)))
    kernel /= kernel.sum(axis=0, keepdims=True)
    new_w = kernel @ weights
    s = float(np.sum(new_w))
    return new_w / s if s > 0 else weights


class ModelAlignedTargetAdaptiveController(ControllerPolicy):
    """Phase 3/4/5 Model-Aligned Target-Adaptive Controller.

    Solves for task difficulty targeting clean task-success probability p*
    using the CANONICAL simulator outcome model, accounting for:
    - Composite effective mastery (85% primary + 15% secondary where applicable);
    - Game item parameters (discrimination, difficulty offset/scale, guessing floor, lapse ceiling);
    - Non-linear difficulty curve (0.25*D + 0.75*D**1.35);
    - Projected session trial counter and configured population fatigue sensitivity prior (0.02);
    - Session time gaps and configured population forgetting rate prior (0.02);
    - Bounded numerical bisection solving on [D_min, D_max];
    - Strict decoupling between mathematical reachability and safety/deadband constraints;
    - Zero simulator truth leakage (controller never reads true mastery, true fatigue, or true forgetting).
    """

    def __init__(
        self,
        task_registry: TaskRegistry,
        target_success_prob: float = 0.75,
        initial_difficulty: float = 0.45,
        deadband: float = 0.04,
        max_step_change: float = 0.08,
        min_difficulty: float = 0.08,
        max_difficulty: float = 0.95,
        fatigue_sensitivity_prior: float = 0.02,
        forgetting_rate_prior: float = 0.02,
        learning_rate_prior: float = 0.08,
        secondary_transfer_ratio: float = 0.30,
        belief_update_mode: str = "heuristic_baseline",
        max_consecutive_failures: int = 3,
        max_session_trials: int = 15,
        cooldown_gap_hours: float = 2.0,
        predictor_type: str = "canonical_clean_model",
        prior_initializer: Optional[MasteryPriorInitializer] = None,
    ):
        if predictor_type != "canonical_clean_model":
            raise ValueError(
                "CRITICAL SEMANTIC SAFEGUARD: Cannot route observed-success predictor P(y_obs) "
                "directly into clean-success controller target P(y_clean). Offline benchmark models "
                "predict P(y_obs=1|features), whereas the closed-loop controller targets clean task success "
                "P(y_clean=1). An explicit, validated observation-noise inversion model is required "
                "to convert P(y_obs) to P(y_clean)."
            )
        super().__init__(
            task_registry,
            max_consecutive_failures=max_consecutive_failures,
            max_session_trials=max_session_trials,
            cooldown_gap_hours=cooldown_gap_hours,
        )
        self.target_prob = target_success_prob
        self.current_difficulty = initial_difficulty
        self.deadband = deadband
        self.max_step_change = max_step_change
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.fatigue_sensitivity_prior = fatigue_sensitivity_prior
        self.forgetting_rate_prior = forgetting_rate_prior
        self.learning_rate_prior = learning_rate_prior
        self.secondary_transfer_ratio = secondary_transfer_ratio
        self.belief_update_mode = belief_update_mode
        self.games = task_registry.game_ids

        # Estimated belief mastery theta_hat per skill in [0.05, 0.95]
        if prior_initializer is not None:
            init_m, _ = prior_initializer.initialize(task_registry.skill_ids)
            self.estimated_mastery = init_m
        else:
            self.estimated_mastery = {
                s: 0.45 for s in task_registry.skill_ids
            }
        self._last_predicted_clean_prob: float = 0.75

        if self.belief_update_mode in ("approx_bayesian_filter", "approx_bayesian_filter_transfer", "robust_adaptive_filter"):
            self._bayesian_filter_state: Dict[str, Dict[str, np.ndarray]] = {}
            for s in task_registry.skill_ids:
                grid = np.linspace(0.05, 0.95, 41)
                m0 = self.estimated_mastery.get(s, 0.45)
                weights = np.exp(-((grid - m0) ** 2) / (2.0 * (0.18 ** 2)))
                w_sum = float(np.sum(weights))
                weights = weights / w_sum if w_sum > 0 else np.ones_like(grid) / len(grid)
                self._bayesian_filter_state[s] = {"grid": grid, "weights": weights}

        if self.belief_update_mode == "robust_adaptive_filter":
            self.heuristic_mastery: Dict[str, float] = copy.deepcopy(self.estimated_mastery)
            self.rel_residual_ema: float = 0.0
            self.base_process_noise: float = 0.012
            self.gap_diffusion_rate: float = 0.020
            self.fallback_threshold: float = -0.07

    def get_effective_mastery(self, game_id: str) -> float:
        """Computes estimated effective mastery blending primary and secondary skills."""
        game = self.registry.games[game_id]
        prim_m = self.estimated_mastery.get(game.primary_skill, 0.45)
        sec_m = self.estimated_mastery.get(game.secondary_skill) if game.secondary_skill else None
        return compute_composite_effective_mastery(prim_m, sec_m)

    def predict_clean_success_prob(
        self, game_id: str, difficulty: float, session_trial_counter: int = 1
    ) -> float:
        """Predicts latent clean task success using the canonical model and population priors."""
        game = self.registry.games[game_id]
        eff_m = self.get_effective_mastery(game_id)
        return compute_game_clean_success_prob(
            game=game,
            effective_mastery=eff_m,
            difficulty=difficulty,
            session_trial_counter=session_trial_counter,
            fatigue_sensitivity=self.fatigue_sensitivity_prior,
        )

    def select_next_task(self, step_idx: int) -> Tuple[str, float, float]:
        game_id = self.games[step_idx % len(self.games)]
        game = self.registry.games[game_id]

        # 1. Project session gap and projected trial counter
        if self.session_trial_count >= self.max_session_trials:
            time_gap = 12.0
            self.session_trial_count = 0
            projected_counter = 1
        else:
            time_gap = 0.02
            projected_counter = self.session_trial_count + 1

        # 2. Safety intervention check
        safety_triggered = False
        if self.consecutive_failures >= self.max_consecutive_failures:
            safety_triggered = True
            self.consecutive_failures = 0
            time_gap = max(time_gap, self.cooldown_gap_hours)
            if time_gap > 2.0:
                projected_counter = 1

        # 3. Apply expected forgetting decay to beliefs once per trial based on planned time gap
        if time_gap > 0.0:
            for s in self.registry.skill_ids:
                decayed = 0.05 + (self.estimated_mastery[s] - 0.05) * math.exp(
                    -self.forgetting_rate_prior * time_gap
                )
                self.estimated_mastery[s] = float(np.clip(decayed, 0.05, 0.95))

                if hasattr(self, "heuristic_mastery") and s in self.heuristic_mastery:
                    h_decay = 0.05 + (self.heuristic_mastery[s] - 0.05) * math.exp(
                        -self.forgetting_rate_prior * time_gap
                    )
                    self.heuristic_mastery[s] = float(np.clip(h_decay, 0.05, 0.95))

                if hasattr(self, "_bayesian_filter_state") and s in getattr(self, "_bayesian_filter_state", {}):
                    grid = self._bayesian_filter_state[s]["grid"]
                    weights = self._bayesian_filter_state[s]["weights"]
                    decayed_pts = 0.05 + (grid - 0.05) * math.exp(-self.forgetting_rate_prior * time_gap)
                    new_w = _project_distribution_onto_grid(decayed_pts, weights, grid)
                    if self.belief_update_mode == "robust_adaptive_filter":
                        gap_sigma = min(0.06, getattr(self, "gap_diffusion_rate", 0.020) * math.sqrt(time_gap))
                        new_w = _diffuse_1d_grid(new_w, grid, gap_sigma)
                    self._bayesian_filter_state[s]["weights"] = new_w
                    self.estimated_mastery[s] = float(np.clip(np.sum(new_w * grid), 0.05, 0.95))


        # 4. Compute estimated effective mastery for selected game
        eff_m = self.get_effective_mastery(game_id)

        # 5. Solve for target difficulty using bounded bisection on canonical model
        target_D_solved, reach_status, p_high, p_low = solve_target_difficulty(
            game=game,
            effective_mastery=eff_m,
            target_prob=self.target_prob,
            session_trial_counter=projected_counter,
            fatigue_sensitivity=self.fatigue_sensitivity_prior,
            min_difficulty=self.min_difficulty,
            max_difficulty=self.max_difficulty,
        )
        is_reachable = (reach_status == "reachable")

        if safety_triggered:
            # Safety cooldown overrides difficulty directly; reachability remains separate
            target_D = max(self.min_difficulty, self.current_difficulty - 0.25)
            self.current_difficulty = target_D
            diff = round(self.current_difficulty, 3)
            pred_p = self.predict_clean_success_prob(game_id, diff, projected_counter)
            self._last_predicted_clean_prob = pred_p
            self.last_diagnostic = {
                "reachability_applicable": True,
                "target_reachable": is_reachable,
                "reachability_status": reach_status,
                "p_high": round(p_high, 4),
                "p_low": round(p_low, 4),
                "target_difficulty_solved": round(target_D_solved, 4),
                "applied_difficulty": diff,
                "deadband_held": 0,
                "step_limited": 0,
                "at_min": diff <= self.min_difficulty + 1e-4,
                "at_max": diff >= self.max_difficulty - 1e-4,
                "safety_intervention": True,
                "predicted_clean_prob": round(pred_p, 4),
                "target_prob": self.target_prob,
                "target_error": round(abs(pred_p - self.target_prob), 4),
                "determination_rule": "safety_cooldown",
            }
            return game_id, diff, time_gap

        # 6. Deadband check: hold steady if predicted success at current difficulty is within deadband
        curr_pred_p = self.predict_clean_success_prob(game_id, self.current_difficulty, projected_counter)
        deadband_held = False
        if abs(curr_pred_p - self.target_prob) <= self.deadband:
            target_D = self.current_difficulty
            deadband_held = True
            rule = "deadband"
        else:
            target_D = target_D_solved
            if reach_status == "unreachable_high":
                rule = "unreachable_high_bound"
            elif reach_status == "unreachable_low":
                rule = "unreachable_low_bound"
            else:
                rule = "target_bisection"

        # 7. Apply maximum step-change constraint
        step_delta = float(np.clip(target_D - self.current_difficulty, -self.max_step_change, self.max_step_change))
        step_limited = abs(step_delta - (target_D - self.current_difficulty)) > 1e-5
        if step_limited and not deadband_held:
            rule = "step_limit"

        self.current_difficulty = float(np.clip(
            self.current_difficulty + step_delta, self.min_difficulty, self.max_difficulty
        ))

        diff = round(self.current_difficulty, 3)
        final_pred_p = self.predict_clean_success_prob(game_id, diff, projected_counter)
        self._last_predicted_clean_prob = final_pred_p
        self.last_diagnostic = {
            "reachability_applicable": True,
            "target_reachable": is_reachable,
            "reachability_status": reach_status,
            "p_high": round(p_high, 4),
            "p_low": round(p_low, 4),
            "target_difficulty_solved": round(target_D_solved, 4),
            "applied_difficulty": diff,
            "deadband_held": int(deadband_held),
            "step_limited": int(step_limited),
            "at_min": diff <= self.min_difficulty + 1e-4,
            "at_max": diff >= self.max_difficulty - 1e-4,
            "safety_intervention": False,
            "predicted_clean_prob": round(final_pred_p, 4),
            "target_prob": self.target_prob,
            "target_error": round(abs(final_pred_p - self.target_prob), 4),
            "determination_rule": rule,
        }
        return game_id, diff, time_gap

    def observe_outcome(
        self,
        game_id: str,
        outcome: int,
        difficulty: float,
        estimated_reliability: Optional[float] = None,
    ):
        super().observe_outcome(game_id, outcome, difficulty, estimated_reliability=estimated_reliability)
        game = self.registry.games[game_id]
        prim_skill = game.primary_skill
        sec_skill = game.secondary_skill

        curr_m = self.estimated_mastery.get(prim_skill, 0.45)

        # 1. Primary skill update
        if self.belief_update_mode in ("expected_transition", "expected_transition_transfer"):
            # A. Expected transition baseline (Part B.A)
            rho = 1.0 if estimated_reliability is None else float(np.clip(estimated_reliability, 0.0, 1.0))
            p = float(np.clip(self._last_predicted_clean_prob, 0.01, 0.99))
            if outcome == 1:
                p_obs_given_clean1 = rho + 0.15 * (1.0 - rho)
                p_obs_given_clean0 = 0.04 * (1.0 - rho)
            else:
                p_obs_given_clean1 = 0.85 * (1.0 - rho)
                p_obs_given_clean0 = rho + 0.96 * (1.0 - rho)
            num = p * p_obs_given_clean1
            den = num + (1.0 - p) * p_obs_given_clean0
            q = float(np.clip(num / max(1e-12, den), 0.0, 1.0))

            expected_feedback = 0.30 + 0.40 * q
            expected_engagement = 0.60 + 0.40 * float(np.clip(difficulty, 0.0, 1.0))
            delta_m = self.learning_rate_prior * (0.95 - curr_m) * expected_feedback * expected_engagement
            new_m = float(np.clip(curr_m + delta_m, 0.05, 0.95))
            self.estimated_mastery[prim_skill] = new_m

        elif self.belief_update_mode in ("approx_bayesian_filter", "approx_bayesian_filter_transfer"):
            # B. Approximate Bayesian filter (Part B.B)
            rho = 1.0 if estimated_reliability is None else float(np.clip(estimated_reliability, 0.0, 1.0))
            grid = self._bayesian_filter_state[prim_skill]["grid"]
            w = self._bayesian_filter_state[prim_skill]["weights"]

            if outcome == 1:
                p_obs_given_clean1 = rho + 0.15 * (1.0 - rho)
                p_obs_given_clean0 = 0.04 * (1.0 - rho)
            else:
                p_obs_given_clean1 = 0.85 * (1.0 - rho)
                p_obs_given_clean0 = rho + 0.96 * (1.0 - rho)

            sec_m = self.estimated_mastery.get(sec_skill) if (sec_skill and sec_skill in self.estimated_mastery) else None
            cand_p_clean = np.array([
                compute_game_clean_success_prob(
                    game=game,
                    effective_mastery=compute_composite_effective_mastery(th, sec_m),
                    difficulty=difficulty,
                    session_trial_counter=self.session_trial_count,
                    fatigue_sensitivity=self.fatigue_sensitivity_prior,
                )
                for th in grid
            ])

            obs_lik = cand_p_clean * p_obs_given_clean1 + (1.0 - cand_p_clean) * p_obs_given_clean0
            w_obs = w * obs_lik
            w_obs_sum = float(np.sum(w_obs))
            w_obs = w_obs / w_obs_sum if w_obs_sum > 0 else w

            q_cand = np.clip((cand_p_clean * p_obs_given_clean1) / np.maximum(1e-12, obs_lik), 0.0, 1.0)
            cand_feedback = 0.30 + 0.40 * q_cand
            cand_engagement = 0.60 + 0.40 * float(np.clip(difficulty, 0.0, 1.0))
            cand_delta = self.learning_rate_prior * (0.95 - grid) * cand_feedback * cand_engagement
            cand_next = np.clip(grid + cand_delta, 0.05, 0.95)

            w_next = _project_distribution_onto_grid(cand_next, w_obs, grid)
            self._bayesian_filter_state[prim_skill]["weights"] = w_next
            self.estimated_mastery[prim_skill] = float(np.clip(np.sum(w_next * grid), 0.05, 0.95))

        elif self.belief_update_mode == "robust_adaptive_filter":
            # Robust Adaptive Filter:
            # 1. Update shadow baseline for tracking conservative bounds
            curr_h = self.heuristic_mastery[prim_skill]
            if outcome == 1:
                new_h = curr_h + self.learning_rate_prior * (0.95 - curr_h)
            else:
                new_h = curr_h - self.learning_rate_prior * (curr_h - 0.05)
            self.heuristic_mastery[prim_skill] = float(np.clip(new_h, 0.05, 0.95))

            # 2. Observable reliability-weighted innovation tracking
            rho = 1.0 if estimated_reliability is None else float(np.clip(estimated_reliability, 0.0, 1.0))
            p_clean_pred = float(np.clip(self._last_predicted_clean_prob, 0.01, 0.99))
            pred_obs = p_clean_pred * (rho + 0.15 * (1.0 - rho)) + (1.0 - p_clean_pred) * (0.04 * (1.0 - rho))
            residual = float(outcome - pred_obs)
            self.rel_residual_ema = 0.85 * self.rel_residual_ema + 0.15 * (rho * residual)

            # 3. Bilateral Bayesian belief update
            grid = self._bayesian_filter_state[prim_skill]["grid"]
            w = self._bayesian_filter_state[prim_skill]["weights"]

            if outcome == 1:
                p_obs_given_clean1 = rho + 0.15 * (1.0 - rho)
                p_obs_given_clean0 = 0.04 * (1.0 - rho)
            else:
                p_obs_given_clean1 = 0.85 * (1.0 - rho)
                p_obs_given_clean0 = rho + 0.96 * (1.0 - rho)

            sec_m = self.estimated_mastery.get(sec_skill) if (sec_skill and sec_skill in self.estimated_mastery) else None
            cand_p_clean = np.array([
                compute_game_clean_success_prob(
                    game=game,
                    effective_mastery=compute_composite_effective_mastery(th, sec_m),
                    difficulty=difficulty,
                    session_trial_counter=self.session_trial_count,
                    fatigue_sensitivity=self.fatigue_sensitivity_prior,
                )
                for th in grid
            ])

            obs_lik = cand_p_clean * p_obs_given_clean1 + (1.0 - cand_p_clean) * p_obs_given_clean0
            w_obs = w * obs_lik
            w_obs_sum = float(np.sum(w_obs))
            w_obs = w_obs / w_obs_sum if w_obs_sum > 0 else w

            q_cand = np.clip((cand_p_clean * p_obs_given_clean1) / np.maximum(1e-12, obs_lik), 0.0, 1.0)

            # Bilateral transition target: shifts towards 0.95 on success, 0.05 on failure
            target_m = q_cand * 0.95 + (1.0 - q_cand) * 0.05
            cand_engagement = 0.60 + 0.40 * float(np.clip(difficulty, 0.0, 1.0))
            cand_delta = self.learning_rate_prior * (target_m - grid) * cand_engagement
            cand_next = np.clip(grid + cand_delta, 0.05, 0.95)

            w_next = _project_distribution_onto_grid(cand_next, w_obs, grid)
            w_next = _diffuse_1d_grid(w_next, grid, self.base_process_noise)
            self._bayesian_filter_state[prim_skill]["weights"] = w_next
            new_bf_m = float(np.clip(np.sum(w_next * grid), 0.05, 0.95))

            # 4. Modulated secondary transfer: scaled by trial clean-success expectation q_mean
            q_mean = float(np.sum(w_obs * q_cand))
            if sec_skill and sec_skill in self.estimated_mastery:
                curr_sec_m = self.estimated_mastery[sec_skill]
                sec_target = q_mean * 0.95 + (1.0 - q_mean) * curr_sec_m
                sec_delta = self.secondary_transfer_ratio * self.learning_rate_prior * (sec_target - curr_sec_m)
                new_sec_m = float(np.clip(curr_sec_m + sec_delta, 0.05, 0.95))
                self.estimated_mastery[sec_skill] = new_sec_m

                if hasattr(self, "_bayesian_filter_state") and sec_skill in getattr(self, "_bayesian_filter_state", {}):
                    sec_grid = self._bayesian_filter_state[sec_skill]["grid"]
                    sec_w = self._bayesian_filter_state[sec_skill]["weights"]
                    sec_delta_grid = self.secondary_transfer_ratio * self.learning_rate_prior * (sec_target - sec_grid)
                    sec_next = np.clip(sec_grid + sec_delta_grid, 0.05, 0.95)
                    w_t = _project_distribution_onto_grid(sec_next, sec_w, sec_grid)
                    w_t = _diffuse_1d_grid(w_t, sec_grid, self.base_process_noise)
                    self._bayesian_filter_state[sec_skill]["weights"] = w_t

            # 5. Gated fallback blending: activates when reliable residual indicates persistent overestimation
            if self.rel_residual_ema < self.fallback_threshold and self.heuristic_mastery[prim_skill] < new_bf_m:
                excess = abs(self.rel_residual_ema - self.fallback_threshold)
                w_blend = min(0.65, excess / 0.10)
                blended = (1.0 - w_blend) * new_bf_m + w_blend * self.heuristic_mastery[prim_skill]
                blended = float(np.clip(blended, 0.05, 0.95))
                shift = blended - new_bf_m
                self.estimated_mastery[prim_skill] = blended
                self._bayesian_filter_state[prim_skill]["weights"] = _project_distribution_onto_grid(grid + shift, w_next, grid)

                if sec_skill and sec_skill in self.estimated_mastery:
                    bf_sec = self.estimated_mastery[sec_skill]
                    h_sec = self.heuristic_mastery.get(sec_skill, 0.45)
                    if h_sec < bf_sec:
                        blended_sec = (1.0 - w_blend) * bf_sec + w_blend * h_sec
                        self.estimated_mastery[sec_skill] = float(np.clip(blended_sec, 0.05, 0.95))
                        s_grid = self._bayesian_filter_state[sec_skill]["grid"]
                        s_w = self._bayesian_filter_state[sec_skill]["weights"]
                        self._bayesian_filter_state[sec_skill]["weights"] = _project_distribution_onto_grid(s_grid + (blended_sec - bf_sec), s_w, s_grid)
            else:
                self.estimated_mastery[prim_skill] = new_bf_m

        elif self.belief_update_mode in ("reliability_aware", "combined_transfer_reliability", "combined"):
            # Observable estimated sensor reliability rho_hat in [0, 1]
            rho = 1.0 if estimated_reliability is None else float(np.clip(estimated_reliability, 0.0, 1.0))
            # Model's own pre-outcome predicted clean probability for this trial
            p = float(np.clip(self._last_predicted_clean_prob, 0.01, 0.99))

            # Likelihoods based on documented physical corruption mechanics (slip=0.85, false_hit=0.04)
            if outcome == 1:
                p_obs_given_clean1 = rho + 0.15 * (1.0 - rho)
                p_obs_given_clean0 = 0.04 * (1.0 - rho)
            else:
                p_obs_given_clean1 = 0.85 * (1.0 - rho)
                p_obs_given_clean0 = rho + 0.96 * (1.0 - rho)
            num = p * p_obs_given_clean1
            den = num + (1.0 - p) * p_obs_given_clean0
            soft_y = float(np.clip(num / max(1e-12, den), 0.0, 1.0))

            target_m = soft_y * 0.95 + (1.0 - soft_y) * 0.05
            new_m = curr_m + self.learning_rate_prior * (target_m - curr_m)
            self.estimated_mastery[prim_skill] = float(np.clip(new_m, 0.05, 0.95))

        else:
            # Baseline heuristic updater (Phase 3 baseline)
            if outcome == 1:
                new_m = curr_m + self.learning_rate_prior * (0.95 - curr_m)
            else:
                new_m = curr_m - self.learning_rate_prior * (curr_m - 0.05)
            self.estimated_mastery[prim_skill] = float(np.clip(new_m, 0.05, 0.95))

        # 2. Secondary skill transfer update (tracks population secondary transfer gain)
        if self.belief_update_mode in (
            "transfer_aware",
            "combined_transfer_reliability",
            "combined",
            "expected_transition_transfer",
            "approx_bayesian_filter_transfer",
        ):
            if sec_skill and sec_skill in self.estimated_mastery:
                curr_sec_m = self.estimated_mastery[sec_skill]
                sec_delta = self.secondary_transfer_ratio * self.learning_rate_prior * (0.95 - curr_sec_m)
                new_sec_m = float(np.clip(curr_sec_m + sec_delta, 0.05, 0.95))
                self.estimated_mastery[sec_skill] = new_sec_m
                if hasattr(self, "_bayesian_filter_state") and sec_skill in getattr(self, "_bayesian_filter_state", {}):
                    sec_grid = self._bayesian_filter_state[sec_skill]["grid"]
                    sec_w = self._bayesian_filter_state[sec_skill]["weights"]
                    sec_next = np.clip(
                        sec_grid + self.secondary_transfer_ratio * self.learning_rate_prior * (0.95 - sec_grid),
                        0.05,
                        0.95,
                    )
                    self._bayesian_filter_state[sec_skill]["weights"] = _project_distribution_onto_grid(sec_next, sec_w, sec_grid)


class RobustAdaptiveFilterController(ModelAlignedTargetAdaptiveController):
    """Production robust adaptive Bayesian filter controller with bilateral practice transitions,
    session gap uncertainty diffusion, success-modulated secondary transfer, and reliability-gated
    conservative fallback for distribution shift and temporal dynamics robustness.
    """

    def __init__(self, task_registry: TaskRegistry, **kwargs):
        kwargs.setdefault("belief_update_mode", "robust_adaptive_filter")
        super().__init__(task_registry, **kwargs)



def calculate_immediate_reversals(diffs: List[float]) -> int:
    """Calculates immediate trial-to-trial direction reversals: (D_t - D_{t-1}) * (D_{t-1} - D_{t-2}) < 0.

    Zero-change (flat) transitions are not reversals; any zero change produces a product of 0
    and resets the immediate direction state.
    """
    if len(diffs) < 3:
        return 0
    deltas = [diffs[i] - diffs[i - 1] for i in range(1, len(diffs))]
    reversals = 0
    for i in range(1, len(deltas)):
        if (deltas[i] * deltas[i - 1]) < -1e-5:
            reversals += 1
    return reversals


def calculate_filtered_direction_reversals(diffs: List[float]) -> int:
    """Calculates sign reversals of consecutive non-zero difficulty changes (filtering zero-step intervals)."""
    deltas = [diffs[i] - diffs[i - 1] for i in range(1, len(diffs)) if abs(diffs[i] - diffs[i - 1]) > 1e-4]
    reversals = 0
    for i in range(1, len(deltas)):
        if (deltas[i] * deltas[i - 1]) < 0.0:
            reversals += 1
    return reversals


def calculate_direction_reversals(diffs: List[float]) -> int:
    """Legacy alias for calculate_filtered_direction_reversals."""
    return calculate_filtered_direction_reversals(diffs)


def run_closed_loop_comparison(
    task_registry: TaskRegistry,
    scenario_cfg: Dict[str, Any],
    num_learners: int = 60,
    trials_per_learner: int = 60,
    seed: int = 42,
    controller_configs: Optional[Dict[str, Any]] = None,
    policies: Optional[List[str]] = None,
    include_legacy_surrogate: bool = False,
    prior_initializer: Optional[MasteryPriorInitializer] = None,
) -> Tuple[Dict[str, List[ClosedLoopRunResult]], pd.DataFrame, pd.DataFrame]:
    """Runs paired closed-loop simulation across controller strategies using identical initial learner profiles.

    Args:
        task_registry: Registry of 9 Grow21 games.
        scenario_cfg: Scenario parameter dictionary.
        num_learners: Cohort size.
        trials_per_learner: Trial horizon per learner.
        seed: Random seed for profile generation.
        controller_configs: Optional nested controller configuration overrides.
        policies: Optional explicit list of policy keys to run.
        include_legacy_surrogate: If True and policies is None, runs 4 policies including legacy surrogate.
        prior_initializer: Optional MasteryPriorInitializer for skill prior initialization.

    Returns:
        (results_dict, events_df, learner_mastery_df)
    """
    rng = np.random.RandomState(seed)
    rel_engine = ReliabilityEngine(task_registry)

    cfg = controller_configs or {}
    fixed_cfg = cfg.get("fixed_difficulty", {})
    streak_cfg = cfg.get("streak_adaptive", {})
    target_cfg = cfg.get("target_adaptive", {})
    legacy_cfg = cfg.get("legacy_surrogate_target_adaptive", target_cfg)

    all_factories = {
        "fixed_difficulty": lambda: FixedDifficultyController(
            task_registry,
            fixed_difficulty=float(fixed_cfg.get("difficulty", 0.50)),
            cooldown_gap_hours=float(fixed_cfg.get("cooldown_gap_hours", 2.0)),
        ),
        "streak_adaptive": lambda: StreakAdaptiveController(
            task_registry,
            initial_difficulty=float(streak_cfg.get("initial_difficulty", 0.45)),
            up_step=float(streak_cfg.get("up_step", 0.06)),
            down_step=float(streak_cfg.get("down_step", 0.08)),
            cooldown_gap_hours=float(streak_cfg.get("cooldown_gap_hours", 2.0)),
        ),
        "target_adaptive": lambda: ModelAlignedTargetAdaptiveController(
            task_registry,
            target_success_prob=float(target_cfg.get("target_success_prob", 0.75)),
            initial_difficulty=float(target_cfg.get("initial_difficulty", 0.45)),
            deadband=float(target_cfg.get("deadband", 0.04)),
            max_step_change=float(target_cfg.get("max_step_change", 0.08)),
            min_difficulty=float(target_cfg.get("min_difficulty", 0.08)),
            max_difficulty=float(target_cfg.get("max_difficulty", 0.95)),
            fatigue_sensitivity_prior=float(target_cfg.get("fatigue_sensitivity_prior", 0.02)),
            forgetting_rate_prior=float(target_cfg.get("forgetting_rate_prior", 0.02)),
            cooldown_gap_hours=float(target_cfg.get("cooldown_gap_hours", 2.0)),
            prior_initializer=prior_initializer,
        ),
        "model_aligned_target_adaptive": lambda: ModelAlignedTargetAdaptiveController(
            task_registry,
            target_success_prob=float(target_cfg.get("target_success_prob", 0.75)),
            initial_difficulty=float(target_cfg.get("initial_difficulty", 0.45)),
            deadband=float(target_cfg.get("deadband", 0.04)),
            max_step_change=float(target_cfg.get("max_step_change", 0.08)),
            min_difficulty=float(target_cfg.get("min_difficulty", 0.08)),
            max_difficulty=float(target_cfg.get("max_difficulty", 0.95)),
            fatigue_sensitivity_prior=float(target_cfg.get("fatigue_sensitivity_prior", 0.02)),
            forgetting_rate_prior=float(target_cfg.get("forgetting_rate_prior", 0.02)),
            learning_rate_prior=float(target_cfg.get("learning_rate_prior", 0.08)),
            secondary_transfer_ratio=float(target_cfg.get("secondary_transfer_ratio", 0.30)),
            belief_update_mode=str(target_cfg.get("belief_update_mode", "heuristic_baseline")),
            cooldown_gap_hours=float(target_cfg.get("cooldown_gap_hours", 2.0)),
            prior_initializer=prior_initializer,
        ),
        "transfer_aware_target_adaptive": lambda: ModelAlignedTargetAdaptiveController(
            task_registry,
            target_success_prob=float(target_cfg.get("target_success_prob", 0.75)),
            initial_difficulty=float(target_cfg.get("initial_difficulty", 0.45)),
            deadband=float(target_cfg.get("deadband", 0.04)),
            max_step_change=float(target_cfg.get("max_step_change", 0.08)),
            min_difficulty=float(target_cfg.get("min_difficulty", 0.08)),
            max_difficulty=float(target_cfg.get("max_difficulty", 0.95)),
            fatigue_sensitivity_prior=float(target_cfg.get("fatigue_sensitivity_prior", 0.02)),
            forgetting_rate_prior=float(target_cfg.get("forgetting_rate_prior", 0.02)),
            learning_rate_prior=float(target_cfg.get("learning_rate_prior", 0.08)),
            secondary_transfer_ratio=float(target_cfg.get("secondary_transfer_ratio", 0.30)),
            belief_update_mode="transfer_aware",
            cooldown_gap_hours=float(target_cfg.get("cooldown_gap_hours", 2.0)),
            prior_initializer=prior_initializer,
        ),
        "reliability_aware_target_adaptive": lambda: ModelAlignedTargetAdaptiveController(
            task_registry,
            target_success_prob=float(target_cfg.get("target_success_prob", 0.75)),
            initial_difficulty=float(target_cfg.get("initial_difficulty", 0.45)),
            deadband=float(target_cfg.get("deadband", 0.04)),
            max_step_change=float(target_cfg.get("max_step_change", 0.08)),
            min_difficulty=float(target_cfg.get("min_difficulty", 0.08)),
            max_difficulty=float(target_cfg.get("max_difficulty", 0.95)),
            fatigue_sensitivity_prior=float(target_cfg.get("fatigue_sensitivity_prior", 0.02)),
            forgetting_rate_prior=float(target_cfg.get("forgetting_rate_prior", 0.02)),
            learning_rate_prior=float(target_cfg.get("learning_rate_prior", 0.08)),
            secondary_transfer_ratio=float(target_cfg.get("secondary_transfer_ratio", 0.30)),
            belief_update_mode="reliability_aware",
            cooldown_gap_hours=float(target_cfg.get("cooldown_gap_hours", 2.0)),
            prior_initializer=prior_initializer,
        ),
        "combined_target_adaptive": lambda: ModelAlignedTargetAdaptiveController(
            task_registry,
            target_success_prob=float(target_cfg.get("target_success_prob", 0.75)),
            initial_difficulty=float(target_cfg.get("initial_difficulty", 0.45)),
            deadband=float(target_cfg.get("deadband", 0.04)),
            max_step_change=float(target_cfg.get("max_step_change", 0.08)),
            min_difficulty=float(target_cfg.get("min_difficulty", 0.08)),
            max_difficulty=float(target_cfg.get("max_difficulty", 0.95)),
            fatigue_sensitivity_prior=float(target_cfg.get("fatigue_sensitivity_prior", 0.02)),
            forgetting_rate_prior=float(target_cfg.get("forgetting_rate_prior", 0.02)),
            learning_rate_prior=float(target_cfg.get("learning_rate_prior", 0.08)),
            secondary_transfer_ratio=float(target_cfg.get("secondary_transfer_ratio", 0.30)),
            belief_update_mode="combined_transfer_reliability",
            cooldown_gap_hours=float(target_cfg.get("cooldown_gap_hours", 2.0)),
            prior_initializer=prior_initializer,
        ),
        "expected_transition": lambda: ModelAlignedTargetAdaptiveController(
            task_registry,
            target_success_prob=float(target_cfg.get("target_success_prob", 0.75)),
            initial_difficulty=float(target_cfg.get("initial_difficulty", 0.45)),
            deadband=float(target_cfg.get("deadband", 0.04)),
            max_step_change=float(target_cfg.get("max_step_change", 0.08)),
            min_difficulty=float(target_cfg.get("min_difficulty", 0.08)),
            max_difficulty=float(target_cfg.get("max_difficulty", 0.95)),
            fatigue_sensitivity_prior=float(target_cfg.get("fatigue_sensitivity_prior", 0.02)),
            forgetting_rate_prior=float(target_cfg.get("forgetting_rate_prior", 0.02)),
            learning_rate_prior=float(target_cfg.get("learning_rate_prior", 0.08)),
            secondary_transfer_ratio=float(target_cfg.get("secondary_transfer_ratio", 0.30)),
            belief_update_mode="expected_transition",
            cooldown_gap_hours=float(target_cfg.get("cooldown_gap_hours", 2.0)),
            prior_initializer=prior_initializer,
        ),
        "expected_transition_transfer": lambda: ModelAlignedTargetAdaptiveController(
            task_registry,
            target_success_prob=float(target_cfg.get("target_success_prob", 0.75)),
            initial_difficulty=float(target_cfg.get("initial_difficulty", 0.45)),
            deadband=float(target_cfg.get("deadband", 0.04)),
            max_step_change=float(target_cfg.get("max_step_change", 0.08)),
            min_difficulty=float(target_cfg.get("min_difficulty", 0.08)),
            max_difficulty=float(target_cfg.get("max_difficulty", 0.95)),
            fatigue_sensitivity_prior=float(target_cfg.get("fatigue_sensitivity_prior", 0.02)),
            forgetting_rate_prior=float(target_cfg.get("forgetting_rate_prior", 0.02)),
            learning_rate_prior=float(target_cfg.get("learning_rate_prior", 0.08)),
            secondary_transfer_ratio=float(target_cfg.get("secondary_transfer_ratio", 0.30)),
            belief_update_mode="expected_transition_transfer",
            cooldown_gap_hours=float(target_cfg.get("cooldown_gap_hours", 2.0)),
            prior_initializer=prior_initializer,
        ),
        "approx_bayesian_filter": lambda: ModelAlignedTargetAdaptiveController(
            task_registry,
            target_success_prob=float(target_cfg.get("target_success_prob", 0.75)),
            initial_difficulty=float(target_cfg.get("initial_difficulty", 0.45)),
            deadband=float(target_cfg.get("deadband", 0.04)),
            max_step_change=float(target_cfg.get("max_step_change", 0.08)),
            min_difficulty=float(target_cfg.get("min_difficulty", 0.08)),
            max_difficulty=float(target_cfg.get("max_difficulty", 0.95)),
            fatigue_sensitivity_prior=float(target_cfg.get("fatigue_sensitivity_prior", 0.02)),
            forgetting_rate_prior=float(target_cfg.get("forgetting_rate_prior", 0.02)),
            learning_rate_prior=float(target_cfg.get("learning_rate_prior", 0.08)),
            secondary_transfer_ratio=float(target_cfg.get("secondary_transfer_ratio", 0.30)),
            belief_update_mode="approx_bayesian_filter",
            cooldown_gap_hours=float(target_cfg.get("cooldown_gap_hours", 2.0)),
            prior_initializer=prior_initializer,
        ),
        "approx_bayesian_filter_transfer": lambda: ModelAlignedTargetAdaptiveController(
            task_registry,
            target_success_prob=float(target_cfg.get("target_success_prob", 0.75)),
            initial_difficulty=float(target_cfg.get("initial_difficulty", 0.45)),
            deadband=float(target_cfg.get("deadband", 0.04)),
            max_step_change=float(target_cfg.get("max_step_change", 0.08)),
            min_difficulty=float(target_cfg.get("min_difficulty", 0.08)),
            max_difficulty=float(target_cfg.get("max_difficulty", 0.95)),
            fatigue_sensitivity_prior=float(target_cfg.get("fatigue_sensitivity_prior", 0.02)),
            forgetting_rate_prior=float(target_cfg.get("forgetting_rate_prior", 0.02)),
            learning_rate_prior=float(target_cfg.get("learning_rate_prior", 0.08)),
            secondary_transfer_ratio=float(target_cfg.get("secondary_transfer_ratio", 0.30)),
            belief_update_mode="approx_bayesian_filter_transfer",
            cooldown_gap_hours=float(target_cfg.get("cooldown_gap_hours", 2.0)),
            prior_initializer=prior_initializer,
        ),
        "robust_adaptive_filter": lambda: ModelAlignedTargetAdaptiveController(
            task_registry,
            target_success_prob=float(target_cfg.get("target_success_prob", 0.75)),
            initial_difficulty=float(target_cfg.get("initial_difficulty", 0.45)),
            deadband=float(target_cfg.get("deadband", 0.04)),
            max_step_change=float(target_cfg.get("max_step_change", 0.08)),
            min_difficulty=float(target_cfg.get("min_difficulty", 0.08)),
            max_difficulty=float(target_cfg.get("max_difficulty", 0.95)),
            fatigue_sensitivity_prior=float(target_cfg.get("fatigue_sensitivity_prior", 0.02)),
            forgetting_rate_prior=float(target_cfg.get("forgetting_rate_prior", 0.02)),
            learning_rate_prior=float(target_cfg.get("learning_rate_prior", 0.08)),
            secondary_transfer_ratio=float(target_cfg.get("secondary_transfer_ratio", 0.30)),
            belief_update_mode="robust_adaptive_filter",
            cooldown_gap_hours=float(target_cfg.get("cooldown_gap_hours", 2.0)),
            prior_initializer=prior_initializer,
        ),
        "legacy_surrogate_target_adaptive": lambda: LegacySurrogateTargetAdaptiveController(
            task_registry,
            target_success_prob=float(legacy_cfg.get("target_success_prob", 0.75)),
            initial_difficulty=float(legacy_cfg.get("initial_difficulty", 0.45)),
            gamma=float(legacy_cfg.get("gamma", 2.4)),
            nu_D=float(legacy_cfg.get("nu_D", 1.0)),
            deadband=float(legacy_cfg.get("deadband", 0.04)),
            max_step_change=float(legacy_cfg.get("max_step_change", 0.08)),
            min_difficulty=float(legacy_cfg.get("min_difficulty", 0.08)),
            max_difficulty=float(legacy_cfg.get("max_difficulty", 0.95)),
            cooldown_gap_hours=float(legacy_cfg.get("cooldown_gap_hours", 2.0)),
        ),
    }

    # Aliases for clean notation
    all_factories["model_aligned_baseline"] = all_factories["model_aligned_target_adaptive"]
    all_factories["transfer_aware"] = all_factories["transfer_aware_target_adaptive"]
    all_factories["reliability_aware"] = all_factories["reliability_aware_target_adaptive"]
    all_factories["combined_transfer_reliability"] = all_factories["combined_target_adaptive"]

    if policies is not None:
        selected_policy_names = policies
    elif include_legacy_surrogate:
        selected_policy_names = [
            "fixed_difficulty",
            "streak_adaptive",
            "legacy_surrogate_target_adaptive",
            "model_aligned_target_adaptive",
        ]
    else:
        selected_policy_names = ["fixed_difficulty", "streak_adaptive", "target_adaptive"]

    controllers_factories = {p: all_factories[p] for p in selected_policy_names}

    results: Dict[str, List[ClosedLoopRunResult]] = {c: [] for c in controllers_factories}
    all_events: List[Dict[str, Any]] = []
    learner_mastery_records: List[Dict[str, Any]] = []

    sim_scenario_params = (
        scenario_cfg["test_params"]
        if scenario_cfg.get("evaluation_type") == "distribution_shift" and "test_params" in scenario_cfg
        else scenario_cfg
    )

    for l_id in range(num_learners):
        base_profile = generate_learner_profile(
            learner_id=l_id,
            skill_ids=task_registry.skill_ids,
            rng=rng,
            scenario_params=sim_scenario_params,
        )

        for c_name, c_factory in controllers_factories.items():
            learner_prof = copy.deepcopy(base_profile)
            sim = VirtualLearnerSimulator(
                profile=learner_prof,
                task_registry=task_registry,
                reliability_engine=rel_engine,
            )
            controller = c_factory()

            init_mastery_dict = dict(sim.current_mastery)
            init_m_mean = float(np.mean(list(init_mastery_dict.values())))

            obs_outcomes = []
            clean_outcomes = []
            diffs = []
            primary_counts: Dict[str, int] = {s: 0 for s in task_registry.skill_ids}
            secondary_counts: Dict[str, int] = {s: 0 for s in task_registry.skill_ids}
            at_min_counts = 0
            at_max_counts = 0
            reachable_counts = 0
            safety_interv_counts = 0
            pred_probs: List[float] = []
            target_errors: List[float] = []
            prediction_gaps: List[float] = []

            # Seeded per-learner stream to ensure fair common random numbers across controllers
            sim_rng = np.random.RandomState(seed * 1000 + l_id)

            for step in range(trials_per_learner):
                gid, diff, gap = controller.select_next_task(step)
                diag = getattr(controller, "last_diagnostic", {})

                # Capture pre-outcome belief state (after time-gap forgetting, before outcome observation)
                est_m_pre = dict(getattr(controller, "estimated_mastery", {}))

                ev = sim.step(
                    game_id=gid,
                    difficulty=diff,
                    time_gap_hours=gap,
                    rng=sim_rng,
                    scenario_params=sim_scenario_params,
                )
                controller.observe_outcome(gid, ev.observed_outcome, diff, estimated_reliability=ev.estimated_reliability)

                # Capture post-update belief state (after outcome observation and belief update)
                est_m_post = dict(getattr(controller, "estimated_mastery", {}))

                obs_outcomes.append(ev.observed_outcome)
                clean_outcomes.append(ev.clean_outcome)
                diffs.append(diff)

                primary_counts[ev.primary_skill] += 1
                if ev.secondary_skill:
                    secondary_counts[ev.secondary_skill] += 1

                if diag.get("at_min", False):
                    at_min_counts += 1
                if diag.get("at_max", False):
                    at_max_counts += 1
                if diag.get("safety_intervention", False):
                    safety_interv_counts += 1
                if diag.get("reachability_applicable", False) and diag.get("target_reachable") is True:
                    reachable_counts += 1

                pred_p = diag.get("predicted_clean_prob")
                target_err = diag.get("target_error")
                sim_p = round(float(ev.p_clean), 4)

                gap_clean = None
                if pred_p is not None:
                    pred_probs.append(pred_p)
                    if target_err is not None:
                        target_errors.append(target_err)
                    gap_clean = round(float(pred_p - sim_p), 4)
                    prediction_gaps.append(gap_clean)

                target_reachable_val = diag.get("target_reachable")

                # 1. Pre-outcome alignment (controller belief before outcome vs simulator latent mastery before trial)
                prim_est_m_pre = est_m_pre.get(ev.primary_skill)
                sec_est_m_pre = est_m_pre.get(ev.secondary_skill) if ev.secondary_skill else None
                prim_true_m_pre = ev.latent_mastery_before.get(ev.primary_skill) if ev.latent_mastery_before else None
                sec_true_m_pre = ev.latent_mastery_before.get(ev.secondary_skill) if (ev.secondary_skill and ev.latent_mastery_before) else None
                prim_err_pre = round(float(prim_est_m_pre - prim_true_m_pre), 4) if (prim_est_m_pre is not None and prim_true_m_pre is not None) else None
                sec_err_pre = round(float(sec_est_m_pre - sec_true_m_pre), 4) if (sec_est_m_pre is not None and sec_true_m_pre is not None) else None

                # 2. Post-update alignment (controller belief after observe_outcome vs simulator latent mastery after step)
                prim_est_m_post = est_m_post.get(ev.primary_skill)
                sec_est_m_post = est_m_post.get(ev.secondary_skill) if ev.secondary_skill else None
                prim_true_m_post = ev.latent_mastery_after.get(ev.primary_skill) if ev.latent_mastery_after else None
                sec_true_m_post = ev.latent_mastery_after.get(ev.secondary_skill) if (ev.secondary_skill and ev.latent_mastery_after) else None
                prim_err_post = round(float(prim_est_m_post - prim_true_m_post), 4) if (prim_est_m_post is not None and prim_true_m_post is not None) else None
                sec_err_post = round(float(sec_est_m_post - sec_true_m_post), 4) if (sec_est_m_post is not None and sec_true_m_post is not None) else None

                all_events.append({
                    "controller": c_name,
                    "learner_id": l_id,
                    "step_idx": step,
                    "game_id": gid,
                    "primary_skill": ev.primary_skill,
                    "secondary_skill": ev.secondary_skill or "none",
                    "difficulty": diff,
                    "time_gap_hours": gap,
                    "observed_outcome": ev.observed_outcome,
                    "clean_outcome": ev.clean_outcome,
                    "simulator_p_clean": sim_p,
                    "predicted_clean_prob": pred_p,
                    "prediction_gap_clean": gap_clean,
                    "target_prob": diag.get("target_prob"),
                    "target_error": target_err,
                    "target_difficulty_solved": diag.get("target_difficulty_solved"),
                    "applied_difficulty": diff,
                    "estimated_reliability": ev.estimated_reliability,
                    "reachability_applicable": int(diag.get("reachability_applicable", False)),
                    "target_reachable": int(target_reachable_val) if target_reachable_val is not None else None,
                    "reachability_status": diag.get("reachability_status", "N/A"),
                    "p_high": diag.get("p_high"),
                    "p_low": diag.get("p_low"),
                    "deadband_held": diag.get("deadband_held"),
                    "step_limited": diag.get("step_limited"),
                    "at_min_bound": int(diag.get("at_min", False)),
                    "at_max_bound": int(diag.get("at_max", False)),
                    "safety_intervention": int(diag.get("safety_intervention", False)),
                    "determination_rule": diag.get("determination_rule", "N/A"),

                    # Time-aligned pre-outcome fields
                    "estimated_mastery_primary_pre": prim_est_m_pre,
                    "true_mastery_primary_pre": prim_true_m_pre,
                    "mastery_error_primary_pre": prim_err_pre,
                    "estimated_mastery_secondary_pre": sec_est_m_pre,
                    "true_mastery_secondary_pre": sec_true_m_pre,
                    "mastery_error_secondary_pre": sec_err_pre,

                    # Time-aligned post-update fields
                    "estimated_mastery_primary_post": prim_est_m_post,
                    "true_mastery_primary_post": prim_true_m_post,
                    "mastery_error_primary_post": prim_err_post,
                    "estimated_mastery_secondary_post": sec_est_m_post,
                    "true_mastery_secondary_post": sec_true_m_post,
                    "mastery_error_secondary_post": sec_err_post,

                    # Unadorned aliases mapped strictly to pre-outcome state
                    "estimated_mastery_primary": prim_est_m_pre,
                    "true_mastery_primary": prim_true_m_pre,
                    "mastery_error_primary": prim_err_pre,
                    "estimated_mastery_secondary": sec_est_m_pre,
                    "true_mastery_secondary": sec_true_m_pre,
                    "mastery_error_secondary": sec_err_pre,

                    "corruption_occurred": int(ev.corruption_occurred),
                })

            final_mastery_dict = dict(sim.current_mastery)
            final_m_mean = float(np.mean(list(final_mastery_dict.values())))

            # Direction reversals (both immediate consecutive and filtered non-zero)
            imm_reversals = calculate_immediate_reversals(diffs)
            filt_reversals = calculate_filtered_direction_reversals(diffs)
            total_variation = sum(abs(diffs[i] - diffs[i - 1]) for i in range(1, len(diffs)))

            # Per skill gains and categorized skill tracking
            per_skill = {s: round(final_mastery_dict[s] - init_mastery_dict[s], 4) for s in task_registry.skill_ids}

            # Export per-learner per-skill records
            for s in task_registry.skill_ids:
                learner_mastery_records.append({
                    "learner_id": l_id,
                    "policy": c_name,
                    "seed": seed,
                    "skill_id": s,
                    "initial_mastery": round(init_mastery_dict[s], 4),
                    "final_mastery": round(final_mastery_dict[s], 4),
                    "mastery_change": per_skill[s],
                    "primary_task_counts": primary_counts[s],
                    "secondary_transfer_counts": secondary_counts[s],
                })

            # Primary skills (practiced as primary at least once)
            prim_gains = [per_skill[s] for s, cnt in primary_counts.items() if cnt > 0]
            # Secondary skills (received transfer practice but never primary)
            sec_gains = [per_skill[s] for s, cnt in secondary_counts.items() if cnt > 0 and primary_counts[s] == 0]
            if not sec_gains:
                sec_gains = [per_skill[s] for s, cnt in secondary_counts.items() if cnt > 0]

            is_target_policy = "target" in c_name
            reach_prop = (
                round(reachable_counts / trials_per_learner, 4)
                if is_target_policy
                else None
            )

            brier = None
            if pred_probs and len(pred_probs) == len(clean_outcomes):
                brier = round(float(np.mean([(p - y) ** 2 for p, y in zip(pred_probs, clean_outcomes)])), 4)

            res = ClosedLoopRunResult(
                controller_name=c_name,
                scenario_name=scenario_cfg.get("description", "custom"),
                learner_id=l_id,
                initial_mastery_mean=round(init_m_mean, 4),
                final_mastery_mean=round(final_m_mean, 4),
                mastery_gain_all_skills=round(final_m_mean - init_m_mean, 4),
                mastery_gain_primary_skills=round(float(np.mean(prim_gains)) if prim_gains else 0.0, 4),
                mastery_gain_secondary_skills=round(float(np.mean(sec_gains)) if sec_gains else 0.0, 4),
                per_skill_gains=per_skill,
                total_trials=trials_per_learner,
                observed_success_rate=round(float(np.mean(obs_outcomes)), 4),
                clean_success_rate=round(float(np.mean(clean_outcomes)), 4),
                immediate_direction_reversals=imm_reversals,
                filtered_direction_reversals=filt_reversals,
                direction_reversals=filt_reversals,
                total_difficulty_variation=round(float(total_variation), 3),
                mean_difficulty=round(float(np.mean(diffs)), 3),
                proportion_at_min_bound=round(at_min_counts / trials_per_learner, 4),
                proportion_at_max_bound=round(at_max_counts / trials_per_learner, 4),
                target_reachable_proportion=reach_prop,
                frustration_violations=controller.frustration_violations,
                break_limit_violations=controller.break_limit_violations,
                safety_interventions=safety_interv_counts,
                mean_predicted_clean_prob=round(float(np.mean(pred_probs)), 4) if pred_probs else None,
                mean_target_error=round(float(np.mean(target_errors)), 4) if target_errors else None,
                mean_prediction_gap=round(float(np.mean(prediction_gaps)), 4) if prediction_gaps else None,
                brier_score_clean=brier,
            )
            results[c_name].append(res)

    events_df = pd.DataFrame(all_events)
    mastery_df = pd.DataFrame(learner_mastery_records)
    return results, events_df, mastery_df


def run_open_loop_replay(
    task_registry: TaskRegistry,
    scenario_cfg: Dict[str, Any],
    num_learners: int = 60,
    trials_per_learner: int = 60,
    seed: int = 42,
    policies: Optional[List[str]] = None,
    prior_initializer: Optional[MasteryPriorInitializer] = None,
    fixed_difficulty_schedule: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs open-loop state estimation replay on pre-specified exogenous task/difficulty sequences.

    Isolates state estimation tracking from controller closed-loop difficulty adaptations.
    All estimators receive identical virtual learner profiles, identical exogenous tasks and difficulties,
    and identical observed responses and estimated reliabilities.

    Returns:
        (replay_events_df, replay_summary_df)
    """
    rng = np.random.RandomState(seed)
    rel_engine = ReliabilityEngine(task_registry)

    sim_scenario_params = (
        scenario_cfg["test_params"]
        if scenario_cfg.get("evaluation_type") == "distribution_shift" and "test_params" in scenario_cfg
        else scenario_cfg
    )

    schedule = fixed_difficulty_schedule or [0.20, 0.35, 0.50, 0.65, 0.80]
    candidate_policies = policies or [
        "model_aligned_baseline",
        "transfer_aware",
        "reliability_aware",
        "combined_transfer_reliability",
        "expected_transition",
        "expected_transition_transfer",
        "approx_bayesian_filter",
        "approx_bayesian_filter_transfer",
    ]

    all_replay_events: List[Dict[str, Any]] = []

    for l_id in range(num_learners):
        base_profile = generate_learner_profile(
            learner_id=l_id,
            skill_ids=task_registry.skill_ids,
            rng=rng,
            scenario_params=sim_scenario_params,
        )

        # 1. Generate the common exogenous simulator sequence for this learner
        sim_rng = np.random.RandomState(seed * 1000 + l_id)
        sim = VirtualLearnerSimulator(
            profile=copy.deepcopy(base_profile),
            task_registry=task_registry,
            reliability_engine=rel_engine,
        )

        exogenous_events: List[SimulatorEvent] = []
        for step in range(trials_per_learner):
            gid = task_registry.game_ids[step % len(task_registry.game_ids)]
            diff = schedule[step % len(schedule)]
            gap = 12.0 if (step > 0 and step % 15 == 0) else 0.02

            ev = sim.step(
                game_id=gid,
                difficulty=diff,
                time_gap_hours=gap,
                rng=sim_rng,
                scenario_params=sim_scenario_params,
            )
            exogenous_events.append(ev)

        # 2. Replay this identical exogenous sequence across all candidate estimators
        for pol in candidate_policies:
            ctrl = ModelAlignedTargetAdaptiveController(
                task_registry=task_registry,
                belief_update_mode=pol,
                prior_initializer=prior_initializer,
            )

            for step, ev in enumerate(exogenous_events):
                gid = ev.game_id
                diff = ev.difficulty
                gap = ev.session_gap_hours
                proj_counter = (step % 15) + 1

                # Forgetting
                if gap > 0.0:
                    for s in task_registry.skill_ids:
                        decayed = 0.05 + (ctrl.estimated_mastery[s] - 0.05) * math.exp(
                            -ctrl.forgetting_rate_prior * gap
                        )
                        ctrl.estimated_mastery[s] = float(np.clip(decayed, 0.05, 0.95))
                        if hasattr(ctrl, "_bayesian_filter_state") and s in getattr(ctrl, "_bayesian_filter_state", {}):
                            grid = ctrl._bayesian_filter_state[s]["grid"]
                            weights = ctrl._bayesian_filter_state[s]["weights"]
                            decayed_pts = 0.05 + (grid - 0.05) * math.exp(-ctrl.forgetting_rate_prior * gap)
                            new_w = _project_distribution_onto_grid(decayed_pts, weights, grid)
                            ctrl._bayesian_filter_state[s]["weights"] = new_w
                            ctrl.estimated_mastery[s] = float(np.clip(np.sum(new_w * grid), 0.05, 0.95))

                # Pre-outcome belief state
                est_m_pre = dict(ctrl.estimated_mastery)
                pred_p = ctrl.predict_clean_success_prob(gid, diff, session_trial_counter=proj_counter)
                ctrl._last_predicted_clean_prob = pred_p

                # Observe outcome
                ctrl.observe_outcome(gid, ev.observed_outcome, diff, estimated_reliability=ev.estimated_reliability)
                est_m_post = dict(ctrl.estimated_mastery)

                # Scored against simulator ground truth before/after
                prim_s = ev.primary_skill
                sec_s = ev.secondary_skill

                prim_est_pre = est_m_pre.get(prim_s)
                prim_true_pre = ev.latent_mastery_before.get(prim_s)
                prim_err_pre = round(float(prim_est_pre - prim_true_pre), 4) if (prim_est_pre is not None and prim_true_pre is not None) else None

                sec_est_pre = est_m_pre.get(sec_s) if sec_s else None
                sec_true_pre = ev.latent_mastery_before.get(sec_s) if (sec_s and ev.latent_mastery_before) else None
                sec_err_pre = round(float(sec_est_pre - sec_true_pre), 4) if (sec_est_pre is not None and sec_true_pre is not None) else None

                prim_est_post = est_m_post.get(prim_s)
                prim_true_post = ev.latent_mastery_after.get(prim_s)
                prim_err_post = round(float(prim_est_post - prim_true_post), 4) if (prim_est_post is not None and prim_true_post is not None) else None

                sec_est_post = est_m_post.get(sec_s) if sec_s else None
                sec_true_post = ev.latent_mastery_after.get(sec_s) if (sec_s and ev.latent_mastery_after) else None
                sec_err_post = round(float(sec_est_post - sec_true_post), 4) if (sec_est_post is not None and sec_true_post is not None) else None

                prob_abs_err = round(abs(float(pred_p - ev.p_clean)), 4)
                prob_gap = round(float(pred_p - ev.p_clean), 4)

                all_replay_events.append({
                    "controller": pol,
                    "learner_id": l_id,
                    "seed": seed,
                    "step_idx": step,
                    "game_id": gid,
                    "difficulty": diff,
                    "time_gap_hours": gap,
                    "primary_skill": prim_s,
                    "secondary_skill": sec_s or "none",
                    "observed_outcome": ev.observed_outcome,
                    "clean_outcome": ev.clean_outcome,
                    "simulator_p_clean": round(float(ev.p_clean), 4),
                    "predicted_clean_prob": round(float(pred_p), 4),
                    "prob_abs_error": prob_abs_err,
                    "prediction_gap_clean": prob_gap,
                    "estimated_reliability": ev.estimated_reliability,
                    "estimated_mastery_primary_pre": prim_est_pre,
                    "true_mastery_primary_pre": prim_true_pre,
                    "mastery_error_primary_pre": prim_err_pre,
                    "estimated_mastery_secondary_pre": sec_est_pre,
                    "true_mastery_secondary_pre": sec_true_pre,
                    "mastery_error_secondary_pre": sec_err_pre,
                    "estimated_mastery_primary_post": prim_est_post,
                    "true_mastery_primary_post": prim_true_post,
                    "mastery_error_primary_post": prim_err_post,
                    "estimated_mastery_secondary_post": sec_est_post,
                    "true_mastery_secondary_post": sec_true_post,
                    "mastery_error_secondary_post": sec_err_post,
                })

    replay_events_df = pd.DataFrame(all_replay_events)

    # Summarize by controller policy
    summary_rows = []
    for pol, p_df in replay_events_df.groupby("controller"):
        p_err_pre = p_df["mastery_error_primary_pre"].dropna().to_numpy()
        p_err_post = p_df["mastery_error_primary_post"].dropna().to_numpy()
        s_err_pre = p_df["mastery_error_secondary_pre"].dropna().to_numpy()
        s_err_post = p_df["mastery_error_secondary_post"].dropna().to_numpy()
        prob_err = p_df["prob_abs_error"].to_numpy()
        prob_gaps = p_df["prediction_gap_clean"].to_numpy()

        summary_rows.append({
            "controller": pol,
            "n_events": len(p_df),
            "primary_mae_pre": float(np.mean(np.abs(p_err_pre))) if len(p_err_pre) > 0 else 0.0,
            "primary_mastery_mae_pre": float(np.mean(np.abs(p_err_pre))) if len(p_err_pre) > 0 else 0.0,
            "primary_rmse_pre": float(np.sqrt(np.mean(p_err_pre ** 2))) if len(p_err_pre) > 0 else 0.0,
            "primary_bias_pre": float(np.mean(p_err_pre)) if len(p_err_pre) > 0 else 0.0,
            "primary_mae_post": float(np.mean(np.abs(p_err_post))) if len(p_err_post) > 0 else 0.0,
            "primary_mastery_mae_post": float(np.mean(np.abs(p_err_post))) if len(p_err_post) > 0 else 0.0,
            "primary_rmse_post": float(np.sqrt(np.mean(p_err_post ** 2))) if len(p_err_post) > 0 else 0.0,
            "primary_bias_post": float(np.mean(p_err_post)) if len(p_err_post) > 0 else 0.0,
            "secondary_mae_pre": float(np.mean(np.abs(s_err_pre))) if len(s_err_pre) > 0 else None,
            "secondary_mae_post": float(np.mean(np.abs(s_err_post))) if len(s_err_post) > 0 else None,
            "clean_prob_mae": float(np.mean(prob_err)) if len(prob_err) > 0 else 0.0,
            "clean_prob_rmse": float(np.sqrt(np.mean(prob_gaps ** 2))) if len(prob_gaps) > 0 else 0.0,
            "signed_prediction_gap": float(np.mean(prob_gaps)) if len(prob_gaps) > 0 else 0.0,
        })

    summary_df = pd.DataFrame(summary_rows)
    return replay_events_df, summary_df

