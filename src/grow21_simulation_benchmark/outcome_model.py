"""Shared canonical task-outcome model for Grow21 simulator and target controller.

Establishes a single source of truth for item-response clean task-success probability,
difficulty curves, composite effective mastery, and bounded target difficulty solving.

Terminology:
- p_clean: Latent probability that a task is completed successfully BEFORE sensor corruption.
- clean_outcome: Sampled latent clean binary outcome (y_clean in {0, 1}) from p_clean.
- observed_outcome: Potentially corrupted interaction label (y_obs in {0, 1}) available
  to the learner and controller after physical sensor noise / motor slip.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np


def compute_difficulty_curve(difficulty: float) -> float:
    """Computes the smooth non-linear difficulty inflection curve.

    curve(D) = 0.25 * D + 0.75 * D**1.35 for D in [0, 1].
    """
    d_clamped = max(0.0, min(1.0, float(difficulty)))
    return 0.25 * d_clamped + 0.75 * (d_clamped ** 1.35)


def compute_difficulty_threshold(
    difficulty_offset: float, difficulty_scale: float, difficulty: float
) -> float:
    """Translates abstract difficulty D in [0, 1] into an item threshold.

    threshold_g(D) = offset_g + scale_g * curve(D).
    """
    curve = compute_difficulty_curve(difficulty)
    return float(difficulty_offset + difficulty_scale * curve)


def compute_composite_effective_mastery(
    primary_mastery: float, secondary_mastery: Optional[float] = None
) -> float:
    """Computes composite effective mastery for a game task.

    If the game has an active secondary skill, blends 85% primary and 15% secondary mastery;
    otherwise uses 100% primary mastery.
    """
    p_m = float(primary_mastery)
    if secondary_mastery is not None:
        return 0.85 * p_m + 0.15 * float(secondary_mastery)
    return p_m


def compute_clean_task_success_prob(
    discrimination: float,
    difficulty_offset: float,
    difficulty_scale: float,
    guess_floor: float,
    lapse_ceiling: float,
    effective_mastery: float,
    difficulty: float,
    session_trial_counter: int = 1,
    fatigue_sensitivity: float = 0.02,
) -> float:
    """Calculates clean task-success probability p_clean before sensor corruption.

    Formula:
      1. threshold_g(D) = offset_g + scale_g * (0.25 * D + 0.75 * D**1.35)
      2. s = sigmoid(discrimination_g * (effective_m - threshold_g(D)))
      3. fatigue_lapse = min(0.15, fatigue_sensitivity * max(0, session_trial_counter - 10))
      4. total_lapse = min(0.35, lapse_ceiling + fatigue_lapse)
      5. total_guess = guess_floor
      6. p_clean = clip(total_guess + (1 - total_guess - total_lapse) * s, 0.01, 0.99)

    Args:
        discrimination: Game item discrimination parameter a_g (> 0).
        difficulty_offset: Base difficulty threshold offset b_offset.
        difficulty_scale: Difficulty threshold scaling factor b_scale.
        guess_floor: Guessing parameter c_g (minimum success floor).
        lapse_ceiling: Base lapse probability l_g.
        effective_mastery: Combined latent skill mastery in [0, 1].
        difficulty: Abstract difficulty parameter D in [0, 1].
        session_trial_counter: Number of trials executed in current session (1-indexed).
        fatigue_sensitivity: Fatigue lapse rate per trial beyond trial 10.

    Returns:
        p_clean: Latent success probability in [0.01, 0.99].
    """
    d_threshold = compute_difficulty_threshold(difficulty_offset, difficulty_scale, difficulty)
    logit_diff = discrimination * (effective_mastery - d_threshold)

    # Numerically stable sigmoid bounded to [-30, 30]
    sigmoid_val = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit_diff))))

    # Fatigue lapse onset after trial 10, capped at +0.15
    trial_excess = max(0, int(session_trial_counter) - 10)
    fatigue_lapse = min(0.15, max(0.0, float(fatigue_sensitivity) * trial_excess))
    total_lapse = min(0.35, float(lapse_ceiling) + fatigue_lapse)
    total_guess = float(guess_floor)

    p_clean = total_guess + (1.0 - total_guess - total_lapse) * sigmoid_val
    return float(np.clip(p_clean, 0.01, 0.99))


def compute_game_clean_success_prob(
    game: Any,
    effective_mastery: float,
    difficulty: float,
    session_trial_counter: int = 1,
    fatigue_sensitivity: float = 0.02,
) -> float:
    """Convenience wrapper accepting a GameDef or dict and passing parameters to compute_clean_task_success_prob."""
    if hasattr(game, "discrimination"):
        disc = game.discrimination
        offset = game.difficulty_offset
        scale = game.difficulty_scale
        guess = game.guess_floor
        lapse = game.lapse_ceiling
    else:
        disc = game["discrimination"]
        offset = game["difficulty_offset"]
        scale = game["difficulty_scale"]
        guess = game["guess_floor"]
        lapse = game["lapse_ceiling"]

    return compute_clean_task_success_prob(
        discrimination=disc,
        difficulty_offset=offset,
        difficulty_scale=scale,
        guess_floor=guess,
        lapse_ceiling=lapse,
        effective_mastery=effective_mastery,
        difficulty=difficulty,
        session_trial_counter=session_trial_counter,
        fatigue_sensitivity=fatigue_sensitivity,
    )


def solve_target_difficulty(
    game: Any,
    effective_mastery: float,
    target_prob: float,
    session_trial_counter: int = 1,
    fatigue_sensitivity: float = 0.02,
    min_difficulty: float = 0.08,
    max_difficulty: float = 0.95,
    tol: float = 1e-4,
    max_iter: int = 40,
) -> Tuple[float, str, float, float]:
    """Solves for difficulty D in [min_difficulty, max_difficulty] such that p_clean(D) = target_prob.

    Because difficulty threshold increases monotonically with difficulty, p_clean is strictly
    decreasing in D. Thus:
      - p_high = p_clean(min_difficulty) is the maximum achievable probability.
      - p_low = p_clean(max_difficulty) is the minimum achievable probability.

    Returns:
        (target_D, reachability_status, p_high, p_low)
        reachability_status is one of:
          - 'reachable': target_prob is within [p_low, p_high], D solved via bisection.
          - 'unreachable_high': target_prob > p_high, clamped to min_difficulty.
          - 'unreachable_low': target_prob < p_low, clamped to max_difficulty.
    """
    p_high = compute_game_clean_success_prob(
        game,
        effective_mastery,
        min_difficulty,
        session_trial_counter,
        fatigue_sensitivity,
    )
    p_low = compute_game_clean_success_prob(
        game,
        effective_mastery,
        max_difficulty,
        session_trial_counter,
        fatigue_sensitivity,
    )

    if target_prob > p_high:
        return float(min_difficulty), "unreachable_high", round(p_high, 4), round(p_low, 4)
    elif target_prob < p_low:
        return float(max_difficulty), "unreachable_low", round(p_high, 4), round(p_low, 4)

    # Reachable: Bisection method on [min_difficulty, max_difficulty]
    low_d = float(min_difficulty)
    high_d = float(max_difficulty)
    mid_d = (low_d + high_d) / 2.0

    for _ in range(max_iter):
        mid_d = (low_d + high_d) / 2.0
        p_mid = compute_game_clean_success_prob(
            game,
            effective_mastery,
            mid_d,
            session_trial_counter,
            fatigue_sensitivity,
        )
        diff = p_mid - target_prob
        if abs(diff) <= tol or (high_d - low_d) < 1e-5:
            break
        # Since p_clean is decreasing in D:
        # If p_mid > target_prob, D needs to be higher to lower p_clean.
        if diff > 0:
            low_d = mid_d
        else:
            high_d = mid_d

    return round(float(mid_d), 4), "reachable", round(p_high, 4), round(p_low, 4)
