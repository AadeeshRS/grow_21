"""Closed-loop adaptive simulation experiment for Grow21.

Compares controller policies (Fixed, Streak Heuristic, and Grow21 Model-Based Adaptive)
acting on virtual learners, measuring latent mastery progression, difficulty oscillations,
bound saturation, and enforcing strict frustration and session break-limit constraints.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .reliability import ReliabilityEngine
from .simulator import (
    SimulatorEvent,
    VirtualLearnerProfile,
    VirtualLearnerSimulator,
    generate_learner_profile,
)
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

    def select_next_task(self, step_idx: int) -> Tuple[str, float, float, Dict[str, Any]]:
        """Returns (game_id, difficulty, time_gap_hours, diagnostics_dict)."""
        raise NotImplementedError

    def observe_outcome(self, game_id: str, outcome: int, difficulty: float):
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

    def select_next_task(self, step_idx: int) -> Tuple[str, float, float, Dict[str, Any]]:
        game_id = self.games[step_idx % len(self.games)]

        # Check session break constraint: breaks after trials 15, 30, 45 (3 breaks = 36h total in 60 trials)
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

        self.last_diagnostic = {
            "reachability_applicable": False,
            "target_reachable": None,
            "at_min": False,
            "at_max": False,
            "safety_intervention": safety_triggered,
        }
        return game_id, round(diff, 3), time_gap


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

    def select_next_task(self, step_idx: int) -> Tuple[str, float, float, Dict[str, Any]]:
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
            "at_min": diff <= self.min_difficulty + 1e-4,
            "at_max": diff >= self.max_difficulty - 1e-4,
            "safety_intervention": safety_triggered,
        }
        return game_id, diff, time_gap

    def observe_outcome(self, game_id: str, outcome: int, difficulty: float):
        super().observe_outcome(game_id, outcome, difficulty)
        if outcome == 1:
            self.consecutive_successes += 1
            if self.consecutive_successes >= 2:
                self.current_difficulty = min(self.max_difficulty, self.current_difficulty + self.up_step)
                self.consecutive_successes = 0
        else:
            self.consecutive_successes = 0
            self.current_difficulty = max(self.min_difficulty, self.current_difficulty - self.down_step)


class Grow21ModelBasedAdaptiveController(ControllerPolicy):
    """Grow21 Model-Based Adaptive Controller.

    Uses Grow21's predicted-success equation:
      p_hat = sigmoid(logit(theta_k) - gamma * D**nu_D)
    to choose difficulty D targeting the Zone of Proximal Development (ZPD target p_star ~ 0.75),
    with explicit feasibility checking, deadband, step-size limits, and frustration/break constraints.
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

    def select_next_task(self, step_idx: int) -> Tuple[str, float, float, Dict[str, Any]]:
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
            # Evaluate reachability independently without conflating safety cooldown with mathematical unreachability
            is_reachable, p_low, p_high = self.check_reachability(prim_skill)
            if self.target_prob > p_high:
                status = "unreachable_high"
            elif self.target_prob < p_low:
                status = "unreachable_low"
            else:
                status = "reachable"
            self.last_diagnostic = {
                "reachability_applicable": True,
                "target_reachable": is_reachable,
                "reachability_status": status,
                "at_min": diff <= self.min_difficulty + 1e-4,
                "at_max": diff >= self.max_difficulty - 1e-4,
                "safety_intervention": True,
                "p_high": round(p_high, 4),
                "p_low": round(p_low, 4),
            }
            return game_id, diff, time_gap

        # Grow21 Model-Based Controller Difficulty Optimization
        # Check reachability across difficulty bounds [D_min, D_max]
        is_reachable, p_low, p_high = self.check_reachability(prim_skill)

        if self.target_prob > p_high:
            # Unreachable-high target: target success prob exceeds maximum achievable success at D_min
            target_D = self.min_difficulty
            status = "unreachable_high"
        elif self.target_prob < p_low:
            # Unreachable-low target: target success prob is below minimum achievable success at D_max
            target_D = self.max_difficulty
            status = "unreachable_low"
        else:
            # Reachable target: solve for target difficulty within [D_min, D_max]
            theta = float(np.clip(self.estimated_mastery.get(prim_skill, 0.45), 0.05, 0.95))
            logit_theta = math.log(theta / (1.0 - theta))
            logit_target = math.log(self.target_prob / (1.0 - self.target_prob))
            req_power = (logit_theta - logit_target) / self.gamma
            target_D = float(np.clip(req_power ** (1.0 / self.nu_D), self.min_difficulty, self.max_difficulty))
            status = "reachable"

        # Check deadband: hold steady if within deadband of target
        current_pred_p = self.predict_success_prob(prim_skill, self.current_difficulty)
        if abs(current_pred_p - self.target_prob) <= self.deadband:
            target_D = self.current_difficulty

        # Apply maximum step change limit (smooth control)
        step_delta = float(np.clip(target_D - self.current_difficulty, -self.max_step_change, self.max_step_change))
        self.current_difficulty = float(np.clip(
            self.current_difficulty + step_delta, self.min_difficulty, self.max_difficulty
        ))

        diff = round(self.current_difficulty, 3)
        self.last_diagnostic = {
            "reachability_applicable": True,
            "target_reachable": is_reachable,
            "reachability_status": status,
            "at_min": diff <= self.min_difficulty + 1e-4,
            "at_max": diff >= self.max_difficulty - 1e-4,
            "safety_intervention": False,
            "p_high": round(p_high, 4),
            "p_low": round(p_low, 4),
        }
        return game_id, diff, time_gap

    def observe_outcome(self, game_id: str, outcome: int, difficulty: float):
        super().observe_outcome(game_id, outcome, difficulty)
        prim_skill = self.registry.games[game_id].primary_skill

        # Online belief update on skill mastery (Bayesian smoothed knowledge update)
        curr_m = self.estimated_mastery.get(prim_skill, 0.45)
        if outcome == 1:
            new_m = curr_m + 0.08 * (0.95 - curr_m)
        else:
            new_m = curr_m - 0.08 * (curr_m - 0.05)
        self.estimated_mastery[prim_skill] = float(np.clip(new_m, 0.05, 0.95))


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
) -> Tuple[Dict[str, List[ClosedLoopRunResult]], pd.DataFrame, pd.DataFrame]:
    """Runs paired closed-loop simulation across controller strategies using identical initial learner profiles.

    Returns:
        (results_dict, events_df, learner_mastery_df)
    """
    rng = np.random.RandomState(seed)
    rel_engine = ReliabilityEngine(task_registry)

    cfg = controller_configs or {}
    fixed_cfg = cfg.get("fixed_difficulty", {})
    streak_cfg = cfg.get("streak_adaptive", {})
    target_cfg = cfg.get("target_adaptive", {})

    controllers_factories = {
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
        "target_adaptive": lambda: Grow21ModelBasedAdaptiveController(
            task_registry,
            target_success_prob=float(target_cfg.get("target_success_prob", 0.75)),
            initial_difficulty=float(target_cfg.get("initial_difficulty", 0.45)),
            gamma=float(target_cfg.get("gamma", 2.4)),
            nu_D=float(target_cfg.get("nu_D", 1.0)),
            deadband=float(target_cfg.get("deadband", 0.04)),
            max_step_change=float(target_cfg.get("max_step_change", 0.08)),
            min_difficulty=float(target_cfg.get("min_difficulty", 0.08)),
            max_difficulty=float(target_cfg.get("max_difficulty", 0.95)),
            cooldown_gap_hours=float(target_cfg.get("cooldown_gap_hours", 2.0)),
        ),
    }

    results: Dict[str, List[ClosedLoopRunResult]] = {c: [] for c in controllers_factories}
    all_events: List[Dict[str, Any]] = []
    learner_mastery_records: List[Dict[str, Any]] = []

    for l_id in range(num_learners):
        base_profile = generate_learner_profile(
            learner_id=l_id,
            skill_ids=task_registry.skill_ids,
            rng=rng,
            scenario_params=scenario_cfg,
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

            # Seeded per-learner stream to ensure fair common random numbers
            sim_rng = np.random.RandomState(seed * 1000 + l_id)

            for step in range(trials_per_learner):
                gid, diff, gap = controller.select_next_task(step)
                diag = getattr(controller, "last_diagnostic", {})
                ev = sim.step(
                    game_id=gid,
                    difficulty=diff,
                    time_gap_hours=gap,
                    rng=sim_rng,
                    scenario_params=scenario_cfg,
                )
                controller.observe_outcome(gid, ev.observed_outcome, diff)

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

                target_reachable_val = diag.get("target_reachable")
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
                    "estimated_reliability": ev.estimated_reliability,
                    "reachability_applicable": int(diag.get("reachability_applicable", False)),
                    "target_reachable": int(target_reachable_val) if target_reachable_val is not None else None,
                    "at_min_bound": int(diag.get("at_min", False)),
                    "at_max_bound": int(diag.get("at_max", False)),
                    "safety_intervention": int(diag.get("safety_intervention", False)),
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

            reach_prop = (
                round(reachable_counts / trials_per_learner, 4)
                if c_name == "target_adaptive"
                else None
            )

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
            )
            results[c_name].append(res)

    events_df = pd.DataFrame(all_events)
    mastery_df = pd.DataFrame(learner_mastery_records)
    return results, events_df, mastery_df
