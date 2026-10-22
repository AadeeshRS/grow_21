"""Live learner runtime: the single owner of per-learner model state between requests.

Wraps the selected estimator/controller (`RobustAdaptiveFilterController`) for live game
events. Nothing from the simulator is used here: no virtual learner, no reliability sampler,
no latent mastery, no clean outcomes and no scenario labels. The controller only receives
observed outcomes, optional observed reliability, and real elapsed wall-clock time.

Protocol invariants (see docs/integration/FRONTEND_BACKEND_API.md):
- At most one task is outstanding per learner; repeated next-task requests return it again.
- Each issued taskId yields at most one model observation; eventId retries are idempotent.
- Every request is validated before any state changes, and changes are applied to a copy
  that replaces the stored state only when the whole operation succeeds.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

from ..closed_loop import RobustAdaptiveFilterController
from ..paths import DEFAULT_CONFIG_PATH
from ..tasks import TaskRegistry

SCHEMA_VERSION = 1

# Game keys used by the React Native module (src/types.ts GameId) -> configured game IDs.
FRONTEND_GAME_KEYS: Dict[str, str] = {
    "trace": "trace_glow",
    "bubble": "bubble_pop_brigade",
    "block": "block_stacker",
    "echo": "echo_safari",
    "sentence": "picture_sentence_strips",
    "audio": "audio_search",
    "sort": "category_sorting",
    "flip": "visual_flip_match",
    "pattern": "pattern_train",
}

# Task outcomes that close a task without providing evidence about the learner.
NO_OBSERVATION_OUTCOMES = ("abandoned", "timeout", "skipped")


class RuntimeRejection(Exception):
    """A request the runtime refuses; the learner state is left unchanged."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass
class LearnerState:
    learner_id: str
    config_digest: str
    controller: RobustAdaptiveFilterController
    game_difficulty: Dict[str, float]  # configured game id -> current difficulty D
    clock_hours: float  # wall-clock hours at which forgetting was last applied
    tasks_issued: int = 0
    session_id: Optional[str] = None
    closed_sessions: List[str] = field(default_factory=list)
    session_ended: bool = False
    last_observation_at: Optional[float] = None  # epoch seconds
    outstanding: Optional[Dict[str, Any]] = None
    observed_task_ids: Set[str] = field(default_factory=set)
    processed_events: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    last_decision: Optional[Dict[str, Any]] = None


class LearnerRuntime:
    def __init__(
        self,
        config: Dict[str, Any],
        state_dir: Optional[Path] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.config = config
        self.registry = TaskRegistry(config)
        if sorted(FRONTEND_GAME_KEYS.values()) != sorted(self.registry.game_ids):
            raise ValueError("FRONTEND_GAME_KEYS does not cover exactly the configured games")
        self.game_keys = {gid: key for key, gid in FRONTEND_GAME_KEYS.items()}
        self.controller_cfg = config["controllers"]["target_adaptive"]
        self.config_digest = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.state_dir = Path(state_dir) if state_dir else None
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._learners: Dict[str, LearnerState] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> "LearnerRuntime":
        config_path = Path(os.environ.get("GROW21_CONFIG", str(DEFAULT_CONFIG_PATH)))
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if os.environ.get("GROW21_DISABLE_COOLDOWN") == "1":
            # Demo switch: no rest pause after a tricky run. The safety difficulty drop still applies.
            config["controllers"]["target_adaptive"]["cooldown_gap_hours"] = 0.0
        state_dir = os.environ.get("GROW21_STATE_DIR") or None
        return cls(config, state_dir=Path(state_dir) if state_dir else None)

    # ------------------------------------------------------------------ public API

    def describe(self) -> Dict[str, Any]:
        c = self.controller_cfg
        return {
            "schemaVersion": SCHEMA_VERSION,
            "estimator": "robust_adaptive_filter",
            "skills": list(self.registry.skill_ids),
            "games": {
                key: {
                    "gameId": gid,
                    "name": self.registry.games[gid].name,
                    "primarySkill": self.registry.games[gid].primary_skill,
                    "secondarySkill": self.registry.games[gid].secondary_skill,
                }
                for key, gid in FRONTEND_GAME_KEYS.items()
            },
            "difficulty": {
                "min": float(c["min_difficulty"]),
                "max": float(c["max_difficulty"]),
                "initial": float(c["initial_difficulty"]),
                "maxStep": float(c["max_step_change"]),
                "deadband": float(c["deadband"]),
                "targetCleanSuccessProb": float(c["target_success_prob"]),
            },
            "safety": {
                "maxConsecutiveFailures": int(c["max_consecutive_failures"]),
                "maxSessionTrials": int(c["max_session_trials"]),
                "cooldownSec": int(round(float(c["cooldown_gap_hours"]) * 3600)),
            },
        }

    def next_task(self, child_id: str, session_id: str, mode: str) -> Dict[str, Any]:
        if mode != "play":
            raise RuntimeRejection(
                422, "unsupported_mode", "Baseline mode has no backend implementation; use mode 'play'."
            )
        with self._lock:
            learner = self._load(child_id)
            if learner is None:
                learner = self._new_learner(child_id)
            working = copy.deepcopy(learner)
            directive = self._next_task(working, session_id)
            self._commit(working)
            return directive

    def submit_telemetry(self, event: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            learner = self._load(event["childId"])
            if learner is None:
                raise RuntimeRejection(404, "unknown_learner", "No task was issued to this learner.")

            digest = hashlib.sha256(json.dumps(event, sort_keys=True).encode("utf-8")).hexdigest()
            prior = learner.processed_events.get(event["eventId"])
            if prior is not None:
                if prior["digest"] != digest:
                    raise RuntimeRejection(
                        409, "event_id_conflict", "eventId was already used for a different event."
                    )
                return {**prior["response"], "duplicate": True}

            task = self._validate_event(learner, event)
            y_obs, reason = self._observed_outcome(event)

            working = copy.deepcopy(learner)
            now = self.clock()
            if y_obs is not None:
                working.controller.observe_outcome(
                    task["gameId"], y_obs, task["D"], estimated_reliability=event.get("reliability")
                )
                working.last_observation_at = now
            response = {"ack": True, "duplicate": False, "observationApplied": y_obs is not None}
            working.observed_task_ids.add(task["taskId"])
            working.outstanding = None
            working.processed_events[event["eventId"]] = {"digest": digest, "response": response}
            working.observations.append(
                {
                    "eventId": event["eventId"],
                    "taskId": task["taskId"],
                    "sessionId": task["sessionId"],
                    "gameId": task["gameId"],
                    "difficulty": task["D"],
                    "observedResponse": y_obs,
                    "reason": reason,
                    "reliability": event.get("reliability"),
                    "receivedAt": now,
                    "clientTimestamp": event["timestampIso"],
                }
            )
            self._commit(working)
            return response

    def learner_state(self, child_id: str) -> Dict[str, Any]:
        with self._lock:
            learner = self._load(child_id)
            if learner is None:
                raise RuntimeRejection(404, "unknown_learner", "Unknown learner.")
            ctrl = learner.controller
            return {
                "schemaVersion": SCHEMA_VERSION,
                "childId": learner.learner_id,
                "mastery": {s: round(float(v), 4) for s, v in ctrl.estimated_mastery.items()},
                "gameDifficulty": {
                    self.game_keys[gid]: round(float(d), 4) for gid, d in learner.game_difficulty.items()
                },
                "sessionId": learner.session_id,
                "sessionEnded": learner.session_ended,
                "sessionTrialsObserved": int(ctrl.session_trial_count),
                "consecutiveFailures": int(ctrl.consecutive_failures),
                "tasksIssued": learner.tasks_issued,
                "observationsRecorded": len(learner.observations),
                "outstandingTaskId": learner.outstanding["taskId"] if learner.outstanding else None,
                "lastDecision": learner.last_decision,
            }

    # ------------------------------------------------------------------ internals

    def _new_learner(self, child_id: str) -> LearnerState:
        c = self.controller_cfg
        controller = RobustAdaptiveFilterController(
            self.registry,
            target_success_prob=float(c["target_success_prob"]),
            initial_difficulty=float(c["initial_difficulty"]),
            deadband=float(c["deadband"]),
            max_step_change=float(c["max_step_change"]),
            min_difficulty=float(c["min_difficulty"]),
            max_difficulty=float(c["max_difficulty"]),
            fatigue_sensitivity_prior=float(c["fatigue_sensitivity_prior"]),
            forgetting_rate_prior=float(c["forgetting_rate_prior"]),
            learning_rate_prior=float(c.get("learning_rate_prior", 0.08)),
            secondary_transfer_ratio=float(c.get("secondary_transfer_ratio", 0.30)),
            max_consecutive_failures=int(c["max_consecutive_failures"]),
            max_session_trials=int(c["max_session_trials"]),
            cooldown_gap_hours=float(c["cooldown_gap_hours"]),
        )
        return LearnerState(
            learner_id=child_id,
            config_digest=self.config_digest,
            controller=controller,
            game_difficulty={gid: float(c["initial_difficulty"]) for gid in self.registry.game_ids},
            clock_hours=self.clock() / 3600.0,
        )

    def _next_task(self, learner: LearnerState, session_id: str) -> Dict[str, Any]:
        now = self.clock()
        self._enter_session(learner, session_id)
        ctrl = learner.controller

        if learner.outstanding is not None:
            return self._task_directive(learner.outstanding)
        if learner.session_ended:
            return {"schemaVersion": SCHEMA_VERSION, "endSession": {"reason": "session_trial_limit"}}
        if ctrl.session_trial_count >= ctrl.max_session_trials:
            learner.session_ended = True
            return {"schemaVersion": SCHEMA_VERSION, "endSession": {"reason": "session_trial_limit"}}
        if ctrl.consecutive_failures >= ctrl.max_consecutive_failures:
            # The simulator models the frustration cooldown as a rest gap of cooldown_gap_hours
            # before the safety-reduced task; live, the rest is measured from the last observation.
            rested = now - (learner.last_observation_at or now)
            remaining = ctrl.cooldown_gap_hours * 3600.0 - rested
            if remaining > 0:
                return {
                    "schemaVersion": SCHEMA_VERSION,
                    "break": {"reason": "frustration_cooldown", "lengthSec": int(math.ceil(remaining))},
                }

        game_id = self.registry.game_ids[learner.tasks_issued % len(self.registry.game_ids)]
        now_hours = now / 3600.0
        elapsed_hours = max(0.0, now_hours - learner.clock_hours)
        learner.clock_hours = max(learner.clock_hours, now_hours)

        # Difficulty is kept per game: each game has its own item parameters, so one game's D
        # is never carried into another game (Integrated Mathematical System v1.1, section 8).
        ctrl.current_difficulty = learner.game_difficulty[game_id]
        difficulty = ctrl.select_difficulty_for_elapsed_time(game_id, elapsed_hours)
        learner.game_difficulty[game_id] = float(ctrl.current_difficulty)

        learner.tasks_issued += 1
        learner.outstanding = {
            "taskId": uuid.uuid4().hex,
            "sessionId": session_id,
            "gameId": game_id,
            "gameKey": self.game_keys[game_id],
            "skillId": self.registry.games[game_id].primary_skill,
            "D": float(difficulty),
        }
        learner.last_decision = {
            "taskId": learner.outstanding["taskId"],
            "gameId": self.game_keys[game_id],
            "elapsedHours": round(elapsed_hours, 6),
            **ctrl.last_diagnostic,
        }
        return self._task_directive(learner.outstanding)

    def _enter_session(self, learner: LearnerState, session_id: str) -> None:
        if learner.session_id == session_id:
            return
        if session_id in learner.closed_sessions:
            raise RuntimeRejection(409, "session_closed", "This session was superseded by a newer session.")
        if learner.session_id is not None:
            learner.closed_sessions.append(learner.session_id)
        learner.session_id = session_id
        learner.session_ended = False
        learner.outstanding = None  # an unplayed task from the previous session is discarded
        learner.controller.session_trial_count = 0

    @staticmethod
    def _task_directive(task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "task": {
                "taskId": task["taskId"],
                "gameId": task["gameKey"],
                "skillId": task["skillId"],
                "D": task["D"],
            },
        }

    def _validate_event(self, learner: LearnerState, event: Dict[str, Any]) -> Dict[str, Any]:
        if event["sessionId"] != learner.session_id:
            raise RuntimeRejection(409, "stale_session", "Event does not belong to the learner's current session.")
        if event["taskId"] in learner.observed_task_ids:
            raise RuntimeRejection(409, "task_already_observed", "This task already has a recorded event.")
        task = learner.outstanding
        if task is None or task["taskId"] != event["taskId"]:
            raise RuntimeRejection(409, "unknown_task", "taskId does not match the task issued to this learner.")
        if event["gameId"] != task["gameKey"] or event["skillId"] != task["skillId"]:
            raise RuntimeRejection(422, "task_mismatch", "gameId/skillId differ from the issued task.")
        if abs(float(event["difficulty"]) - task["D"]) > 1e-9:
            raise RuntimeRejection(422, "task_mismatch", "difficulty differs from the issued task.")
        if event["isBaseline"]:
            raise RuntimeRejection(422, "unsupported_mode", "Baseline events have no backend implementation.")
        if event["scaffoldStartLevel"] != 0:
            raise RuntimeRejection(422, "invalid_event", "The backend did not request a starting scaffold level.")
        attempts = event["attempts"]
        if event["outcome"] not in NO_OBSERVATION_OUTCOMES:
            if not attempts:
                raise RuntimeRejection(422, "invalid_event", "A success/failure event needs at least one attempt.")
            any_success = any(a["outcome"] == "success" for a in attempts)
            if (event["outcome"] == "success") != any_success:
                raise RuntimeRejection(422, "invalid_event", "Task outcome is inconsistent with its attempts.")
        if attempts and attempts[0]["scaffoldLevel"] != event["scaffoldStartLevel"]:
            raise RuntimeRejection(422, "invalid_event", "First attempt must be at the starting scaffold level.")
        return task

    @staticmethod
    def _observed_outcome(event: Dict[str, Any]) -> Tuple[Optional[int], str]:
        """Maps a task event to the estimator's binary observed response y_obs.

        The canonical outcome model predicts unassisted success at the presented difficulty,
        so y_obs is the first attempt (made before any in-task scaffold escalation).
        Abandoned/timeout/skipped tasks and stalled first attempts carry no observation.
        """
        if event["outcome"] in NO_OBSERVATION_OUTCOMES:
            return None, f"no_observation_{event['outcome']}"
        first = event["attempts"][0]["outcome"]
        if first == "success":
            return 1, "first_attempt_success"
        if first == "failure":
            return 0, "first_attempt_failure"
        return None, "no_observation_stalled"

    def _path(self, child_id: str) -> Path:
        assert self.state_dir is not None
        return self.state_dir / f"{child_id}.pkl"

    def _load(self, child_id: str) -> Optional[LearnerState]:
        learner = self._learners.get(child_id)
        if learner is None and self.state_dir and self._path(child_id).exists():
            with open(self._path(child_id), "rb") as f:
                learner = pickle.load(f)
            self._learners[child_id] = learner
        if learner is not None and learner.config_digest != self.config_digest:
            raise RuntimeRejection(
                409, "state_config_mismatch", "Stored learner state was created with a different model config."
            )
        return learner

    def _commit(self, learner: LearnerState) -> None:
        if self.state_dir:
            tmp = self._path(learner.learner_id).with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(learner, f)
            os.replace(tmp, self._path(learner.learner_id))
        self._learners[learner.learner_id] = learner
