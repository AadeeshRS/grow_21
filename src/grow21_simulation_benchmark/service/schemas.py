"""Wire schemas for the Grow21 learner service.

These mirror the React Native module's `src/types.ts` (NextTaskRequest, TelemetryEvent,
TaskDirective). Unknown fields are rejected so the two definitions cannot drift silently.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import Annotated

GameKey = Literal["trace", "bubble", "block", "echo", "sentence", "audio", "sort", "flip", "pattern"]
Outcome = Literal["success", "failure", "abandoned", "timeout", "skipped"]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
NonNegative = Annotated[float, Field(ge=0)]
ScaffoldLevel = Annotated[int, Field(ge=0, le=3)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Availability(_Strict):
    microphone: bool
    affectSignal: bool


class NextTaskRequest(_Strict):
    schemaVersion: Literal[1]
    childId: Identifier
    sessionId: Identifier
    mode: Literal["baseline", "play"]
    sessionActiveMs: NonNegative
    tasksCompletedThisSession: Annotated[int, Field(ge=0)]
    lastTaskId: Optional[str] = None
    lastOutcome: Optional[Outcome] = None
    availability: Availability


class AttemptRecord(_Strict):
    attemptIndex: Annotated[int, Field(ge=0)]
    scaffoldLevel: ScaffoldLevel
    outcome: Literal["success", "failure", "stalled"]
    latencyMs: Optional[NonNegative] = None
    durationMs: Optional[NonNegative] = None
    motor: Optional[Dict[str, Any]] = None
    speech: Optional[Dict[str, Any]] = None
    task: Optional[Dict[str, Any]] = None


class Affect(_Strict):
    value: Annotated[float, Field(ge=0, le=1)]


class TelemetryEvent(_Strict):
    schemaVersion: Literal[1]
    eventId: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    childId: Identifier
    sessionId: Identifier
    gameId: GameKey
    skillId: str
    taskId: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    taskIndexInSession: Annotated[int, Field(ge=0)]
    isBaseline: bool
    baselineSubtask: Optional[str] = None
    timestampIso: str
    tPresentedMs: float
    tFirstResponseMs: Optional[float] = None
    tCompletedMs: Optional[float] = None
    outcome: Outcome
    latencyMs: Optional[NonNegative] = None
    durationMs: Optional[NonNegative] = None
    retries: Annotated[int, Field(ge=0)]
    difficulty: Annotated[float, Field(ge=0, le=1)]
    scaffoldStartLevel: ScaffoldLevel
    scaffoldFinalLevel: ScaffoldLevel
    assisted: bool
    scaffoldExhausted: bool
    attempts: List[AttemptRecord]
    gameConfig: Dict[str, Union[float, str, bool]]
    confirmationTrial: Optional[bool] = None
    motor: Optional[Dict[str, Any]] = None
    speech: Optional[Dict[str, Any]] = None
    affect: Optional[Affect] = None
    task: Optional[Dict[str, Any]] = None
    # Observed reliability rho_hat in [0, 1], only when a game has a real uncertainty estimate.
    # Omitted -> the controller's documented convention (no explicit uncertainty model: rho = 1).
    reliability: Optional[Annotated[float, Field(ge=0, le=1)]] = None


class DirectiveTask(_Strict):
    taskId: str
    gameId: GameKey
    skillId: str
    D: Annotated[float, Field(ge=0, le=1)]


class DirectiveBreak(_Strict):
    reason: str
    lengthSec: Annotated[int, Field(gt=0)]


class DirectiveEnd(_Strict):
    reason: str


class TaskDirective(_Strict):
    schemaVersion: Literal[1]
    task: Optional[DirectiveTask] = None
    # `break` is a Python keyword; the wire name is kept via alias.
    break_: Optional[DirectiveBreak] = Field(default=None, alias="break")
    endSession: Optional[DirectiveEnd] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TelemetryAck(_Strict):
    ack: Literal[True]
    duplicate: bool
    observationApplied: bool
