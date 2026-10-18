"""Live learner service: API contract, state transitions, and safety behaviour.

All events are synthetic. These tests verify software behaviour of the runtime adapter around
the selected estimator; they say nothing about real-child accuracy.
"""

from __future__ import annotations

import copy
import pickle

import pytest
import yaml
from fastapi.testclient import TestClient

from grow21_simulation_benchmark.closed_loop import RobustAdaptiveFilterController
from grow21_simulation_benchmark.paths import DEFAULT_CONFIG_PATH
from grow21_simulation_benchmark.service.app import create_app
from grow21_simulation_benchmark.service.runtime import FRONTEND_GAME_KEYS, LearnerRuntime
from grow21_simulation_benchmark.tasks import TaskRegistry


class FakeClock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def load_config():
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def runtime(clock):
    return LearnerRuntime(load_config(), clock=clock)


@pytest.fixture
def client(runtime):
    return TestClient(create_app(runtime))


def next_req(child="kid1", session="s1", mode="play"):
    return {
        "schemaVersion": 1,
        "childId": child,
        "sessionId": session,
        "mode": mode,
        "sessionActiveMs": 0,
        "tasksCompletedThisSession": 0,
        "availability": {"microphone": True, "affectSignal": False},
    }


def event_for(task, *, child="kid1", session="s1", outcome="success", first=None, event_id=None, **extra):
    first = first or ("success" if outcome == "success" else "failure")
    attempts = [{"attemptIndex": 0, "scaffoldLevel": 0, "outcome": first, "latencyMs": 800.0}]
    if outcome == "success" and first != "success":
        attempts.append({"attemptIndex": 1, "scaffoldLevel": 0, "outcome": "success", "latencyMs": 1500.0})
    if outcome in ("abandoned", "timeout", "skipped"):
        attempts = []
    ev = {
        "schemaVersion": 1,
        "eventId": event_id or f"ev-{task['taskId']}",
        "childId": child,
        "sessionId": session,
        "gameId": task["gameId"],
        "skillId": task["skillId"],
        "taskId": task["taskId"],
        "taskIndexInSession": 0,
        "isBaseline": False,
        "timestampIso": "2026-10-18T10:00:00.000Z",
        "tPresentedMs": 10.0,
        "tFirstResponseMs": 810.0,
        "tCompletedMs": 900.0,
        "outcome": outcome,
        "latencyMs": 800.0,
        "durationMs": 890.0,
        "retries": max(0, len(attempts) - 1),
        "difficulty": task["D"],
        "scaffoldStartLevel": 0,
        "scaffoldFinalLevel": 0,
        "assisted": False,
        "scaffoldExhausted": False,
        "attempts": attempts,
        "gameConfig": {"pathWidthDp": 80.0, "shapeType": "zigzag"},
    }
    ev.update(extra)
    return ev


def get_task(client, **kw):
    r = client.post("/api/v1/task/next", json=next_req(**kw))
    assert r.status_code == 200, r.text
    body = r.json()
    assert "task" in body, body
    return body["task"]


def play(client, outcome="success", **kw):
    """Requests the next task and submits one event for it. Returns (task, ack)."""
    child = kw.pop("child", "kid1")
    session = kw.pop("session", "s1")
    task = get_task(client, child=child, session=session)
    r = client.post("/api/v1/telemetry/submit", json=event_for(task, child=child, session=session, outcome=outcome, **kw))
    assert r.status_code == 200, r.text
    return task, r.json()


def state(client, child="kid1"):
    r = client.get(f"/api/v1/learners/{child}/state")
    assert r.status_code == 200, r.text
    return r.json()


def frozen(runtime, child="kid1"):
    """Byte-level snapshot of a learner's full model state."""
    return pickle.dumps(runtime._learners[child])


# --------------------------------------------------------------------------- contract


def test_health_and_config_reflect_configured_games_and_skills(client):
    assert client.get("/health").json() == {"status": "ok", "schemaVersion": 1, "estimator": "robust_adaptive_filter"}
    cfg = client.get("/api/v1/config").json()
    reg = TaskRegistry(load_config())
    assert cfg["skills"] == reg.skill_ids and len(cfg["skills"]) == 6
    assert set(cfg["games"]) == set(FRONTEND_GAME_KEYS) and len(cfg["games"]) == 9
    for key, g in cfg["games"].items():
        gdef = reg.games[FRONTEND_GAME_KEYS[key]]
        assert (g["gameId"], g["primarySkill"], g["secondarySkill"]) == (gdef.game_id, gdef.primary_skill, gdef.secondary_skill)
    assert cfg["difficulty"]["min"] == 0.08 and cfg["difficulty"]["max"] == 0.95
    assert cfg["safety"] == {"maxConsecutiveFailures": 3, "maxSessionTrials": 15, "cooldownSec": 7200}


def test_task_directive_shape_and_bounds(client):
    seen = []
    for _ in range(9):
        task, _ = play(client)
        assert set(task) == {"taskId", "gameId", "skillId", "D"}
        assert 0.08 <= task["D"] <= 0.95
        assert task["skillId"] == TaskRegistry(load_config()).games[FRONTEND_GAME_KEYS[task["gameId"]]].primary_skill
        seen.append(task["gameId"])
    assert sorted(seen) == sorted(FRONTEND_GAME_KEYS)  # configured round-robin covers all nine games


@pytest.mark.parametrize(
    "mutate,status,code",
    [
        (lambda e: e.update(gameId="chess"), 422, "invalid_request"),
        (lambda e: e.update(outcome="won"), 422, "invalid_request"),
        (lambda e: e.update(difficulty=1.5), 422, "invalid_request"),
        (lambda e: e.update(difficulty=float("nan")), 422, None),
        (lambda e: e.update(unexpectedField=1), 422, "invalid_request"),
        (lambda e: e.pop("attempts"), 422, "invalid_request"),
        (lambda e: e.update(skillId="phonological_awareness"), 422, "task_mismatch"),
        (lambda e: e.update(gameId="echo"), 422, "task_mismatch"),
        (lambda e: e.update(difficulty=0.9), 422, "task_mismatch"),
        (lambda e: e.update(taskId="not-issued"), 409, "unknown_task"),
        (lambda e: e.update(sessionId="other"), 409, "stale_session"),
        (lambda e: e.update(isBaseline=True), 422, "unsupported_mode"),
        (lambda e: e.update(scaffoldStartLevel=1), 422, "invalid_event"),
        (lambda e: e.update(attempts=[]), 422, "invalid_event"),
        (lambda e: e.update(outcome="failure"), 422, "invalid_event"),  # attempts contain a success
        (lambda e: e.update(reliability=1.2), 422, "invalid_request"),
    ],
)
def test_rejected_events_do_not_mutate_state(client, runtime, mutate, status, code):
    task = get_task(client)
    before = frozen(runtime)
    ev = event_for(task)
    mutate(ev)
    if code is None:  # NaN cannot be sent as strict JSON; post raw text
        body = '{"difficulty": NaN}'
        r = client.post("/api/v1/telemetry/submit", content=body, headers={"Content-Type": "application/json"})
    else:
        r = client.post("/api/v1/telemetry/submit", json=ev)
    assert r.status_code == status, r.text
    if code:
        assert r.json()["error"]["code"] == code
    assert frozen(runtime) == before
    # The issued task is still open and can be completed normally afterwards.
    ok = client.post("/api/v1/telemetry/submit", json=event_for(task))
    assert ok.status_code == 200 and ok.json()["observationApplied"] is True


def test_unknown_learner_and_baseline_mode(client):
    r = client.post("/api/v1/telemetry/submit", json=event_for({"taskId": "x", "gameId": "trace", "skillId": "fine_motor", "D": 0.4}, child="ghost"))
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_learner"
    assert client.get("/api/v1/learners/ghost/state").status_code == 404
    r = client.post("/api/v1/task/next", json=next_req(mode="baseline"))
    assert r.status_code == 422 and r.json()["error"]["code"] == "unsupported_mode"
    r = client.post("/api/v1/task/next", json=next_req(child="../etc"))
    assert r.status_code == 422


# --------------------------------------------------------------------------- idempotency


def test_duplicate_event_and_retry_semantics(client, runtime):
    task = get_task(client)
    ev = event_for(task)
    first = client.post("/api/v1/telemetry/submit", json=ev).json()
    assert first == {"ack": True, "duplicate": False, "observationApplied": True}
    after_first = frozen(runtime)

    again = client.post("/api/v1/telemetry/submit", json=ev).json()
    assert again == {"ack": True, "duplicate": True, "observationApplied": True}
    assert frozen(runtime) == after_first
    assert state(client)["sessionTrialsObserved"] == 1  # counted once

    changed = dict(ev, outcome="failure", attempts=[{"attemptIndex": 0, "scaffoldLevel": 0, "outcome": "failure"}])
    r = client.post("/api/v1/telemetry/submit", json=changed)
    assert r.status_code == 409 and r.json()["error"]["code"] == "event_id_conflict"

    r = client.post("/api/v1/telemetry/submit", json=dict(ev, eventId="other-event"))
    assert r.status_code == 409 and r.json()["error"]["code"] == "task_already_observed"
    assert frozen(runtime) == after_first


def test_next_task_is_idempotent_while_a_task_is_outstanding(client, runtime, clock):
    t1 = get_task(client)
    snapshot = frozen(runtime)
    clock.advance(30)
    t2 = get_task(client)
    assert t1 == t2
    assert frozen(runtime) == snapshot  # no extra decay, step, or task counter change


# --------------------------------------------------------------------------- model semantics


def _fresh(clock_t=1_800_000_000.0):
    clk = FakeClock(clock_t)
    rt = LearnerRuntime(load_config(), clock=clk)
    return TestClient(create_app(rt)), rt, clk


def test_shared_skill_evidence_carries_across_games():
    # trace_glow and bubble_pop_brigade share the primary skill fine_motor.
    results = {}
    for outcome in ("success", "failure"):
        client, _, _ = _fresh()
        trace, _ = play(client, outcome=outcome)
        assert trace["gameId"] == "trace"
        bubble = get_task(client)
        assert bubble["gameId"] == "bubble" and bubble["skillId"] == trace["skillId"] == "fine_motor"
        st = state(client)
        results[outcome] = (st["mastery"]["fine_motor"], st["lastDecision"]["predicted_clean_prob"])
    assert results["success"][0] > results["failure"][0]
    assert results["success"][1] > results["failure"][1]


def test_observation_only_updates_the_games_configured_skills(client, runtime):
    for _ in range(3):  # advance round-robin to echo_safari
        play(client)
    echo = get_task(client)
    assert echo["gameId"] == "echo"
    before = dict(runtime._learners["kid1"].controller.estimated_mastery)
    client.post("/api/v1/telemetry/submit", json=event_for(echo))
    after = runtime._learners["kid1"].controller.estimated_mastery
    changed = {s for s in before if abs(after[s] - before[s]) > 1e-12}
    assert changed == {"phonological_awareness", "working_memory"}  # primary + configured secondary


def test_secondary_transfer_is_success_modulated():
    sec = {}
    for outcome in ("success", "failure"):
        client, rt, _ = _fresh()
        task = get_task(client)
        before = rt._learners["kid1"].controller.estimated_mastery["visual_discrimination"]
        client.post("/api/v1/telemetry/submit", json=event_for(task, outcome=outcome))
        sec[outcome] = rt._learners["kid1"].controller.estimated_mastery["visual_discrimination"] - before
    assert sec["success"] > 0
    assert sec["success"] > sec["failure"] >= -1e-12


def test_missing_reliability_uses_convention_and_low_reliability_dampens_update():
    deltas = {}
    for label, extra in {"missing": {}, "explicit_one": {"reliability": 1.0}, "low": {"reliability": 0.3}}.items():
        client, rt, _ = _fresh()
        task = get_task(client)
        before = rt._learners["kid1"].controller.estimated_mastery["fine_motor"]
        ack = client.post("/api/v1/telemetry/submit", json=event_for(task, outcome="failure", **extra)).json()
        assert ack["observationApplied"] is True
        deltas[label] = rt._learners["kid1"].controller.estimated_mastery["fine_motor"] - before
    assert deltas["missing"] == pytest.approx(deltas["explicit_one"], abs=1e-12)
    assert deltas["missing"] < 0
    assert abs(deltas["low"]) < abs(deltas["missing"])


@pytest.mark.parametrize("outcome", ["abandoned", "timeout", "skipped"])
def test_non_observations_close_the_task_without_touching_the_model(client, runtime, outcome):
    task = get_task(client)
    ctrl_before = pickle.dumps(runtime._learners["kid1"].controller)
    ack = client.post("/api/v1/telemetry/submit", json=event_for(task, outcome=outcome)).json()
    assert ack == {"ack": True, "duplicate": False, "observationApplied": False}
    assert pickle.dumps(runtime._learners["kid1"].controller) == ctrl_before
    assert state(client)["outstandingTaskId"] is None
    assert get_task(client)["taskId"] != task["taskId"]


def test_first_attempt_defines_the_observation():
    # A task eventually solved after an unassisted first-attempt failure is evidence of failure at D.
    client, rt, _ = _fresh()
    task = get_task(client)
    client.post("/api/v1/telemetry/submit", json=event_for(task, outcome="success", first="failure"))
    assert rt._learners["kid1"].controller.consecutive_failures == 1
    assert rt._learners["kid1"].observations[-1]["observedResponse"] == 0


def test_forgetting_applies_with_real_elapsed_time_only(client, runtime, clock):
    play(client)
    task = get_task(client)
    client.post("/api/v1/telemetry/submit", json=event_for(task))
    m_before = dict(state(client)["mastery"])

    clock.advance(7 * 24 * 3600)
    get_task(client, session="s2")
    st = state(client)
    assert st["lastDecision"]["elapsedHours"] == pytest.approx(7 * 24, rel=1e-6)
    for s, v in m_before.items():
        assert 0.05 <= st["mastery"][s] < v, s


def test_live_selection_matches_simulator_selection_path():
    reg = TaskRegistry(load_config())
    sim = RobustAdaptiveFilterController(reg)
    live = RobustAdaptiveFilterController(reg)
    outcomes = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1]  # includes a 3-failure safety trigger
    for step, y in enumerate(outcomes):
        gid, d_sim, gap = sim.select_next_task(step)
        d_live = live.select_difficulty_for_elapsed_time(gid, gap)
        assert d_live == d_sim and live.last_diagnostic == sim.last_diagnostic
        assert live.estimated_mastery == sim.estimated_mastery
        sim.observe_outcome(gid, y, d_sim, estimated_reliability=0.8)
        live.observe_outcome(gid, y, d_live, estimated_reliability=0.8)
    assert live.estimated_mastery == sim.estimated_mastery


# --------------------------------------------------------------------------- safety, breaks, sessions


def test_consecutive_failures_trigger_cooldown_break_then_safety_drop_and_recovery(client, runtime, clock):
    for _ in range(3):
        play(client, outcome="failure")
    r = client.post("/api/v1/task/next", json=next_req()).json()
    assert r == {"schemaVersion": 1, "break": {"reason": "frustration_cooldown", "lengthSec": 7200}}

    clock.advance(3600)  # resuming early keeps the break, with the remaining time
    r = client.post("/api/v1/task/next", json=next_req()).json()
    assert r["break"]["lengthSec"] == 3600
    assert state(client)["outstandingTaskId"] is None

    clock.advance(3600)
    untouched_d = runtime._learners["kid1"].game_difficulty["category_sorting"]
    task = get_task(client)
    st = state(client)
    assert st["lastDecision"]["safety_intervention"] is True
    assert st["lastDecision"]["determination_rule"] == "safety_cooldown"
    assert st["consecutiveFailures"] == 0
    assert task["gameId"] == "echo"  # 4th game in the configured rotation, first play at initial D
    assert task["D"] == pytest.approx(max(0.08, 0.45 - 0.25), abs=1e-3)
    assert runtime._learners["kid1"].game_difficulty["category_sorting"] == untouched_d == 0.45

    client.post("/api/v1/telemetry/submit", json=event_for(task, outcome="success"))
    assert state(client)["consecutiveFailures"] == 0
    assert "task" in client.post("/api/v1/task/next", json=next_req()).json()


def test_session_trial_limit_ends_session_and_new_session_resumes(client, runtime):
    for _ in range(15):
        play(client)
    r = client.post("/api/v1/task/next", json=next_req()).json()
    assert r == {"schemaVersion": 1, "endSession": {"reason": "session_trial_limit"}}
    assert client.post("/api/v1/task/next", json=next_req()).json() == r  # stays ended

    mastery = state(client)["mastery"]
    task = get_task(client, session="s2")
    st = state(client)
    assert st["sessionId"] == "s2" and st["sessionTrialsObserved"] == 0 and not st["sessionEnded"]
    assert st["mastery"] == pytest.approx(mastery, abs=1e-3)  # shared state carries across sessions

    r = client.post("/api/v1/task/next", json=next_req(session="s1"))
    assert r.status_code == 409 and r.json()["error"]["code"] == "session_closed"
    r = client.post("/api/v1/telemetry/submit", json=event_for(task, session="s1"))
    assert r.status_code == 409


def test_restoring_the_same_session_returns_the_outstanding_task(client):
    play(client)
    t = get_task(client)
    # App remount with the same identifiers resumes rather than resetting.
    assert get_task(client) == t
    assert state(client)["sessionTrialsObserved"] == 1


def test_learners_are_isolated():
    solo_client, solo_rt, _ = _fresh()
    for outcome in ("success", "failure", "success", "success"):
        play(solo_client, outcome=outcome, child="b")

    client, rt, _ = _fresh()
    for a_outcome, b_outcome in zip(("failure", "failure", "success", "failure"), ("success", "failure", "success", "success")):
        play(client, outcome=a_outcome, child="a")
        play(client, outcome=b_outcome, child="b")
    assert state(client, "b")["mastery"] == state(solo_client, "b")["mastery"]
    assert state(client, "b")["gameDifficulty"] == state(solo_client, "b")["gameDifficulty"]
    assert state(client, "a")["mastery"] != state(client, "b")["mastery"]

    task_b = get_task(client, child="b")
    r = client.post("/api/v1/telemetry/submit", json=event_for(task_b, child="a"))
    assert r.status_code == 409  # a's events cannot close b's task


def test_state_persists_across_service_restart(tmp_path, clock):
    rt1 = LearnerRuntime(load_config(), state_dir=tmp_path, clock=clock)
    c1 = TestClient(create_app(rt1))
    play(c1)
    play(c1, outcome="failure")
    outstanding = get_task(c1)
    snapshot = state(c1)

    rt2 = LearnerRuntime(load_config(), state_dir=tmp_path, clock=clock)
    c2 = TestClient(create_app(rt2))
    assert state(c2) == snapshot
    assert get_task(c2) == outstanding
    assert c2.post("/api/v1/telemetry/submit", json=event_for(outstanding)).json()["observationApplied"]

    changed = load_config()
    changed["controllers"]["target_adaptive"]["max_step_change"] = 0.05
    c3 = TestClient(create_app(LearnerRuntime(changed, state_dir=tmp_path, clock=clock)))
    r = c3.get("/api/v1/learners/kid1/state")
    assert r.status_code == 409 and r.json()["error"]["code"] == "state_config_mismatch"


def test_runtime_is_deterministic_for_identical_inputs():
    outs = []
    for _ in range(2):
        client, _, clk = _fresh()
        for i, o in enumerate(("success", "failure", "success", "failure", "failure", "success")):
            clk.advance(45 + i)
            play(client, outcome=o, reliability=0.9)
        st = state(client)
        outs.append((st["mastery"], st["gameDifficulty"]))
    assert outs[0] == outs[1]
