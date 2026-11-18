"""End-to-end: does a motor game's telemetry actually reach the controller as
a trust weight, and is the default behaviour unchanged when the flag is off?"""

import pytest
from fastapi.testclient import TestClient

from grow21_simulation_benchmark.service.app import create_app
from grow21_simulation_benchmark.service.runtime import LearnerRuntime

from .test_service_api import FakeClock, event_for, load_config, next_req

SLIPPY_BLOCKS = {"releaseMisses": 5, "wrongTarget": 0, "orderErrors": 0,
                 "drags": 5, "taps": 0, "blocks": 3, "toleranceDp": 40}


def _last_observation(runtime, child="kid1"):
    """The public state endpoint only reports a count, so read the recorded
    observation off the stored learner."""
    return runtime._load(child).observations[-1]


def _client(motor_reliability):
    runtime = LearnerRuntime(load_config(), clock=FakeClock(),
                             motor_reliability=motor_reliability)
    return TestClient(create_app(runtime)), runtime


def _play_until_block_stacker(client, max_tries=30):
    """The controller picks the game, so keep asking until it offers the one
    with a motor channel we want.

    A fresh session each time, because after a set number of trials the
    controller returns a rest directive instead of a task and there is nothing
    to submit against.
    """
    for i in range(max_tries):
        r = client.post("/api/v1/task/next", json=next_req(session="s%d" % i))
        assert r.status_code == 200, r.text
        body = r.json()
        if "task" not in body:
            continue                      # rest or break directive, try again
        task = body["task"]
        # the directive carries the frontend key ("block"); the stored task
        # the runtime matches against uses the internal id ("block_stacker")
        if task["gameId"] == "block":
            return task, "s%d" % i
        client.post("/api/v1/telemetry/submit",
                    json=event_for(task, session="s%d" % i, outcome="success"))
    raise AssertionError("controller never offered the block game in %d tries" % max_tries)


def test_flag_off_leaves_reliability_untouched():
    client, runtime = _client(False)
    task, session = _play_until_block_stacker(client)
    ev = event_for(task, session=session, outcome="failure", motor=SLIPPY_BLOCKS)
    assert client.post("/api/v1/telemetry/submit", json=ev).status_code == 200

    obs = _last_observation(runtime)
    assert obs["reliability"] is None
    assert obs["reliabilitySource"] == "none"


def test_flag_on_derives_reliability_from_motor_telemetry():
    client, runtime = _client(True)
    task, session = _play_until_block_stacker(client)
    ev = event_for(task, session=session, outcome="failure", motor=SLIPPY_BLOCKS)
    assert client.post("/api/v1/telemetry/submit", json=ev).status_code == 200

    obs = _last_observation(runtime)
    assert obs["reliabilitySource"] == "motor_model"
    assert 0.0 < obs["reliability"] < 0.5, "all slips, so this should barely count"


def test_a_value_sent_by_the_app_always_wins():
    client, runtime = _client(True)
    task, session = _play_until_block_stacker(client)
    ev = event_for(task, session=session, outcome="failure",
                   motor=SLIPPY_BLOCKS, reliability=0.9)
    assert client.post("/api/v1/telemetry/submit", json=ev).status_code == 200

    obs = _last_observation(runtime)
    assert obs["reliability"] == 0.9
    assert obs["reliabilitySource"] == "client"
