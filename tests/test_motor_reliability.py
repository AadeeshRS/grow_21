"""Tests for deriving observation reliability from motor telemetry."""

import pytest

from grow21_simulation_benchmark.motor_reliability import (MOTOR_GAMES,
                                                           estimate_reliability)


def test_games_without_a_motor_channel_return_none():
    """Seven of the games are tap-to-choose and have no trajectory, so there
    is no slip to detect and the engine must keep trusting them fully."""
    for game in ("pattern_train", "category_sorting", "echo_safari",
                 "audio_search", "visual_flip_match", "picture_sentence_strips"):
        assert game not in MOTOR_GAMES
        assert estimate_reliability(game, {"anything": 1}, outcome=0) is None


def test_a_success_is_always_trusted():
    """How the hand moved does not change what a correct answer tells us."""
    motor = {"meanMissDistanceDp": 5, "bubbleSizeDp": 60}
    assert estimate_reliability("bubble_pop_brigade", motor, outcome=1) == 1.0


def test_missing_or_malformed_motor_payload_is_ignored():
    assert estimate_reliability("trace_glow", None, 0) is None
    assert estimate_reliability("trace_glow", {}, 0) is None
    assert estimate_reliability("trace_glow", {"meanDeviationDp": "x"}, 0) is None
    # a zero-size target would divide by zero
    assert estimate_reliability("bubble_pop_brigade",
                                {"meanMissDistanceDp": 10, "bubbleSizeDp": 0}, 0) is None


def test_block_stacker_separates_slips_from_deliberate_errors():
    """This game counts the two cases itself, so rho is the share of errors
    that look deliberate."""
    slips = estimate_reliability(
        "block_stacker", {"releaseMisses": 4, "wrongTarget": 0, "orderErrors": 0}, 0)
    intent = estimate_reliability(
        "block_stacker", {"releaseMisses": 0, "wrongTarget": 3, "orderErrors": 1}, 0)
    mixed = estimate_reliability(
        "block_stacker", {"releaseMisses": 3, "wrongTarget": 1, "orderErrors": 0}, 0)
    assert intent == 1.0
    assert slips < mixed < intent


def test_a_near_miss_is_trusted_less_than_a_wild_one():
    near = estimate_reliability(
        "bubble_pop_brigade",
        {"meanMissDistanceDp": 20, "bubbleSizeDp": 60, "misses": 5, "escaped": 0}, 0)
    wild = estimate_reliability(
        "bubble_pop_brigade",
        {"meanMissDistanceDp": 180, "bubbleSizeDp": 60, "misses": 1, "escaped": 5}, 0)
    assert near < wild


def test_abandoning_the_task_is_not_treated_as_a_slip():
    """Stopping a fifth of the way along the path is a choice, not the hand."""
    common = {"meanDeviationDp": 12, "pathWidthDp": 40, "offPathExits": 3, "lifts": 2}
    finished = estimate_reliability("trace_glow", {**common, "coverage": 0.9}, 0)
    gave_up = estimate_reliability("trace_glow", {**common, "coverage": 0.2}, 0)
    assert gave_up > finished


@pytest.mark.parametrize("game,motor", [
    ("block_stacker", {"releaseMisses": 9, "wrongTarget": 0, "orderErrors": 0}),
    ("bubble_pop_brigade", {"meanMissDistanceDp": 0, "bubbleSizeDp": 60, "misses": 20}),
    ("trace_glow", {"meanDeviationDp": 0, "pathWidthDp": 40, "offPathExits": 20,
                    "lifts": 20, "coverage": 1.0}),
])
def test_rho_stays_in_range_and_never_discards_an_observation(game, motor):
    rho = estimate_reliability(game, motor, 0)
    assert 0.0 < rho <= 1.0
    assert rho >= 0.15, "an observation should never be thrown away completely"
