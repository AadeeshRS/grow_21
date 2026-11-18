"""Estimates observation reliability (rho) from a game's motor telemetry.

Background
----------
`TelemetryEvent` already carries an optional `reliability` field, and the
controller already weights an observation by it in `observe_outcome`. But
nothing ever fills it in, so it arrives as None and every answer is trusted
completely. That is wrong for children with reduced hand control: a child can
know the answer, aim at the right target and miss it, and the engine then
records that as "does not know" and makes the work easier for the wrong reason.

This module fills the field. It reads the per-game `motor` payload the
frontend already sends and returns rho in [0, 1], where a low value means
"this failure looks like the hand, not the head".

Scope
-----
Only three of the games have a motor channel at all, because only they require
a motor action where a slip is possible:

    block_stacker        drag a block onto a slot
    bubble_pop_brigade   tap a moving bubble
    trace_glow           slide along a path

For every other game this returns None, and the engine keeps its current
behaviour of trusting the observation fully. That is the right default: in a
tap-to-choose game there is no trajectory, so there is no slip to detect.

Status of the numbers
---------------------
The thresholds below are provisional, taken from a simulation study, not from
children. Replace them with values fitted to collected data before relying on
them. The structure is what matters here; the constants are placeholders and
are marked as such.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Games that expose a motor channel. Anything else has no slip signal.
MOTOR_GAMES = frozenset({"block_stacker", "bubble_pop_brigade", "trace_glow"})

# Provisional. Fit to real collected data before trusting these.
_SPATIAL_WEIGHT = 0.55      # how much a near-miss counts as evidence of a slip
_EVENT_WEIGHT = 0.30        # how much repeated excursions count
_RHO_FLOOR = 0.15           # never discard an observation completely


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _num(d: Dict[str, Any], key: str) -> Optional[float]:
    v = d.get(key)
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None          # reject NaN


def estimate_reliability(game_id: str, motor: Optional[Dict[str, Any]],
                         outcome: Optional[int] = None) -> Optional[float]:
    """Returns rho in [0, 1], or None when the game carries no motor signal.

    rho = 1.0 means trust the observation fully, which is what the engine does
    today. Lower values mean the failure looks like a motor slip.

    A success is always trusted: if the child got it right, how their hand
    moved does not change what we learned.
    """
    if game_id not in MOTOR_GAMES or not isinstance(motor, dict):
        return None
    if outcome == 1:
        return 1.0

    if game_id == "block_stacker":
        return _block_stacker(motor)
    if game_id == "bubble_pop_brigade":
        return _bubble_pop(motor)
    return _trace_glow(motor)


def _block_stacker(motor: Dict[str, Any]) -> Optional[float]:
    """This game already distinguishes the two cases in its own counters.

    releaseMisses  the block was dropped near the right slot but not on it,
                   which is the hand failing
    wrongTarget    the block was placed on a different slot on purpose
    orderErrors    the blocks were stacked in the wrong order

    So rho is simply the share of errors that look deliberate. No model needed.
    """
    slips = _num(motor, "releaseMisses") or 0.0
    intent = (_num(motor, "wrongTarget") or 0.0) + (_num(motor, "orderErrors") or 0.0)
    total = slips + intent
    if total <= 0:
        return None                        # failed for some other reason
    return _clamp(intent / total, _RHO_FLOOR, 1.0)


def _bubble_pop(motor: Dict[str, Any]) -> Optional[float]:
    """Tapping a moving bubble. A slip lands close to the bubble; giving up or
    not trying lands nowhere near it, or lets bubbles escape untouched."""
    miss_dp = _num(motor, "meanMissDistanceDp")
    size_dp = _num(motor, "bubbleSizeDp")
    misses = _num(motor, "misses") or 0.0
    escaped = _num(motor, "escaped") or 0.0

    if miss_dp is None or not size_dp:
        return None

    # how far off, in bubble widths. under 1.0 means the tap grazed the bubble
    offset = miss_dp / max(size_dp, 1.0)
    near_miss = _clamp(1.0 - offset)                     # 1 when right on the edge
    repeated = _clamp(misses / 6.0)                      # several near misses in a row
    gave_up = _clamp(escaped / 6.0)                      # bubbles left untouched

    slip_evidence = _SPATIAL_WEIGHT * near_miss + _EVENT_WEIGHT * repeated
    slip_evidence *= (1.0 - 0.5 * gave_up)               # not trying is not a slip
    return _clamp(1.0 - slip_evidence, _RHO_FLOOR, 1.0)


def _trace_glow(motor: Dict[str, Any]) -> Optional[float]:
    """Sliding along a path. Small wobbles just outside a narrow path are the
    hand; abandoning the path early is not."""
    dev_dp = _num(motor, "meanDeviationDp")
    width_dp = _num(motor, "pathWidthDp")
    exits = _num(motor, "offPathExits") or 0.0
    lifts = _num(motor, "lifts") or 0.0
    coverage = _num(motor, "coverage")

    if dev_dp is None or not width_dp:
        return None

    offset = dev_dp / max(width_dp / 2.0, 1.0)           # in half-widths
    near_miss = _clamp(1.5 - offset)
    wobble = _clamp((exits + lifts) / 8.0)

    slip_evidence = _SPATIAL_WEIGHT * near_miss + _EVENT_WEIGHT * wobble
    if coverage is not None and coverage < 0.4:
        slip_evidence *= 0.4                             # stopped early, not a slip
    return _clamp(1.0 - slip_evidence, _RHO_FLOOR, 1.0)
