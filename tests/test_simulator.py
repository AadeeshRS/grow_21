"""Unit tests verifying simulator bounds, mastery dynamics, and parameter mappings."""

from pathlib import Path
import numpy as np
import pytest

from grow21_simulation_benchmark.dataset import SimulationDatasetGenerator
from grow21_simulation_benchmark.reliability import ReliabilityEngine
from grow21_simulation_benchmark.simulator import (
    VirtualLearnerSimulator,
    generate_learner_profile,
)
from grow21_simulation_benchmark.tasks import TaskRegistry


@pytest.fixture
def config_and_registry():
    cfg_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    registry = TaskRegistry.from_yaml(cfg_path)
    return cfg_path, registry


def test_game_definitions_and_params(config_and_registry):
    _, registry = config_and_registry
    assert len(registry.games) == 9
    expected_games = {
        "trace_glow",
        "bubble_pop_brigade",
        "block_stacker",
        "echo_safari",
        "picture_sentence_strips",
        "audio_search",
        "category_sorting",
        "visual_flip_match",
        "pattern_train",
    }
    assert set(registry.game_ids) == expected_games

    # Test abstract difficulty mapping for all games
    for gid in registry.game_ids:
        params_low = registry.map_abstract_difficulty_to_params(gid, 0.1)
        params_high = registry.map_abstract_difficulty_to_params(gid, 0.9)
        assert isinstance(params_low, dict)
        assert isinstance(params_high, dict)


def test_learner_profile_and_mastery_bounds(config_and_registry):
    _, registry = config_and_registry
    rng = np.random.RandomState(42)
    scenario_params = {
        "initial_mastery_mean": 0.45,
        "initial_mastery_std": 0.18,
        "learning_rate_mean": 0.08,
        "learning_rate_std": 0.03,
        "forgetting_rate_mean": 0.02,
        "forgetting_rate_std": 0.01,
    }

    profile = generate_learner_profile(
        learner_id=1,
        skill_ids=registry.skill_ids,
        rng=rng,
        scenario_params=scenario_params,
        mastery_min=0.05,
        mastery_max=0.95,
    )

    for skill, m in profile.initial_mastery.items():
        assert 0.05 <= m <= 0.95, f"Mastery for {skill} out of bounds: {m}"

    reliability_engine = ReliabilityEngine(registry)
    sim = VirtualLearnerSimulator(
        profile=profile,
        task_registry=registry,
        reliability_engine=reliability_engine,
        mastery_floor=0.05,
        mastery_ceiling=0.95,
    )

    # Run 100 simulation steps
    for step in range(100):
        gid = registry.game_ids[step % len(registry.game_ids)]
        diff = float(np.clip(rng.uniform(0.0, 1.0), 0.0, 1.0))
        gap = float(rng.choice([0.02, 12.0, 48.0]))
        ev = sim.step(
            game_id=gid,
            difficulty=diff,
            time_gap_hours=gap,
            rng=rng,
            scenario_params={"corruption_multiplier": 1.0},
        )

        assert 0.0 <= ev.difficulty <= 1.0
        assert 0.0 <= ev.p_clean <= 1.0
        assert ev.observed_outcome in (0, 1)
        assert ev.clean_outcome in (0, 1)
        assert 0.0 <= ev.estimated_reliability <= 1.0
        assert 0.0 <= ev.true_reliability <= 1.0

        for skill, m in ev.latent_mastery_after.items():
            assert 0.05 <= m <= 0.95, f"Evolved mastery out of bounds: {m}"


def test_inter_session_forgetting(config_and_registry):
    _, registry = config_and_registry
    rng = np.random.RandomState(123)
    profile = generate_learner_profile(
        learner_id=2,
        skill_ids=registry.skill_ids,
        rng=rng,
        scenario_params={"initial_mastery_mean": 0.8, "initial_mastery_std": 0.02},
        mastery_min=0.05,
        mastery_max=0.95,
    )
    reliability_engine = ReliabilityEngine(registry)
    sim = VirtualLearnerSimulator(
        profile=profile,
        task_registry=registry,
        reliability_engine=reliability_engine,
        mastery_floor=0.05,
        mastery_ceiling=0.95,
    )

    initial_m = sim.current_mastery["fine_motor"]
    # Apply long 72-hour gap
    sim.apply_forgetting(72.0)
    decayed_m = sim.current_mastery["fine_motor"]

    assert decayed_m < initial_m, "Mastery should decay over time gap"
    assert decayed_m >= 0.05, "Decayed mastery must not drop below floor"
