"""Task, game, and skill specifications for Grow21 simulation.

Defines game metadata, skill relationships, active modality configurations,
and difficulty mapping functions without hardcoding a permanent one-game-equals-one-skill rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass(frozen=True)
class Skill:
    id: str
    name: str


@dataclass(frozen=True)
class GameDef:
    game_id: str
    name: str
    primary_skill: str
    secondary_skill: Optional[str]
    active_modalities: List[str]
    guess_floor: float
    lapse_ceiling: float
    discrimination: float
    difficulty_offset: float
    difficulty_scale: float


@dataclass(frozen=True)
class ModalityDef:
    id: str
    description: str
    base_corruption_rate: float
    estimation_noise_std: float
    sensor_dropout_rate: float


class TaskRegistry:
    """Registry managing skills, games, modalities, and task parameter mappings."""

    def __init__(self, config: Dict[str, Any]):
        self.skills: Dict[str, Skill] = {
            s["id"]: Skill(id=s["id"], name=s["name"]) for s in config.get("skills", [])
        }
        self.skill_ids: List[str] = list(self.skills.keys())
        self.skill_to_idx: Dict[str, int] = {sid: i for i, sid in enumerate(self.skill_ids)}

        self.games: Dict[str, GameDef] = {}
        for gid, gdata in config.get("games", {}).items():
            self.games[gid] = GameDef(
                game_id=gid,
                name=gdata.get("name", gid),
                primary_skill=gdata["primary_skill"],
                secondary_skill=gdata.get("secondary_skill"),
                active_modalities=list(gdata.get("active_modalities", [])),
                guess_floor=float(gdata.get("guess_floor", 0.1)),
                lapse_ceiling=float(gdata.get("lapse_ceiling", 0.05)),
                discrimination=float(gdata.get("discrimination", 3.0)),
                difficulty_offset=float(gdata.get("difficulty_offset", 0.0)),
                difficulty_scale=float(gdata.get("difficulty_scale", 1.0)),
            )
        self.game_ids: List[str] = list(self.games.keys())
        self.game_to_idx: Dict[str, int] = {gid: i for i, gid in enumerate(self.game_ids)}

        self.modalities: Dict[str, ModalityDef] = {}
        for mid, mdata in config.get("modalities", {}).items():
            self.modalities[mid] = ModalityDef(
                id=mid,
                description=mdata.get("description", mid),
                base_corruption_rate=float(mdata.get("base_corruption_rate", 0.05)),
                estimation_noise_std=float(mdata.get("estimation_noise_std", 0.05)),
                sensor_dropout_rate=float(mdata.get("sensor_dropout_rate", 0.05)),
            )
        self.modality_ids: List[str] = list(self.modalities.keys())
        self.modality_to_idx: Dict[str, int] = {mid: i for i, mid in enumerate(self.modality_ids)}

        sim_cfg = config.get("simulation", {})
        self.default_reliability_convention = float(
            sim_cfg.get("default_reliability_convention", 1.0)
        )
        self.missing_modality_default_uncertainty = float(
            sim_cfg.get("missing_modality_default_uncertainty", 0.25)
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskRegistry:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    def calculate_difficulty_threshold(self, game_id: str, difficulty: float) -> float:
        """Translates abstract difficulty D in [0, 1] to an item threshold.

        Uses a smooth nonlinear bounded representation distinct from the controller's
        internal gamma * D**nu power predictor.
        """
        g = self.games[game_id]
        d_clamped = max(0.0, min(1.0, float(difficulty)))
        # Non-linear convex/concave inflection without replicating controller equation:
        curve = 0.25 * d_clamped + 0.75 * (d_clamped ** 1.35)
        return g.difficulty_offset + g.difficulty_scale * curve

    def map_abstract_difficulty_to_params(
        self, game_id: str, difficulty: float
    ) -> Dict[str, Any]:
        """Provides simulated gameplay parameters mapped from abstract difficulty D."""
        d = max(0.0, min(1.0, float(difficulty)))
        if game_id == "trace_glow":
            return {
                "path_width_px": round(40.0 - 24.0 * d, 1),
                "curvature_points": int(3 + round(5 * d)),
                "speed_tolerance_ms": int(4000 - 2200 * d),
            }
        elif game_id == "bubble_pop_brigade":
            return {
                "bubble_radius_px": round(50.0 - 28.0 * d, 1),
                "lifespan_sec": round(3.5 - 2.0 * d, 2),
                "distractor_bubbles": int(round(4 * d)),
            }
        elif game_id == "block_stacker":
            return {
                "target_height": int(3 + round(5 * d)),
                "block_width_tolerance_pct": round(25.0 - 15.0 * d, 1),
                "jitter_amplitude_px": round(8.0 * d, 1),
            }
        elif game_id == "echo_safari":
            # Echo Safari detects vocalization onset and duration, not exact phoneme ASR
            return {
                "required_vocal_duration_sec": round(0.4 + 0.8 * d, 2),
                "acoustic_snr_threshold_db": round(10.0 + 8.0 * d, 1),
                "prompt_repeat_allowed": d < 0.6,
            }
        elif game_id == "picture_sentence_strips":
            return {
                "sentence_tokens": int(2 + round(3 * d)),
                "foil_options": int(1 + round(3 * d)),
            }
        elif game_id == "audio_search":
            return {
                "target_sound_library_size": int(4 + round(8 * d)),
                "audio_cue_length_sec": round(1.8 - 0.9 * d, 2),
                "ambient_distractor_level": round(0.3 * d, 2),
            }
        elif game_id == "category_sorting":
            return {
                "num_categories": 2 if d < 0.5 else 3,
                "items_to_sort": int(4 + round(6 * d)),
                "semantic_distance": round(0.9 - 0.5 * d, 2),
            }
        elif game_id == "visual_flip_match":
            return {
                "grid_cards": 4 if d < 0.35 else (6 if d < 0.75 else 8),
                "preview_time_sec": round(4.0 - 2.5 * d, 2),
            }
        elif game_id == "pattern_train":
            return {
                "pattern_period": 2 if d < 0.4 else (3 if d < 0.8 else 4),
                "missing_elements": 1 if d < 0.6 else 2,
            }
        return {"abstract_difficulty": d}
