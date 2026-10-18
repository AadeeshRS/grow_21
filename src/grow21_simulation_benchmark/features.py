"""Feature extraction for the Logistic Regression baseline in Grow21.

Extracts historical and pre-response task features strictly up to event t-1
to ensure no leakage of the target outcome or future events.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .simulator import SimulatorEvent
from .tasks import TaskRegistry


class FeatureExtractor:
    """Extracts strictly causal, pre-response features for each event."""

    def __init__(self, task_registry: TaskRegistry):
        self.registry = task_registry
        self.games = task_registry.game_ids
        self.skills = task_registry.skill_ids
        self.modalities = task_registry.modality_ids

    def extract_sequence_features(
        self, sequence: List[SimulatorEvent]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Extracts feature rows and target labels for a single learner sequence.

        Returns:
            X: Array of shape [num_events, num_features]
            y: Array of shape [num_events]
            diagnostics: List of metadata dicts for oracle evaluations
        """
        # Learner running history states (initialized empty)
        total_attempts = 0
        total_successes = 0

        skill_attempts: Dict[str, int] = {s: 0 for s in self.skills}
        skill_successes: Dict[str, int] = {s: 0 for s in self.skills}
        skill_last_time: Dict[str, float] = {s: -1.0 for s in self.skills}

        game_attempts: Dict[str, int] = {g: 0 for g in self.games}
        game_successes: Dict[str, int] = {g: 0 for g in self.games}

        recent_outcomes: List[int] = []
        skill_recent_outcomes: Dict[str, List[int]] = {s: [] for s in self.skills}

        consecutive_successes = 0
        consecutive_failures = 0

        feature_rows: List[List[float]] = []
        labels: List[int] = []
        diagnostics: List[Dict[str, Any]] = []

        for ev in sequence:
            # 1. PRE-RESPONSE CURRENT TASK FEATURES (known before outcome occurs)
            row: List[float] = []

            # One-hot game ID
            for g in self.games:
                row.append(1.0 if ev.game_id == g else 0.0)

            # One-hot primary skill
            for s in self.skills:
                row.append(1.0 if ev.primary_skill == s else 0.0)

            # Task difficulty & timing
            row.append(float(ev.difficulty))
            row.append(float(min(72.0, ev.session_gap_hours)))
            row.append(float(min(200.0, ev.time_hours)))
            row.append(float(ev.estimated_reliability))

            # Active modalities: uncertainties and availability flags
            for mid in self.modalities:
                p_hat = ev.modality_p_hats.get(mid, 0.0)
                is_avail = 1.0 if ev.modality_available.get(mid, False) else 0.0
                is_active = 1.0 if mid in ev.game_params else (
                    1.0 if mid in self.registry.games[ev.game_id].active_modalities else 0.0
                )
                row.append(float(p_hat))
                row.append(is_avail)
                row.append(is_active)
                # Explicit missingness indicator when active but unavailable
                missing_active = 1.0 if (is_active == 1.0 and is_avail == 0.0) else 0.0
                row.append(missing_active)

            # 2. CAUSAL HISTORY FEATURES (strictly from events 0 to t-1)
            # Global history
            row.append(float(total_attempts))
            row.append(float(total_successes))
            global_smoothed_rate = (total_successes + 1.0) / (total_attempts + 2.0)
            row.append(float(global_smoothed_rate))

            # Primary skill history (Performance Factors Analysis core)
            s_att = skill_attempts[ev.primary_skill]
            s_succ = skill_successes[ev.primary_skill]
            s_fail = s_att - s_succ
            s_smoothed_rate = (s_succ + 1.0) / (s_att + 2.0)
            row.append(float(s_att))
            row.append(float(s_succ))
            row.append(float(s_fail))
            row.append(float(s_smoothed_rate))

            # Time since primary skill last practiced
            last_t = skill_last_time[ev.primary_skill]
            gap_since_skill = (ev.time_hours - last_t) if last_t >= 0.0 else 99.0
            row.append(float(min(99.0, gap_since_skill)))

            # Game-specific history
            g_att = game_attempts[ev.game_id]
            g_succ = game_successes[ev.game_id]
            g_smoothed_rate = (g_succ + 1.0) / (g_att + 2.0)
            row.append(float(g_att))
            row.append(float(g_succ))
            row.append(float(g_smoothed_rate))

            # Recency features
            if len(recent_outcomes) > 0:
                last_outcome = float(recent_outcomes[-1])
                has_prior = 1.0
            else:
                last_outcome = 0.5  # neutral prior
                has_prior = 0.0
            row.append(last_outcome)
            row.append(has_prior)

            # Rolling window 3 & 5 global success rate
            w3 = recent_outcomes[-3:] if len(recent_outcomes) >= 3 else recent_outcomes
            w3_rate = (sum(w3) + 1.0) / (len(w3) + 2.0) if w3 else 0.5
            row.append(float(w3_rate))

            w5 = recent_outcomes[-5:] if len(recent_outcomes) >= 5 else recent_outcomes
            w5_rate = (sum(w5) + 1.0) / (len(w5) + 2.0) if w5 else 0.5
            row.append(float(w5_rate))

            # Skill-specific recency
            sk_rec = skill_recent_outcomes[ev.primary_skill]
            sk_w3 = sk_rec[-3:] if len(sk_rec) >= 3 else sk_rec
            sk_w3_rate = (sum(sk_w3) + 1.0) / (len(sk_w3) + 2.0) if sk_w3 else 0.5
            row.append(float(sk_w3_rate))

            # Streaks
            row.append(float(consecutive_successes))
            row.append(float(consecutive_failures))

            # Record feature row and target
            feature_rows.append(row)
            labels.append(ev.observed_outcome)
            diagnostics.append(
                {
                    "learner_id": ev.learner_id,
                    "step_idx": ev.step_idx,
                    "clean_outcome": ev.clean_outcome,
                    "p_clean": ev.p_clean,
                    "true_reliability": ev.true_reliability,
                    "estimated_reliability": ev.estimated_reliability,
                }
            )

            # 3. POST-RESPONSE STATE UPDATE (for future events t+1 onwards ONLY)
            y = ev.observed_outcome
            total_attempts += 1
            total_successes += y

            skill_attempts[ev.primary_skill] += 1
            skill_successes[ev.primary_skill] += y
            skill_last_time[ev.primary_skill] = ev.time_hours
            skill_recent_outcomes[ev.primary_skill].append(y)

            game_attempts[ev.game_id] += 1
            game_successes[ev.game_id] += y

            recent_outcomes.append(y)

            if y == 1:
                consecutive_successes += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                consecutive_successes = 0

        return np.array(feature_rows, dtype=np.float32), np.array(labels, dtype=np.int64), diagnostics


class TabularDatasetPipeline:
    """Processes datasets of learner sequences into scaled tabular matrices for Logistic Regression."""

    def __init__(self, task_registry: TaskRegistry):
        self.extractor = FeatureExtractor(task_registry)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_from_sequences(
        self, sequences: List[List[SimulatorEvent]]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        all_x: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        all_diag: List[Dict[str, Any]] = []

        for seq in sequences:
            x_seq, y_seq, diag_seq = self.extractor.extract_sequence_features(seq)
            all_x.append(x_seq)
            all_y.append(y_seq)
            all_diag.extend(diag_seq)

        X = np.vstack(all_x) if all_x else np.empty((0, 0))
        y = np.concatenate(all_y) if all_y else np.empty(0)
        return X, y, all_diag

    def fit_transform_train(
        self, train_sequences: List[List[SimulatorEvent]]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        X_train, y_train, diag = self.extract_from_sequences(train_sequences)
        X_scaled = self.scaler.fit_transform(X_train)
        self.is_fitted = True
        return X_scaled, y_train, diag

    def transform_eval(
        self, eval_sequences: List[List[SimulatorEvent]]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training sequences first.")
        X_eval, y_eval, diag = self.extract_from_sequences(eval_sequences)
        X_scaled = self.scaler.transform(X_eval)
        return X_scaled, y_eval, diag
