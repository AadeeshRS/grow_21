"""Logistic Regression baseline for Grow21 knowledge-tracing benchmark.

Competent feature-based baseline tracking historical success/failure counts,
recency, streaks, difficulty, session gaps, and pre-response uncertainty risk.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression

from ..features import TabularDatasetPipeline
from ..simulator import SimulatorEvent
from ..tasks import TaskRegistry


class LogisticBaselineModel:
    """Logistic Regression baseline with causal history features and standardized scaling."""

    def __init__(
        self,
        task_registry: TaskRegistry,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: int = 42,
    ):
        self.registry = task_registry
        self.pipeline = TabularDatasetPipeline(task_registry)
        self.clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(
        self,
        train_sequences: List[List[SimulatorEvent]],
        val_sequences: Optional[List[List[SimulatorEvent]]] = None,
    ) -> "LogisticBaselineModel":
        """Fits the scaler and logistic regression classifier on training sequences."""
        X_train, y_train, _ = self.pipeline.fit_transform_train(train_sequences)
        self.clf.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(
        self, eval_sequences: List[List[SimulatorEvent]]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Predicts probabilities P(y_obs = 1) for evaluation events.

        Returns:
            probs: 1D array of predicted probabilities in [0, 1]
            y_true: 1D array of true observed outcomes in {0, 1}
            diagnostics: List of metadata dicts for oracle inspection
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        X_eval, y_eval, diagnostics = self.pipeline.transform_eval(eval_sequences)
        probs = self.clf.predict_proba(X_eval)[:, 1]
        return probs, y_eval, diagnostics

    def predict(self, eval_sequences: List[List[SimulatorEvent]]) -> np.ndarray:
        probs, _, _ = self.predict_proba(eval_sequences)
        return (probs >= 0.5).astype(np.int64)

    def count_parameters(self) -> int:
        if not self.is_fitted:
            return 0
        return int(self.clf.coef_.size + self.clf.intercept_.size)
