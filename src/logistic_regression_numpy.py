"""A small binary logistic regression implementation using only NumPy."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegressionNumpy(ClassifierMixin, BaseEstimator):
    """Binary logistic regression trained with batch gradient descent."""

    def __init__(self, learning_rate: float = 0.1, epochs: int = 2_000) -> None:
        if learning_rate <= 0 or epochs <= 0:
            raise ValueError("learning_rate and epochs must be positive")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -500, 500)
        return 1.0 / (1.0 + np.exp(-values))

    def fit(self, features: np.ndarray, target: np.ndarray) -> "LogisticRegressionNumpy":
        features = np.asarray(features, dtype=float)
        target = np.asarray(target, dtype=float).reshape(-1)
        if features.ndim != 2 or len(features) != len(target):
            raise ValueError("features must be 2-D and match target length")
        if not np.all(np.isin(target, [0.0, 1.0])):
            raise ValueError("target must contain only 0 and 1")

        samples = len(features)
        self.weights_ = np.zeros(features.shape[1], dtype=float)
        self.bias_ = 0.0
        self.loss_history_ = []
        for _ in range(self.epochs):
            probabilities = self._sigmoid(features @ self.weights_ + self.bias_)
            error = probabilities - target
            self.weights_ -= self.learning_rate * (features.T @ error) / samples
            self.bias_ -= self.learning_rate * float(np.mean(error))
            loss = -np.mean(
                target * np.log(probabilities + 1e-12)
                + (1 - target) * np.log(1 - probabilities + 1e-12)
            )
            self.loss_history_.append(float(loss))
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("fit must be called before prediction")
        features = np.asarray(features, dtype=float)
        probabilities = self._sigmoid(features @ self.weights_ + self.bias_)
        return np.column_stack((1 - probabilities, probabilities))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)
