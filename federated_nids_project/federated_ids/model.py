"""A small NumPy logistic-regression model trained using FedAvg."""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return sigmoid(X @ weights + bias)


def local_train(
    X: np.ndarray,
    y: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    epochs: int = 2,
    learning_rate: float = 0.12,
    batch_size: int = 64,
    regularization: float = 0.001,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Train on one client's local records; return updated parameters only."""
    rng = np.random.default_rng(seed)
    weights = initial_weights.copy()
    bias = float(initial_bias)
    n = len(y)

    for _ in range(epochs):
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = indices[start : start + batch_size]
            xb = X[batch]
            yb = y[batch]
            probabilities = predict_proba(xb, weights, bias)
            error = probabilities - yb
            grad_w = (xb.T @ error) / len(batch) + regularization * weights
            grad_b = float(np.mean(error))
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b
    return weights, bias


