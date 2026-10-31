"""Synthetic network-flow data generator for the federated IDS demo.

The data is generated locally for each client. It is intentionally synthetic
and is suitable for demonstrating the workflow, not for benchmarking a real IDS.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "duration_norm",
    "packet_count_norm",
    "byte_count_norm",
    "failed_logins_norm",
    "unusual_port_score_norm",
    "connection_rate_norm",
]


def make_traffic_data(
    n_samples: int,
    attack_rate: float,
    client_shift: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create labelled, normalized traffic features for one simulated network."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, attack_rate, size=n_samples).astype(np.float64)

    benign_mean = np.array([0.30, 0.38, 0.35, 0.10, 0.18, 0.28])
    attack_mean = np.array([0.49, 0.55, 0.58, 0.40, 0.52, 0.55])
    std = np.array([0.22, 0.23, 0.23, 0.21, 0.22, 0.22])

    # Small client-specific shifts simulate different local traffic profiles.
    shift_vector = np.array([1.0, -0.5, 0.7, 0.4, -0.7, 0.8]) * client_shift
    means = np.where(y[:, None] == 1, attack_mean, benign_mean) + shift_vector
    X = rng.normal(loc=means, scale=std, size=(n_samples, len(FEATURE_NAMES)))
    X = np.clip(X, 0.0, 1.0).astype(np.float64)
    return X, y


def build_federated_clients(
    n_clients: int = 4,
    samples_per_client: int = 900,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build clients with slightly different attack prevalence and traffic profiles."""
    attack_rates = [0.25, 0.35, 0.45, 0.55]
    shifts = [-0.06, -0.02, 0.025, 0.06]
    clients = []
    for client_id in range(n_clients):
        X, y = make_traffic_data(
            n_samples=samples_per_client,
            attack_rate=attack_rates[client_id % len(attack_rates)],
            client_shift=shifts[client_id % len(shifts)],
            seed=seed + client_id,
        )
        clients.append((X, y))
    return clients
