"""Federated Learning for Network Intrusion Detection - educational prototype."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from federated_ids.data import FEATURE_NAMES, build_federated_clients, make_traffic_data
from federated_ids.metrics import evaluate
from federated_ids.model import predict_proba
from federated_ids.network import federated_train_tcp

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def save_plots(history: list[dict[str, float]], matrix: list[list[int]]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    history_df = pd.DataFrame(history)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history_df["round"], history_df["test_accuracy"], marker="o", label="Test accuracy")
    ax.set_title("Federated Training Progress")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_accuracy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(np.array(matrix), cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], ["Benign", "Attack"])
    ax.set_yticks([0, 1], ["Benign", "Attack"])
    for (row, col), value in np.ndenumerate(np.array(matrix)):
        ax.text(col, row, str(value), ha="center", va="center", fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    clients = build_federated_clients(n_clients=4, samples_per_client=900, seed=42)
    X_test, y_test = make_traffic_data(
        n_samples=1200, attack_rate=0.40, client_shift=0.0, seed=2026
    )

    print("=" * 62)
    print(" FEDERATED LEARNING FOR NETWORK INTRUSION DETECTION")
    print("=" * 62)
    print(f"Simulated clients : {len(clients)}")
    print(f"Training records  : {sum(len(y) for _, y in clients):,}")
    print(f"Test records      : {len(y_test):,}")
    print(f"Input features    : {len(FEATURE_NAMES)}")
    print("Transport          : TCP sockets over 127.0.0.1")
    print("Data exchange     : JSON model parameters only (no raw records)")
    print("\nTraining global model using FedAvg...\n")

    weights, bias, history = federated_train_tcp(
        clients,
        X_test,
        y_test,
        rounds=25,
        local_epochs=2,
        learning_rate=0.12,
    )
    probabilities = predict_proba(X_test, weights, bias)
    predictions = (probabilities >= 0.5).astype(np.int64)
    results = evaluate(y_test.astype(np.int64), predictions)

    print("Training completed.")
    print(f"Final test accuracy : {results['accuracy'] * 100:.2f}%")
    print(f"Precision           : {results['precision'] * 100:.2f}%")
    print(f"Recall              : {results['recall'] * 100:.2f}%")
    print(f"F1-score            : {results['f1_score'] * 100:.2f}%")
    print("\nConfusion matrix [actual rows: benign, attack]:")
    print(np.array(results["confusion_matrix"]))
    print(f"\nOutputs saved to: {OUTPUT_DIR.relative_to(ROOT)}")

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "round_metrics.csv", index=False)
    with (OUTPUT_DIR / "evaluation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    with (OUTPUT_DIR / "final_model.json").open("w", encoding="utf-8") as file:
        json.dump({"weights": weights.tolist(), "bias": bias, "features": FEATURE_NAMES}, file, indent=2)
    save_plots(history, results["confusion_matrix"])


if __name__ == "__main__":
    main()
