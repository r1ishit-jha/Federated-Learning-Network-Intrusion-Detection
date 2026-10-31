"""TCP client/coordinator transport for the federated-learning demonstration."""
from __future__ import annotations

import json
import queue
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .model import local_train

HOST = "127.0.0.1"  # Loopback keeps the demo self-contained on one computer.


def _read_json_line(stream) -> dict:
    raw = stream.readline()
    if not raw:
        raise ConnectionError("Client closed the connection before sending a message.")
    return json.loads(raw.decode("utf-8"))


def _coordinator_round(
    expected_clients: int,
    ready_queue: queue.Queue,
    result_queue: queue.Queue,
) -> None:
    """Accept one update from each client, aggregate with FedAvg, reply to clients."""
    connections: list[tuple[socket.socket, object]] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((HOST, 0))
            server.listen(expected_clients)
            server.settimeout(25)
            ready_queue.put(server.getsockname()[1])

            updates = []
            for _ in range(expected_clients):
                conn, _address = server.accept()
                conn.settimeout(25)
                stream = conn.makefile("rwb")
                payload = _read_json_line(stream)
                updates.append(payload)
                connections.append((conn, stream))

            total_samples = sum(int(item["n_samples"]) for item in updates)
            if total_samples <= 0:
                raise ValueError("Coordinator received no valid training samples.")

            global_weights = np.zeros(len(updates[0]["weights"]), dtype=np.float64)
            global_bias = 0.0
            for item in updates:
                weight = int(item["n_samples"]) / total_samples
                global_weights += weight * np.asarray(item["weights"], dtype=np.float64)
                global_bias += weight * float(item["bias"])

            response = {
                "type": "global_model",
                "weights": global_weights.tolist(),
                "bias": global_bias,
            }
            response_bytes = (json.dumps(response) + "\n").encode("utf-8")
            for conn, stream in connections:
                stream.write(response_bytes)
                stream.flush()
            result_queue.put((global_weights, global_bias))
    except Exception as exc:  # Surface socket and protocol errors to the caller.
        result_queue.put(exc)
    finally:
        for conn, stream in connections:
            try:
                stream.close()
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass


def _client_round(
    client_id: int,
    X_local: np.ndarray,
    y_local: np.ndarray,
    global_weights: np.ndarray,
    global_bias: float,
    port: int,
    round_index: int,
    local_epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Train locally and exchange only the model update over a TCP socket."""
    local_weights, local_bias = local_train(
        X_local,
        y_local,
        global_weights,
        global_bias,
        epochs=local_epochs,
        learning_rate=learning_rate,
        seed=seed + round_index * 100 + client_id,
    )
    message = {
        "type": "client_update",
        "client_id": client_id + 1,
        "n_samples": int(len(y_local)),
        "weights": local_weights.tolist(),
        "bias": local_bias,
    }

    with socket.create_connection((HOST, port), timeout=25) as client_socket:
        client_socket.settimeout(25)
        stream = client_socket.makefile("rwb")
        stream.write((json.dumps(message) + "\n").encode("utf-8"))
        stream.flush()
        response = _read_json_line(stream)
        stream.close()

    if response.get("type") != "global_model":
        raise ConnectionError("Coordinator returned an unexpected message.")
    return (
        np.asarray(response["weights"], dtype=np.float64),
        float(response["bias"]),
    )


def federated_train_tcp(
    clients: list[tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int = 25,
    local_epochs: int = 2,
    learning_rate: float = 0.12,
    seed: int = 7,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    """Run FedAvg with real TCP exchanges between local client workers/server."""
    n_features = clients[0][0].shape[1]
    global_weights = np.zeros(n_features, dtype=np.float64)
    global_bias = 0.0
    history: list[dict[str, float]] = []

    for round_index in range(rounds):
        ready_queue: queue.Queue = queue.Queue(maxsize=1)
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        server_thread = threading.Thread(
            target=_coordinator_round,
            args=(len(clients), ready_queue, result_queue),
            daemon=True,
        )
        server_thread.start()
        ready_result = ready_queue.get(timeout=10)
        if isinstance(ready_result, Exception):
            raise ready_result
        port = int(ready_result)

        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            futures = [
                executor.submit(
                    _client_round,
                    client_id,
                    X_local,
                    y_local,
                    global_weights,
                    global_bias,
                    port,
                    round_index,
                    local_epochs,
                    learning_rate,
                    seed,
                )
                for client_id, (X_local, y_local) in enumerate(clients)
            ]
            client_models = [future.result() for future in futures]

        server_thread.join(timeout=30)
        if server_thread.is_alive():
            raise TimeoutError("Coordinator did not finish the TCP aggregation round.")
        aggregate = result_queue.get(timeout=5)
        if isinstance(aggregate, Exception):
            raise aggregate

        # Each client receives the identical aggregated global model from the server.
        global_weights, global_bias = client_models[0]

        from .model import predict_proba
        probabilities = predict_proba(X_test, global_weights, global_bias)
        predictions = (probabilities >= 0.5).astype(np.int64)
        accuracy = float(np.mean(predictions == y_test))
        clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
        loss = float(
            -np.mean(y_test * np.log(clipped) + (1.0 - y_test) * np.log(1.0 - clipped))
        )
        history.append({"round": round_index + 1, "test_accuracy": accuracy, "test_log_loss": loss})

    return global_weights, global_bias, history
