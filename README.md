# Federated-Learning-Network-Intrusion-Detection
Federated Learning based Network Intrusion Detection using Python
Federated Learning for Network Intrusion Detection
A small, beginner-friendly Computer Networks mini-project. It demonstrates how multiple network clients can train a shared intrusion-detection model without sending their raw traffic records to a central server.

What the project does
Creates four simulated network clients, each with its own local traffic records.
Generates six normalized traffic features and a binary label: 0 = benign, 1 = attack.
Trains a logistic-regression model locally at each client.
Sends only a JSON model update (client ID, sample count, weights, and bias) to the coordinator over a TCP socket on 127.0.0.1.
Uses sample-weighted Federated Averaging (FedAvg) to combine client updates and sends the global parameters back to each client over TCP. This repeats for 25 rounds.
Evaluates the final global model and saves metrics and plots in outputs/.
Important: This is an educational prototype using synthetic data. Four client workers and the coordinator run on one computer and communicate using real TCP sockets over the loopback interface (127.0.0.1); the demo does not use separate physical machines. It is not trained or validated on live network traffic or a standard intrusion-detection benchmark, and its scores must not be interpreted as real-world security performance. The prototype has no authentication or encryption. Federated learning reduces the need to centralize raw records, but model updates can still reveal information; production systems need secure aggregation, authentication, encryption, and privacy protections.

Requirements
Python 3.10 or newer
NumPy
pandas
Matplotlib
Run the project
Open a terminal in this folder and run:

python -m venv .venv
Activate the environment:

Windows PowerShell

.\.venv\Scripts\Activate.ps1
Windows Command Prompt

.venv\Scripts\activate.bat
Install dependencies and run:

python -m pip install -r requirements.txt
python main.py
The program prints accuracy, precision, recall, F1-score and a confusion matrix. It creates:

outputs/round_metrics.csv - test metrics for each federated round
outputs/evaluation_metrics.json - final evaluation metrics
outputs/final_model.json - final model parameters and feature names
outputs/training_accuracy.png - training progress chart
outputs/confusion_matrix.png - final confusion matrix
Project structure
federated_nids_project/
├── federated_ids/
│   ├── __init__.py
│   ├── data.py
│   ├── metrics.py
│   ├── model.py
│   └── network.py
├── outputs/
├── main.py
├── requirements.txt
└── README.md
Model inputs
Feature	Meaning
duration_norm	Normalized connection duration
packet_count_norm	Normalized packet count
byte_count_norm	Normalized bytes transferred
failed_logins_norm	Normalized failed-login count
unusual_port_score_norm	Normalized score indicating unusual destination-port activity
connection_rate_norm	Normalized connection rate
Simplified FedAvg equation
For client model parameters (w_k), local sample count (n_k), and total sample count (N), the server calculates:

[ w_{global} = \sum_{k=1}^{K} \frac{n_k}{N} w_k ]

The same weighted averaging is applied to the model bias. The clients keep their local traffic records; the coordinator receives model parameters only over the TCP connection.

Suggested demonstration
Explain the TCP client-server flow, run python main.py, point out that each client has a different simulated traffic distribution, and show the saved training curve and confusion matrix. Be clear that the data is synthetic and the TCP exchange uses the local loopback interface.
