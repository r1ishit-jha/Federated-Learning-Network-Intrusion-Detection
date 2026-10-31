# Federated Learning for Network Intrusion Detection

### Computer Networks – Mini Project

**Project Phase – 1**

A machine learning-based network intrusion detection prototype using Federated Learning, Logistic Regression, TCP socket communication, and Federated Averaging (FedAvg).

---

## 1. Project Overview

Network intrusion detection is an important part of cybersecurity. Traditional intrusion detection systems often rely on collecting network traffic data at a central location for analysis and model training. However, sharing sensitive network data can create privacy and security concerns.

This project demonstrates **Federated Learning for Network Intrusion Detection**, where multiple independent clients train their own machine learning models locally and share only their model parameters with a central server.

The central server combines these updates using Federated Averaging (FedAvg) to create a global model capable of classifying network traffic as either benign or malicious.

The project simulates four independent clients communicating with a central coordinator through TCP sockets.

## 2. Objectives

- Understand the concept and working of Federated Learning.
- Demonstrate distributed machine learning across multiple clients.
- Implement network communication using TCP sockets.
- Apply Logistic Regression for binary network traffic classification.
- Implement Federated Averaging (FedAvg).
- Evaluate the global model using standard classification metrics.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Scikit-learn | Machine learning |
| NumPy | Numerical operations |
| Matplotlib | Graphs and visualisation |
| TCP Sockets | Client-server communication |
| JSON | Data and model parameter exchange |
| Federated Learning | Distributed model training |

## 4. System Architecture

The project consists of four simulated clients and one central server.

1. Each client receives its own synthetic network traffic dataset.
2. Each client trains a Logistic Regression model locally.
3. Clients send their model parameters to the central server using TCP sockets.
4. The server aggregates the model parameters using Federated Averaging.
5. The updated global model is returned to the clients.
6. The process repeats over multiple communication rounds.
7. The final global model is evaluated using separate test data.

**Important:** The original training records remain local to each simulated client. Only model updates are exchanged.

## 5. Machine Learning Model

The project uses Logistic Regression for binary classification.

The model classifies network traffic into two categories:

- **Benign (0):** Normal network traffic.
- **Attack (1):** Potentially malicious network traffic.

The model uses six traffic features:

- Connection duration
- Packet count
- Byte count
- Failed login attempts
- Unusual port score
- Connection rate

## 6. Federated Averaging (FedAvg)

Federated Averaging is the aggregation algorithm used by the central server.

Instead of collecting the original training data, the server receives model parameters from participating clients and calculates a weighted average.

The aggregation formula is:

\[
W_{global}=\sum_{k=1}^{K}\frac{n_k}{N}W_k
\]

Where:

- \(W_{global}\) = Global model parameters
- \(W_k\) = Local model parameters
- \(n_k\) = Number of training samples at client \(k\)
- \(N\) = Total training samples
- \(K\) = Total number of clients

## 7. Project Results

The demonstration was evaluated using a separate synthetic test dataset.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 91.92% |
| Precision | 93.45% |
| Recall | 87.01% |
| F1-Score | 90.11% |

### Confusion Matrix

| Actual / Predicted | Benign | Attack |
|---|---:|---:|
| Benign | 661 | 31 |
| Attack | 66 | 442 |

The model achieved 91.92% classification accuracy in the demonstration.

**Note:** These results are based on synthetically generated network traffic and should not be interpreted as real-world intrusion detection performance.

## 8. Project Structure

```text
Federated-Learning-Network-Intrusion-Detection/
│
├── federated_ids/
│
├── outputs/
│   ├── training_accuracy.png
│   ├── confusion_matrix.png
│   ├── evaluation_metrics.json
│   ├── final_model.json
│   ├── round_metrics.csv
│   └── run_log.txt
│
├── main.py
├── requirements.txt
└── README.md
```

## 9. Limitations

- The project uses synthetic network traffic rather than a real-world intrusion detection dataset.
- All four clients are simulated on a single computer using localhost.
- Logistic Regression is used as a simple baseline model.
- Additional security mechanisms such as differential privacy and secure aggregation are not implemented.

## 10. Future Scope

- Integrate real-world datasets such as CIC-IDS2017 or UNSW-NB15.
- Deploy clients across multiple physical computers.
- Experiment with advanced models such as Random Forest and Neural Networks.
- Implement secure aggregation and differential privacy.
- Extend the system to identify different categories of network attacks.
- Develop a real-time network intrusion monitoring dashboard.

## 11. Team Members

| Name | Role | Student ID |
|---|---|---|
| Rishit Jha | Team Lead | KU2407U181 |
| Team Member 2 | Team Member | KU2407U145 |
| Team Member 3 | Team Member | KU2407U214 |
| Team Member 4 | Team Member | KU2407U138 |

## 12. Conclusion

This project demonstrates how Federated Learning can be applied to network intrusion detection.

By allowing multiple clients to train models locally and share model parameters through TCP communication, the system demonstrates a distributed approach to machine learning without centrally collecting the original training data.

The project provides a practical introduction to the combination of Computer Networks, Machine Learning, and Cybersecurity.

---

**Academic Project:** Computer Networks Mini Project  
**Institution:** Unitedworld Institute of Technology, Karnavati University  
**Department:** Computer Science Engineering
