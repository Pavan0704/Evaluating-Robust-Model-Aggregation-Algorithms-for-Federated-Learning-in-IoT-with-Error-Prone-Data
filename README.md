# Evaluating-Robust-Model-Aggregation-Algorithms-for-Federated-Learning-in-IoT-with-Error-Prone-Data
A comprehensive Federated Learning (FL) simulation pipeline designed to evaluate the robustness of various aggregation algorithms against data corruption in IoT environments. This repository implements and compares FedAvg, Trimmed Mean, and Krum strategies using a PyTorch-based Multi-Layer Perceptron (MLP).


# Key Features:
Robust Aggregation: Implements Byzantine-robust algorithms (Krum, Trimmed Mean) alongside standard FedAvg to handle malicious or noisy client updates.Data Corruption Simulation: Evaluates model performance across multiple data scenarios, including clean, noisy, missing values, and outlier-contaminated datasets.Preprocessing Pipeline: Includes scripts for data normalization using RobustScaler and median imputation to handle anomalies before training.Statistical Analysis: Tools for exploratory data analysis (EDA), including feature correlation checks and statistical distribution profiling.Performance Metrics: visualizing convergence trajectories (MSE, $R^2$) and cross-validation results to quantify robustness degradation.
