"""
fl_iot_simulation.py

Full simulation pipeline for federated learning experiments:
- Synthetic IoT data generation (clean + noisy + missing + outliers)
- Partition to clients (n_clients)
- Lightweight MLP model (input dim=3, hidden=8, output=1)
- Local training (SGD, 1 local epoch)
- Aggregation: FedAvg, Trimmed Mean (per-parameter), Krum (Byzantine-robust)
- Evaluation: MSE and R^2 on held-out clean test set
- Cross-validation and plotting
- Saves CSVs for datasets and PNG plots.

Adjust parameters in the "CONFIG" section below.
"""

import os
import random
from copy import deepcopy
import numpy as np
#import numpy as np
# import pandas as pd # Removed due to environment issues
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# CONFIG (canonical)
# -----------------------
RNG_SEED = 42

n_clients = 10
samples_per_client = 1200            # approx samples per client
input_dim = 3                        # temperature, humidity, pressure
hidden_units = 8
local_epochs = 5
local_lr = 0.01 # Updated to match user preference
l2_lambda = 1e-5                     # weight decay (L2)
communication_rounds = 100            # T
trim_beta = 0.20
krum_f = 2                           # assumed Byzantine clients for Krum
cv_folds = 5

# corruption parameters (canonical)
noise_sigma = 0.3                    # gaussian noise std
missing_rate = 0.05
outlier_rate = 0.02
outlier_range = (-10, 10)

# file outputs
OUT_DIR = "fl_sim_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# reproducibility
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

# -----------------------
# -----------------------
# Data generation helpers (REMOVED)
# -----------------------
# (Data generation logic removed as per user request; loading existing datasets instead)


# -----------------------
# MLP model (PyTorch)
# -----------------------

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=3, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # return (batch,)

def model_state_vector(state_dict):
    """Flatten state dict (tensor parameters) into 1D numpy vector and return shapes for remap."""
    vecs = []
    shapes = {}
    for k, v in state_dict.items():
        arr = v.cpu().numpy().reshape(-1)
        shapes[k] = v.shape
        vecs.append(arr)
    flat = np.concatenate(vecs)
    return flat, shapes

def vector_to_state_dict(vec, template_state):
    """Reconstruct a state dict from a flat vector using the template shapes/order."""
    out = {}
    offset = 0
    for k, v in template_state.items():
        numel = int(np.prod(v.shape))
        part = vec[offset:offset + numel]
        out[k] = torch.tensor(part.reshape(v.shape), dtype=torch.float32)
        offset += numel
    return out

def state_dict_to_numpy_vector(state_dict):
    """Convert state_dict OrderedDict to numpy vector, keeping consistent order."""
    parts = []
    for k, v in state_dict.items():
        parts.append(v.cpu().numpy().reshape(-1))
    return np.concatenate(parts)

# -----------------------
# Local training
# -----------------------

def local_train(model, X_local, y_local, epochs=5, lr=0.001, l2_lambda=0.0001, device='cpu'):
    """Train a model copy on local data and return updated state_dict."""
    model_local = deepcopy(model).to(device)
    model_local.train()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model_local.parameters(), lr=lr, weight_decay=l2_lambda)
    X_tensor = torch.tensor(X_local, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_local, dtype=torch.float32).to(device)

    # full-batch training (one epoch)
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model_local(X_tensor)
        loss = criterion(preds, y_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()
        
        # Clip gradients to prevent explosion with outliers
        torch.nn.utils.clip_grad_norm_(model_local.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Check for divergence
        if torch.isnan(loss):
            print("Warning: Local training divergence (NaN loss). Returning initial weights.")
            return deepcopy(model.state_dict())
    return deepcopy(model_local.state_dict())

# -----------------------
# Aggregation functions
# -----------------------

def fedavg(updates, weights=None):
    """Weighted average of parameter vectors (numpy arrays). updates: list of numpy arrays"""
    updates = np.array(updates)
    if weights is None:
        return np.mean(updates, axis=0)
    weights = np.array(weights).reshape(-1, 1)
    wsum = weights.sum()
    return (updates * (weights / (wsum))).sum(axis=0)

def trimmed_mean(updates, beta=trim_beta):
    """Per-parameter trimmed mean. updates is (n_clients, param_dim) numpy array"""
    updates = np.array(updates)
    n, d = updates.shape
    n_trim = int(np.floor(beta * n))
    if 2 * n_trim >= n:
        # degenerate: fallback to mean
        return np.mean(updates, axis=0)
    # for each coordinate, sort and trim
    sorted_coords = np.sort(updates, axis=0)
    trimmed = sorted_coords[n_trim: n - n_trim, :]
    return np.mean(trimmed, axis=0)

def krum(updates, f=krum_f, num_to_select=1):
    """Krum as per Blanchard et al. Selects single client update (most consistent).
       updates: numpy array (n_clients, dim)
       returns: chosen update vector (dim,)
    """
    updates = np.array(updates)
    n = updates.shape[0]
    if n <= num_to_select + 2:
        return np.mean(updates, axis=0)
    # compute pairwise distances
    dist = cdist(updates, updates, metric='euclidean')
    scores = []
    num_closest = n - f - 2
    for i in range(n):
        dists_i = np.sort(dist[i])
        # exclude self-distance at position 0
        scores.append(np.sum(dists_i[1: num_closest + 1]))
    krum_idx = int(np.argmin(scores))
    return updates[krum_idx]

# -----------------------
# Evaluation helpers
# -----------------------

def evaluate_model_on_dataset(state_dict, model_template, X_test, y_test):
    """Load state_dict into model, compute MSE and R2 on (X_test, y_test)."""
    model = deepcopy(model_template)
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v for k, v in state_dict.items()})
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_t).numpy()
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mse, r2, preds

# -----------------------
# Federated training routine
# -----------------------

def run_federated_experiment(df_clients, agg_method="fedavg", rounds=communication_rounds, device='cpu', return_trajectories=True, l2_lambda=0.01):
    """
    df_clients: list of client DataFrames (each with columns: temp, hum, pres, aggregator index)
    agg_method: 'fedavg', 'trimmed', or 'krum'
    returns: dict with train_mse/val_mse trajectories and final model state
    """
    # Build clean global test set: from original clean df in df_clients if provided separately; here we assume user passes a separate test or we create by sampling across clients
    # For simplicity, build test set by combining the first 10% of rows from each client (no corruption) OR accept externally
    model_template = SimpleMLP(input_dim=input_dim, hidden=hidden_units)
    global_model = SimpleMLP(input_dim=input_dim, hidden=hidden_units)
    global_state = global_model.state_dict()
    # Prepare client datasets (lists of numpy arrays, and sizes)
    # Prepare client datasets (lists of numpy arrays, and sizes)
    # df_clients is now list of (X, y) tuples
    client_data = df_clients 
    # (Previously it was list of DataFrames, now we pass pre-processed arrays)


    # Create global test set as a clean holdout: sample 10% per client (we assume df_clients correspond to clean or user-specified test)
    X_all = np.vstack([c[0] for c in client_data])
    y_all = np.concatenate([c[1] for c in client_data])
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.10, random_state=RNG_SEED)

    # Distribute training set across same client partitions proportionally (for evaluation only)
    # For simplicity we reuse the client_data arrays for local training and use the holdout X_test_all for final evaluation

    train_mse_traj = []
    val_mse_traj = []
    train_r2_traj = []
    val_r2_traj = []

    # compute client weights for FedAvg (by samples)
    client_sizes = [len(c[1]) for c in client_data]
    total_samples = sum(client_sizes)
    client_weights = [s / total_samples for s in client_sizes]

    # flatten template state to get param vector mapping
    # use torch state dict as template
    template_state = global_state

    for rnd in range(1, rounds + 1):
        # Each client trains locally and returns updated state
        updates = []
        for (Xc, yc) in client_data:
            # local training on copy of global model
            global_model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in global_state.items()})
            updated_state = local_train(global_model, Xc, yc, epochs=local_epochs, lr=local_lr, l2_lambda=l2_lambda, device=device)
            # convert to numpy vector
            vec = state_dict_to_numpy_vector(updated_state)
            updates.append(vec)
        updates = np.array(updates)  # shape (n_clients, dim)

        # Apply chosen aggregation
        if agg_method.lower() == "fedavg":
            new_vec = fedavg(updates, weights=client_sizes)
        elif agg_method.lower() == "trimmed":
            new_vec = trimmed_mean(updates, beta=trim_beta)
        elif agg_method.lower() == "krum":
            new_vec = krum(updates, f=krum_f)
        else:
            raise ValueError("Unknown agg method")

        # Map new_vec to state_dict format
        # Build mapping to reconstruct keys from template_state
        # First get template parameter ordering and shapes
        # Convert template_state (torch tensors) to an ordered dict list
        keys = list(template_state.keys())
        shapes = [template_state[k].shape for k in keys]
        # reconstruct state dict
        offset = 0
        new_state = {}
        for k, s in zip(keys, shapes):
            numel = int(np.prod(s))
            part = new_vec[offset: offset + numel]
            new_state[k] = torch.tensor(part.reshape(s), dtype=torch.float32)
            offset += numel
        global_state = new_state

        # Evaluate global model on training union and test holdout (we report test holdout metrics)
        train_mse, train_r2, _ = evaluate_model_on_dataset(global_state, model_template, X_train_all, y_train_all)
        val_mse, val_r2, _ = evaluate_model_on_dataset(global_state, model_template, X_test_all, y_test_all)

        train_mse_traj.append(train_mse)
        val_mse_traj.append(val_mse)
        train_r2_traj.append(train_r2)
        val_r2_traj.append(val_r2)

        # Optional: print progress
        if rnd % 5 == 0 or rnd == 1 or rnd == rounds:
            print(f"[{agg_method}] Round {rnd}/{rounds}  Train MSE {train_mse:.4f}  Val MSE {val_mse:.4f}  Val R2 {val_r2:.4f}")

    results = {
        "final_state": global_state,
        "train_mse_traj": train_mse_traj,
        "val_mse_traj": val_mse_traj,
        "train_r2_traj": train_r2_traj,
        "val_r2_traj": val_r2_traj
    }
    return results

# -----------------------
# Cross-validation routine (per-aggregation evaluation)
# -----------------------

def cross_val_experiment(data_all, agg_method="fedavg", n_splits=cv_folds, l2_lambda=0.01, device='cpu'):
    """Perform k-fold cross-validation. data_all is (X, y) tuple."""
    X_all, y_all = data_all
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
    fold_mses = []
    fold_r2s = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        # Build per-client partition from training indices
        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        
        # Split into n_clients chunks
        n_train = len(train_idx)
        chunk_size = n_train // n_clients
        client_data_list = []
        for i in range(n_clients):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_clients - 1 else n_train
            Xc = X_train[start:end]
            yc = y_train[start:end]
            client_data_list.append((Xc, yc))
            
        # run federated training
        res = run_federated_experiment(client_data_list, agg_method=agg_method, rounds=communication_rounds, l2_lambda=l2_lambda, device=device, return_trajectories=True)
        # evaluate final model
        model_template = SimpleMLP(input_dim=input_dim, hidden=hidden_units)
        final_state = res["final_state"]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]
        mse, r2, _ = evaluate_model_on_dataset(final_state, model_template, X_test, y_test)
        fold_mses.append(mse)
        fold_r2s.append(r2)
        print(f"CV fold {fold+1}/{n_splits} (agg={agg_method})  MSE={mse:.4f}  R2={r2:.4f}")

    return {"fold_mses": fold_mses, "fold_r2s": fold_r2s}

# -----------------------
# Plotting helpers
# -----------------------

def plot_train_val_trajectories(results_dict, outpath):
    """Plot training and validation MSE/R2 trajectories for multiple aggregation methods.
       results_dict: {method_name: results returned by run_federated_experiment}
    """
    plt.figure(figsize=(10,6))
    for method, res in results_dict.items():
        rounds = np.arange(1, len(res["train_mse_traj"]) + 1)
        plt.plot(rounds, res["train_mse_traj"], label=f"{method} - Train MSE")
        plt.plot(rounds, res["val_mse_traj"], linestyle='--', label=f"{method} - Val MSE")
    plt.xlabel("Communication rounds")
    plt.ylabel("MSE")
    plt.title("Training and Validation MSE per Communication Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved plot:", outpath)

def plot_cv_bars(cv_results, outpath):
    """cv_results: dict {method_name: {'fold_mses': [...], 'fold_r2s':[...]} }"""
    methods = list(cv_results.keys())
    mses = [np.mean(cv_results[m]['fold_mses']) for m in methods]
    stds = [np.std(cv_results[m]['fold_mses']) for m in methods]
    x = np.arange(len(methods))
    plt.figure(figsize=(8,5))
    plt.bar(x, mses, yerr=stds, capsize=5)
    plt.xticks(x, methods)
    plt.ylabel("Mean MSE (CV)")
    plt.title("Cross-Validation MSE by Aggregation Method")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved CV bar plot:", outpath)

def plot_comparison(results_clean, results_corrupted, dataset_name, outpath):
    """Plot Clean vs Corrupted convergence trajectories."""
    plt.figure(figsize=(12, 7))
    colors = {'fedavg': 'blue', 'trimmed': 'green', 'krum': 'red'}
    
    # Plot Clean (Solid)
    for method, res in results_clean.items():
        rounds = np.arange(1, len(res["train_mse_traj"]) + 1)
        plt.plot(rounds, res["val_mse_traj"], linestyle='-', color=colors.get(method, 'black'), alpha=0.6, label=f"{method} (Clean)")
        
    # Plot Corrupted (Dashed)
    for method, res in results_corrupted.items():
        rounds = np.arange(1, len(res["train_mse_traj"]) + 1)
        plt.plot(rounds, res["val_mse_traj"], linestyle='--', linewidth=2, color=colors.get(method, 'black'), label=f"{method} ({dataset_name})")

    plt.xlabel("Communication rounds")
    plt.ylabel("Validation MSE")
    plt.title(f"Convergence Comparison: Clean vs {dataset_name.capitalize()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved comparison plot:", outpath)

def plot_train_val_r2(results_dict, outpath):
    """Plot training and validation R2 trajectories for multiple aggregation methods.
       results_dict: {method_name: results returned by run_federated_experiment}
    """
    plt.figure(figsize=(10,6))
    for method, res in results_dict.items():
        rounds = np.arange(1, len(res["train_r2_traj"]) + 1)
        # Plot Train (Solid)
        plt.plot(rounds, res["train_r2_traj"], linestyle='-', label=f"{method} - Train R2")
        # Plot Val (Dashed)
        plt.plot(rounds, res["val_r2_traj"], linestyle='--', label=f"{method} - Val R2")
    
    plt.xlabel("Communication rounds")
    plt.ylabel("Accuracy (R2 Score)")
    plt.title("Training and Validation Accuracy (R2) per Communication Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved Train/Val R2 plot:", outpath)

def plot_r2_trajectories(results_dict, outpath):
    """Plot validation R2 trajectories for multiple aggregation methods."""
    plt.figure(figsize=(10,6))
    for method, res in results_dict.items():
        rounds = np.arange(1, len(res["val_r2_traj"]) + 1)
        plt.plot(rounds, res["val_r2_traj"], linewidth=2, label=f"{method} - Val R2")
    
    plt.xlabel("Communication rounds")
    plt.ylabel("R2 Score")
    plt.title("Validation R2 Score per Communication Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved R2 plot:", outpath)

def plot_l2_comparison(results_with_l2, results_no_l2, dataset_name, outpath):
    """Plot bar chart comparing final R2 score With vs Without L2."""
    # results_with_l2/no_l2 are dicts: {method: result_dict}
    methods = list(results_with_l2.keys())
    
    # Extract final R2 scores
    r2_with = [results_with_l2[m]["val_r2_traj"][-1] for m in methods]
    r2_no = [results_no_l2[m]["val_r2_traj"][-1] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    plt.figure(figsize=(9, 6))
    
    # Match standard colors: Blue (Without L2), Orange (With L2)
    plt.bar(x - width/2, r2_no, width, label='Without L2', color='#1f77b4')
    plt.bar(x + width/2, r2_with, width, label='With L2', color='#ff7f0e')
    
    plt.ylabel('R² Score (Accuracy)')
    
    # Formatting title: "L2 Regularization on X Dataset"
    # Handle "Missing" -> "Missing Values" if needed to match example strictly
    ds_title = dataset_name.capitalize()
    if ds_title.lower() == "missing":
        ds_title = "Missing Values"
        
    plt.title(f'L2 Regularization on {ds_title} Dataset')
    plt.xticks(x, methods)
    
    # Remove fixed ylim to show negative values if present
    # plt.ylim(0, 1.05)
    print(f"DEBUG: {dataset_name} | No L2: {r2_no} | With L2: {r2_with}")
    
    plt.legend()
    # Add grid lines for y-axis
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved L2 comparison plot:", outpath)

def plot_l2_cv_comparison(cv_results_with_l2, cv_results_no_l2, dataset_name, outpath):
    """Plot bar chart comparing Mean CV R2 score With vs Without L2."""
    # cv_results_* are dicts: {method: {'fold_mses': [], 'fold_r2s': []}}
    methods = list(cv_results_with_l2.keys())
    
    means_with = [np.mean(cv_results_with_l2[m]['fold_r2s']) for m in methods]
    stds_with = [np.std(cv_results_with_l2[m]['fold_r2s']) for m in methods]
    
    means_no = [np.mean(cv_results_no_l2[m]['fold_r2s']) for m in methods]
    stds_no = [np.std(cv_results_no_l2[m]['fold_r2s']) for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    plt.figure(figsize=(9, 6))
    
    # Match standard colors: Blue (Without L2), Orange (With L2)
    # Adding error bars (capsize=5) for CV variability
    plt.bar(x - width/2, means_no, width, yerr=stds_no, capsize=5, label='Without L2', color='#1f77b4', alpha=0.9)
    plt.bar(x + width/2, means_with, width, yerr=stds_with, capsize=5, label='With L2', color='#ff7f0e', alpha=0.9)
    
    plt.ylabel('Mean Cross-Validation R² Score')
    
    ds_title = dataset_name.capitalize()
    if ds_title.lower() == "missing":
        ds_title = "Missing Values"
        
    plt.title(f'L2 Regularization Impact on {ds_title} Dataset (CV R²)')
    plt.xticks(x, methods)
    
    # Print values for verification
    print(f"DEBUG: {dataset_name} (CV Mean R2)")
    print(f"  No L2: {means_no}")
    print(f"  With L2: {means_with}")
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved L2 CV Comparison plot:", outpath)

# -----------------------
# Main experiment driver
# -----------------------

def load_data_numpy(filepath):
    """Load X (3 cols) and y (1 col) from CSV using numpy."""
    # timestamp, temperature_C, humidity_pct, pressure_hPa, aggregator_sensor_index
    # We ignore timestamp (col 0).
    # Cols: 1, 2, 3 are features. 4 is target.
    # Skip header.
    try:
        raw = np.genfromtxt(filepath, delimiter=',', skip_header=1, encoding='utf-8')
        # If encoding fails or timestamp is string, genfromtxt might return nan for string cols if not specified.
        # But we only need cols 1-4.
        # Let's specify usecols to be safe and avoid string parsing issues.
        data_numeric = np.genfromtxt(filepath, delimiter=',', skip_header=1, usecols=(1,2,3,4))
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        # Fallback using standard python read if genfromtxt fails on strings
        with open(filepath, 'r') as f:
            lines = f.readlines()[1:] # skip header
        data = []
        for line in lines:
            parts = line.strip().split(',')
            # timestamp is parts[0], we want parts[1:5]
            vals = [float(p) for p in parts[1:5]]
            data.append(vals)
        data_numeric = np.array(data)
        
    X = data_numeric[:, :3].astype(np.float32)
    y = data_numeric[:, 3].astype(np.float32)
    
    # Impute NaNs in X if any (simple column mean)
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        
    # Impute NaNs in y if any
    if np.isnan(y).any():
        y_mean = np.nanmean(y)
        y[np.isnan(y)] = y_mean
        
    # Normalize features to prevent exploding gradients (Pressure ~1013 vs Temp ~20)
    # Use nanmean/nanstd to be safe, though imputation should have handled it.
    # Normalize features to prevent exploding gradients (Pressure ~1013 vs Temp ~20)
    # Use nanmean/nanstd to be safe, though imputation should have handled it.
    # mean = np.nanmean(X, axis=0)
    # std = np.nanstd(X, axis=0) + 1e-9
    # X = (X - mean) / std
    pass
    
    # Final safety check for X and y
    if not np.isfinite(X).all():
        print(f"Warning: Non-finite values found in X of {filepath}. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
    if not np.isfinite(y).all():
        print(f"Warning: Non-finite values found in y of {filepath}. Replacing with mean.")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y

def split_into_clients(X, y, n_clients):
    """Split X, y into n_clients chunks."""
    n = len(y)
    chunk_size = n // n_clients
    clients = []
    for i in range(n_clients):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_clients - 1 else n
        clients.append((X[start:end], y[start:end]))
    return clients

def main():
    # Load existing datasets
    print("Loading datasets from", OUT_DIR)
    
    files = {
        "clean": "clean_normalized.csv",
        "noisy": "noisy_normalized.csv",
        "missing": "missing_normalized.csv",
        "outliers": "outliers_normalized.csv"
    }
    
    datasets = {}
    for key, fname in files.items():
        path = os.path.join(OUT_DIR, fname)
        print(f"Loading {key} from {path}...")
        X, y = load_data_numpy(path)
        datasets[key] = (X, y)

    # Clean dataset for canonical test/CV
    data_clean = datasets['clean']
    data_noisy = datasets['noisy']
    data_missing = datasets['missing']
    data_outliers = datasets['outliers']

    # Partition into clients
    client_dfs_clean = split_into_clients(*data_clean, n_clients)
    client_dfs_noisy = split_into_clients(*data_noisy, n_clients)
    client_dfs_missing = split_into_clients(*data_missing, n_clients)
    client_dfs_outliers = split_into_clients(*data_outliers, n_clients)

    # Run federated experiment on clean dataset for three aggregation methods
    methods = ["fedavg", "trimmed", "krum"]
    results_clean = {}
    for m in methods:
        print("\n=== Running on CLEAN dataset, method:", m)
        res = run_federated_experiment(client_dfs_clean, agg_method=m, rounds=communication_rounds)
        results_clean[m] = res

    # Plot training/val trajectories (one plot with three methods)
    # Plot training/val trajectories (one plot with three methods)
    plot_train_val_trajectories(results_clean, os.path.join(OUT_DIR, "train_val_mse_clean.png"))
    plot_train_val_r2(results_clean, os.path.join(OUT_DIR, "train_val_r2_clean.png"))
    plot_r2_trajectories(results_clean, os.path.join(OUT_DIR, "val_r2_clean.png"))

    # L2 Comparison for Clean
    print("\n=== Running L2 comparison for CLEAN dataset (Without L2) ===")
    results_clean_no_l2 = {}
    for m in methods:
        res = run_federated_experiment(client_dfs_clean, agg_method=m, rounds=communication_rounds, l2_lambda=0.0)
        results_clean_no_l2[m] = res
    plot_l2_comparison(results_clean, results_clean_no_l2, "clean", os.path.join(OUT_DIR, "l2_comparison_clean.png"))

    # Cross-validation for Clean dataset
    print("\n*** Cross-validation for CLEAN dataset")
    cv_res_clean = {}
    for m in methods:
        cv = cross_val_experiment(data_clean, agg_method=m)
        cv_res_clean[m] = cv
    plot_cv_bars(cv_res_clean, os.path.join(OUT_DIR, "cv_mse_clean.png"))

    # Run experiments on corrupted datasets to measure degradation (one dataset at a time)
    dataset_map = {
        # "noisy": client_dfs_noisy,
        # "missing": client_dfs_missing,
        "outliers": client_dfs_outliers
    }
    # map to full data for CV
    full_data_map = {
        "noisy": data_noisy,
        "missing": data_missing,
        "outliers": data_outliers
    }

    all_results = {} # Store all results for robustness calc
    cv_results = {}

    for ds_name, client_list in dataset_map.items():
        print(f"\n*** Running experiments on dataset: {ds_name}")
        per_method_results = {}
        for m in methods:
            res = run_federated_experiment(client_list, agg_method=m, rounds=communication_rounds)
            per_method_results[m] = res
        
        all_results[ds_name] = per_method_results
        
        # Save per-dataset plot
        plot_train_val_trajectories(per_method_results, os.path.join(OUT_DIR, f"train_val_mse_{ds_name}.png"))
        plot_train_val_r2(per_method_results, os.path.join(OUT_DIR, f"train_val_r2_{ds_name}.png"))
        plot_r2_trajectories(per_method_results, os.path.join(OUT_DIR, f"val_r2_{ds_name}.png"))
        
        # Save Comparison Plot (Clean vs This Data)
        plot_comparison(results_clean, per_method_results, ds_name, os.path.join(OUT_DIR, f"comparison_clean_vs_{ds_name}.png"))

        # L2 Comparison for Corrupted
        print(f"\n=== Running L2 comparison for {ds_name} dataset (Without L2) ===")
        results_no_l2 = {}
        for m in methods:
            res = run_federated_experiment(client_list, agg_method=m, rounds=communication_rounds, l2_lambda=0.0)
            results_no_l2[m] = res
        plot_l2_comparison(per_method_results, results_no_l2, ds_name, os.path.join(OUT_DIR, f"l2_comparison_{ds_name}.png"))

        # Cross-validate on whole corrupted combined df for this dataset
        print(f"\n*** Cross-validation for dataset {ds_name}")
        cv_res = {}
        data_all = full_data_map[ds_name]
        for m in methods:
            cv = cross_val_experiment(data_all, agg_method=m)
            cv_res[m] = cv
        plot_cv_bars(cv_res, os.path.join(OUT_DIR, f"cv_mse_{ds_name}.png"))
        cv_results[ds_name] = cv_res

    # -----------------------
    # Robustness Score Evaluation
    # -----------------------
    print("\n" + "="*80)
    print("ROBUSTNESS SCORE EVALUATION TABLE")
    print("="*80)
    print(f"| Dataset | Method | Clean MSE | Corrupted MSE | Robustness Score (Degradation %) |")
    print(f"| :--- | :--- | :--- | :--- | :--- |")
    
    # helper to get final MSE
    def get_final_mse(results_dict, method):
        # last val mse
        return results_dict[method]["val_mse_traj"][-1]

    # Iterate over datasets including clean
    table_lines = []
    table_lines.append("| Dataset | Method | Clean MSE | Corrupted MSE | Robustness Score (Degradation %) |")
    table_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    for ds_name in ["clean", "noisy", "missing", "outliers"]:
        for m in methods:
            try:
                clean_mse = get_final_mse(results_clean, m)
                if ds_name == "clean":
                    corr_mse = clean_mse
                else:
                    corr_mse = get_final_mse(all_results[ds_name], m)
                degradation = ((corr_mse - clean_mse) / clean_mse) * 100.0
                line = f"| {ds_name.capitalize()} | {m} | {clean_mse:.4f} | {corr_mse:.4f} | {degradation:.2f}% |"
            except KeyError:
                line = f"| {ds_name.capitalize()} | {m} | N/A | N/A | Error |"
            table_lines.append(line)
            print(line)

    # Save table to file
    with open(os.path.join(OUT_DIR, "robustness_table.md"), "w") as f:
        f.write("\n".join(table_lines))
    print(f"\nRobustness table saved to {os.path.join(OUT_DIR, 'robustness_table.md')}")


    print("\nAll experiments complete. Output directory:", OUT_DIR)

if __name__ == "__main__":
    main()
