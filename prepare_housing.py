"""
prepare_housing.py — READ-ONLY (do not modify)

Housing price data pipeline, analogous to prepare.py in the LLM autoresearch.
Generates a synthetic housing CSV on first run, then handles:
  - Feature normalization (fit on train, apply to val)
  - Train / val split
  - DataLoader creation
  - evaluate_rmse() — the single ground-truth metric

The agent must NOT modify this file. It is the fixed evaluation harness.
"""

from __future__ import annotations

import csv
import os
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Constants (mirroring prepare.py layout)
# ---------------------------------------------------------------------------
CSV_PATH       = "housing.csv"
TARGET_COL     = "price_k"           # price in $1000s
FEATURE_COLS   = ["sqft_living", "bedrooms", "bathrooms", "age", "has_garage", "floors"]
N_SAMPLES      = 300
EVAL_FRACTION  = 0.20                # 20 % held-out validation
RANDOM_SEED    = 42
TIME_BUDGET    = 60                  # seconds of training wall-clock time


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_housing_csv(path: str = CSV_PATH, n: int = N_SAMPLES, seed: int = RANDOM_SEED) -> None:
    """
    Generate a small synthetic housing dataset and write it to *path*.
    Price formula (in $1000s):
        price = 150 + 0.10*sqft + 5*beds + 15*baths - 1.5*age + 20*garage + 10*floors + N(0,25)
    """
    rng = np.random.default_rng(seed)

    sqft     = rng.uniform(500.0, 4000.0, n)
    bedrooms = rng.integers(1, 6, n).astype(float)          # 1–5
    bathrooms = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], n)
    age      = rng.uniform(0.0, 50.0, n)
    garage   = rng.integers(0, 2, n).astype(float)          # 0 or 1
    floors   = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], n)

    price = (
        150.0
        + 0.10  * sqft
        + 5.0   * bedrooms
        + 15.0  * bathrooms
        - 1.5   * age
        + 20.0  * garage
        + 10.0  * floors
        + rng.normal(0, 25, n)
    ).clip(80.0, 950.0)

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FEATURE_COLS + [TARGET_COL])
        for i in range(n):
            writer.writerow([
                round(sqft[i], 1),
                int(bedrooms[i]),
                bathrooms[i],
                round(age[i], 1),
                int(garage[i]),
                floors[i],
                round(price[i], 1),
            ])

    print(f"[prepare_housing] Generated {n} rows → {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, int, float, float]:
    """
    Returns:
        train_loader, val_loader, n_features, y_mean, y_std

    Features are z-scored (fit on train split only).
    Targets are also z-scored; y_mean / y_std let you denormalize predictions.
    """
    if not os.path.exists(CSV_PATH):
        generate_housing_csv(CSV_PATH)

    # ---- load ---------------------------------------------------------------
    rows: list[list[str]] = []
    with open(CSV_PATH, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    n_total = len(rows)
    X = np.array([[float(r[c]) for c in FEATURE_COLS] for r in rows], dtype=np.float32)
    y = np.array([float(r[TARGET_COL]) for r in rows], dtype=np.float32)

    # ---- shuffle & split ----------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(n_total)
    X, y = X[idx], y[idx]

    n_val = max(1, int(n_total * EVAL_FRACTION))
    X_val,   y_val   = X[:n_val],  y[:n_val]
    X_train, y_train = X[n_val:],  y[n_val:]

    # ---- feature normalisation (fit on train) --------------------------------
    x_mean = X_train.mean(axis=0)
    x_std  = X_train.std(axis=0) + 1e-8
    X_train = (X_train - x_mean) / x_std
    X_val   = (X_val   - x_mean) / x_std

    # ---- target normalisation -----------------------------------------------
    y_mean = float(y_train.mean())
    y_std  = float(y_train.std() + 1e-8)
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm   = (y_val   - y_mean) / y_std

    # ---- tensors & loaders --------------------------------------------------
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train_norm).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val_norm).unsqueeze(1),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    n_features = X_train.shape[1]
    print(
        f"[prepare_housing] {n_total} rows | "
        f"train={len(train_ds)} val={len(val_ds)} | "
        f"features={n_features} | "
        f"y_mean={y_mean:.1f}k y_std={y_std:.1f}k"
    )
    return train_loader, val_loader, n_features, y_mean, y_std


# ---------------------------------------------------------------------------
# Evaluation metric — the ground-truth RMSE (never changes)
# ---------------------------------------------------------------------------

def evaluate_rmse(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> float:
    """
    Compute RMSE on the validation set, denormalized back to $1000s.
    Lower is better (mirrors val_bpb).
    """
    model.eval()
    sq_err_sum = 0.0
    n_total    = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            pred_real = pred * y_std + y_mean
            y_real    = y_batch * y_std + y_mean
            sq_err_sum += ((pred_real - y_real) ** 2).sum().item()
            n_total    += y_batch.size(0)
    model.train()
    return (sq_err_sum / n_total) ** 0.5
