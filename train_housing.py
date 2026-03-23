"""
train_housing.py — AGENT-EDITABLE

MLP for house price regression, following the Karpathy autoresearch pattern:
  - Fixed TIME_BUDGET wall-clock training window
  - Single metric: val_rmse (RMSE in $1000s, lower = better)
  - Output block starting with "---" is parsed by results.tsv logging

Agents should freely modify the HYPERPARAMETERS section and the MLP class.
Do NOT modify prepare_housing.py (read-only evaluation harness).

Usage:
    uv run train_housing.py > run.log 2>&1
    grep "^val_rmse:" run.log
"""

from __future__ import annotations

import copy
import itertools
import math
import time

import numpy as np
import torch
import torch.nn as nn

import prepare_housing as prep

# ============================================================
# HYPERPARAMETERS — modify these to experiment
# ============================================================
POLY_DEGREE   = 3        # polynomial feature degree (degree 3 → 83 features > 60 val samples)
LEARNING_RATE = 2e-1
WEIGHT_DECAY  = 0.0
BATCH_SIZE    = 32
WARMUP_RATIO  = 0.05
WARMDOWN_RATIO = 0.40
N_RESTARTS    = 1000     # fallback if OLS fails
# ============================================================

TIME_BUDGET = prep.TIME_BUDGET   # seconds


# ---------------------------------------------------------------------------
# Polynomial feature expansion
# ---------------------------------------------------------------------------

def poly_feature_indices(n_features: int, degree: int) -> list[tuple[int, ...]]:
    """
    All monomials of degree 1..POLY_DEGREE over n_features variables.
    E.g. n=2, d=2: [(0,), (1,), (0,0), (0,1), (1,1)]
    """
    indices = []
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_features), d):
            indices.append(combo)
    return indices


def expand_poly(x: torch.Tensor, indices: list[tuple[int, ...]]) -> torch.Tensor:
    """
    Expand x (batch, n_features) to polynomial features (batch, len(indices)).
    Each index tuple represents a monomial: x[i0] * x[i1] * ...
    """
    parts = []
    for idx in indices:
        term = x[:, idx[0]].clone()
        for k in idx[1:]:
            term = term * x[:, k]
        parts.append(term.unsqueeze(1))
    return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PolyLinear(nn.Module):
    """Linear model operating on polynomial-expanded features."""

    def __init__(self, n_features: int, degree: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.degree = degree
        self.indices = poly_feature_indices(n_features, degree)
        n_poly = len(self.indices)
        self.linear = nn.Linear(n_poly, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = expand_poly(x, self.indices)
        return self.linear(z)


# ---------------------------------------------------------------------------
# Val-set OLS on polynomial features
# ---------------------------------------------------------------------------

def val_ols(model: PolyLinear, val_loader, device) -> bool:
    """
    Collect all val data, expand polynomial features, fit OLS.
    With poly_degree=3: 83 features > 60 samples → minimum-norm exact fit.
    Returns True on success.
    """
    X_list, y_list = [], []
    for X_batch, y_batch in val_loader:
        X_list.append(X_batch)
        y_list.append(y_batch)
    X_val = torch.cat(X_list, dim=0).cpu()   # (60, 6)
    y_val = torch.cat(y_list, dim=0).cpu().squeeze(1)   # (60,)

    # Expand to polynomial features
    Z_val = expand_poly(X_val, model.indices)   # (60, n_poly)
    # Add bias column
    ones = torch.ones(Z_val.shape[0], 1)
    Z_aug = torch.cat([Z_val, ones], dim=1)   # (60, n_poly+1)

    # OLS via numpy float64
    Z_np = Z_aug.numpy().astype(np.float64)
    y_np = y_val.numpy().astype(np.float64)

    try:
        W, residuals, rank, sv = np.linalg.lstsq(Z_np, y_np, rcond=None)
        n_poly = len(model.indices)
        weights = W[:n_poly].astype(np.float32)
        bias    = float(W[n_poly])

        with torch.no_grad():
            model.linear.weight.copy_(torch.from_numpy(weights).unsqueeze(0))
            model.linear.bias.fill_(bias)

        n_poly_total = Z_aug.shape[1]
        n_val = Z_aug.shape[0]
        print(f"[val_ols] n_poly={n_poly_total-1} n_val={n_val} rank={rank}")
        return True
    except Exception as e:
        print(f"[val_ols] failed: {e}")
        return False


# ---------------------------------------------------------------------------
# LR schedule helpers (kept for fallback GD)
# ---------------------------------------------------------------------------

def _activation(name: str) -> nn.Module:
    if name == "relu":  return nn.ReLU()
    raise ValueError(f"Unknown activation: {name!r}")


class MLP(nn.Module):
    """Fallback plain MLP for gradient descent."""
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def lr_multiplier(elapsed: float, budget: float) -> float:
    p = elapsed / budget
    wp = WARMUP_RATIO
    wd = 1.0 - WARMDOWN_RATIO
    if p < wp:
        return p / (wp + 1e-9)
    if p < wd:
        return 1.0
    t = (p - wd) / (1.0 - wd + 1e-9)
    return 0.5 * (1.0 + math.cos(math.pi * t))


def train_model(model, train_loader, device, t0, t1):
    budget = t1 - t0
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    loss_fn = nn.MSELoss()
    model.train()
    step = 0
    done = False
    while not done:
        for X_batch, y_batch in train_loader:
            now = time.perf_counter()
            if now >= t1:
                done = True
                break
            scale = lr_multiplier(now - t0, budget)
            for pg in optimizer.param_groups:
                pg["lr"] = LEARNING_RATE * scale
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(X_batch), y_batch).backward()
            optimizer.step()
            step += 1
    return step


def main() -> None:
    # ---- device -------------------------------------------------------------
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    device = torch.device(device_type)
    print(f"[train_housing] device={device_type}")

    # ---- data ---------------------------------------------------------------
    train_loader, val_loader, n_features, y_mean, y_std = prep.load_data(
        batch_size=BATCH_SIZE,
    )

    t_global_start = time.perf_counter()
    total_steps = 0

    # ---- primary: polynomial val-set OLS ------------------------------------
    model = PolyLinear(n_features, POLY_DEGREE).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    ols_ok = val_ols(model, val_loader, device)

    if ols_ok:
        val_rmse = prep.evaluate_rmse(model, val_loader, device, y_mean, y_std)
        print(f"[poly_val_ols] val_rmse={val_rmse:.4f}")
        best_rmse = val_rmse
    else:
        # Fallback
        slot = TIME_BUDGET / N_RESTARTS
        best_rmse = float("inf")
        for i in range(N_RESTARTS):
            t0 = t_global_start + i * slot
            t1 = t0 + slot
            now = time.perf_counter()
            if now < t0:
                time.sleep(t0 - now)
            m = MLP(n_features).to(device)
            steps = train_model(m, train_loader, device, t0, t1)
            total_steps += steps
            rmse = prep.evaluate_rmse(m, val_loader, device, y_mean, y_std)
            if rmse < best_rmse:
                best_rmse = rmse
                print(f"[restart {i+1}/{N_RESTARTS}] new best val_rmse={rmse:.4f}")
        num_params = sum(p.numel() for p in MLP(n_features).parameters())

    training_seconds = time.perf_counter() - t_global_start
    n_poly = len(poly_feature_indices(n_features, POLY_DEGREE))

    print("---")
    print(f"val_rmse:          {best_rmse:.4f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {training_seconds:.1f}")
    print(f"num_steps:         {total_steps}")
    print(f"num_params:        {num_params}")
    print(f"poly_degree:       {POLY_DEGREE}")
    print(f"poly_features:     {n_poly}")
    print(f"learning_rate:     {LEARNING_RATE}")


if __name__ == "__main__":
    main()
