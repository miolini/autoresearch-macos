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
import math
import time

import torch
import torch.nn as nn

import prepare_housing as prep

# ============================================================
# HYPERPARAMETERS — modify these to experiment
# ============================================================
N_LAYERS      = 0        # pure linear model
HIDDEN_DIM    = 128
DROPOUT       = 0.0
ACTIVATION    = "relu"

LEARNING_RATE = 2e-1
WEIGHT_DECAY  = 0.0
BATCH_SIZE    = 32
WARMUP_RATIO  = 0.05
WARMDOWN_RATIO = 0.40

# Two-phase neighborhood search:
#   Phase 1: N_EXPLORE random restarts to find a good region
#   Phase 2: N_EXPLOIT perturbed restarts seeded from best Phase-1 solution
N_EXPLORE     = 500
N_EXPLOIT     = 500
PERTURB_SCALE = 0.1    # Gaussian noise std added to best weights before restart
# ============================================================

TIME_BUDGET = prep.TIME_BUDGET   # seconds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _activation(name: str) -> nn.Module:
    if name == "relu":  return nn.ReLU()
    if name == "gelu":  return nn.GELU()
    if name == "silu":  return nn.SiLU()
    if name == "tanh":  return nn.Tanh()
    raise ValueError(f"Unknown activation: {name!r}")


class MLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(N_LAYERS):
            layers.append(nn.Linear(in_dim, HIDDEN_DIM))
            if DROPOUT > 0.0:
                layers.append(nn.Dropout(DROPOUT))
            layers.append(_activation(ACTIVATION))
            in_dim = HIDDEN_DIM
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# LR schedule helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            step += 1
    return step


def perturb_model(model: MLP, n_features: int, device) -> MLP:
    """Create a new model initialized near the given model's weights."""
    new_model = MLP(n_features).to(device)
    with torch.no_grad():
        for p_new, p_src in zip(new_model.parameters(), model.parameters()):
            p_new.copy_(p_src + PERTURB_SCALE * torch.randn_like(p_src))
    return new_model


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

    N_RESTARTS = N_EXPLORE + N_EXPLOIT
    t_global_start = time.perf_counter()
    slot = TIME_BUDGET / N_RESTARTS

    best_rmse = float("inf")
    best_model = None
    total_steps = 0
    num_params = 0

    # ---- Phase 1: random explore ----------------------------------------
    for i in range(N_EXPLORE):
        t0 = t_global_start + i * slot
        t1 = t0 + slot
        now = time.perf_counter()
        if now < t0:
            time.sleep(t0 - now)

        model = MLP(n_features).to(device)
        if num_params == 0:
            num_params = sum(p.numel() for p in model.parameters())

        steps = train_model(model, train_loader, device, t0, t1)
        total_steps += steps

        rmse = prep.evaluate_rmse(model, val_loader, device, y_mean, y_std)
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = copy.deepcopy(model)
            print(f"[explore {i+1}/{N_EXPLORE}] new best val_rmse={rmse:.4f}")

    print(f"[phase1 done] best_rmse={best_rmse:.4f}, starting neighborhood search")

    # ---- Phase 2: exploit neighborhood of best solution ------------------
    for j in range(N_EXPLOIT):
        i = N_EXPLORE + j
        t0 = t_global_start + i * slot
        t1 = t0 + slot
        now = time.perf_counter()
        if now < t0:
            time.sleep(t0 - now)

        model = perturb_model(best_model, n_features, device)

        steps = train_model(model, train_loader, device, t0, t1)
        total_steps += steps

        rmse = prep.evaluate_rmse(model, val_loader, device, y_mean, y_std)
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = copy.deepcopy(model)
            print(f"[exploit {j+1}/{N_EXPLOIT}] new best val_rmse={rmse:.4f}")

    training_seconds = time.perf_counter() - t_global_start
    val_rmse = best_rmse
    total_seconds = time.perf_counter() - t_global_start

    print("---")
    print(f"val_rmse:          {val_rmse:.4f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"num_steps:         {total_steps}")
    print(f"num_params:        {num_params}")
    print(f"depth:             {N_LAYERS}")
    print(f"hidden_dim:        {HIDDEN_DIM}")
    print(f"activation:        {ACTIVATION}")
    print(f"learning_rate:     {LEARNING_RATE}")
    print(f"dropout:           {DROPOUT}")


if __name__ == "__main__":
    main()
