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

import math
import time

import torch
import torch.nn as nn

import prepare_housing as prep

# ============================================================
# HYPERPARAMETERS — modify these to experiment
# ============================================================
HIDDEN_DIM    = 128      # width of each hidden layer (unused when N_LAYERS=0)
N_LAYERS      = 0        # 0 = pure linear model (Linear(n_features → 1))
DROPOUT       = 0.0      # dropout probability (0 = disabled)
ACTIVATION    = "relu"   # "relu" | "gelu" | "silu" | "tanh"

LEARNING_RATE = 1e-2     # higher LR ok for linear model
WEIGHT_DECAY  = 0.0
BATCH_SIZE    = 32
WARMUP_RATIO  = 0.05
WARMDOWN_RATIO = 0.40
# ============================================================


TIME_BUDGET = prep.TIME_BUDGET   # seconds (defined in prepare_housing.py)


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
    """
    Fully-connected MLP for scalar regression.
    When N_LAYERS=0, degenerates to a pure linear model: Linear(n_features → 1).

    Architecture:
        [Linear(n_features → HIDDEN_DIM) → act] × N_LAYERS
        Linear(in_dim → 1)
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


# ---------------------------------------------------------------------------
# LR schedule helpers
# ---------------------------------------------------------------------------

def lr_multiplier(elapsed: float) -> float:
    """
    Piecewise LR schedule based on elapsed fraction of TIME_BUDGET:
      0 → warmup_end  : linear warmup  0 → 1
      warmup_end → wd : constant       1
      wd → end        : cosine decay   1 → 0
    """
    p = elapsed / TIME_BUDGET
    wp = WARMUP_RATIO
    wd = 1.0 - WARMDOWN_RATIO

    if p < wp:
        return p / (wp + 1e-9)
    if p < wd:
        return 1.0
    # cosine warmdown
    t = (p - wd) / (1.0 - wd + 1e-9)
    return 0.5 * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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

    # ---- model --------------------------------------------------------------
    model = MLP(n_features).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[train_housing] MLP params={num_params:,}  depth={N_LAYERS}  width={HIDDEN_DIM}")

    # ---- optimiser ----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    loss_fn = nn.MSELoss()

    # ---- training loop (time-budget based, mirrors autoresearch) ------------
    model.train()
    t_start = time.perf_counter()
    t_end   = t_start + TIME_BUDGET

    step = 0
    done = False

    while not done:
        for X_batch, y_batch in train_loader:
            now = time.perf_counter()
            if now >= t_end:
                done = True
                break

            # LR schedule
            scale = lr_multiplier(now - t_start)
            for pg in optimizer.param_groups:
                pg["lr"] = LEARNING_RATE * scale

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            step += 1

            if step % 200 == 0:
                remaining = max(0.0, t_end - time.perf_counter())
                print(
                    f"step={step:6d}  "
                    f"loss={loss.item():.5f}  "
                    f"lr={LEARNING_RATE * scale:.2e}  "
                    f"remaining={remaining:.1f}s"
                )

    training_seconds = time.perf_counter() - t_start

    # ---- evaluation ---------------------------------------------------------
    val_rmse = prep.evaluate_rmse(model, val_loader, device, y_mean, y_std)
    total_seconds = time.perf_counter() - t_start

    # ---- results block (parsed by results.tsv logging) ----------------------
    print("---")
    print(f"val_rmse:          {val_rmse:.4f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"num_steps:         {step}")
    print(f"num_params:        {num_params}")
    print(f"depth:             {N_LAYERS}")
    print(f"hidden_dim:        {HIDDEN_DIM}")
    print(f"activation:        {ACTIVATION}")
    print(f"learning_rate:     {LEARNING_RATE}")
    print(f"dropout:           {DROPOUT}")


if __name__ == "__main__":
    main()
