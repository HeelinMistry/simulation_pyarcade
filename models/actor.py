"""
models/actor.py
───────────────
Discrete SAC Actor: maps a 92-d state vector to a probability
distribution over 4 actions (LONG, SHORT, CLOSE, HOLD).

Design choices:
  - LayerNorm instead of BatchNorm: stable with small batches and
    single-sample inference (no batch statistics needed at live time).
  - Two hidden layers of 256 units: sufficient for 92-d input without
    over-parameterising for 4 outputs.
  - No tanh on output — softmax directly on logits is numerically cleaner.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


ACTION_NAMES = {0: "LONG", 1: "SHORT", 2: "CLOSE", 3: "HOLD"}


class Actor(nn.Module):
    def __init__(self, state_dim: int = 92, hidden_dim: int = 256, action_dim: int = 4):
        super().__init__()
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Initialise final layer with small weights so the initial policy
        # starts close to uniform — avoids baking in early bias.
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns action probabilities. Shape: (batch, action_dim)."""
        logits = self.net(state)
        return F.softmax(logits, dim=-1)

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        Sample (or argmax) an action and return supporting tensors.

        Returns
        -------
        action    : LongTensor  (batch,)
        probs     : FloatTensor (batch, action_dim)
        log_probs : FloatTensor (batch, action_dim)  — per-action log probs,
                    used directly in the SAC entropy term without re-sampling.
        """
        probs = self.forward(state)
        log_probs = torch.log(probs + 1e-8)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.distributions.Categorical(probs).sample()

        return action, probs, log_probs

    # ── Convenience for live inference ───────────────────────────────────────
    def act(self, state_np: np.ndarray, deterministic: bool = False, device: str = "cpu"):
        """
        Numpy-in / numpy-out wrapper for the live trading loop.
        Returns (action_int, probs_np).
        """
        with torch.no_grad():
            s = torch.FloatTensor(state_np).unsqueeze(0).to(device)
            action, probs, _ = self.get_action(s, deterministic=deterministic)
        return int(action.item()), probs.squeeze(0).cpu().numpy()