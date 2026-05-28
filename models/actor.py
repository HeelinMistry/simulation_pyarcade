"""
models/actor.py
───────────────
Discrete SAC Actor: maps a state vector to a probability
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
    def __init__(self, state_dim: int = 50, hidden_dim: int = 128, action_dim: int = 4):
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

    def forward(self, state: torch.Tensor, action_mask=None) -> torch.Tensor:
        logits = self.net(state)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return F.softmax(logits, dim=-1)

    def get_action(self, state: torch.Tensor, deterministic: bool = False,
                   action_mask=None):
        probs = self.forward(state, action_mask)
        log_probs = torch.log(probs + 1e-8)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.distributions.Categorical(probs).sample()
        return action, probs, log_probs

    def act(self, state_np: np.ndarray, deterministic: bool = False,
            device: str = "cpu", action_mask=None):
        with torch.no_grad():
            s = torch.FloatTensor(state_np).unsqueeze(0).to(device)
            mask = action_mask.unsqueeze(0).to(device) if action_mask is not None else None
            action, probs, _ = self.get_action(s, deterministic=deterministic,
                                               action_mask=mask)
        return int(action.item()), probs.squeeze(0).cpu().numpy()