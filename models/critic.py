"""
models/critic.py
────────────────
Dual Q-network for discrete SAC.

Why two critics?
  In standard Q-learning the value target is computed by the same network
  being updated → systematic overestimation bias → policy chases phantom
  rewards. Using two independent critics and taking min(Q1, Q2) as the
  bootstrap target gives pessimistic value estimates, which has been shown
  to prevent the kind of overconfident policy collapse you saw with the
  world model's value head.

Why state-only input (no action concat)?
  For discrete action spaces, a single forward pass returns Q-values for
  ALL actions simultaneously. This is more efficient than the continuous
  SAC design that concatenates (state, action) and outputs a scalar — we
  can compute the full Bellman target and the actor loss in one pass each.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Single Q-network: state → Q(s, a) for all actions."""

    def __init__(self, state_dim: int = 92, hidden_dim: int = 256, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Small init on final layer for stable early training
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions. Shape: (batch, action_dim)."""
        return self.net(state)


class DualCritic(nn.Module):
    """
    Two independent Q-networks.
    Used as a unit so both share the same optimiser, making saving/loading
    and target-network management cleaner.
    """

    def __init__(self, state_dim: int = 92, hidden_dim: int = 256, action_dim: int = 4):
        super().__init__()
        self.q1 = QNetwork(state_dim, hidden_dim, action_dim)
        self.q2 = QNetwork(state_dim, hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        """Returns (Q1_values, Q2_values). Shape each: (batch, action_dim)."""
        return self.q1(state), self.q2(state)

    def min_q(self, state: torch.Tensor) -> torch.Tensor:
        """Pessimistic Q — used for actor loss computation."""
        q1, q2 = self.forward(state)
        return torch.min(q1, q2)