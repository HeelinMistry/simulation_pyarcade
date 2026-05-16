"""
agents/replay_buffer.py
───────────────────────
Off-policy experience replay for SAC.

SAC is off-policy: transitions collected under any policy can be reused,
so we maintain a large circular buffer and sample randomly. This breaks
temporal correlation (which destabilises on-policy methods like PPO) and
allows the agent to revisit rare events (large moves, position closes)
many times.

Capacity guidance for your dataset:
  46 347 rows × ~1 transition each ≈ 46k transitions per full pass.
  A capacity of 200 000 holds ~4 passes of data, enough that the critic
  never forgets early market regimes while still reflecting recent updates.
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store a single transition."""
        self.buffer.append((
            state.astype(np.float32),
            int(action),
            float(reward),
            next_state.astype(np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        """
        Returns five numpy arrays ready to be converted to tensors.
        Raises ValueError if buffer is too small — caller should check
        len(buffer) >= batch_size before calling.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self.buffer)} transitions, need {batch_size}."
            )

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),   # (B, state_dim)
            np.array(actions,     dtype=np.int64),      # (B,)
            np.array(rewards,     dtype=np.float32),    # (B,)
            np.array(next_states, dtype=np.float32),    # (B, state_dim)
            np.array(dones,       dtype=np.float32),    # (B,)
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size