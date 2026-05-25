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

SIGNAL_THRESHOLD = 0.0005  # |reward| > 0.05% is a "signal" transition

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
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer too small: {len(self.buffer)} < {batch_size}")

        signal = [t for t in self.buffer if abs(t[2]) > SIGNAL_THRESHOLD]
        noise = [t for t in self.buffer if abs(t[2]) <= SIGNAL_THRESHOLD]

        n_signal = min(len(signal), batch_size // 2)
        n_noise = batch_size - n_signal

        selected = []
        if n_signal > 0:
            selected.extend(random.sample(signal, n_signal))
        if n_noise > 0:
            pool = noise if len(noise) >= n_noise else list(self.buffer)
            selected.extend(random.sample(pool, n_noise))

        random.shuffle(selected)
        states, actions, rewards, next_states, dones = zip(*selected[:batch_size])
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size