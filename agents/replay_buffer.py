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
        Sample with soft action balancing — prevents one action from
        dominating the batch and causing one-sided Q-value collapse.
        Each action gets at most 40% of the batch; remainder filled randomly.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer too small: {len(self.buffer)} < {batch_size}")

        by_action = {a: [] for a in range(4)}
        for t in self.buffer:
            by_action[t[1]].append(t)  # t[1] is the action

        max_per_action = int(batch_size * 0.40)
        selected = []
        for a in range(4):
            pool = by_action[a]
            n = min(len(pool), max_per_action)
            if n > 0:
                selected.extend(random.sample(pool, n))

        # Pad to batch_size with fully random samples if needed
        remaining = batch_size - len(selected)
        if remaining > 0:
            selected.extend(random.sample(list(self.buffer), remaining))

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