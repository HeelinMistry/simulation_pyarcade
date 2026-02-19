import numpy as np


class MultiPaceAgent:
    def __init__(self, pace, max_history=8):
        self.pace = pace
        self.history = []
        self.max_history = max_history

    def update(self, indicators):
        self.history.append(indicators)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_state(self):
        # Default empty state if no history
        if len(self.history) == 0:
            return np.zeros(8, dtype=np.float32)

        h = np.array(self.history)
        cur = h[-1]

        # Need at least 2 ticks to calculate slope
        if len(self.history) < 2:
            return np.concatenate([cur, np.zeros_like(cur)])

        mean = h.mean(axis=0)
        std = h.std(axis=0) + 1e-6
        slope = (cur - mean) / std

        # Returns 8 features: 4 current + 4 slope
        return np.concatenate([cur, slope]).astype(np.float32)