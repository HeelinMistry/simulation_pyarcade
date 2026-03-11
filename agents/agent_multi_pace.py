import numpy as np


class MultiPaceAgent:
    def __init__(self, pace, max_history=8):
        self.pace = pace
        self.history = []
        self.max_history = max_history

    def warm_up(self, all_indicators, current_idx):
        """
        Pre-fills history by looking BACKWARDS from the current_idx.
        Ensures the brain has 'context' immediately after a jump.
        """
        self.history = []
        # We need max_history number of points, spaced by 'pace'
        for i in range(self.max_history):
            lookback_idx = current_idx - ((self.max_history - i) * self.pace)
            if lookback_idx >= 0:
                self.history.append(all_indicators[lookback_idx])
            else:
                # If we are too close to the start of the file, use zero-padding
                self.history.append(np.zeros_like(all_indicators[0]))

    def update(self, indicators):
        self.history.append(indicators)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_state(self):
        if len(self.history) == 0:
            # Now returning 12 features: 4 cur, 4 slope, 4 std
            return np.zeros(12, dtype=np.float32)

        h = np.array(self.history)
        cur = h[-1]

        if len(self.history) < 2:
            return np.concatenate([cur, np.zeros_like(cur), np.zeros_like(cur)])

        mean = h.mean(axis=0)
        std = h.std(axis=0) + 1e-6
        slope = (cur - mean) / std

        # We include the raw 'std' as a volatility measure
        # Since indicators are already scaled, std will be in a usable range (0 to 1)
        return np.concatenate([cur, slope, std]).astype(np.float32)