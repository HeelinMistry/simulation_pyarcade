import numpy as np


class MultiPaceAgent:
    def __init__(self, pace, max_history=8, num_indicators=6):  # Added num_indicators
        self.pace = pace
        self.history = []
        self.max_history = max_history
        self.num_indicators = num_indicators  # Store the expected number of indicators

    def warm_up(self, all_indicators, current_idx):
        """
        Pre-fills history by looking BACKWARDS from the current_idx.
        """
        self.history = []
        # Ensure num_indicators is set if not already
        if self.num_indicators == 0 and len(all_indicators) > 0:
            self.num_indicators = len(all_indicators[0])

        for i in range(self.max_history):
            lookback_idx = current_idx - ((self.max_history - i) * self.pace)
            if lookback_idx >= 0:
                self.history.append(all_indicators[lookback_idx])
            else:
                # Use num_indicators to create a zero-padded array of correct size
                self.history.append(np.zeros(self.num_indicators, dtype=np.float32))

    def update(self, indicators):
        # Ensure num_indicators is set if not already
        if self.num_indicators == 0 and len(indicators) > 0:
            self.num_indicators = len(indicators)

        self.history.append(indicators)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_state(self):
        # Dynamically determine size based on stored num_indicators
        if len(self.history) == 0:
            # Return a zero-filled array of the expected size (num_indicators * 3)
            return np.zeros(self.num_indicators * 3, dtype=np.float32)

        h = np.array(self.history)
        cur = h[-1]

        if len(self.history) < 2:
            # Return [cur, zeros, zeros]
            return np.concatenate([cur, np.zeros(self.num_indicators), np.zeros(self.num_indicators)])

        mean = h.mean(axis=0)
        std = h.std(axis=0) + 1e-6
        slope = (cur - mean) / std

        # Concatenate 3 states per indicator: [Current, Slope, Volatility]
        return np.concatenate([cur, slope, std]).astype(np.float32)