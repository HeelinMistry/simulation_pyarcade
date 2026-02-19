import numpy as np
from agents.state_aggregator import StateAggregator


class UnifiedExecutor:
    def __init__(self, name, brain, paces=(1, 2, 4, 8, 16)):
        self.name = name
        self.brain = brain
        self.aggregator = StateAggregator(paces)
        self.inventory = []
        self.total_reward = 0.0
        self.last_probs = np.array([0.0, 0.0, 1.0])  # Default HOLD

    def get_state(self, indicator_input):
        self.aggregator.update(indicator_input)
        return self.aggregator.get_state()

    def step(self, indicators: np.ndarray, price: float, tick: int):
        """
        Called by SimulationEnv/Live execution.
        Returns (action, probabilities).
        """
        state = self.get_state(indicators)
        self.last_probs = self.brain.get_probs(state)
        action = self.brain.act_deterministic(state)

        # Track PnL for the environment status
        if action == 0 and len(self.inventory) == 0:
            self.inventory.append(price)
        elif action == 1 and len(self.inventory) > 0:
            entry = self.inventory.pop(0)
            self.total_reward += (price - entry) / entry

        return action, self.last_probs

    def get_status(self):
        """Returns dictionary for UI rendering."""
        return {
            "position": "LONG" if len(self.inventory) > 0 else "FLAT",
            "pnl": self.total_reward
        }