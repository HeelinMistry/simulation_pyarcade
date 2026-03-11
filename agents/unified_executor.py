import numpy as np
from agents.state_aggregator import StateAggregator


class UnifiedExecutor:
    def __init__(self, name, brain, paces=(1, 2, 4, 8, 16), max_trade_len=100):
        self.name = name
        self.brain = brain
        self.aggregator = StateAggregator(paces)
        self.inventory = []  # Still used to store entry price
        self.current_side = None  # Tracks "LONG" or "SHORT"
        self.total_reward = 0.0
        # Updated for 4 actions: [LONG=0, SHORT=1, CLOSE=2, HOLD=3]
        self.last_probs = np.array([0.0, 0.0, 0.0, 1.0])

    def _calculate_internal_state(self, current_price, current_tick):
        """Calculates 3 vital metrics: Side, uPnL, and Duration."""
        if len(self.inventory) > 0:
            entry_price = self.inventory[0]

            if self.current_side == "LONG":
                side_val = 1.0
                u_pnl = (current_price - entry_price) / entry_price
            else:  # SHORT
                side_val = -1.0
                u_pnl = (entry_price - current_price) / entry_price

            u_pnl = np.clip(u_pnl, -0.05, 0.05)
        else:
            side_val, u_pnl, duration = 0.0, 0.0, 0.0

        return {
            'position': side_val,
            'unrealized_pnl': u_pnl
        }

    def step(self, indicators: np.ndarray, price: float, tick: int):
        state = self.get_state(indicators, price, tick)
        self.last_probs = self.brain.get_probs(state)
        action = self.brain.act_deterministic(state)

        trade_reward = 0.0

        # Action 0: Go LONG (closes Short if exists)
        if action == 0 and self.current_side != "LONG":
            trade_reward = self._close_position(price)
            self.inventory = [price]
            self.current_side = "LONG"

        # Action 1: Go SHORT (closes Long if exists)
        elif action == 1 and self.current_side != "SHORT":
            trade_reward = self._close_position(price)
            self.inventory = [price]
            self.current_side = "SHORT"

        # Action 2: CLOSE to Neutral
        elif action == 2 and self.current_side is not None:
            trade_reward = self._close_position(price)

        # Action 3: HOLD (Nothing to do)

        self.total_reward += trade_reward
        return action, self.last_probs

    def _close_position(self, price):
        if not self.inventory:
            return 0.0

        entry = self.inventory.pop(0)
        if self.current_side == "LONG":
            reward = (price - entry) / entry
        else:  # SHORT
            reward = (entry - price) / entry

        self.current_side = None
        self.entry_tick = 0
        return reward

    def get_status(self):
        return {
            "position": self.current_side if self.current_side else "FLAT",
            "pnl": self.total_reward
        }

    def get_state(self, indicator_input, current_price, current_tick):
        # Update market indicators
        self.aggregator.update(indicator_input)

        # Get internal context
        portfolio_info = self._calculate_internal_state(current_price, current_tick)

        # Pass both to the aggregator (matches the new StateAggregator logic)
        return self.aggregator.get_state(portfolio_info)
