import numpy as np
import cupy as cp
from agents.state_aggregator import StateAggregator


class UnifiedExecutor:
    def __init__(self, name, planner, paces=(1, 2, 4, 8, 12)):
        self.name = name
        self.planner = planner
        self.aggregator = StateAggregator(paces)
        self.inventory = []
        self.current_side = None
        self.total_reward = 0.0
        self.last_probs = np.array([0.0, 0.0, 0.0, 1.0])
        self.tick = 0
        
        # --- Live Execution Filters ---
        self.live_temp = 0.15 # Higher temp than training to prevent "stuck" decisions
        self.min_conviction = 0.40 # Must be 40% sure to open a position

    def _calculate_internal_state(self, current_price):
        if self.current_side is not None and len(self.inventory) > 0:
            entry_price = self.inventory[0]
            if self.current_side == "LONG":
                side_val, u_pnl = 1.0, (current_price - entry_price) / entry_price
            else:
                side_val, u_pnl = -1.0, (entry_price - current_price) / entry_price
            u_pnl = np.clip(u_pnl, -0.05, 0.05)
        else:
            side_val, u_pnl = 0.0, 0.0

        return {'position': side_val, 'unrealized_pnl': u_pnl}

    def get_state(self, indicator_input, current_price):
        self.aggregator.update(indicator_input)
        portfolio_info = self._calculate_internal_state(current_price)
        return self.aggregator.get_state(portfolio_info)

    def step(self, indicators: np.ndarray, price: float, tick: int):
        self.tick = tick
        raw_features = self.get_state(indicators, price)

        # 1. Search with Live Temperature
        action, probs = self.planner.search_best_action(raw_features, temp=self.live_temp)
        self.last_probs = probs

        trade_reward = 0.0

        # 2. Conviction Filter: Don't open new positions unless sure
        if self.current_side is None:
            if action in (0, 1) and probs[action] < self.min_conviction:
                action = 3 # Force HOLD

        # 3. Logic: 0: LONG, 1: SHORT, 2: CLOSE, 3: HOLD
        if action == 0 and self.current_side != "LONG":
            if self.current_side == "SHORT": trade_reward = self._close_position(price)
            self.inventory, self.current_side = [price], "LONG"
        elif action == 1 and self.current_side != "SHORT":
            if self.current_side == "LONG": trade_reward = self._close_position(price)
            self.inventory, self.current_side = [price], "SHORT"
        elif action == 2 and self.current_side is not None:
            trade_reward = self._close_position(price)

        self.total_reward += trade_reward
        return action, self.last_probs

    def _close_position(self, price):
        if not self.inventory: return 0.0
        entry = self.inventory.pop(0)
        reward = (price - entry) / entry if self.current_side == "LONG" else (entry - price) / entry
        self.current_side = None
        return reward

    def get_status(self):
        return {"position": self.current_side if self.current_side else "FLAT", "pnl": self.total_reward}

    def predict_trajectory(self, raw_features, price, horizon=15): # Modified signature
        model = self.planner.model
        s_latent = model.get_initial_state(raw_features)
        probs, _ = model.predict(s_latent)
        probs_np = cp.asnumpy(probs)[0]
        bias = probs_np[0] - probs_np[1]
        path, current_s = [], s_latent
        current_hallucinated_price = price
        for _ in range(horizon):
            next_s, expected_reward = model.simulate_next(current_s, 0.0)
            current_hallucinated_price *= (1 + float(cp.asnumpy(expected_reward).item()))
            path.append(current_hallucinated_price)
            current_s = next_s
        return np.array(path), bias
