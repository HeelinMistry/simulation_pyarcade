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

    def _calculate_internal_state(self, current_price):
        """Calculates position and unrealized P/L."""
        if self.current_side is not None and len(self.inventory) > 0:
            entry_price = self.inventory[0]

            if self.current_side == "LONG":
                side_val = 1.0
                u_pnl = (current_price - entry_price) / entry_price
            else:  # SHORT
                side_val = -1.0
                u_pnl = (entry_price - current_price) / entry_price

            u_pnl = np.clip(u_pnl, -0.05, 0.05)
        else:
            side_val, u_pnl = 0.0, 0.0

        return {
            'position': side_val,
            'unrealized_pnl': u_pnl
        }

    def get_state(self, indicator_input, current_price):
        self.aggregator.update(indicator_input)
        portfolio_info = self._calculate_internal_state(current_price)
        return self.aggregator.get_state(portfolio_info)

    def step(self, indicators: np.ndarray, price: float, tick: int):
        self.tick = tick
        raw_features = self.get_state(indicators, price)

        # THE SEARCH: Use the MCTS Planner
        action, probs = self.planner.search_best_action(raw_features)
        self.last_probs = probs

        trade_reward = 0.0

        # Action 0: LONG, 1: SHORT, 2: CLOSE, 3: HOLD
        if action == 0 and self.current_side != "LONG":
            # If we were SHORT, close it first
            if self.current_side == "SHORT":
                trade_reward = self._close_position(price)
            
            self.inventory = [price]
            self.current_side = "LONG"

        elif action == 1 and self.current_side != "SHORT":
            # If we were LONG, close it first
            if self.current_side == "LONG":
                trade_reward = self._close_position(price)
                
            self.inventory = [price]
            self.current_side = "SHORT"

        elif action == 2:
            # ONLY close if we actually have a position
            if self.current_side is not None:
                trade_reward = self._close_position(price)
            else:
                # If model chooses CLOSE while FLAT, treat as NO-OP (no reward)
                pass

        # Action 3: HOLD (Intrinsic No-Op)

        self.total_reward += trade_reward
        return action, self.last_probs

    def _close_position(self, price):
        if not self.inventory or self.current_side is None:
            return 0.0

        entry = self.inventory.pop(0)
        if self.current_side == "LONG":
            reward = (price - entry) / entry
        else:  # SHORT
            reward = (entry - price) / entry

        self.current_side = None
        return reward

    def get_status(self):
        return {
            "position": self.current_side if self.current_side else "FLAT",
            "pnl": self.total_reward
        }

    def predict_trajectory(self, indicators, price, horizon=15):
        """
        Generates a projection path by RECURSIVELY HALLUCINATING using the Dynamics Network.
        This ensures the UI shows EXACTLY what the model is thinking.
        """
        raw_features = self.get_state(indicators, price)
        model = self.planner.model

        # 1. Get current latent state
        s_latent = model.get_initial_state(raw_features)
        
        # 2. Get directional bias from the prediction head
        probs, _ = model.predict(s_latent)
        probs_np = cp.asnumpy(probs)[0]
        bias = probs_np[0] - probs_np[1] # LONG - SHORT

        path = []
        current_hallucinated_price = price
        
        # 3. Step forward in "Mental Time"
        current_s = s_latent
        for _ in range(horizon):
            # Use HOLD signal (0.0) to see market evolution
            next_s, expected_reward = model.simulate_next(current_s, 0.0)
            
            reward_val = float(cp.asnumpy(expected_reward).item())
            current_hallucinated_price *= (1 + reward_val)
            path.append(current_hallucinated_price)
            
            current_s = next_s

        return np.array(path), bias
