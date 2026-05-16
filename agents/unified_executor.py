import numpy as np
import cupy as cp
import collections
from agents.state_aggregator import StateAggregator

# Matches PositionManager commission used during training
COMMISSION = 0.00015


class UnifiedExecutor:
    def __init__(self, name, planner, paces=(1, 2, 4, 8, 12)):
        self.name = name
        self.planner = planner
        self.aggregator = StateAggregator(paces)
        self.inventory = collections.deque(maxlen=100)
        self.current_side = None
        self.total_reward = 0.0
        self.last_probs = np.array([0.0, 0.0, 0.0, 1.0])
        self.tick = 0

        # --- Live Execution Filters ---
        # live_temp is slightly above training temp (0.05) to allow marginal
        # flexibility without drifting far from the trained decision boundary.
        self.live_temp = 0.08
        # Matches the conviction threshold used in the training loop exactly.
        self.min_conviction = 0.45

    def _calculate_internal_state(self, current_price):
        if self.current_side is not None and len(self.inventory) > 0:
            entry_price = self.inventory[0]
            if self.current_side == "LONG":
                side_val = 1.0
                u_pnl = (current_price - entry_price) / entry_price
            else:
                side_val = -1.0
                u_pnl = (entry_price - current_price) / entry_price
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

        # 2. Conviction Filter: Don't open new positions unless sure.
        # Threshold matches training exactly so the model operates in the
        # same decision regime it was rewarded for.
        if action in (0, 1) and probs[action] < self.min_conviction:
            action = 3  # Force HOLD

        # 3. Execute action — 0: LONG, 1: SHORT, 2: CLOSE, 3: HOLD
        if action == 0 and self.current_side != "LONG":
            if self.current_side == "SHORT":
                trade_reward = self._close_position(price)
            self.inventory.append(price * (1 + COMMISSION))  # entry with commission
            self.current_side = "LONG"
        elif action == 1 and self.current_side != "SHORT":
            if self.current_side == "LONG":
                trade_reward = self._close_position(price)
            self.inventory.append(price * (1 - COMMISSION))  # entry with commission
            self.current_side = "SHORT"
        elif action == 2 and self.current_side is not None:
            trade_reward = self._close_position(price)

        self.total_reward += trade_reward
        return action, self.last_probs

    def _close_position(self, price):
        """
        Closes the current position applying commission on exit.
        Commission is also baked into entry price (set at open),
        matching the PositionManager used during training exactly.
        """
        if not self.inventory:
            return 0.0
        entry = self.inventory.popleft()
        if self.current_side == "LONG":
            exit_price = price * (1 - COMMISSION)
            reward = (exit_price - entry) / entry
        else:
            exit_price = price * (1 + COMMISSION)
            reward = (entry - exit_price) / entry
        self.current_side = None
        return reward

    def get_status(self):
        pnl_str = f"{self.total_reward:.4%}" if self.total_reward != 0.0 else "0.0000%"
        return {
            "position": self.current_side if self.current_side else "FLAT",
            "pnl": self.total_reward,
            "pnl_str": pnl_str,
            "entry": self.inventory[0] if self.inventory else None
        }

    def predict_trajectory(self, raw_features, price, horizon=15):
        model = self.planner.model
        assert raw_features.shape[-1] == model.W_repr1.shape[0], (
            f"predict_trajectory received latent vector "
            f"(shape {raw_features.shape[-1]}), "
            f"expected raw state (shape {model.W_repr1.shape[0]})"
        )
        s_latent = model.get_initial_state(raw_features)
        probs, _ = model.predict(s_latent)
        probs_np = cp.asnumpy(probs)[0]
        bias = probs_np[0] - probs_np[1]
        path, current_s = [], s_latent
        current_hallucinated_price = price
        for _ in range(horizon):
            next_s, expected_reward = model.simulate_next(current_s, 3)
            current_hallucinated_price *= (
                1 + float(cp.asnumpy(expected_reward).item())
            )
            path.append(current_hallucinated_price)
            current_s = next_s
        return np.array(path), bias