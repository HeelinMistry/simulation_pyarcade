import numpy as np
import cupy as cp
from agents.state_aggregator import StateAggregator


class UnifiedExecutor:
    def __init__(self, name, planner, paces=(1, 2, 4, 8, 16)):
        self.name = name
        # 1. We replace 'brain' with 'planner'
        self.planner = planner
        self.aggregator = StateAggregator(paces)
        self.inventory = []
        self.current_side = None
        self.total_reward = 0.0
        self.last_probs = np.array([0.0, 0.0, 0.0, 1.0])
        self.tick = 0

    def _calculate_internal_state(self, current_price):
        """Calculates 2 vital metrics: Side and uPnL."""
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
            side_val, u_pnl = 0.0, 0.0

        return {
            'position': side_val,
            'unrealized_pnl': u_pnl
        }

    def get_state(self, indicator_input, current_price):
        # Update market indicators
        self.aggregator.update(indicator_input)

        # Get internal context
        portfolio_info = self._calculate_internal_state(current_price)

        # Pass both to the aggregator to get the raw 63-feature array
        return self.aggregator.get_state(portfolio_info)

    def step(self, indicators: np.ndarray, price: float, tick: int):
        self.tick = tick

        # 1. Extract the raw multi-pace features
        raw_features = self.get_state(indicators, price)

        # 2. THE SEARCH: Ask the MCTS Planner to simulate futures and pick the best action
        action, probs = self.planner.search_best_action(raw_features)
        self.last_probs = probs

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
        return reward

    def get_status(self):
        return {
            "position": self.current_side if self.current_side else "FLAT",
            "pnl": self.total_reward
        }

    def predict_trajectory(self, indicators, price, horizon=15):
        """
        Generates an aggregate projection line using the World Model's
        internal representation of the market state.
        """
        raw_features = self.get_state(indicators, price)

        # 1. Use the World Model to encode the state and predict probabilities
        root_state = self.planner.model.get_initial_state(raw_features)
        probs, _ = self.planner.model.predict(root_state)
        probs = cp.asnumpy(probs)[0]

        # Calculate Directional Bias: (P_Long - P_Short)
        bias = probs[0] - probs[1]

        # Extract Slopes from all agents in the aggregator
        all_slopes = []
        for agent in self.aggregator.agents:
            agent_state = agent.get_state()
            if len(agent_state) >= 8:
                all_slopes.append(np.mean(agent_state[4:8]))

        avg_momentum = np.mean(all_slopes) if all_slopes else 0

        # Generate the Aggregate Line (The Consensus Path)
        path = []
        current_sim_price = price
        for t in range(1, horizon + 1):
            change = (bias * avg_momentum) * (t * 0.001)
            current_sim_price += change
            path.append(current_sim_price)

        return np.array(path), bias