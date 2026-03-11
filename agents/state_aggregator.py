import numpy as np
from agents.agent_multi_pace import MultiPaceAgent

class StateAggregator:
    def __init__(self, paces=(1, 2, 4, 8, 16), window=8):
        self.paces = paces
        self.agents = [MultiPaceAgent(pace=p, max_history=window) for p in paces]
        self.tick = 0

    def update(self, indicators):
        self.tick += 1
        for agent in self.agents:
            if self.tick % agent.pace == 0:
                agent.update(indicators.copy())

    def get_state(self, portfolio_info=None):
        """
        Args:
            portfolio_info: dict containing:
                - 'position': -1 (short), 0 (flat), 1 (long)
                - 'unrealized_pnl': float (e.g., -0.02 for -2%)
        """
        # 1. Get Market States (12 features per agent * 5 agents = 60 features)
        market_parts = [agent.get_state() for agent in self.agents]
        market_vector = np.concatenate(market_parts)

        # 2. Get Portfolio State (3 features)
        if portfolio_info is None:
            # Default to 'Flat' state if no info provided
            portfolio_vector = np.array([0.0, 0.0], dtype=np.float32)
        else:
            portfolio_vector = np.array([
                portfolio_info.get('position', 0),
                portfolio_info.get('unrealized_pnl', 0)
            ], dtype=np.float32)

        # 3. Final Concatenation (Total: 63 features)
        return np.concatenate([market_vector, portfolio_vector]).astype(np.float32)

    def warm_up_all(self, indicators, idx):
        self.tick = 0
        for agent in self.agents:
            agent.warm_up(indicators, idx)
