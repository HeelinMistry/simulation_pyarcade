import numpy as np
from agents.agent_multi_pace import MultiPaceAgent

class StateAggregator:
    """
    Orchestrates multiple MultiPaceAgents and concatenates their outputs.
    """
    def __init__(self, paces=(1, 2, 4, 8, 16), window=8):
        self.paces = paces
        self.agents = [MultiPaceAgent(pace=p, max_history=window) for p in paces]
        self.tick = 0

    def update(self, indicators):
        self.tick += 1
        for agent in self.agents:
            if self.tick % agent.pace == 0:
                agent.update(indicators.copy())

    def get_state(self):
        state_parts = []
        for agent in self.agents:
            state_parts.append(agent.get_state())
        return np.concatenate(state_parts).astype(np.float32)