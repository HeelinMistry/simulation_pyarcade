import numpy as np

from agents.agent_monte_carlo import MCAgent as Agent


class MasterObserver(Agent):
    def act(self, row):
        state = self.get_state(row)
        probs = self.brain.get_probs(state)

        # DETERMINISTIC: No np.random.choice.
        # We take the absolute best guess the brain has.
        action = np.argmax(probs)
        return state, action