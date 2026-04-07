import cupy as cp


class MCTSPlanner:
    def __init__(self, world_model, lookahead_depth=1):
        self.model = world_model
        self.depth = lookahead_depth

    def search_best_action(self, raw_features):
        """Runs the mental simulation to find the best immediate move."""
        # 1. Encode the raw multi-pace data into the hidden 'root' state
        root_state = self.model.get_initial_state(raw_features)

        # 2. Get the 'gut reaction' probabilities for training later
        base_probs, _ = self.model.predict(root_state)

        # Array to store the simulated value of each action
        action_scores = cp.zeros(4)  # [LONG=0, SHORT=1, CLOSE=2, HOLD=3]

        # 3. Branching: Simulate the outcome of each possible action
        for action_idx in range(4):
            # Convert discrete action to a continuous signal
            action_signal = self._action_to_signal(action_idx)

            # Hallucinate the next market state and immediate reward
            next_state, expected_reward = self.model.simulate_next(root_state, action_signal)

            # Ask the prediction network how valuable that resulting future is
            _, future_value = self.model.predict(next_state)

            # The Total Score = (Reward of the action) + (Value of the resulting state)
            action_scores[action_idx] = expected_reward[0, 0] + future_value[0, 0]

        # 4. Pick the action that yielded the best hallucinated outcome
        best_action = int(cp.argmax(action_scores))

        # Convert CuPy array to NumPy for the executor
        return best_action, cp.asnumpy(base_probs)[0]

    def _action_to_signal(self, action_idx):
        """
        Translates discrete choices into signals the Dynamics net understands.
        Each action MUST have a unique mathematical signature.
        """
        mapping = {
            0: 1.0,   # LONG
            1: -1.0,  # SHORT
            2: 0.5,   # CLOSE (Exit Intent)
            3: 0.0    # HOLD (Patience/Maintenance Intent)
        }
        return mapping[action_idx]
