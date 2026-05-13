import cupy as cp
import random # For epsilon-greedy
import numpy as np # Added for stochastic rollouts


class MCTSPlanner:
    def __init__(self, world_model, lookahead_depth=10, c_puct=1.0): # INTERVENTION 2.1: Rollout Truncation
        self.model = world_model
        self.depth = lookahead_depth
        self.gamma = 0.90  # INTERVENTION 2.2: Reduced Gamma for Loss Weighting
        self.c_puct = c_puct # New parameter for PUCT-style bonus

    def search_best_action(self, raw_features, temp=0.05, epsilon=0.0): # INTERVENTION 1.3: Epsilon-greedy
        """
        Runs a 'Mental Rollout' to find the best action.
        'temp' controls decisiveness of policy targets.
        'epsilon' controls exploration during rollout.
        """
        root_state = self.model.get_initial_state(raw_features)

        # Get initial policy probabilities and value for the root state
        root_policy_probs, root_value = self.model.predict(root_state)
        root_policy_probs_np = cp.asnumpy(root_policy_probs[0]) # Assuming batch size 1

        action_scores = cp.zeros(4)

        for action_idx in range(4):
            action_sig = self._action_to_signal(action_idx)
            current_s, immediate_r = self.model.simulate_next(root_state, action_sig)
            total_path_reward = float(cp.asnumpy(immediate_r).item())

            # Rollout: Hallucinate 'self.depth' steps into the future
            for d in range(self.depth):
                if random.random() < epsilon: # Epsilon-greedy exploration
                    best_future_action = random.randrange(4)
                else:
                    probs, _ = self.model.predict(current_s)
                    # Use greedy (argmax) rather than stochastic sampling during MCTS rollouts
                    best_future_action = int(cp.argmax(probs)) # Changed from np.random.choice
                
                future_sig = self._action_to_signal(best_future_action)
                current_s, expected_r = self.model.simulate_next(current_s, future_sig)
                
                total_path_reward += (self.gamma ** (d + 1)) * float(cp.asnumpy(expected_r).item())
            
            _, final_v = self.model.predict(current_s)
            
            # Add the raw policy probability as a prior bonus on root action scores (PUCT style)
            action_scores[action_idx] = total_path_reward + \
                                        (self.gamma ** self.depth) * float(cp.asnumpy(final_v).item()) + \
                                        self.c_puct * root_policy_probs_np[action_idx]

        mcts_probs = self._softmax(action_scores, temp=temp)
        
        best_action = int(cp.argmax(action_scores))
        return best_action, cp.asnumpy(mcts_probs)

    def _softmax(self, x, temp=1.0):
        x = x / temp
        e_x = cp.exp(x - cp.max(x))
        return e_x / e_x.sum()

    def _action_to_signal(self, action_idx):
        mapping = {0: 1.0, 1: -1.0, 2: 0.8, 3: 0.0}
        return mapping[action_idx]
