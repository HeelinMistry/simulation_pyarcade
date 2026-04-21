import cupy as cp


class MCTSPlanner:
    def __init__(self, world_model, lookahead_depth=20):
        self.model = world_model
        self.depth = lookahead_depth
        self.gamma = 0.99 

    def search_best_action(self, raw_features, temp=0.05):
        """
        Runs the mental simulation. 
        'temp' allows us to control how 'decisive' the resulting probabilities are.
        """
        root_state = self.model.get_initial_state(raw_features)
        action_scores = cp.zeros(4) 
        
        for action_idx in range(4):
            action_sig = self._action_to_signal(action_idx)
            current_s, immediate_r = self.model.simulate_next(root_state, action_sig)
            total_path_reward = float(cp.asnumpy(immediate_r).item())
            
            for d in range(self.depth):
                probs, _ = self.model.predict(current_s)
                best_future_action = int(cp.argmax(probs))
                future_sig = self._action_to_signal(best_future_action)
                current_s, expected_r = self.model.simulate_next(current_s, future_sig)
                total_path_reward += (self.gamma ** (d + 1)) * float(cp.asnumpy(expected_r).item())
            
            _, final_v = self.model.predict(current_s)
            action_scores[action_idx] = total_path_reward + (self.gamma ** self.depth) * float(cp.asnumpy(final_v).item())

        # Softmax with the provided temperature
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
