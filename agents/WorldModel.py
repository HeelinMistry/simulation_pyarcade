class WorldModel(cp.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        # 1. Representation Network: Raw Data -> State s
        self.repr_net = MLP(input_dim, hidden_dim)

        # 2. Dynamics Network: (s, action) -> (next_s, reward)
        self.dyn_net = MLP(hidden_dim + 1, hidden_dim + 1)

        # 3. Prediction Network: s -> (Policy Probs, Value V)
        self.pred_net = MLP(hidden_dim, num_actions + 1)

    def representation(self, raw_features):
        return self.repr_net(raw_features)

    def next_step(self, s, action):
        return self.dyn_net(concat(s, action))

    def predict(self, s):
        return self.pred_net(s)