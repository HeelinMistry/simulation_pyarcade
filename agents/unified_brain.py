import cupy as cp
import pickle
import os

import numpy as np


class UnifiedBrain:
    def __init__(self, input_size, lr=3e-4):
        self.lr = lr
        self.W1 = cp.random.randn(input_size, 64) * 0.1
        self.W2 = cp.random.randn(64, 3) * 0.1

    def forward(self, x):
        x = cp.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)  # Handle single-state inputs

        h = cp.tanh(x @ self.W1)
        z = h @ self.W2

        # Numerically stable softmax
        z = z - cp.max(z, axis=1, keepdims=True)
        exp = cp.exp(z)
        probs = exp / cp.sum(exp, axis=1, keepdims=True)
        return probs, h

    def act(self, state):
        """Stochastic action for training exploration."""
        probs, _ = self.forward(state)
        p = cp.asnumpy(probs)[0]
        return int(np.random.choice(3, p=p))

    def get_probs(self, state):
        """Returns the raw probability distribution for UI/Simulation."""
        probs, _ = self.forward(state)
        return cp.asnumpy(probs)[0]

    def act_deterministic(self, state, min_conf=0.05):
        """Greedy action with confidence margin for live/simulated trading."""
        p = self.get_probs(state)
        best = int(p.argmax())
        second = np.partition(p, -2)[-2]
        if p[best] - second < min_conf:
            return 2  # HOLD
        return best

    def learn(self, states, actions, rewards):
        S = cp.asarray(states)
        A = cp.asarray(actions)
        R = cp.asarray(rewards)

        probs, h = self.forward(S)

        # Baseline
        R = R - R.mean()

        onehot = cp.zeros_like(probs)
        onehot[cp.arange(len(A)), A] = 1

        delta = (onehot - probs) * R[:, None]
        dW2 = h.T @ delta
        dh = delta @ self.W2.T
        dW1 = S.T @ (dh * (1 - h ** 2))

        self.W1 += self.lr * dW1
        self.W2 += self.lr * dW2

    def save(self, path="outcomes/unified_brain.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"W1": cp.asnumpy(self.W1), "W2": cp.asnumpy(self.W2)}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path="outcomes/unified_brain.pkl"):
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.W1 = cp.asarray(data["W1"])
        self.W2 = cp.asarray(data["W2"])