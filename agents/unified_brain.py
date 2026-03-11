import random
from collections import deque
import cupy as cp
import pickle
import os
import numpy as np


class UnifiedBrain:
    def __init__(self, input_size=63, lr=3e-4):
        self.input_size = input_size
        self.lr = lr
        # Initializing weights for the new input dimension
        self.W1 = cp.random.randn(input_size, 64) * cp.sqrt(1. / input_size)
        self.W2 = cp.random.randn(64, 4) * cp.sqrt(1. / 64)

        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def forward(self, x):
        x = cp.asarray(x)
        # Ensure input matches expected dimensions
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Hidden layer with Tanh activation
        h = cp.tanh(x @ self.W1)
        z = h @ self.W2

        # Numerically stable softmax for action probabilities
        z = z - cp.max(z, axis=1, keepdims=True)
        exp = cp.exp(z)
        probs = exp / cp.sum(exp, axis=1, keepdims=True)
        return probs, h

    def act(self, state):
        probs, _ = self.forward(state)
        p = cp.asnumpy(probs)[0]
        return int(np.random.choice(4, p=p))

    def get_probs(self, state):
        probs, _ = self.forward(state)
        return cp.asnumpy(probs)[0]

    def act_deterministic(self, state, min_conf=0.05):
        p = self.get_probs(state)
        best = int(p.argmax())
        # Sort to find the difference between the top two choices
        second = np.partition(p, -2)[-2]
        if p[best] - second < min_conf:
            return 2  # Action 2 is HOLD
        return best

    def learn_from_memory(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards = zip(*batch)

        S = cp.array(states, dtype=cp.float32)
        A = cp.array(actions, dtype=cp.int32)
        R = cp.array(rewards, dtype=cp.float32)

        # Advantage-like normalization
        if R.std() > 1e-6:
            R = (R - R.mean()) / (R.std() + 1e-6)

        probs, h = self.forward(S)

        # Compute Policy Gradient
        onehot = cp.zeros_like(probs)
        onehot[cp.arange(self.batch_size), A] = 1
        delta = (onehot - probs) * R[:, None]

        # Backprop
        dW2 = h.T @ delta
        dh = delta @ self.W2.T
        dW1 = S.T @ (dh * (1 - h ** 2))

        # Gradient Ascent (maximizing reward)
        self.W1 += self.lr * dW1
        self.W2 += self.lr * dW2

    def record(self, state, action, reward):
        self.memory.append((state, action, reward))

    def save(self, path="outcomes/unified_brain.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "W1": cp.asnumpy(self.W1),
            "W2": cp.asnumpy(self.W2),
            "input_size": self.input_size  # Save the shape metadata
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path="outcomes/best_unified_brain.pkl"):
        if not os.path.exists(path):
            print("No brain file found. Starting with fresh weights.")
            return

        with open(path, "rb") as f:
            data = pickle.load(f)

        # Check if the saved model matches our current architecture
        saved_w1 = cp.asarray(data["W1"])
        if saved_w1.shape[0] != self.input_size:
            print(f"Shape Mismatch: Saved={saved_w1.shape[0]}, Expected={self.input_size}.")
            print("Retraining is required because the state space has expanded.")
            return

        self.W1 = saved_w1
        self.W2 = cp.asarray(data["W2"])
        print("Brain weights loaded successfully.")