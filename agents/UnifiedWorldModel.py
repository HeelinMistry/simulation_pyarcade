import cupy as cp
from collections import deque
import random
import os
import pickle


class UnifiedWorldModel:
    def __init__(self, input_size=63, hidden_size=128, lr=1e-4):
        # 1. REPRESENTATION: Raw Features -> Hidden State (s)
        self.W_repr = self._init_weights(input_size, hidden_size)

        # 2. DYNAMICS: (State s + Action a) -> (Next State s' + Reward r)
        self.W_dyn = self._init_weights(hidden_size + 1, hidden_size + 1)

        # 3. PREDICTION: State s -> (Policy Probs + Value V)
        self.W_pred = self._init_weights(hidden_size, 4 + 1)

        self.lr = lr
        self.memory = deque(maxlen=20000)
        self.batch_size = 64

    def _init_weights(self, i, o):
        return cp.random.randn(i, o) * cp.sqrt(2.0 / i)

    def get_initial_state(self, x):
        return cp.tanh(cp.asarray(x) @ self.W_repr)

    def simulate_next(self, s, action):
        if s.ndim == 1:
            s = s.reshape(1, -1)
        action_input = cp.asarray(action).reshape(-1, 1)
        combined = cp.concatenate([s, action_input], axis=1)
        out = cp.tanh(combined @ self.W_dyn)
        next_s = out[:, :-1]
        predicted_reward = out[:, -1:]
        return next_s, predicted_reward

    def predict(self, s):
        if s.ndim == 1:
            s = s.reshape(1, -1)
        out = s @ self.W_pred
        probs = self._softmax(out[:, :4])
        value = out[:, 4:]
        return probs, value

    def _softmax(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)
        z = z - cp.max(z, axis=1, keepdims=True)
        exp = cp.exp(z)
        return exp / cp.sum(exp, axis=1, keepdims=True)

    def record(self, s, a, next_s, r, target_pi):
        self.memory.append((s, a, next_s, r, target_pi))

    def learn_from_memory(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        raw_s, actions, raw_next_s, rewards, target_pis = zip(*batch)

        S = cp.array(raw_s, dtype=cp.float32)
        Next_S = cp.array(raw_next_s, dtype=cp.float32)
        A_discrete = cp.array(actions, dtype=cp.int32)
        R = cp.array(rewards, dtype=cp.float32).reshape(-1, 1)
        Pi = cp.array(target_pis, dtype=cp.float32)

        mapping = cp.array([1.0, -1.0, 0.5, 0.0], dtype=cp.float32)
        A_continuous = mapping[A_discrete].reshape(-1, 1)

        # Forward
        s_latent = cp.tanh(S @ self.W_repr)
        true_next_latent = cp.tanh(Next_S @ self.W_repr)
        pred_out = s_latent @ self.W_pred
        pred_probs = self._softmax(pred_out[:, :4])
        pred_value = pred_out[:, 4:]

        dyn_input = cp.concatenate([s_latent, A_continuous], axis=1)
        dyn_out = cp.tanh(dyn_input @ self.W_dyn)
        pred_next_latent = dyn_out[:, :-1]
        pred_reward = dyn_out[:, -1:]

        # Backprop (simplified for brevity, maintaining logic)
        dZ_policy = (pred_probs - Pi) / self.batch_size
        dZ_value = (pred_value - R) / self.batch_size
        dZ_pred = cp.concatenate([dZ_policy, dZ_value], axis=1)
        dW_pred = s_latent.T @ dZ_pred
        dS_from_pred = dZ_pred @ self.W_pred.T

        d_next_s = (pred_next_latent - true_next_latent) / self.batch_size
        d_reward = (pred_reward - R) / self.batch_size
        dZ_dyn = cp.concatenate([d_next_s, d_reward], axis=1) * (1 - dyn_out ** 2)
        dW_dyn = dyn_input.T @ dZ_dyn
        dS_from_dyn = (dZ_dyn @ self.W_dyn.T)[:, :-1]

        dW_repr = S.T @ ((dS_from_pred + dS_from_dyn) * (1 - s_latent ** 2))

        # Update using persisted self.lr
        self.W_pred -= self.lr * dW_pred
        self.W_dyn -= self.lr * dW_dyn
        self.W_repr -= self.lr * dW_repr

    def save(self, path="outcomes/best_world_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "W_repr": cp.asnumpy(self.W_repr),
            "W_dyn": cp.asnumpy(self.W_dyn),
            "W_pred": cp.asnumpy(self.W_pred),
            "input_size": self.W_repr.shape[0],
            "lr": self.lr  # PERSIST LR
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path="outcomes/best_world_model.pkl"):
        if not os.path.exists(path): return
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        if data.get("input_size", 0) != self.W_repr.shape[0]: return

        self.W_repr = cp.asarray(data["W_repr"])
        self.W_dyn = cp.asarray(data["W_dyn"])
        self.W_pred = cp.asarray(data["W_pred"])
        self.lr = data.get("lr", self.lr) # RESTORE LR
        print(f"World Model loaded. Resuming at LR: {self.lr:.2e}")
