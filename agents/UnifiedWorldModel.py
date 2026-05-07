import cupy as cp
from collections import deque
import random
import os
import pickle


class UnifiedWorldModel:
    def __init__(self, input_size=92, hidden_size=128, lr=1e-4):
        self.W_repr = self._init_weights(input_size, hidden_size)
        self.W_dyn_state = self._init_weights(hidden_size + 1, hidden_size)
        self.W_dyn_reward = self._init_weights(hidden_size + 1, 1)
        self.W_pred = self._init_weights(hidden_size, 4 + 1)

        self.lr = lr
        self.memory = deque(maxlen=20000)
        self.batch_size = 64
        self.hidden_size = hidden_size # Store hidden_size for later use
        
        # --- INTERVENTION 1: POLICY COLLAPSE FIX ---
        self.entropy_beta = 0.05  # Changed to positive to ADD entropy bonus (target > 1.5 bits)
        self.action_penalty_lambda = 0.01 # New: Penalty for repeating same action

        # --- INTERVENTION 4: AGGRESSIVE SPARSITY & BOTTLENECKING ---
        self.l1_lambda_repr = 8e-4 # INCREASED to force Repr Sparsity > 10%
        self.l1_lambda_pred = 5e-4 # Increased L1 for Prediction head (target 20-40% sparsity)
        self.l1_lambda_dyn = 5e-4  # INCREASED to clean up the Hallucination Engine
        self.latent_lambda = 5e-3  # Increased latent sparsity
        self.temporal_contrastive_lambda = 0.01 # New: Forces z_t and predicted z_t+1 to be close

        self.prune_threshold = 4.5e-3 # INCREASED: Shaving more 'average' noise
        self.dropout_rate = 0.20   # INCREASED: Forces model to find "Alpha Leaders"
        self.max_grad_norm = 1.0

        # --- INTERVENTION 5: DIAGNOSTIC ABLATION ---
        self.feature_ablation_mask = None # Mask to zero-out specific features

    def _init_weights(self, i, o):
        return cp.random.randn(i, o) * cp.sqrt(2.0 / i)

    def get_initial_state(self, x):
        """Inference mode (No Dropout)"""
        return cp.tanh(cp.asarray(x) @ self.W_repr)

    def simulate_next(self, s, action):
        if s.ndim == 1: s = s.reshape(1, -1)
        batch_size = s.shape[0]
        action_arr = cp.asarray(action)
        action_input = cp.full((batch_size, 1), action_arr, dtype=cp.float32) if action_arr.size == 1 else action_arr.reshape(batch_size, 1)
        combined = cp.concatenate([s, action_input], axis=1)
        
        pred_next_latent = cp.tanh(combined @ self.W_dyn_state)
        pred_reward = combined @ self.W_dyn_reward # Reward head typically doesn't use tanh for direct reward prediction
        
        return pred_next_latent, pred_reward

    def predict(self, s):
        if s.ndim == 1: s = s.reshape(1, -1)
        out = s @ self.W_pred
        return self._softmax(out[:, :4]), out[:, 4:]

    def _softmax(self, z):
        if z.ndim == 1: z = z.reshape(1, -1)
        z = z - cp.max(z, axis=1, keepdims=True)
        exp = cp.exp(z)
        return exp / cp.sum(exp, axis=1, keepdims=True)

    def record(self, s, a, prev_a, next_s, r, target_pi): # Modified to accept prev_a
        self.memory.append((s, a, prev_a, next_s, r, target_pi))

    def learn_from_memory(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        # Unpack prev_a from the batch
        raw_s, actions, prev_actions, raw_next_s, rewards, target_pis = zip(*batch)

        S, Next_S = cp.array(raw_s, dtype=cp.float32), cp.array(raw_next_s, dtype=cp.float32)
        R, Pi = cp.array(rewards, dtype=cp.float32).reshape(-1, 1), cp.array(target_pis, dtype=cp.float32)
        A_discrete = cp.array(actions, dtype=cp.int32)
        Prev_A_discrete = cp.array(prev_actions, dtype=cp.int32) # New: Previous actions
        A_continuous = cp.array([1.0, -1.0, 0.8, 0.0], dtype=cp.float32)[A_discrete].reshape(-1, 1)

        # --- INTERVENTION 5: FEATURE ABLATION ---
        S_processed = S
        Next_S_processed = Next_S # Initialize for case where dropout_rate is 0
        if self.feature_ablation_mask is not None:
            # Apply the mask to zero-out specific features
            S_processed = S * cp.asarray(self.feature_ablation_mask, dtype=cp.float32)
            Next_S_processed = Next_S * cp.asarray(self.feature_ablation_mask, dtype=cp.float32) # Apply to next state too

        # --- FEATURE DROPOUT (Specialization Trigger) ---
        if self.dropout_rate > 0:
            mask = cp.random.choice([0.0, 1.0], size=S_processed.shape, p=[self.dropout_rate, 1-self.dropout_rate])
            S_processed = S_processed * mask
            Next_S_processed = Next_S_processed * mask # Apply the same mask to Next_S_processed
        
        s_latent = cp.tanh(S_processed @ self.W_repr)
        
        # Forward
        pred_out = s_latent @ self.W_pred
        pred_probs, pred_value = self._softmax(pred_out[:, :4]), pred_out[:, 4:]
        
        dyn_input = cp.concatenate([s_latent, A_continuous], axis=1)
        
        # Split dynamics forward pass
        pred_next_latent = cp.tanh(dyn_input @ self.W_dyn_state)
        pred_reward = dyn_input @ self.W_dyn_reward
        
        true_next_latent = cp.tanh(Next_S_processed @ self.W_repr) # Note: true_next_latent is derived from masked Next_S_processed

        # --- LOSSES ---
        dZ_policy = (pred_probs - Pi) / self.batch_size
        
        # INTERVENTION 1.2: ENTROPY BONUS (flipped sign)
        # Encourages exploration by adding entropy to the loss
        entropy_grad = pred_probs * (cp.log(pred_probs + 1e-9) - cp.mean(cp.log(pred_probs + 1e-9), axis=1, keepdims=True))
        dZ_policy -= (self.entropy_beta * entropy_grad) / self.batch_size # Note: -= instead of +=

        # INTERVENTION 1.1: ACTION PENALTY (for repeating same action)
        # Discourage mode locking by penalizing if current action == previous action
        action_repeat_penalty = cp.zeros_like(pred_probs)
        for i in range(self.batch_size):
            if A_discrete[i] == Prev_A_discrete[i] and A_discrete[i] in [0, 1]: # Only penalize repeating LONG/SHORT
                action_repeat_penalty[i, A_discrete[i]] = self.action_penalty_lambda
        dZ_policy += action_repeat_penalty / self.batch_size

        dZ_value = (pred_value - R) / self.batch_size
        dZ_pred = cp.concatenate([dZ_policy, dZ_value], axis=1)
        
        dW_pred, dS_from_pred = s_latent.T @ dZ_pred, dZ_pred @ self.W_pred.T

        # Split dynamics backward pass
        dZ_pred_next_latent = (pred_next_latent - true_next_latent) / self.batch_size * (1 - pred_next_latent ** 2) # Apply tanh derivative
        dZ_pred_reward = (pred_reward - R) / self.batch_size

        dW_dyn_state, dS_from_dyn_state = dyn_input.T @ dZ_pred_next_latent, dZ_pred_next_latent @ self.W_dyn_state.T
        dW_dyn_reward, dS_from_dyn_reward = dyn_input.T @ dZ_pred_reward, dZ_pred_reward @ self.W_dyn_reward.T

        dS_from_dyn = dS_from_dyn_state + dS_from_dyn_reward

        # INTERVENTION 4.2: TEMPORAL CONTRASTIVE LOSS
        # Forces s_latent and pred_next_latent to be close (reduces drift)
        temporal_loss_grad = (s_latent - pred_next_latent) * self.temporal_contrastive_lambda
        # Fix: Only add temporal_loss_grad to the state portion of dS_from_dyn
        dS_from_dyn[:, :self.hidden_size] += temporal_loss_grad # Add to gradient flowing back to s_latent from dynamics

        dW_repr = S_processed.T @ ((dS_from_pred + dS_from_dyn[:, :self.hidden_size] + (self.latent_lambda * cp.sign(s_latent))) * (1 - s_latent ** 2))

        for grad in [dW_pred, dW_dyn_state, dW_dyn_reward, dW_repr]: cp.clip(grad, -self.max_grad_norm, self.max_grad_norm, out=grad)
        
        # --- WEIGHT UPDATES + L1 REGULARIZATION ---
        # INTERVENTION 4.1: Increased L1 for Repr and Pred heads
        self.W_pred -= self.lr * (dW_pred + self.l1_lambda_pred * cp.sign(self.W_pred))
        self.W_dyn_state -= self.lr * (dW_dyn_state + self.l1_lambda_dyn * cp.sign(self.W_dyn_state))
        self.W_dyn_reward -= self.lr * (dW_dyn_reward + self.l1_lambda_dyn * cp.sign(self.W_dyn_reward)) # Apply L1 to reward head too
        self.W_repr -= self.lr * (dW_repr + self.l1_lambda_repr * cp.sign(self.W_repr))

        # HARD PROXIMAL PRUNING
        self.W_pred = cp.where(cp.abs(self.W_pred) < self.prune_threshold, 0, self.W_pred)
        self.W_dyn_state = cp.where(cp.abs(self.W_dyn_state) < self.prune_threshold, 0, self.W_dyn_state)
        self.W_dyn_reward = cp.where(cp.abs(self.W_dyn_reward) < self.prune_threshold, 0, self.W_dyn_reward)
        self.W_repr = cp.where(cp.abs(self.W_repr) < self.prune_threshold, 0, self.W_repr)

    def save(self, path="outcomes/best_world_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"W_repr": cp.asnumpy(self.W_repr), "W_dyn_state": cp.asnumpy(self.W_dyn_state), "W_dyn_reward": cp.asnumpy(self.W_dyn_reward), "W_pred": cp.asnumpy(self.W_pred), "input_size": self.W_repr.shape[0], "lr": self.lr}
        with open(path, "wb") as f: pickle.dump(data, f)

    def load(self, path="outcomes/best_world_model.pkl"):
        if not os.path.exists(path): return
        with open(path, "rb") as f: data = pickle.load(f)
        if data.get("input_size", 0) != self.W_repr.shape[0]: return
        self.W_repr, self.W_dyn_state, self.W_dyn_reward, self.W_pred, self.lr = cp.asarray(data["W_repr"]), cp.asarray(data["W_dyn_state"]), cp.asarray(data["W_dyn_reward"]), cp.asarray(data["W_pred"]), data.get("lr", self.lr)
        print(f"World Model loaded. Resuming at LR: {self.lr:.2e}")
