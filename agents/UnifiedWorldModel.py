import cupy as cp
from collections import deque
import random
import os
import pickle


class UnifiedWorldModel:
    def __init__(self, input_size=92, hidden_size=128, lr=1e-4):
        self.W_repr1 = self._init_weights(input_size, hidden_size)
        self.W_repr2 = self._init_weights(hidden_size, hidden_size * 2) # New layer
        self.W_repr3 = self._init_weights(hidden_size * 2, hidden_size) # New layer
        self.W_dyn_state = self._init_weights(hidden_size + 4, hidden_size) # Changed from hidden_size + 1 to hidden_size + 4
        self.W_dyn_reward = self._init_weights(hidden_size + 4, 1) # Changed from hidden_size + 1 to hidden_size + 4
        self.W_policy = self._init_weights(hidden_size, 4) # Policy head
        self.W_value = self._init_weights(hidden_size, 1) # Value head

        self.lr = lr
        self.lr_value = lr * 3 # Higher learning rate for value head
        self.memory = deque(maxlen=20000)
        self.batch_size = 64
        self.hidden_size = hidden_size # Store hidden_size for later use
        
        # --- INTERVENTION 1: POLICY COLLAPSE FIX ---
        self.entropy_beta = 0.005  # Changed to positive to ADD entropy bonus (target > 1.5 bits)
        self.action_penalty_lambda = 0.15 # New: Penalty for repeating same action, increased and will be asymmetric
        self.short_repeat_penalty_multiplier = 3.0 # Heavier penalty for repeated SHORT

        # --- INTERVENTION 4: AGGRESSIVE SPARSITY & BOTTLENECKING ---
        self.l1_lambda_repr = 1.5e-3 # INCREASED to force Repr Sparsity > 10%
        self.l1_lambda_policy = 5e-4 # Increased L1 for Policy head (target 20-40% sparsity)
        self.l1_lambda_value = 0.0 # New: Lower L1 for value head to prevent aggressive pruning (set to 0 as requested)
        self.l1_lambda_dyn = 5e-4  # INCREASED to clean up the Hallucination Engine
        self.latent_lambda = 5e-3  # Increased latent sparsity
        self.temporal_contrastive_lambda = 0.01 # New: Forces z_t and predicted z_t+1 to be close
        self.temporal_contrastive_lambda_non_hold = 0.001 # New: Lower lambda for non-HOLD transitions

        self.prune_threshold = 8e-3 # INCREASED: Shaving more 'average' noise
        self.dropout_rate = 0.20   # INCREASED: Forces model to find "Alpha Leaders"
        self.max_grad_norm = 1.0

        # --- INTERVENTION 5: DIAGNOSTIC ABLATION ---
        self.feature_ablation_mask = None # Mask to zero-out specific features

        # --- KL Divergence Regularization ---
        self.kl_penalty_lambda = 0.02 # Lambda for uniform prior KL divergence

    def _init_weights(self, i, o):
        return cp.random.randn(i, o) * cp.sqrt(2.0 / i)

    def get_initial_state(self, x):
        """Inference mode (No Dropout)"""
        # Forward pass through the new representation layers
        h1 = cp.tanh(cp.asarray(x) @ self.W_repr1)
        h2 = cp.tanh(h1 @ self.W_repr2)
        return cp.tanh(h2 @ self.W_repr3)

    def simulate_next(self, s, action):
        if s.ndim == 1: s = s.reshape(1, -1)
        batch_size = s.shape[0]
        # Convert scalar action to one-hot encoding
        action_one_hot = cp.zeros((batch_size, 4), dtype=cp.float32)
        action_one_hot[cp.arange(batch_size), cp.asarray(action, dtype=cp.int32)] = 1.0
        
        combined = cp.concatenate([s, action_one_hot], axis=1)
        
        pred_next_latent = cp.tanh(combined @ self.W_dyn_state)
        pred_reward = combined @ self.W_dyn_reward # Reward head typically doesn't use tanh for direct reward prediction
        
        return pred_next_latent, pred_reward

    def predict(self, s):
        if s.ndim == 1: s = s.reshape(1, -1)
        # Separate policy and value predictions
        policy_out = s @ self.W_policy
        value_out = s @ self.W_value
        return self._softmax(policy_out), value_out

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
        
        # Convert discrete actions to one-hot encoding for dynamics network
        A_one_hot = cp.zeros((self.batch_size, 4), dtype=cp.float32)
        A_one_hot[cp.arange(self.batch_size), A_discrete] = 1.0

        # --- INTERVENTION 5: FEATURE ABLATION ---
        S_processed = S
        Next_S_processed = Next_S # Initialize for case where dropout_rate is 0
        if self.feature_ablation_mask is not None:
            # Apply the mask to zero-out specific features
            S_processed = S * cp.asarray(self.feature_ablation_mask, dtype=cp.float32)
            Next_S_processed = Next_S * cp.asarray(self.feature_ablation_mask, dtype=cp.float32) # Apply to next state too

        # --- FEATURE DROPOUT (Specialization Trigger) ---
        if self.dropout_rate > 0:
            mask_s = cp.random.choice([0.0, 1.0], size=S_processed.shape, p=[self.dropout_rate, 1-self.dropout_rate])
            S_processed = S_processed * mask_s / (1 - self.dropout_rate) # Apply inverted dropout scaling
            
            mask_next_s = cp.random.choice([0.0, 1.0], size=Next_S_processed.shape, p=[self.dropout_rate, 1-self.dropout_rate])
            Next_S_processed = Next_S_processed * mask_next_s / (1 - self.dropout_rate) # Apply a new mask and scaling to Next_S_processed
        
        # Forward pass for representation network
        h1 = S_processed @ self.W_repr1
        h1_activated = cp.tanh(h1)
        h2 = h1_activated @ self.W_repr2
        h2_activated = cp.tanh(h2)
        s_latent = cp.tanh(h2_activated @ self.W_repr3)
        
        # Forward for policy and value heads
        pred_probs = self._softmax(s_latent @ self.W_policy)
        pred_value = s_latent @ self.W_value
        
        dyn_input = cp.concatenate([s_latent, A_one_hot], axis=1) # Use one-hot encoded actions
        
        # Split dynamics forward pass
        pred_next_latent = cp.tanh(dyn_input @ self.W_dyn_state)
        pred_reward = dyn_input @ self.W_dyn_reward
        
        # True next latent state calculation
        true_h1 = Next_S_processed @ self.W_repr1
        true_h1_activated = cp.tanh(true_h1)
        true_h2 = true_h1_activated @ self.W_repr2
        true_h2_activated = cp.tanh(true_h2)
        true_next_latent = cp.tanh(true_h2_activated @ self.W_repr3)

        # --- LOSSES ---
        dZ_policy = (pred_probs - Pi) / self.batch_size
        
        # INTERVENTION 1.2: ENTROPY BONUS (flipped sign)
        # Encourages exploration by adding entropy to the loss
        entropy_grad = pred_probs * (cp.log(pred_probs + 1e-9) - cp.mean(cp.log(pred_probs + 1e-9), axis=1, keepdims=True))
        dZ_policy -= (self.entropy_beta * entropy_grad) / self.batch_size # Note: -= instead of +=

        # INTERVENTION 1.1: ACTION PENALTY (for repeating same action)
        # Discourage mode locking by penalizing if current action == previous action
        action_repeat_penalty = cp.zeros_like(pred_probs)
        
        # Apply base penalty for repeating LONG or SHORT
        repeat_long_mask = (A_discrete == Prev_A_discrete) & (A_discrete == 0) # LONG
        action_repeat_penalty[cp.arange(self.batch_size)[repeat_long_mask], A_discrete[repeat_long_mask]] = self.action_penalty_lambda
        
        # Apply heavier penalty for repeating SHORT
        repeat_short_mask = (A_discrete == Prev_A_discrete) & (A_discrete == 1) # SHORT
        action_repeat_penalty[cp.arange(self.batch_size)[repeat_short_mask], A_discrete[repeat_short_mask]] = self.action_penalty_lambda * self.short_repeat_penalty_multiplier
        
        dZ_policy += action_repeat_penalty / self.batch_size

        # KL Divergence Regularization
        uniform_prior = cp.full_like(pred_probs, 0.25)
        kl_grad = (pred_probs - uniform_prior) * self.kl_penalty_lambda
        dZ_policy += kl_grad / self.batch_size

        # Bootstrap value target: r_t + gamma * V(s_{t+1})
        # with cp.no_grad(): # CuPy does not have a no_grad context manager
        _, next_values = self.predict(self.get_initial_state(Next_S_processed))
        td_target = R + 0.90 * next_values  # gamma=0.90 matches MCTSPlanner
        dZ_value = (pred_value - td_target) / self.batch_size
        
        # Gradients for policy and value heads
        dW_policy, dS_from_policy = s_latent.T @ dZ_policy, dZ_policy @ self.W_policy.T
        dW_value, dS_from_value = s_latent.T @ dZ_value, dZ_value @ self.W_value.T

        # Split dynamics backward pass
        dZ_pred_next_latent = (pred_next_latent - true_next_latent) / self.batch_size * (1 - pred_next_latent ** 2) # Apply tanh derivative
        dZ_pred_reward = (pred_reward - R) / self.batch_size

        dW_dyn_state, dS_from_dyn_state = dyn_input.T @ dZ_pred_next_latent, dZ_pred_next_latent @ self.W_dyn_state.T
        dW_dyn_reward, dS_from_dyn_reward = dyn_input.T @ dZ_pred_reward, dZ_pred_reward @ self.W_dyn_reward.T

        dS_from_dyn = dS_from_dyn_state + dS_from_dyn_reward

        # INTERVENTION 4.2: TEMPORAL CONTRASTIVE LOSS
        # Forces s_latent and pred_next_latent to be close (reduces drift)
        temporal_loss_grad = cp.zeros_like(s_latent)
        
        # Apply stronger lambda for HOLD actions
        hold_mask = (A_discrete == 3) # Action 3 is HOLD
        if hold_mask.any():
            temporal_loss_grad[hold_mask] = (
                (s_latent[hold_mask] - pred_next_latent[hold_mask]) 
                * self.temporal_contrastive_lambda
            )
        
        # Apply weaker lambda for non-HOLD actions
        non_hold_mask = (A_discrete != 3)
        if non_hold_mask.any():
            temporal_loss_grad[non_hold_mask] = (
                (s_latent[non_hold_mask] - pred_next_latent[non_hold_mask]) 
                * self.temporal_contrastive_lambda_non_hold
            )
        
        dS_from_dyn[:, :self.hidden_size] += temporal_loss_grad

        # Backpropagate through the representation network
        dS_from_pred_combined = dS_from_policy + dS_from_value # Combine gradients from policy and value heads
        dS_from_repr = dS_from_pred_combined + dS_from_dyn[:, :self.hidden_size] + (self.latent_lambda * cp.sign(s_latent))

        # dW_repr3
        d_h2_activated = dS_from_repr * (1 - s_latent ** 2)
        dW_repr3 = h2_activated.T @ d_h2_activated

        # dW_repr2
        d_h2 = d_h2_activated @ self.W_repr3.T
        d_h1_activated = d_h2 * (1 - h2_activated ** 2)
        dW_repr2 = h1_activated.T @ d_h1_activated

        # dW_repr1
        d_h1 = d_h1_activated @ self.W_repr2.T
        d_S_processed = d_h1 * (1 - h1_activated ** 2)
        dW_repr1 = S_processed.T @ d_S_processed

        for grad in [dW_policy, dW_value, dW_dyn_state, dW_dyn_reward, dW_repr1, dW_repr2, dW_repr3]: cp.clip(grad, -self.max_grad_norm, self.max_grad_norm, out=grad)
        
        # --- WEIGHT UPDATES + L1 REGULARIZATION ---
        # Update policy and value heads separately
        self.W_policy -= self.lr * (dW_policy + self.l1_lambda_policy * cp.sign(self.W_policy))
        self.W_value -= self.lr_value * (dW_value + self.l1_lambda_value * cp.sign(self.W_value)) # Use lr_value and l1_lambda_value

        self.W_dyn_state -= self.lr * (dW_dyn_state + self.l1_lambda_dyn * cp.sign(self.W_dyn_state))
        self.W_dyn_reward -= self.lr * (dW_dyn_reward + self.l1_lambda_dyn * cp.sign(self.W_dyn_reward)) # Apply L1 to reward head too
        
        # Apply L1 and update for new representation layers
        self.W_repr1 -= self.lr * (dW_repr1 + self.l1_lambda_repr * cp.sign(self.W_repr1))
        self.W_repr2 -= self.lr * (dW_repr2 + self.l1_lambda_repr * cp.sign(self.W_repr2))
        self.W_repr3 -= self.lr * (dW_repr3 + self.l1_lambda_repr * cp.sign(self.W_repr3))

        # HARD PROXIMAL PRUNING
        self.W_policy = cp.where(cp.abs(self.W_policy) < self.prune_threshold, 0, self.W_policy)
        self.W_value = cp.where(cp.abs(self.W_value) < self.prune_threshold, 0, self.W_value) # Prune value head
        self.W_dyn_state = cp.where(cp.abs(self.W_dyn_state) < self.prune_threshold, 0, self.W_dyn_state)
        self.W_dyn_reward = cp.where(cp.abs(self.W_dyn_reward) < self.prune_threshold, 0, self.W_dyn_reward)
        
        # Prune new representation layers
        self.W_repr1 = cp.where(cp.abs(self.W_repr1) < self.prune_threshold, 0, self.W_repr1)
        self.W_repr2 = cp.where(cp.abs(self.W_repr2) < self.prune_threshold, 0, self.W_repr2)
        self.W_repr3 = cp.where(cp.abs(self.W_repr3) < self.prune_threshold, 0, self.W_repr3)

    def save(self, path="outcomes/best_world_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "W_repr1": cp.asnumpy(self.W_repr1),
            "W_repr2": cp.asnumpy(self.W_repr2),
            "W_repr3": cp.asnumpy(self.W_repr3),
            "W_dyn_state": cp.asnumpy(self.W_dyn_state),
            "W_dyn_reward": cp.asnumpy(self.W_dyn_reward),
            "W_policy": cp.asnumpy(self.W_policy), # Save W_policy
            "W_value": cp.asnumpy(self.W_value),   # Save W_value
            "input_size": self.W_repr1.shape[0], # Use W_repr1 for input_size
            "lr": self.lr
        }
        with open(path, "wb") as f: pickle.dump(data, f)

    def load(self, path="outcomes/best_world_model.pkl"):
        if not os.path.exists(path): return
        with open(path, "rb") as f: data = pickle.load(f)
        if data.get("input_size", 0) != self.W_repr1.shape[0]: # Use W_repr1 for input_size check
            print(f"⚠️  Checkpoint input_size mismatch "
                  f"(file={data.get('input_size')}, expected={self.W_repr1.shape[0]}). "
                  f"Load skipped — training from scratch.")
            return
        self.W_repr1 = cp.asarray(data["W_repr1"])
        self.W_repr2 = cp.asarray(data["W_repr2"])
        self.W_repr3 = cp.asarray(data["W_repr3"])
        self.W_dyn_state = cp.asarray(data["W_dyn_state"])
        self.W_dyn_reward = cp.asarray(data["W_dyn_reward"])
        self.W_policy = cp.asarray(data.get("W_policy", self.W_policy)) # Load W_policy, with fallback
        self.W_value = cp.asarray(data.get("W_value", self.W_value))   # Load W_value, with fallback
        self.lr = data.get("lr", self.lr)
        print(f"World Model loaded. Resuming at LR: {self.lr:.2e}")
