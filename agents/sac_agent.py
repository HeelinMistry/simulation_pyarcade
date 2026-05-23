"""
agents/sac_agent.py
───────────────────
Discrete Soft Actor-Critic (Christodoulou 2019) with automatic entropy tuning.

Key properties vs the previous World Model:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Problem              Old approach           SAC fix             │
  │─────────────────────────────────────────────────────────────────│
  │ Entropy oscillation  Manual entropy_beta    Auto temperature α  │
  │ Value head failure   Shared W_pred          Separate Q-networks │
  │ Moving target        No target network      EMA target critic   │
  │ Overestimation bias  Single value head      min(Q1, Q2)         │
  │ Policy/value coupling Shared repr grads     Independent nets    │
  └─────────────────────────────────────────────────────────────────┘

SAC-Discrete update equations (per batch):
  ① Critic:  Q_target = r + γ * Σ_a' π(a'|s') * [min_Q(s',a') - α * log π(a'|s')]
             L_critic = MSE(Q1(s,a), Q_target) + MSE(Q2(s,a), Q_target)

  ② Actor:   L_actor  = Σ_a π(a|s) * [α * log π(a|s) - min_Q(s,a)]
             (minimise → maximise Q while keeping entropy high)

  ③ Temp:    L_α = α * (H[π(·|s)] - H_target)
             where H_target = 0.70 * log(|A|) ≈ 0.97 nats for |A|=4
             (α increases if current entropy < target, decreases if >)

  ④ Target:  θ_target ← τ·θ + (1-τ)·θ_target   (soft EMA, every step)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pickle

from models.actor import Actor
from models.critic import DualCritic


class SACAgent:
    def __init__(
        self,
        state_dim:      int   = 50,
        action_dim:     int   = 4,
        hidden_dim:     int   = 256,
        lr:             float = 3e-4,
        gamma:          float = 0.97,
        tau:            float = 0.005,   # soft target update rate
        target_entropy: float = None,    # None → auto (0.98 * log|A|)
        device:         str   = None,
    ):
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma      = gamma
        self.tau        = tau
        self.action_dim = action_dim

        # ── Networks ────────────────────────────────────────────────────────
        self.actor = Actor(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = DualCritic(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic_target  = DualCritic(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target critic is never trained directly — freeze to prevent
        # accidental gradient flow.
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── Optimisers ──────────────────────────────────────────────────────
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=lr, weight_decay=1e-4
        )

        # ── Automatic entropy temperature ────────────────────────────────────
        # H_target in nats: we want the policy to maintain at least 98% of
        # maximum-entropy (uniform) behaviour, expressed as a lower bound.
        # For |A|=4:  0.98 * ln(4) ≈ 1.355 nats  ≈ 1.96 bits
        # log_alpha is the learnable scalar; alpha = exp(log_alpha) is always +.
        if target_entropy is None:
            self.target_entropy = 0.5 * np.log(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        # ── Diagnostics ─────────────────────────────────────────────────────
        self.training_steps  = 0
        self.last_actor_loss  = 0.0
        self.last_critic_loss = 0.0
        self.last_alpha       = 1.0
        self.last_entropy     = 0.0         # nats

        print(f"[SACAgent] device={self.device}  |  "
              f"target_entropy={self.target_entropy:.3f} nats "
              f"({self.target_entropy/np.log(2):.2f} bits)  |  "
              f"γ={gamma}  τ={tau}  lr={lr:.1e}")

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def alpha(self) -> torch.Tensor:
        """Current temperature — always positive via exp."""
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Inference-time action selection.
        Returns (action_int, probs_np).
        """
        return self.actor.act(state, deterministic=deterministic, device=self.device)

    def update(self, replay_buffer, batch_size: int = 256):
        """
        One gradient step on all three components.
        Call this every N environment steps (N=4 is a good default).
        Safe to call more often than once — multiple updates per step
        are common in SAC and are fine because it's off-policy.
        """
        if not replay_buffer.is_ready(batch_size):
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        S  = torch.FloatTensor(states).to(self.device)
        A  = torch.LongTensor(actions).to(self.device)
        R  = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        S_ = torch.FloatTensor(next_states).to(self.device)
        D  = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ── ① Critic update ─────────────────────────────────────────────────
        with torch.no_grad():
            next_probs     = self.actor(S_)                        # (B, 4)
            next_log_probs = torch.log(next_probs + 1e-8)          # (B, 4)

            q1_next, q2_next = self.critic_target(S_)              # (B, 4) each
            min_q_next = torch.min(q1_next, q2_next)               # (B, 4)

            # SAC-Discrete soft-value: expectation over all next actions
            # V(s') = Σ_a π(a|s') * [Q(s',a) - α * log π(a|s')]
            soft_v_next = (
                next_probs * (min_q_next - self.alpha * next_log_probs)
            ).sum(dim=1, keepdim=True)                             # (B, 1)

            td_target = R + self.gamma * (1.0 - D) * soft_v_next  # (B, 1)

        q1, q2     = self.critic(S)                                 # (B, 4) each
        q1_taken   = q1.gather(1, A.unsqueeze(1))                   # (B, 1)
        q2_taken   = q2.gather(1, A.unsqueeze(1))                   # (B, 1)

        critic_loss = F.mse_loss(q1_taken, td_target) + \
                      F.mse_loss(q2_taken, td_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── ② Actor update ───────────────────────────────────────────────────
        probs     = self.actor(S)                                   # (B, 4)
        log_probs = torch.log(probs + 1e-8)                         # (B, 4)

        # Critic gradients must not flow into the actor update —
        # use detach() rather than no_grad() so actor grads are unaffected.
        with torch.no_grad():
            q1_pi, q2_pi = self.critic(S)
            min_q_pi = torch.min(q1_pi, q2_pi)                     # (B, 4)

        # Actor loss = expectation of [α * log π - Q]  (minimise → maximise Q-entropy)
        actor_loss = (
            probs * (self.alpha.detach() * log_probs - min_q_pi)
        ).sum(dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── ③ Temperature update ─────────────────────────────────────────────
        # Current policy entropy H[π(·|s)]  (in nats, detached from graph)
        with torch.no_grad():
            probs_fresh = self.actor(S)
            log_probs_fresh = torch.log(probs_fresh + 1e-8)
        entropy = -(probs_fresh * log_probs_fresh).sum(dim=1).mean()

        # α increases when entropy < target (policy too deterministic)
        # α decreases when entropy > target (policy too random)
        alpha_loss = self.log_alpha * (entropy - self.target_entropy).detach()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=-4.0, max=2.0)

        # ── ④ Soft target update ─────────────────────────────────────────────
        # θ_target ← τ·θ + (1-τ)·θ_target
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(),
                             self.critic_target.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * p.data)

        # ── Diagnostics bookkeeping ──────────────────────────────────────────
        self.training_steps   += 1
        self.last_critic_loss  = float(critic_loss.item())
        self.last_actor_loss   = float(actor_loss.item())
        self.last_alpha        = float(self.alpha.item())
        self.last_entropy      = float(entropy.item())

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = "outcomes/sac_agent.pt", replay_buffer=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor":          self.actor.state_dict(),
            "critic":         self.critic.state_dict(),
            "critic_target":  self.critic_target.state_dict(),
            "log_alpha":      self.log_alpha.data,
            "actor_opt":      self.actor_opt.state_dict(),
            "critic_opt":     self.critic_opt.state_dict(),
            "alpha_opt":      self.alpha_opt.state_dict(),
            "training_steps": self.training_steps,
            "target_entropy": self.target_entropy,
        }, path)
        if replay_buffer is not None:
            buffer_path = path.replace(".pt", "_buffer.pkl")
            with open(buffer_path, "wb") as f:
                pickle.dump(list(replay_buffer.buffer), f)
            print(f"[SACAgent] ✅ Saved replay buffer → {buffer_path}")
        print(f"[SACAgent] ✅ Saved → {path}  (step {self.training_steps})")

    def load(self, path: str = "outcomes/sac_agent.pt", replay_buffer=None):
        if not os.path.exists(path):
            print(f"[SACAgent] ⚠️  No checkpoint at {path} — starting fresh.")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data    = ckpt["log_alpha"].to(self.device)
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        self.training_steps    = ckpt.get("training_steps", 0)
        self.target_entropy    = ckpt.get("target_entropy", self.target_entropy)
        print(f"[SACAgent] ✅ Loaded ← {path}  "
              f"(step {self.training_steps}  α={self.last_alpha:.4f})")

        self.log_alpha.data.fill_(0.0)  # reset α to 1.0 in exp-space, but now with lower target it will quickly drop
        print(f"[SACAgent] log_alpha reset to 0.0 for fresh entropy tuning")

        if replay_buffer is not None:
            buffer_path = path.replace(".pt", "_buffer.pkl")
            if os.path.exists(buffer_path):
                with open(buffer_path, "rb") as f:
                    for transition in pickle.load(f):
                        replay_buffer.buffer.append(transition)
                print(f"[SACAgent] ✅ Loaded replay buffer ← {buffer_path}")
            else:
                print(f"[SACAgent] ⚠️  No replay buffer checkpoint at {buffer_path}.")
