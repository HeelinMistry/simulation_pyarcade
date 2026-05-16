"""
agents/unified_executor.py  (SAC refactor)
──────────────────────────────────────────
Thin execution layer that sits between the SAC agent and the environment.

What changed vs the World Model version:
  REMOVED  — MCTSPlanner reference and 10-step rollout
  REMOVED  — conviction filter (SAC's temperature handles decisiveness)
  KEPT     — StateAggregator (92-d vector unchanged)
  KEPT     — position/PnL tracking and commission model
  KEPT     — get_status() and get_state() signatures (live_unified.py compatible)
  CHANGED  — step() calls agent.select_action() instead of planner.search_best_action()

Conviction filter note:
  In the World Model you needed a hard 45% filter because entropy was
  uncontrolled. SAC's temperature α regulates this automatically — when
  the model is uncertain, α is high and probabilities are spread; when
  confident, α is low and the argmax is clear. You can optionally re-add
  a soft filter here (e.g., skip LONG/SHORT if max_prob < 0.35) once you
  observe the trained conviction distribution, but don't hard-code it
  during training or it corrupts the reward signal.
"""

import numpy as np
import collections
from agents.state_aggregator import StateAggregator

COMMISSION = 0.00015  # Matches training — do not change without retraining


class UnifiedExecutor:
    """
    Wraps a SACAgent for step-by-step interaction with market data.

    Parameters
    ----------
    name        : identifier used for logging and checkpoint naming
    agent       : SACAgent instance (already loaded/initialised)
    paces       : pace tuple forwarded to StateAggregator (must match training)
    deterministic: use argmax policy (live) vs sampled policy (training)
    """

    def __init__(
        self,
        name:          str,
        agent,                          # SACAgent — avoids circular import
        paces:         tuple = (1, 2, 4, 8, 12),
        deterministic: bool  = False,
        num_indicators: int = 6, # Added num_indicators
    ):
        self.name          = name
        self.agent         = agent
        self.aggregator    = StateAggregator(paces, num_indicators=num_indicators) # Pass num_indicators
        self.deterministic = deterministic

        # Position state
        self.inventory     = collections.deque(maxlen=100)
        self.current_side  = None   # "LONG" | "SHORT" | None

        # P/L tracking
        self.total_reward  = 0.0
        self.tick          = 0

        # For environment.py / diagnostics compatibility
        self.last_probs    = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # ── State construction ────────────────────────────────────────────────────

    def portfolio_info(self, current_price: float) -> dict:
        if self.current_side is not None and self.inventory:
            entry = self.inventory[0]
            if self.current_side == "LONG":
                side_val = 1.0
                u_pnl    = (current_price - entry) / entry
            else:
                side_val = -1.0
                u_pnl    = (entry - current_price) / entry
            u_pnl = np.clip(u_pnl, -0.05, 0.05)
        else:
            side_val, u_pnl = 0.0, 0.0

        return {"position": side_val, "unrealized_pnl": u_pnl}

    def get_state(self, indicators: np.ndarray, price: float) -> np.ndarray:
        """Build and return the current 92-d state vector."""
        self.aggregator.update(indicators)
        portfolio_info = self.portfolio_info(price)
        return self.aggregator.get_state(portfolio_info)

    # ── Core step ────────────────────────────────────────────────────────────

    def step(self, indicators: np.ndarray, price: float, tick: int):
        """
        Execute one market tick.

        Returns
        -------
        action : int   0=LONG 1=SHORT 2=CLOSE 3=HOLD
        probs  : np.ndarray (4,)
        reward : float  realised P/L this tick (0 if no close event)
        state  : np.ndarray (92,) the current state vector
        """
        self.tick = tick
        state     = self.get_state(indicators, price) # State is computed here and returned

        action, probs = self.agent.select_action(state, deterministic=self.deterministic)
        self.last_probs = probs

        reward = self._execute(action, price)
        self.total_reward += reward
        return action, probs, reward, state

    # ── Execution logic ───────────────────────────────────────────────────────

    def _execute(self, action: int, price: float) -> float:
        """Apply action to position, return realised P/L (0 if no close)."""
        reward = 0.0

        if action == 0:   # LONG
            if self.current_side == "SHORT":
                reward = self._close(price)
            if self.current_side != "LONG":
                self.inventory.append(price * (1 + COMMISSION))
                self.current_side = "LONG"

        elif action == 1:  # SHORT
            if self.current_side == "LONG":
                reward = self._close(price)
            if self.current_side != "SHORT":
                self.inventory.append(price * (1 - COMMISSION))
                self.current_side = "SHORT"

        elif action == 2:  # CLOSE
            if self.current_side is not None:
                reward = self._close(price)

        # action == 3: HOLD — no change

        return reward

    def _close(self, price: float) -> float:
        """Close current position with exit commission. Returns net P/L."""
        if not self.inventory:
            return 0.0
        entry = self.inventory.popleft()
        if self.current_side == "LONG":
            pnl = (price * (1 - COMMISSION) - entry) / entry
        else:
            pnl = (entry - price * (1 + COMMISSION)) / entry
        self.current_side = None
        return float(pnl)

    # ── Status / compatibility ────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "position": self.current_side or "FLAT",
            "pnl":      self.total_reward,
            "pnl_str":  f"{self.total_reward:.4%}",
            "entry":    self.inventory[0] if self.inventory else None,
        }

    def reset_position(self):
        """Force-close any open position without recording P/L (e.g. end of epoch)."""
        self.inventory.clear()
        self.current_side = None