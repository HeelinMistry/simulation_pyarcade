import numpy as np
import pandas as pd


class LongOnlyAgent:
    def __init__(self, name, pace, brain):
        self.name = name
        self.pace = pace
        self.brain = brain

        # Wallet & Inventory
        self.history = []
        self.max_history = pace * 2  # Keep enough data to represent its timeframe
        self.inventory = []
        self.max_slots = 1
        # self.commission = 0.0002
        self.commission = 0.0
        self.previous_unrealized_profit = 0.0
        self.total_reward = 0.0

        self.gamma = 0.95 ** self.pace
        self.trade_history = []  # Stores (state, action) pairs during a trade

    def get_avg_unrealized(self, current_price):
        if not self.inventory:
            return 0.0
        profits = [(current_price - p) / p for p in self.inventory]
        return sum(profits) / len(profits)

    def get_state(self, row):
        # 1. Update history
        self.history.append(row)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # 2. Convert history to a temporary DataFrame for easy averaging
        hist_df = pd.DataFrame(self.history)

        # 3. Smooth the indicators based on pace
        rsi_smooth = hist_df['RSI_Scaled'].mean()
        rsi_slope = hist_df['RSI_Scaled'].iloc[-1] - hist_df['RSI_Scaled'].mean()

        macd_smooth = hist_df['MACD_Scaled'].mean()
        macd_slope = hist_df['MACD_Scaled'].iloc[-1] - hist_df['MACD_Scaled'].mean()

        bb_smooth = hist_df['BB_Scaled'].mean()
        bb_slope = hist_df['BB_Scaled'].iloc[-1] - hist_df['BB_Scaled'].mean()

        obv_smooth = hist_df['OBV_Scaled'].mean()
        obv_slope = hist_df['OBV_Scaled'].iloc[-1] - hist_df['OBV_Scaled'].mean()

        def threshold(val, limit=0.02):
            return val if abs(val) > limit else 0.0

        return np.array([
            rsi_smooth,
            threshold(rsi_slope),
            macd_smooth,
            threshold(macd_slope),
            bb_smooth,
            threshold(bb_slope),
            obv_smooth,
            threshold(obv_slope),
            1.0
        ])

    def act(self, row):
        state = self.get_state(row)
        probs = self.brain.get_probs(state)

        # Ensure we are using 64-bit precision for the math
        probs = np.array(probs, dtype=np.float64)

        # 1. Action Masking: 0=BUY, 1=SELL, 2=HOLD
        if len(self.inventory) >= self.max_slots:
            probs[0] = 0.0
        if len(self.inventory) == 0:
            probs[1] = 0.0

            # 2. Robust Normalization
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs /= prob_sum
        else:
            # Fallback if all probs are zeroed (unlikely but safe)
            probs = np.array([0.0, 0.0, 1.0])

        # 3. Final Precision Fix: Force the sum to be exactly 1.0
        # We subtract the sum of the first two from 1.0 to set the third
        probs[-1] = 1.0 - np.sum(probs[:-1])

        # Ensure no negative probabilities (rare edge case of the math above)
        probs = np.maximum(probs, 0)
        probs /= probs.sum()  # Final pass to ensure sum=1.0

        action = np.random.choice([0, 1, 2], p=probs)
        return state, action

    def handle_reward(self, state, action, current_price):
        current_unrealized = self.get_avg_unrealized(current_price)
        reward = 0.0

        if action == 0:  # BUY
            self.inventory.append(current_price * (1 + self.commission))
            reward = 0.0  # Neutral entry

        elif action == 1:  # SELL
            # The ultimate 'Truth' signal
            if self.inventory:
                entry_price = self.inventory.pop(0)
                realized_profit = ((current_price * (1 + self.commission)) - entry_price) / entry_price
                # Stronger weight on the final realization to "lock in" the logic
                reward = realized_profit * 5.0

        elif action == 2:  # HOLD
            if self.inventory:
                # Reward/Penalize the 'Change' since the last tick
                # This teaches the brain to stay in trades that are still growing
                delta = current_unrealized - self.previous_unrealized_profit
                reward = delta * 2.0
            else:
                reward = 0.0  # No penalty for being flat

        self.previous_unrealized_profit = current_unrealized
        self.total_reward += reward

        normalized_reward = reward / np.sqrt(self.pace)
        clipped_force = np.clip(normalized_reward, -0.05, 0.05)
        self.brain.learn(state, action, clipped_force)
        return reward
