import numpy as np
import pandas as pd


class RegressionAgent:
    def __init__(self, name, pace, brain, threshold=0.005):
        self.name = name
        self.pace = pace
        self.brain = brain

        # Action Threshold: Predicted profit required to trigger a BUY
        # e.g., 0.005 means we need the brain to predict a 0.5% move
        self.threshold = threshold

        self.history = []
        self.max_history = pace * 5  # Increased history for better smoothing
        self.inventory = []
        self.max_slots = 1
        self.total_realized_pl = 0.0  # Track actual money made

    def get_avg_unrealized(self, current_price):
        if not self.inventory: return 0.0
        profits = [(current_price - p) / p for p in self.inventory]
        return sum(profits) / len(profits)

    def get_state(self, row):
        self.history.append(row)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        hist_df = pd.DataFrame(self.history)

        # Calculate Slope: Current value vs. the average of the history window
        # This gives us the "Directional Momentum" relative to the agent's pace
        def get_slope(col):
            if len(hist_df) < 0.02: return 0.0
            return row[col] - hist_df[col].mean()

        return np.array([
            row['RSI_Scaled'], get_slope('RSI_Scaled'),
            row['MACD_Scaled'], get_slope('MACD_Scaled'),
            row['BB_Scaled'], get_slope('BB_Scaled'),
            row['OBV_Scaled'], get_slope('OBV_Scaled'),
            1.0  # Bias
        ])

    def act(self, row):
        state = self.get_state(row)

        # The Brain now predicts a continuous number: Expected Profit
        predicted_profit = self.brain.predict(state)

        # ACTION LOGIC (Threshold-Based)
        # 0: BUY, 1: SELL, 2: HOLD
        action = 2  # Default to HOLD

        if not self.inventory:
            # Entry Logic: Is the indicators' "geometry" promising?
            if predicted_profit > self.threshold:
                action = 0  # BUY
        else:
            # Exit Logic: Has the profit expectancy turned negative?
            # We exit if the brain predicts we are about to lose 0.1% or more
            if predicted_profit < -0.001:
                action = 1  # SELL

        return state, action

    def handle_reward(self, state, action, current_price):
        """
        In Regression, 'reward' is simply the ground truth:
        What was the actual unrealized profit change?
        """
        unrealized = self.get_avg_unrealized(current_price)

        # 1. Update Inventory
        if action == 0:
            self.inventory.append(current_price)
        elif action == 1 and self.inventory:
            entry = self.inventory.pop(0)
            self.total_realized_pl += (current_price - entry) / entry

        # 2. TEACH THE BRAIN:
        # We tell the brain: "This state resulted in THIS much unrealized profit"
        # We don't need 'reward' logic anymore, just the raw profit value.
        self.brain.learn(state, unrealized)

        return unrealized