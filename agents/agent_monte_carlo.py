import numpy as np
import pandas as pd


class MCAgent:
    def __init__(self, name, pace, brain):
        self.name = name
        self.pace = pace
        self.brain = brain

        # --- Wallet & Inventory ---
        self.inventory = []
        self.max_slots = 1
        self.commission = 0.0  # Set to 0.0002 for realistic simulation

        # --- Memory Management ---
        # Stores tuples of (state, action) for the CURRENT active trade only
        self.episode_memory = []

        # History for state generation (Technical Indicators)
        self.history = []
        self.max_history = pace * 3  # Increased slightly for better smoothing calculations

        # --- Monte Carlo Settings ---
        # Gamma: Discount factor. 0.99 means the 'Buy' action 100 steps ago
        # still claims ~36% of the credit for the Sell reward.
        self.current_index = 0
        self.gamma = 0.99

        self.total_reward = 0.0

    def get_state(self, row):
        # 1. Update history
        self.history.append(row)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        if len(self.history) < 2:
            return np.zeros(9)

        try:
            # We extract directly to numpy
            rsi_vec = np.array([h['RSI_Scaled'] for h in self.history], dtype=np.float64)
            macd_vec = np.array([h['MACD_Scaled'] for h in self.history], dtype=np.float64)
            bb_vec = np.array([h['BB_Scaled'] for h in self.history], dtype=np.float64)
            obv_vec = np.array([h['OBV_Scaled'] for h in self.history], dtype=np.float64)
        except KeyError as e:
            return np.zeros(9)

        # Note: We added 'pace' as a parameter here
        def get_features(vec, pace):
            val = vec[-1]
            # Amplifies tiny micro-moves for fast agents so the brain can 'feel' them
            slope = (val - np.mean(vec)) * np.sqrt(pace)
            return val, slope

        def threshold(val, limit=0.01):  # Lowered limit slightly for micro-moves
            return val if abs(val) > limit else 0.0

        # 5. Calculate Features using the Agent's specific pace
        r_val, r_slope = get_features(rsi_vec, self.pace)
        m_val, m_slope = get_features(macd_vec, self.pace)
        b_val, b_slope = get_features(bb_vec, self.pace)
        o_val, o_slope = get_features(obv_vec, self.pace)

        return np.array([
            r_val, threshold(r_slope),
            m_val, threshold(m_slope),
            b_val, threshold(b_slope),
            o_val, threshold(o_slope),
            1.0  # Bias
        ], dtype=np.float64)

    def act(self, row):
        """
        Decides on an action (Buy/Sell/Hold) based on the current state.
        """
        state = self.get_state(row)
        probs = self.brain.get_probs(state)

        # 1. Action Masking (prevent illegal moves)
        if len(self.inventory) >= self.max_slots:
            probs[0] = 0.0  # Cannot BUY
        if len(self.inventory) == 0:
            probs[1] = 0.0  # Cannot SELL

        # 2. Renormalize Probabilities
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs /= prob_sum
        else:
            probs = np.array([0.0, 0.0, 1.0])  # Default to HOLD

        # 3. Select Action
        action = np.random.choice([0, 1, 2], p=probs)

        return state, action

    def handle_reward(self, state, action, current_price):
        """
        Processes the action result.
        CRITICAL: Does NOT learn immediately. Stores memory until the trade closes.
        """
        reward = 0.0

        # --- CASE 1: BUY ---
        if action == 0:
            self.inventory.append(current_price * (1 + self.commission))
            # Start the episode
            self.episode_memory.append((state, action))

        # --- CASE 2: HOLD ---
        elif action == 2:
            # Only record 'Hold' actions if we are actually in a trade.
            # We don't care about holding when we have no money at stake.
            if self.inventory:
                self.episode_memory.append((state, action))

        # --- CASE 3: SELL (The Trigger) ---
        elif action == 1 and self.inventory:
            entry_price = self.inventory.pop(0)

            # 1. Calculate the 'True' result of the episode
            raw_profit = (current_price - entry_price) / entry_price
            net_profit = raw_profit * (1 - self.commission)

            # 2. Trigger Monte Carlo Learning
            self._finalize_episode(net_profit)

            # 3. Update stats
            self.total_reward += net_profit
            reward = net_profit

        return reward

    def _finalize_episode(self, final_profit):
        """
        Backpropagates the final trade result to all steps in the episode history.
        """
        # Identify Trade Duration
        trade_duration = len(self.episode_memory)
        if trade_duration == 0:
            return

        # --- DYNAMIC SCALING ---
        # 1. Pace Normalization:
        # High pace (8h) agents have fewer trades but bigger % moves.
        # We divide by sqrt(pace) to keep their gradients generic enough for the shared brain.
        normalized_profit = final_profit / np.sqrt(self.pace if self.pace > 0 else 1)

        # 2. Clip gradients to prevent exploding weights during market crashes/pumps
        clipped_reward = np.clip(normalized_profit, -0.1, 0.1)

        # --- BACKPROPAGATION LOOP ---
        # We iterate backwards? No, we just iterate and apply power of gamma.
        # G_t = Reward * Gamma^(Steps from end)

        for i, (state, action) in enumerate(self.episode_memory):
            # Calculate how 'far' this action was from the result
            steps_from_end = trade_duration - i - 1

            # Discount the reward based on distance
            discounted_reward = clipped_reward * (self.gamma ** steps_from_end)

            # Teach the brain
            self.brain.learn(state, action, discounted_reward)

        # Clear memory for the next trade
        self.episode_memory = []

    def teleport(self, new_index, df):
        """Resets the agent's position and history to prevent data leakage."""
        self.current_index = new_index
        self.inventory = []
        self.episode_memory = []

        # Pre-fill history so the next 'get_state' is accurate for the new location
        start_lookback = max(0, new_index - self.max_history)
        self.history = df.iloc[start_lookback:new_index].to_dict('records')

    def reset_and_teleport(self, df):
        """Jumps to a random location in the data and refills history."""
        # 1. Pick a random index, leaving room for history lookback and a trade duration
        margin = self.max_history + 500
        self.current_index = np.random.randint(margin, len(df) - 500)

        # 2. Reset trade-specific memory
        self.inventory = []
        self.episode_memory = []

        # 3. Pre-fill technical history so the first state is accurate
        lookback_start = self.current_index - self.max_history
        # Convert to list of dicts to keep your get_state logic happy
        self.history = df.iloc[lookback_start:self.current_index].to_dict('records')