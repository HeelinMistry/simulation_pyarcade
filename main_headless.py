import numpy as np

from agents.unified_brain import UnifiedBrain
from agents.unified_executor import UnifiedExecutor
from data.data_manager import update_master_data


# -------------------------------------------------------
# Utility: Simple Position Manager
# -------------------------------------------------------
class PositionManager:
    def __init__(self, commission=0.0002):
        self.position = None  # "LONG", "SHORT", or None
        self.entry = 0.0
        self.commission = commission

    def step(self, action, price):
        reward = 0.0

        # Action 0: Go LONG (or switch from Short to Long)
        if action == 0 and self.position != "LONG":
            reward = self._close_current(price)
            self.position = "LONG"
            self.entry = price * (1 + self.commission)

        # Action 1: Go SHORT (or switch from Long to Short)
        elif action == 1 and self.position != "SHORT":
            reward = self._close_current(price)
            self.position = "SHORT"
            self.entry = price * (1 - self.commission)

        # Action 2: CLOSE to Neutral
        elif action == 2 and self.position is not None:
            reward = self._close_current(price)

        # Action 3: HOLD (Do nothing)
        return reward

    def _close_current(self, price):
        if self.position is None:
            return 0.0

        if self.position == "LONG":
            exit_price = price * (1 - self.commission)
            pnl = (exit_price - self.entry) / self.entry
        else:  # SHORT
            exit_price = price * (1 + self.commission)
            pnl = (self.entry - exit_price) / self.entry

        self.position = None
        self.entry = 0.0
        return pnl

    def reset(self):
        self.position = None
        self.entry = 0.0


# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------
def shape_reward(pnl, side):
    """
    pnl: The raw percentage change (positive or negative)
    duration: How many ticks the trade lasted
    side: 'LONG' or 'SHORT'
    """
    # 1. Base Scaling
    # We use a higher multiplier for shorts to account for the 'friction'
    # of betting against the general market trend.
    # multiplier = 100
    multiplier = 120 if side == "SHORT" else 100
    reward = pnl * multiplier

    # 2. Asymmetric Risk Penalty
    # We punish losses harder than we reward gains to force the brain
    # to find high-probability 'Circles'.
    if pnl < 0:
        # Shorts are punished slightly more for large drawdowns
        penalty_scale = 1.8 if side == "SHORT" else 1.5
        reward *= penalty_scale

    # # 3. Time-Decay (The 'Passive' Penalty)
    # # If a trade lasts too long without hitting profit, it's dead money.
    # # This prevents the brain from 'praying' for a reversal.
    # time_penalty = -0.0001 * duration
    # reward += time_penalty

    # 4. Success Bonus
    # A small fixed bonus for any closed trade that survived commissions
    # if pnl > 0.001: # > 0.1% profit
    #     reward += 0.05

    return reward


def run_stochastic_epoch(executor, indicators, prices, num_trades=15, train=True):
    data_len = len(prices)
    total_reward = 0.0
    trades_completed = 0
    pos_mgr = PositionManager()

    WALK_DURATION = 2000

    while trades_completed < num_trades:
        start_idx = np.random.randint(200, data_len - WALK_DURATION)
        executor.aggregator.warm_up_all(indicators, start_idx)

        active_trade_sequence = []

        for i in range(WALK_DURATION):
            idx = start_idx + i

            # FIX 1: Pass current_price and current_tick (idx) to get_state
            # This allows the Brain to see its Unrealized PnL and Duration
            state = executor.get_state(indicators[idx], prices[idx], idx)

            # Decision logic
            action = executor.brain.act(state) if train else executor.brain.act_deterministic(state)

            # Execute in the PositionManager (Ground Truth)
            pnl = pos_mgr.step(action, prices[idx])

            # IMPORTANT: Sync the executor's internal inventory with the trainer's PositionManager
            # This ensures the Brain's 'Internal State' (seen in get_state) is accurate
            if action == 0 and pos_mgr.position == "LONG":
                executor.inventory = [prices[idx]]
                executor.entry_tick = idx
            elif pnl != 0.0:
                executor.inventory = []
                executor.entry_tick = 0

            if train:
                active_trade_sequence.append((state, action))

            if pnl != 0.0:
                # TRADE CLOSED
                shaped_reward = shape_reward(pnl, pos_mgr.position)

                if train:
                    for s, a in active_trade_sequence:
                        executor.brain.record(s, a, shaped_reward)
                    executor.brain.learn_from_memory()

                total_reward += pnl
                trades_completed += 1
                active_trade_sequence = []
                pos_mgr.reset()

            if trades_completed >= num_trades:
                break

    return total_reward


# -------------------------------------------------------
# Main Simulation
# -------------------------------------------------------
def run_sim():
    # --------------------------------
    # Load Data
    # --------------------------------
    print("Loading master data...")
    df = update_master_data()
    features = [
        "RSI_Scaled",
        "MACD_Scaled",
        "BB_Scaled",
        "OBV_Scaled"
    ]
    indicators = df[features].values.astype(np.float32)
    prices = df["Close"].values.astype(np.float32)

    # --------------------------------
    # Train / Val Split
    # --------------------------------
    split = int(len(indicators) * 0.8)

    train_X, train_P = indicators[:split], prices[:split]
    val_X, val_P = indicators[split:], prices[split:]

    # --------------------------------
    # Training Hyperparameters
    # --------------------------------
    initial_lr = 5e-5
    min_lr = 1e-5
    decay_rate = 0.98  # Reduce LR by 2% every epoch

    paces = (1, 2, 3, 4, 5)

    # FIX 2: Update Input Size
    # (5 agents * 12 features each) + 2 internal portfolio features = 38
    input_size = (len(paces) * 12) + 2

    brain = UnifiedBrain(input_size=input_size, lr=initial_lr)
    brain.load()

    # Note: brain.load() now has a safety check to ensure it doesn't

    executor = UnifiedExecutor(name="MainExecutor", brain=brain, paces=paces)

    # --------------------------------
    # Robust Early Stopping
    # --------------------------------
    best_val = -np.inf
    patience = 50  # Increased from 10
    bad_epochs = 0
    epoch = 1
    val_history = []  # To track the moving average

    print("🚀 Starting Extended Training...")

    while True:
        # 1. Training Step with Input Augmentation
        # Add 1% random noise to training indicators to prevent overfitting
        # noisy_train_X = train_X + np.random.normal(0, 0.01, train_X.shape)

        # 1. Distributed Training
        train_pnl = run_stochastic_epoch(
            executor, train_X, train_P, num_trades=200, train=True
        )

        # 2. Distributed Validation
        # We do more trades here to get a statistically significant score
        val_pnl = run_stochastic_epoch(
            executor, val_X, val_P, num_trades=100, train=False
        )

        # 3. Update History and Check for Positivity
        val_history.append(val_pnl)
        if len(val_history) > 10: val_history.pop(0)
        smoothed_val = np.mean(val_history)

        # Inside your while True: loop in run_sim
        if smoothed_val > best_val:
            best_val = smoothed_val
            print(f"🏆 NEW BEST SMOOTHED P/L: {best_val:+.2%}. Saving brain...")
            brain.save("outcomes/best_unified_brain.pkl")  # Save a protected copy
            brain.save()  # Overwrite current working brain
        else:
            # bad_epochs += 1
            pass

        # 5. Learning Rate Decay
        # Slowly lower the heat as the brain matures
        brain.lr = max(min_lr, brain.lr * decay_rate)

        print(
            f"Epoch {epoch:03d} | LR: {brain.lr:.2e} | Val P/L: {val_pnl:+.2%} (Smooth: {smoothed_val:+.2%})")

        if bad_epochs >= patience:
            print(f"🛑 Early stopping reached. Best Smoothed Val: {best_val:.4f}")
            break

        if epoch % 50 == 0:
            brain.memory.clear()

        epoch += 1


if __name__ == "__main__":
    run_sim()
