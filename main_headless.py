import numpy as np

from agents.unified_brain import UnifiedBrain
from agents.unified_executor import UnifiedExecutor
from data.data_manager import update_master_data


# -------------------------------------------------------
# Utility: Simple Position Manager
# -------------------------------------------------------
class PositionManager:
    def __init__(self, commission=0.0002):
        self.position = None
        self.entry = 0.0
        self.commission = commission

    def step(self, action, price):
        """
        Executes action and returns reward if trade closes.
        """
        reward = 0.0

        # BUY
        if action == 0 and self.position is None:
            self.position = "LONG"
            self.entry = price * (1 + self.commission)

        # SELL
        elif action == 1 and self.position == "LONG":
            exit_price = price * (1 - self.commission)
            reward = (exit_price - self.entry) / self.entry
            self.position = None
            self.entry = 0.0

        # HOLD = 2 → do nothing
        return reward

    def reset(self):
        self.position = None
        self.entry = 0.0


# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------
def run_stochastic_epoch(
        executor: UnifiedExecutor,
        indicators: np.ndarray,
        prices: np.ndarray,
        steps=2000,
        train=True
):
    data_len = len(prices)
    max_steps = min(steps, data_len - 200)
    if max_steps <= 0:
        return 0.0

    pos_mgr = PositionManager()
    start = np.random.randint(50, data_len - max_steps)

    memory = []
    total_reward = 0.0

    for i in range(max_steps):
        idx = start + i

        # 1. Delegate state aggregation to the Executor
        state = executor.get_state(indicators[idx])

        # 2. Brain chooses action
        action = executor.brain.act(state)

        # 3. Environment step
        reward = pos_mgr.step(action, prices[idx])
        total_reward += reward

        if train:
            memory.append((state, action, reward))

        # 4. Episode reset
        if reward != 0.0:
            pos_mgr.reset()

    # --------------------------------
    # Batch Learning Phase
    # --------------------------------
    if train and memory:
        states, actions, rewards = zip(*memory)
        executor.brain.learn(states, actions, rewards)

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
    # Initialize Unified Architecture
    # --------------------------------
    paces = (1, 2, 4, 8, 16)

    # StateAggregator outputs 4 features per pace.
    # Therefore, input_size = len(paces) * 8 features
    input_size = len(paces) * 8

    brain = UnifiedBrain(input_size=input_size, lr=3e-4)
    brain.load()  # Load previous weights if they exist

    executor = UnifiedExecutor(name="MainExecutor", brain=brain, paces=paces)

    # --------------------------------
    # Early Stopping
    # --------------------------------
    best_val = -np.inf
    patience = 10
    bad_epochs = 0
    epoch = 1

    # --------------------------------
    # Training Loop
    # --------------------------------
    print("Starting simulation...")
    while True:
        train_pl = run_stochastic_epoch(
            executor, train_X, train_P, steps=3000, train=True
        )
        val_pl = run_stochastic_epoch(
            executor, val_X, val_P, steps=800, train=False
        )

        # --------------------------------
        # Model Selection
        # --------------------------------
        if val_pl > best_val:
            best_val = val_pl
            bad_epochs = 0
            brain.save()
            print(f"✔ Epoch {epoch} | Saved | Val P/L: {best_val:.3%} | Train P/L: {train_pl:.3%}")
        else:
            bad_epochs += 1
            print(f"✖ Epoch {epoch} | No Impr | Val P/L: {val_pl:.3%} | Train P/L: {train_pl:.3%}")

        if bad_epochs >= patience:
            print("Early stopping reached.")
            break
        epoch += 1


if __name__ == "__main__":
    run_sim()
