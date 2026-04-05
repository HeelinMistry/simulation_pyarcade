import numpy as np
import cupy as cp
from agents.UnifiedWorldModel import UnifiedWorldModel
from data.data_manager import update_master_data
from agents.state_aggregator import StateAggregator


def run_world_model_epoch(model, aggregator, indicators, prices, num_samples=1000):
    data_len = len(prices)
    total_loss = 0

    print(f"--- Starting World Model Training Phase ({num_samples} samples) ---")

    for _ in range(num_samples):
        # Pick a random point in history
        idx = np.random.randint(200, data_len - 50)

        # 1. Capture State S
        aggregator.warm_up_all(indicators, idx)
        s_raw = aggregator.get_state({'position': 0, 'unrealized_pnl': 0})

        # 2. Pick a random action (Exploring the 'physics' of the market)
        action = np.random.randint(0, 4)

        # 3. Observe Next State S' and Reward R
        # (Simplified reward: change in price for that action)
        aggregator.update(indicators[idx + 1])
        next_s_raw = aggregator.get_state({'position': 0, 'unrealized_pnl': 0})

        price_change = (prices[idx + 1] - prices[idx]) / prices[idx]
        actual_reward = price_change if action == 0 else (-price_change if action == 1 else 0)

        # 4. Store in Replay Buffer
        # We use a dummy target_pi (random) for now until MCTS is active
        target_pi = np.zeros(4)
        target_pi[action] = 1.0

        model.record(s_raw, action, next_s_raw, actual_reward, target_pi)

        # 5. Optimization Step
        if len(model.memory) > model.batch_size:
            model.learn_from_memory()

    print("✅ Epoch Complete. Model internal 'physics' updated.")


if __name__ == "__main__":
    df = update_master_data()
    indicators = df[["RSI_Scaled", "MACD_Scaled", "BB_Scaled", "OBV_Scaled"]].values.astype(np.float32)
    prices = df["Close"].values.astype(np.float32)

    # Setup Architecture
    paces = (1, 2, 4, 8, 12)
    input_size = (len(paces) * 12) + 2

    model = UnifiedWorldModel(input_size=input_size)
    aggregator = StateAggregator(paces=paces)

    # Run Test Training
    run_world_model_epoch(model, aggregator, indicators, prices)
    model.save("outcomes/test_world_model.pkl")