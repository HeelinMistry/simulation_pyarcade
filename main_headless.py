import numpy as np
import os

from agents.UnifiedWorldModel import UnifiedWorldModel
from agents.MCTSPlanner import MCTSPlanner  # The new lookahead engine
from agents.unified_executor import UnifiedExecutor
from data.data_manager import update_master_data


# -------------------------------------------------------
# Utility: Simple Position Manager
# -------------------------------------------------------
class PositionManager:
    def __init__(self, commission=0.00015):  # Added synthetic friction
        self.position = None
        self.entry = 0.0
        self.commission = commission

    def step(self, action, price):
        reward = 0.0
        if action == 0 and self.position != "LONG":
            reward = self._close_current(price)
            self.position = "LONG"
            self.entry = price * (1 + self.commission)
        elif action == 1 and self.position != "SHORT":
            reward = self._close_current(price)
            self.position = "SHORT"
            self.entry = price * (1 - self.commission)
        elif action == 2 and self.position is not None:
            reward = self._close_current(price)
        return reward

    def _close_current(self, price):
        if self.position is None: return 0.0
        if self.position == "LONG":
            exit_price = price * (1 - self.commission)
            pnl = (exit_price - self.entry) / self.entry
        else:
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
def shape_reward(pnl, side, prediction_corr=0):
    multiplier = 120 if side == "SHORT" else 100
    reward = pnl * multiplier

    if pnl < 0:
        penalty_scale = 1.8 if side == "SHORT" else 1.5
        reward *= penalty_scale

    # World Model Bonus: Did the hallucination match reality?
    if prediction_corr > 0.7:
        reward += 0.05

    return reward


def run_stochastic_epoch(executor, indicators, prices, num_trades=15, train=True):
    data_len = len(prices)
    total_reward = 0.0
    trades_completed = 0
    pos_mgr = PositionManager()

    WALK_DURATION = 2000
    correlation_scores = []

    while trades_completed < num_trades:
        start_idx = np.random.randint(200, data_len - WALK_DURATION)
        executor.aggregator.warm_up_all(indicators, start_idx)

        active_trade_sequence = []

        for i in range(WALK_DURATION):
            idx = start_idx + i

            # Deep Analysis Projection
            if i % 50 == 0 and (idx + 15) < len(prices):
                proj_path, bias = executor.predict_trajectory(indicators[idx], prices[idx], horizon=15)
                real_path = prices[idx + 1:idx + 16]

                if np.std(proj_path) > 1e-9 and np.std(real_path) > 1e-9:
                    corr = np.corrcoef(proj_path, real_path)[0, 1]
                    correlation_scores.append(corr)

            # 1. Get current raw features
            current_raw_features = executor.get_state(indicators[idx], prices[idx])

            # 2. Planning: MCTS simulates futures and returns action + expected policy
            if train:
                action, mcts_probs = executor.planner.search_best_action(current_raw_features)
            else:
                # Deterministic fallback for validation
                action, mcts_probs = executor.planner.search_best_action(current_raw_features)
                action = int(np.argmax(mcts_probs))
                if mcts_probs[action] < 0.35: action = 3  # Force HOLD on low conviction

            # 3. Environment Step
            pnl = pos_mgr.step(action, prices[idx])

            # 4. Get the *Actual* Next State to train the Dynamics Network
            next_idx = min(idx + 1, data_len - 1)
            next_raw_features = executor.get_state(indicators[next_idx], prices[next_idx])

            if action == 0 and pos_mgr.position == "LONG":
                executor.inventory = [prices[idx]]
            elif pnl != 0.0:
                executor.inventory = []

            # STORE SEARCH STATISTICS instead of just (state, action)
            if train:
                active_trade_sequence.append({
                    'state': current_raw_features,
                    'action': action,
                    'next_state': next_raw_features,
                    'mcts_probs': mcts_probs
                })

            if pnl != 0.0:
                shaped_reward = shape_reward(pnl, pos_mgr.position)

                # Record the full sequence to the World Model
                if train:
                    for step_data in active_trade_sequence:
                        # We train the Dynamics net to predict the 'next_state' and 'shaped_reward'
                        # We train the Prediction net to output the 'mcts_probs' and 'shaped_reward'
                        executor.planner.model.record(
                            s=step_data['state'],
                            a=step_data['action'],
                            next_s=step_data['next_state'],
                            r=shaped_reward,
                            target_pi=step_data['mcts_probs']
                        )
                    executor.planner.model.learn_from_memory()

                total_reward += pnl
                trades_completed += 1
                active_trade_sequence = []
                pos_mgr.reset()

            if trades_completed >= num_trades:
                break

    avg_corr = np.mean(correlation_scores) if correlation_scores else 0
    return total_reward, avg_corr


# -------------------------------------------------------
# Main Simulation
# -------------------------------------------------------
def run_sim():
    print("Loading master data...")
    df = update_master_data()
    features = ["RSI_Scaled", "MACD_Scaled", "BB_Scaled", "OBV_Scaled"]
    indicators = df[features].values.astype(np.float32)
    prices = df["Close"].values.astype(np.float32)

    split = int(len(indicators) * 0.8)
    train_X, train_P = indicators[:split], prices[:split]
    val_X, val_P = indicators[split:], prices[split:]

    paces = (1, 2, 4, 8, 12)
    input_size = (len(paces) * 12) + 2

    # --- NEW WORLD MODEL INITIALIZATION ---
    world_model = UnifiedWorldModel(input_size=input_size, hidden_size=128)

    # Load weights if they exist (requires updated save/load logic in WorldModel)
    if os.path.exists("outcomes/best_world_model.pkl"):
        world_model.load("outcomes/best_world_model.pkl")

    planner = MCTSPlanner(world_model, lookahead_depth=20)
    executor = UnifiedExecutor(name="MainExecutor", planner=planner, paces=paces)

    best_val = -np.inf
    patience = 50
    bad_epochs = 0
    epoch = 1
    val_history = []

    print("🚀 Starting World Model Training...")

    while True:
        train_pnl, avg_corr = run_stochastic_epoch(
            executor, train_X, train_P, num_trades=200, train=True
        )

        val_pnl, avg_corr = run_stochastic_epoch(
            executor, val_X, val_P, num_trades=200, train=False
        )

        val_history.append(val_pnl)
        if len(val_history) > 10: val_history.pop(0)
        smoothed_val = np.mean(val_history)

        if smoothed_val > best_val:
            best_val = smoothed_val
            print(f"🏆 NEW BEST SMOOTHED P/L: {best_val:+.2%}. Saving World Model...")
            world_model.save("outcomes/best_world_model.pkl")
            bad_epochs = 0
        else:
            bad_epochs += 1

        world_model.lr = max(1e-5, world_model.lr * 0.98)

        print(
            f"Epoch {epoch:03d} | LR: {world_model.lr:.2e} | Val P/L: {val_pnl:+.2%} (Smooth: {smoothed_val:+.2%}) (Corr: {avg_corr:+.2%})")

        if bad_epochs >= patience:
            print(f"🛑 Early stopping reached. Best Smoothed Val: {best_val:.4f}")
            break

        epoch += 1


if __name__ == "__main__":
    run_sim()