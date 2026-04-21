import numpy as np
import os

from agents.UnifiedWorldModel import UnifiedWorldModel
from agents.MCTSPlanner import MCTSPlanner
from agents.unified_executor import UnifiedExecutor
from data.data_manager import update_master_data


# -------------------------------------------------------
# Utility: Simple Position Manager
# -------------------------------------------------------
class PositionManager:
    def __init__(self, commission=0.00015):
        self.position = None
        self.entry = 0.0
        self.commission = commission
        self.trade_duration = 0

    def step(self, action, price):
        reward = 0.0
        self.trade_duration += 1
        
        if action == 0 and self.position != "LONG":
            reward = self._close_current(price)
            self.position = "LONG"
            self.entry = price * (1 + self.commission)
            self.trade_duration = 0
        elif action == 1 and self.position != "SHORT":
            reward = self._close_current(price)
            self.position = "SHORT"
            self.entry = price * (1 - self.commission)
            self.trade_duration = 0
        elif action == 2:
            reward = self._close_current(price)
            self.trade_duration = 0
        elif action == 3:
            pass
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
        self.trade_duration = 0


# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------
def shape_reward(pnl, side, trade_duration=0):
    multiplier = 400 
    reward = pnl * multiplier

    # 1. Commitment Penalty: Penalize trades lasting less than 15 ticks
    if trade_duration < 15 and pnl != 0.0:
        reward -= 0.1

    # 2. Drawdown Avoidance (Risk Shaping)
    if pnl < -0.02:  # 2% loss is the "Danger Zone"
        reward *= 5.0 # Hyper-penalty for major drawdown
    elif pnl < 0:
        reward *= 2.0 # Standard loss penalty

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

            if i % 50 == 0 and (idx + 15) < len(prices):
                proj_path, bias = executor.predict_trajectory(indicators[idx], prices[idx], horizon=15)
                real_path = prices[idx + 1:idx + 16]
                if np.std(proj_path) > 1e-9 and np.std(real_path) > 1e-9:
                    corr = np.corrcoef(proj_path, real_path)[0, 1]
                    correlation_scores.append(corr)

            current_raw_features = executor.get_state(indicators[idx], prices[idx])
            action, mcts_probs = executor.planner.search_best_action(current_raw_features)
            
            if action in (0, 1) and mcts_probs[action] < 0.45:
                action = 3

            if not train:
                action = int(np.argmax(mcts_probs))
                if mcts_probs[action] < 0.35: action = 3

            pnl = pos_mgr.step(action, prices[idx])
            next_idx = min(idx + 1, data_len - 1)
            next_raw_features = executor.get_state(indicators[next_idx], prices[next_idx])

            if train:
                active_trade_sequence.append({
                    'state': current_raw_features,
                    'action': action,
                    'next_state': next_raw_features,
                    'mcts_probs': mcts_probs
                })

            if pnl != 0.0:
                shaped_reward = shape_reward(pnl, pos_mgr.position, pos_mgr.trade_duration)
                if train:
                    for step_data in active_trade_sequence:
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
            if trades_completed >= num_trades: break

    avg_corr = np.mean(correlation_scores) if correlation_scores else 0
    return total_reward, avg_corr


def run_sim():
    df = update_master_data()
    features = ["RSI_Scaled", "MACD_Scaled", "BB_Scaled", "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled"]
    indicators = df[features].values.astype(np.float32)
    prices = df["Close"].values.astype(np.float32)

    split = int(len(indicators) * 0.8)
    train_X, train_P = indicators[:split], prices[:split]
    val_X, val_P = indicators[split:], prices[split:]

    paces = (1, 2, 4, 8, 12)
    input_size = (len(features) * 3 * len(paces)) + 2 
    world_model = UnifiedWorldModel(input_size=input_size, hidden_size=128)

    if os.path.exists("outcomes/best_world_model.pkl"):
        world_model.load("outcomes/best_world_model.pkl")

    planner = MCTSPlanner(world_model, lookahead_depth=100)
    executor = UnifiedExecutor(name="MainExecutor", planner=planner, paces=paces)

    best_val = -np.inf
    patience = 50
    bad_epochs = 0
    epoch = 1
    val_history = []

    print(f"🚀 Training for DRAWDOWN AVOIDANCE | Input Size: {input_size}")

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
        print(f"Epoch {epoch:03d} | LR: {world_model.lr:.2e} | Val P/L: {val_pnl:+.2%} (Smooth: {smoothed_val:+.2%}) (Corr: {avg_corr:+.2%})")

        if bad_epochs >= patience: break
        epoch += 1

if __name__ == "__main__":
    run_sim()
