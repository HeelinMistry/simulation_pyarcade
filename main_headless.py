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
        self.last_action = 3  # Default to HOLD

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

        self.last_action = action  # Update last action after step
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
        self.last_action = 3  # Reset to HOLD


# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------
def shape_reward(pnl, trade_duration=0):
    multiplier = 400
    reward = pnl * multiplier

    if trade_duration < 15 and pnl != 0.0:
        reward -= 0.1

    # Symmetric scaling for both gains and losses
    # No special multipliers for negative PNL, it's already scaled by 'multiplier'

    # Normalize the reward to be within a reasonable range, e.g., [-1, 1]
    # Assuming max possible reward is around 20 (0.05 * 400) and min is -20 (-0.05 * 400)
    # Dividing by 20 brings most values into [-1, 1]
    reward /= 20.0 # Adjusted normalization factor

    return reward


def run_stochastic_epoch(executor, indicators_param, prices_param, num_trades=15, train=True):
    total_reward = 0.0  # Initialized here
    trades_completed = 0
    correlation_scores = []

    # print(f"DEBUG: total_reward initialized to {total_reward}") # NEW DEBUG PRINT
    # print(f"DEBUG: trades_completed initialized to {trades_completed}") # NEW DEBUG PRINT
    # print(f"DEBUG: correlation_scores initialized to {correlation_scores}") # NEW DEBUG PRINT

    # Use len(indicators_param) directly for data_len consistency
    data_len = len(indicators_param)
    # print(f"DEBUG: run_stochastic_epoch - id(indicators_param): {id(indicators_param)}, len(indicators_param): {len(indicators_param)}, len(prices_param): {len(prices_param)}, data_len: {data_len}")

    pos_mgr = PositionManager()

    WALK_DURATION = 2000
    min_start_idx = 200

    # Ensure data_len is sufficient for WALK_DURATION and min_start_idx
    upper_bound_for_start_idx = data_len - WALK_DURATION
    if upper_bound_for_start_idx < min_start_idx:
        print(
            f"ERROR: data_len ({data_len}) is too small for WALK_DURATION ({WALK_DURATION}) and min_start_idx ({min_start_idx}). Adjust WALK_DURATION or data split. Returning 0.0, 0.0.")
        return 0.0, 0.0  # Return zero PNL and correlation for this epoch

    # INTERVENTION 1.3: Epsilon for MCTS exploration
    epsilon = 0.1 if train else 0.0  # Only explore during training

    while trades_completed < num_trades:
        start_idx = np.random.randint(min_start_idx, upper_bound_for_start_idx)
        # print(f"DEBUG: run_stochastic_epoch - start_idx: {start_idx}, upper_bound_for_start_idx: {upper_bound_for_start_idx}")
        executor.aggregator.warm_up_all(indicators_param, start_idx)
        executor.current_side = None
        executor.inventory = []
        active_trade_sequence = []

        # Initialize next_raw_features for the first iteration
        next_raw_features = executor.get_state(indicators_param[start_idx], prices_param[start_idx])

        for i in range(WALK_DURATION):
            idx = start_idx + i

            # Ensure all_indicators is correctly indexed
            if idx >= data_len:
                print(f"ERROR: idx ({idx}) exceeded data_len ({data_len}) during inner loop. Breaking.")
                break  # Break inner loop if index goes out of bounds

            # current_raw_features for this iteration is the next_raw_features from the previous iteration
            current_raw_features = next_raw_features

            if i % 50 == 0 and (idx + 15) < data_len:  # Use data_len for bounds check
                proj_path, bias = executor.predict_trajectory(current_raw_features, prices_param[idx], horizon=15)
                real_path = prices_param[idx + 1:idx + 16]
                if np.std(proj_path) > 1e-9 and np.std(real_path) > 1e-9:
                    corr = np.corrcoef(proj_path, real_path)[0, 1]
                    correlation_scores.append(corr)

            action, mcts_probs = executor.planner.search_best_action(current_raw_features, epsilon=epsilon)

            if action in (0, 1) and mcts_probs[action] < 0.45:
                action = 3

            if not train:
                action = int(np.argmax(mcts_probs))
                if mcts_probs[action] < 0.35: action = 3

            prev_action_for_record = pos_mgr.last_action      # ← save BEFORE step updates it
            trade_duration_before_step = pos_mgr.trade_duration # Save duration before step() resets it
            pnl = pos_mgr.step(action, prices_param[idx])

            # After pos_mgr.step(), sync executor position so portfolio features are live
            executor.current_side = pos_mgr.position
            executor.inventory = [pos_mgr.entry] if pos_mgr.position is not None else []

            # Calculate the next_raw_features for the *next* iteration
            next_idx = min(idx + 1, data_len - 1)
            next_raw_features = executor.get_state(indicators_param[next_idx], prices_param[next_idx])

            if train:
                # Cap active_trade_sequence at ~100 entries (rolling window)
                if len(active_trade_sequence) >= 100:
                    active_trade_sequence.pop(0)
                active_trade_sequence.append({
                    'state': current_raw_features,
                    'action': action,
                    'prev_action': prev_action_for_record,     # ← use saved value
                    'next_state': next_raw_features,
                    'mcts_probs': mcts_probs
                })

            if pnl != 0.0:
                shaped_reward = shape_reward(pnl, trade_duration_before_step) # Use the saved duration
                if train:
                    # Assign 0 reward to all intermediate steps
                    for step_data in active_trade_sequence[:-1]:
                        executor.planner.model.record(
                            s=step_data['state'],
                            a=step_data['action'],
                            prev_a=step_data['prev_action'],
                            next_s=step_data['next_state'],
                            r=0.0, # Intermediate steps get 0 reward
                            target_pi=step_data['mcts_probs']
                        )
                    # Assign shaped_reward only to the last step (the one that closed the trade)
                    last_step_data = active_trade_sequence[-1]
                    executor.planner.model.record(
                        s=last_step_data['state'],
                        a=last_step_data['action'],
                        prev_a=last_step_data['prev_action'],
                        next_s=last_step_data['next_state'],
                        r=shaped_reward, # Only the closing step gets the shaped reward
                        target_pi=last_step_data['mcts_probs']
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

    # print(f"DEBUG: run_sim - len(indicators) (full): {len(indicators)}, len(prices) (full): {len(prices)}")
    # print(f"DEBUG: run_sim - len(train_X): {len(train_X)}, len(train_P): {len(train_P)}")
    # print(f"DEBUG: run_sim - len(val_X): {len(val_X)}, len(val_P): {len(val_P)}")

    paces = (1, 2, 4, 8, 12)
    input_size = (len(features) * 3 * len(paces)) + 2
    world_model = UnifiedWorldModel(input_size=input_size, hidden_size=128)

    if os.path.exists("outcomes/best_world_model.pkl"):
        world_model.load("outcomes/best_world_model.pkl")

    planner = MCTSPlanner(world_model, lookahead_depth=10)
    executor = UnifiedExecutor(name="MainExecutor", planner=planner, paces=paces)

    best_val = -np.inf
    patience = 50
    bad_epochs = 0
    epoch = 1
    val_history = []

    print(f"🚀 Training for GENERALIZED LOGIC | Input Size: {input_size}")

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

        if bad_epochs >= patience: break
        epoch += 1


if __name__ == "__main__":
    run_sim()