"""
main_sac.py
───────────
Headless training loop for the SAC-Discrete agent.

Epoch structure
───────────────
One epoch = one full pass through the processed CSV (train split).
SAC is off-policy so we don't reset the replay buffer between epochs —
transitions from epoch 1 are still valid for critic updates in epoch 50.

Reward design
─────────────
Sparse realised P/L is used as the reward signal, with one small shaping
term to help the critic bootstrap before any positions are closed.

  r_t = realised_pnl     if position closed this tick  (dense signal)
      + unrealized_delta * 0.1                          (shaping: 10% weight)
      + 0                otherwise

The shaping coefficient (0.1) keeps the unrealized signal subordinate to
actual P/L so the agent doesn't optimise for paper gains over realised ones.
Set it to 0.0 if you want purely sparse rewards (slower but cleaner).

Update schedule
───────────────
SAC is updated every UPDATE_EVERY steps once the buffer is ready.
Multiple updates per data step are valid and common — we use
UPDATES_PER_STEP=2 to help the critic catch up to the fast-moving
financial time series.
"""

import time

import numpy as np
import pandas as pd

from agents.replay_buffer import ReplayBuffer
from agents.sac_agent import SACAgent
from agents.unified_executor import UnifiedExecutor
from data.data_manager import update_master_data

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MASTER_CSV       = "data/processed/XRPUSDT_master_processed.csv"
CHECKPOINT_PATH  = "outcomes/sac_agent.pt"
BEST_PATH        = "outcomes/sac_agent_best.pt"

FEATURES   = ["RSI_Scaled", "MACD_Scaled", "BB_Scaled",
              "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled"]
PACES      = (1, 4, 16, 64)
STATE_DIM  = (len(FEATURES) * 2 * len(PACES)) + 2   # 50 (6 indicators * 2 features per indicator * 4 paces + 2 portfolio features)
ACTION_DIM = 4

# Training hyperparameters
NUM_EPOCHS       = 200
TRAIN_SPLIT      = 0.8          # first 80% for training, last 20% for val
WARMUP_IDX       = 512          # aggregator warm-up lookback rows

PATIENCE = 8          # epochs without improvement before stopping
WARMUP_EPOCHS = 5
MIN_IMPROVE   = 0.001   # val PnL must improve by 0.5pp to reset patience

BUFFER_CAPACITY  = 500_000
BATCH_SIZE       = 256
UPDATE_EVERY     = 8            # update SAC every N environment steps
UPDATES_PER_STEP = 1            # gradient steps per update call
LR               = 3e-4
GAMMA            = 0.97
TAU              = 0.005

# Reward shaping
SHAPING_COEFF    = 0.1          # weight on unrealized PnL delta; set 0 for sparse-only

# Logging
LOG_EVERY_TICKS  = 2_000        # console print frequency within an epoch
SAVE_EVERY_EPOCH = 5


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

# Remove reward_scaled entirely — keep reward in fraction units
INVALID_ACTION_PENALTY  = -0.003    # was -0.0005; stronger flat-CLOSE deterrent
MICRO_HOLD_COST         = 0.000005  # 0.5 bp/tick when in position — variance only
EPSILON_START           = 0.05      # was 0.10; less noise killing SHORT
EPSILON_END             = 0.005     # was 0.01

def compute_shaped_reward(realised_pnl, prev_unrealized, curr_unrealized,
                          is_holding, is_invalid_close, in_position):
    unrealized_delta = curr_unrealized - prev_unrealized
    shaped = realised_pnl + SHAPING_COEFF * unrealized_delta

    if is_invalid_close:
        shaped += INVALID_ACTION_PENALTY

    if in_position and is_holding:
        shaped -= MICRO_HOLD_COST          # tiny per-tick cost, keeps variance alive

    if curr_unrealized < -0.02:
        shaped -= 0.0002 * abs(curr_unrealized) * 10

    return shaped   # NO ×100 — stay in fraction units


def get_unrealized(state: np.ndarray) -> float:
    """
    Extracts unrealized P/L from the state vector.
    Assumes unrealized P/L is the last element of the state vector.
    """
    return float(state[-1])


def run_epoch(executor: UnifiedExecutor, df: pd.DataFrame,
              replay_buffer: ReplayBuffer, agent: SACAgent,
              train: bool = True, epsilon=0.0) -> dict:
    """
    Single pass through df.
    If train=True: push to buffer and call agent.update().
    If train=False: evaluation pass only (no buffer writes, no updates).
    Returns a metrics dict.
    """
    indicators_arr = df[FEATURES].values.astype(np.float32)
    prices_arr     = df["Close"].values.astype(np.float32)
    n              = len(df)

    executor.total_reward = 0.0
    executor.inventory.clear()
    executor.current_side = None

    # Warm up aggregator so slope/std features are populated from the start
    executor.aggregator.tick = 0
    executor.aggregator.warm_up_all(indicators_arr, WARMUP_IDX)

    total_realised   = 0.0
    n_trades         = 0
    action_counts    = [0, 0, 0, 0]
    update_count     = 0
    prev_unrealized  = 0.0

    # Initialize for the first iteration
    # Before loop:
    prev_state = executor.aggregator.get_state(executor.portfolio_info(prices_arr[WARMUP_IDX]))
    prev_action, prev_reward_scaled, prev_done = None, 0.0, False

    for i in range(WARMUP_IDX + 1, n):
        indicators = indicators_arr[i]
        price = prices_arr[i]

        action, probs, realised_pnl, s_t = executor.step(
            indicators, price, tick=i, epsilon=epsilon
        )        # s_t = state actor used = market features at tick i + pre-execute portfolio

        curr_unrealized = get_unrealized(s_t)
        action_counts[action] += 1

        if realised_pnl != 0.0:
            total_realised += realised_pnl
            n_trades += 1
            prev_unrealized = 0.0

        was_flat_before = (executor.current_side is None and action == 2 and realised_pnl == 0.0)

        reward = compute_shaped_reward(
            realised_pnl, prev_unrealized, curr_unrealized,
            is_holding=(executor.current_side is not None and realised_pnl == 0.0),
            is_invalid_close=was_flat_before,
            in_position=(executor.current_side is not None),
        )

        prev_unrealized = curr_unrealized if executor.current_side else 0.0
        done = (i == n - 1)

        if train and prev_action is not None:
            # Transition: agent was in prev_state, took prev_action, got prev_reward, landed in s_t
            replay_buffer.push(prev_state, prev_action, reward, s_t, done)
            if (i % UPDATE_EVERY == 0) and replay_buffer.is_ready(BATCH_SIZE):
                for _ in range(UPDATES_PER_STEP):
                    agent.update(replay_buffer, batch_size=BATCH_SIZE)
                update_count += 1

        prev_state = s_t
        prev_action = action

        # ── Console heartbeat ────────────────────────────────────────────────
        if train and (i % LOG_EVERY_TICKS == 0):
            total_ticks = i - WARMUP_IDX
            pct = ", ".join(f"{c/total_ticks:.0%}" for c in action_counts)
            print(
                f"  tick {i:>6}  |  realised P/L: {total_realised:+.4%}  "
                f"| trades: {n_trades}  |  actions [L/S/C/H]: {pct}  "
                f"| α={agent.last_alpha:.4f}  H={agent.last_entropy/np.log(2):.2f}b"
            )

    return {
        "realised_pnl":   total_realised,
        "n_trades":        n_trades,
        "action_counts":   action_counts,
        "update_count":    update_count,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    df = update_master_data()

    # ── Load data ────────────────────────────────────────────────────────────
    df = df[["Open_time", "Close"] + FEATURES].dropna().reset_index(drop=True)
    split = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df   = df.iloc[split:].reset_index(drop=True)
    print(f"  Train rows: {len(train_df):,}  |  Val rows: {len(val_df):,}")

    # ── Initialise components ────────────────────────────────────────────────
    agent = SACAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        hidden_dim=128, 
        lr=LR, gamma=GAMMA, tau=TAU,
    )
    agent.load(CHECKPOINT_PATH)   # resumes if checkpoint exists

    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

    train_executor = UnifiedExecutor("Train", agent, paces=PACES, deterministic=False, num_indicators=len(FEATURES))
    val_executor   = UnifiedExecutor("Val",   agent, paces=PACES, deterministic=True, num_indicators=len(FEATURES))

    best_val_pnl = -np.inf
    no_improve = 0

    # ── Training loop ────────────────────────────────────────────────────────
    EPSILON_START = 0.10  # 10% random actions in epoch 1
    EPSILON_END = 0.01  # 1% random actions by epoch 20
    EPSILON_DECAY = (EPSILON_END / EPSILON_START) ** (1 / 20)

    epsilon = EPSILON_START
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_metrics = run_epoch(
            train_executor, train_df, replay_buffer, agent,
            train=True, epsilon=epsilon
        )
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Validation pass (no updates, deterministic policy)
        val_metrics = run_epoch(
            val_executor, val_df, replay_buffer, agent, train=False
        )

        elapsed = time.time() - t0
        t_pnl   = train_metrics["realised_pnl"]
        v_pnl   = val_metrics["realised_pnl"]
        t_ac    = train_metrics["action_counts"]
        v_ac    = val_metrics["action_counts"]

        def pct_str(counts):
            total = max(sum(counts), 1)
            return "/".join(f"{c/total:.0%}" for c in counts)

        print(f"\n  ── Results ──────────────────────────────────────────")
        print(f"  Train P/L : {t_pnl:+.4%}  |  trades={train_metrics['n_trades']}")
        print(f"  Val   P/L : {v_pnl:+.4%}  |  trades={val_metrics['n_trades']}")
        print(f"  Train actions [L/S/C/H]: {pct_str(t_ac)}")
        print(f"  Val   actions [L/S/C/H]: {pct_str(v_ac)}")
        print(f"  SAC   α={agent.last_alpha:.4f}  "
              f"H={agent.last_entropy/np.log(2):.2f}b  "
              f"critic_loss={agent.last_critic_loss:.4f}  "
              f"actor_loss={agent.last_actor_loss:.4f}")
        print(f"  Time  : {elapsed:.1f}s  |  updates this epoch: {train_metrics['update_count']}")

        # Early stopping on val PnL degradation
        if v_pnl > best_val_pnl + MIN_IMPROVE:
            best_val_pnl = v_pnl
            no_improve = 0
            agent.save(BEST_PATH)
            print(f"  ⭐ New best val P/L: {best_val_pnl:+.4%}")
        else:
            no_improve += 1

        if epoch >= WARMUP_EPOCHS and no_improve >= PATIENCE:
            print(f"  ⚠ No val improvement for {PATIENCE} epochs — early stop")
            break

        # Periodic checkpoint
        if epoch % SAVE_EVERY_EPOCH == 0:
            agent.save(CHECKPOINT_PATH)

    print(f"\n✅ Training complete.  Best val P/L: {best_val_pnl:+.4%}")
    agent.save(CHECKPOINT_PATH)


if __name__ == "__main__":
    main()

# copy outcomes\sac_agent_best.pt outcomes\sac_agent.pt
# del outcomes\sac_agent_buffer.pkl