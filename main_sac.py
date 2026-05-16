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

import os
import time
import numpy as np
import pandas as pd

from agents.sac_agent      import SACAgent
from agents.replay_buffer  import ReplayBuffer
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
PACES      = (1, 2, 4, 8, 12)
STATE_DIM  = (len(FEATURES) * 3 * len(PACES)) + 2   # 92
ACTION_DIM = 4

# Training hyperparameters
NUM_EPOCHS       = 200
TRAIN_SPLIT      = 0.8          # first 80% for training, last 20% for val
WARMUP_IDX       = 200          # aggregator warm-up lookback rows

BUFFER_CAPACITY  = 200_000
BATCH_SIZE       = 256
UPDATE_EVERY     = 4            # update SAC every N environment steps
UPDATES_PER_STEP = 2            # gradient steps per update call
LR               = 3e-4
GAMMA            = 0.99
TAU              = 0.005

# Reward shaping
SHAPING_COEFF    = 0.1          # weight on unrealized PnL delta; set 0 for sparse-only

# Logging
LOG_EVERY_TICKS  = 2_000        # console print frequency within an epoch
SAVE_EVERY_EPOCH = 5


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def compute_shaped_reward(
    realised_pnl: float,
    prev_unrealized: float,
    curr_unrealized: float,
) -> float:
    """
    Combine realised P/L with a small unrealized-delta shaping term.
    If the position was just closed, realised_pnl is non-zero and
    prev_unrealized/curr_unrealized are both 0 (position gone).
    """
    unrealized_delta = curr_unrealized - prev_unrealized
    return realised_pnl + SHAPING_COEFF * unrealized_delta


def get_unrealized(executor: UnifiedExecutor, price: float) -> float:
    info = executor._portfolio_info(price)
    return float(info["unrealized_pnl"])


def run_epoch(executor: UnifiedExecutor, df: pd.DataFrame,
              replay_buffer: ReplayBuffer, agent: SACAgent,
              train: bool = True) -> dict:
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

    prev_state = executor.aggregator.get_state(
        executor._portfolio_info(prices_arr[WARMUP_IDX])
    )

    for i in range(WARMUP_IDX + 1, n):
        indicators = indicators_arr[i]
        price      = prices_arr[i]

        # ── Environment step ────────────────────────────────────────────────
        action, probs, realised_pnl, curr_state = executor.step(indicators, price, tick=i)
        curr_unrealized = get_unrealized(executor, price)
        action_counts[action] += 1

        if realised_pnl != 0.0:
            total_realised += realised_pnl
            n_trades       += 1

        # ── Reward shaping ──────────────────────────────────────────────────
        if realised_pnl != 0.0:
            prev_unrealized = 0.0   # position just closed; unrealized was already credited

        reward = compute_shaped_reward(realised_pnl, prev_unrealized, curr_unrealized)
        prev_unrealized = curr_unrealized if executor.current_side else 0.0

        done = (i == n - 1)

        # ── Build next state ─────────────────────────────────────────────────
        # We need s_t for buffer but get_state() already advanced the aggregator.
        # We stored prev_state before the step, so:
        curr_state = executor.aggregator.get_state(executor._portfolio_info(price))

        if train:
            replay_buffer.push(prev_state, action, reward, curr_state, done)

            # Update SAC on schedule
            if (i % UPDATE_EVERY == 0) and replay_buffer.is_ready(BATCH_SIZE):
                for _ in range(UPDATES_PER_STEP):
                    agent.update(replay_buffer, batch_size=BATCH_SIZE)
                update_count += 1

        prev_state = curr_state

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
    features = ["RSI_Scaled", "MACD_Scaled", "BB_Scaled", "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled"]
    indicators = df[features].values.astype(np.float32)
    prices = df["Close"].values.astype(np.float32)

    # ── Load data ────────────────────────────────────────────────────────────
    df = df[["Open_time", "Close"] + FEATURES].dropna().reset_index(drop=True)
    split = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df   = df.iloc[split:].reset_index(drop=True)
    print(f"  Train rows: {len(train_df):,}  |  Val rows: {len(val_df):,}")

    # ── Initialise components ────────────────────────────────────────────────
    agent  = SACAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        lr=LR, gamma=GAMMA, tau=TAU,
    )
    agent.load(CHECKPOINT_PATH)   # resumes if checkpoint exists

    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

    train_executor = UnifiedExecutor("Train", agent, paces=PACES, deterministic=False)
    val_executor   = UnifiedExecutor("Val",   agent, paces=PACES, deterministic=True)

    best_val_pnl = -np.inf

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        print(f"\n{'═'*60}")
        print(f"  Epoch {epoch}/{NUM_EPOCHS}  |  buffer={len(replay_buffer):,}")
        print(f"{'═'*60}")

        # Training pass
        train_metrics = run_epoch(
            train_executor, train_df, replay_buffer, agent, train=True
        )

        # Validation pass (no updates, deterministic policy)
        val_metrics = run_epoch(
            val_executor, val_df, replay_buffer, agent, train=False
        )

        elapsed = time.time() - t0
        t_pnl   = train_metrics["realised_pnl"]
        v_pnl   = val_metrics["realised_pnl"]
        t_ac    = train_metrics["action_counts"]
        v_ac    = val_metrics["action_counts"]
        t_n     = sum(t_ac) - t_ac[3]   # non-HOLD ticks

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

        # Save best model on val P/L
        if v_pnl > best_val_pnl:
            best_val_pnl = v_pnl
            agent.save(BEST_PATH)
            print(f"  ⭐ New best val P/L: {best_val_pnl:+.4%}")

        # Periodic checkpoint
        if epoch % SAVE_EVERY_EPOCH == 0:
            agent.save(CHECKPOINT_PATH)

    print(f"\n✅ Training complete.  Best val P/L: {best_val_pnl:+.4%}")
    agent.save(CHECKPOINT_PATH)


if __name__ == "__main__":
    main()