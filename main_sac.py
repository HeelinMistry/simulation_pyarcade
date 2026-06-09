"""
main_sac.py
───────────
Headless training loop for the SAC-Discrete agent.

Candle timeframe: 4 h  (one tick = 4 hours)

Epoch structure
───────────────
One epoch = one full pass through the processed CSV (train split).
SAC is off-policy so we don't reset the replay buffer between epochs —
transitions from epoch 1 are still valid for critic updates in epoch 50.

Reward design
─────────────
Sparse realised P/L is used as the reward signal.  A small hold cost
penalises sitting in a position without closing it, proportional to
tick frequency (0.5 bp / 4 h tick).

  r_t = realised_pnl  if position closed this tick  (dense signal)
      - MICRO_HOLD_COST  each tick while in position (−0.5 bp / tick)
      + 0               otherwise

Update schedule
───────────────
SAC is updated every UPDATE_EVERY steps once the buffer is ready.
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
PACES      = (1, 6, 42, 90)
STATE_DIM  = (len(FEATURES) * 2 * len(PACES)) + 2   # 50 (6 indicators * 2 features per indicator * 4 paces + 2 portfolio features)
ACTION_DIM = 4

# Training hyperparameters
NUM_EPOCHS       = 100
WARMUP_IDX       = 128          # aggregator warm-up rows (= max_pace × max_history)

PATIENCE         = 2            # epochs without improvement before stopping
WARMUP_EPOCHS    = 2
MIN_IMPROVE      = 0.005        # val PnL must improve by 0.1 pp to reset patience

BUFFER_CAPACITY  = 30_000       # ~7 epochs of 4 h data (9 486 rows × 0.8 ≈ 7 588 / epoch)
BATCH_SIZE       = 128
UPDATE_EVERY     = 4            # update SAC every N environment steps
UPDATES_PER_STEP = 4            # gradient steps per update call
LR               = 3e-4
GAMMA            = 0.97         # at 4 h / tick: 0.97^6 ≈ 83 % weight over 1 day

TAU              = 0.005

# Logging
LOG_EVERY_TICKS  = 500          # ~14 heartbeats per epoch over ~7 000 train ticks
SAVE_EVERY_EPOCH = 5


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

# Remove reward_scaled entirely — keep reward in fraction units
MICRO_HOLD_COST = 0.000005  # 0.5 bp / tick (4 h candle) while in position
EPSILON_START   = 0.10      # 10 % random actions in epoch 1
EPSILON_END     = 0.01      #  1 % random actions by epoch 20

def compute_shaped_reward(realised_pnl, is_holding, in_position):
    shaped = realised_pnl
    if realised_pnl > 0.001:
        shaped += realised_pnl * 0.1
    if in_position and is_holding:
        shaped -= MICRO_HOLD_COST
    return shaped


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

    # Initialize for the first iteration
    # Before loop:
    prev_state = executor.aggregator.get_state(executor.portfolio_info(prices_arr[WARMUP_IDX]))
    prev_action, prev_done = None, False
    prev_reward = 0.0

    for i in range(WARMUP_IDX + 1, n):
        indicators = indicators_arr[i]
        price = prices_arr[i]

        was_in_position = executor.current_side is not None

        action, probs, realised_pnl, s_t = executor.step(
            indicators, price, tick=i, epsilon=epsilon
        )        # s_t = state actor used = market features at tick i + pre-execute portfolio

        action_counts[action] += 1

        if realised_pnl != 0.0:
            total_realised += realised_pnl
            n_trades += 1

        was_flat_before = (not was_in_position and action == 2 and realised_pnl == 0.0)

        reward = compute_shaped_reward(
            realised_pnl,
            is_holding=(executor.current_side is not None and realised_pnl == 0.0),
            in_position=(executor.current_side is not None),
        )

        done = (i == n - 1)

        if train and prev_action is not None:
            replay_buffer.push(prev_state, prev_action, prev_reward, s_t, done)
            if (i % UPDATE_EVERY == 0) and replay_buffer.is_ready(BATCH_SIZE):
                for _ in range(UPDATES_PER_STEP):
                    agent.update(replay_buffer, batch_size=BATCH_SIZE)
                update_count += 1

        prev_state = s_t
        prev_action = action
        prev_reward = reward

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
    df['year'] = pd.to_datetime(df['Open_time']).dt.year

    # Val: 2022 bear + 2025 (out-of-sample recent)
    # Train: everything else
    val_mask = df['year'].isin([2022, 2025])
    train_mask = ~val_mask

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    print(f"Train: {len(train_df):,} rows  |  Val: {len(val_df):,} rows")
    print(f"Train years: {sorted(df[train_mask]['year'].unique().tolist())}")
    print(f"Val years:   {sorted(df[val_mask]['year'].unique().tolist())}")

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
    # Epsilon decays from EPSILON_START → EPSILON_END over 20 epochs.
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

        bear_df = df[df['year'] == 2022].reset_index(drop=True)
        bear_executor = UnifiedExecutor("Bear", agent, paces=PACES,
                                        deterministic=True, num_indicators=len(FEATURES))

        # In the epoch loop, after val_metrics:
        bear_metrics = run_epoch(bear_executor, bear_df, replay_buffer, agent, train=False)
        b_pnl = bear_metrics["realised_pnl"]
        b_short_pct = bear_metrics["action_counts"][1] / max(sum(bear_metrics["action_counts"]), 1)
        print(f"  Bear  P/L : {b_pnl:+.4%}  |  SHORT={b_short_pct:.0%}")

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

        short_pct = t_ac[1] / max(sum(t_ac), 1)
        if epoch > WARMUP_EPOCHS and short_pct < 0.03:
            print(f"  ⛔ SHORT collapsed to {short_pct:.1%} — stopping immediately")
            break

        short_pct_val = v_ac[1] / max(sum(v_ac), 1)
        long_pct_val = v_ac[0] / max(sum(v_ac), 1)

        # Model must use both directions to be saved as "best"
        is_directional = short_pct_val >= 0.05 and long_pct_val >= 0.03

        if is_directional and v_pnl > best_val_pnl + MIN_IMPROVE:
            best_val_pnl = v_pnl
            no_improve = 0
            agent.save(BEST_PATH)
            print(f"  ⭐ New best val P/L: {best_val_pnl:+.4%}  "
                  f"(L={long_pct_val:.0%} S={short_pct_val:.0%})")
        elif not is_directional:
            print(f"  ↷ Skipped save — directional collapse "
                  f"(L={long_pct_val:.0%} S={short_pct_val:.0%})")
            no_improve += 1
        else:
            no_improve += 1

        if epoch >= WARMUP_EPOCHS and no_improve >= PATIENCE:
            print(f"  ⚠ No val improvement for {PATIENCE} epochs — early stop")
            break

        short_pct = t_ac[1] / max(sum(t_ac), 1)
        if epoch > WARMUP_EPOCHS and short_pct < 0.05:
            print(f"  ⚠ SHORT at {short_pct:.1%} in training — watch for collapse")

        # Periodic checkpoint
        if epoch % SAVE_EVERY_EPOCH == 0:
            agent.save(CHECKPOINT_PATH)

    print(f"\n✅ Training complete.  Best val P/L: {best_val_pnl:+.4%}")
    agent.save(CHECKPOINT_PATH)


if __name__ == "__main__":
    main()

# copy outcomes\sac_agent_best.pt outcomes\sac_agent.pt
# del outcomes\sac_agent_buffer.pkl