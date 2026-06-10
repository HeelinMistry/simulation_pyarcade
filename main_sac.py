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
Sparse realised P/L is used as the reward signal, scaled ×100 for the
critic buffer only (keeps Q in 2–10 range so α can compete meaningfully).
A small hold cost penalises sitting in a position without acting.

  r_buffer = (realised_pnl + bonus − hold_cost) × REWARD_SCALE
  r_display = realised_pnl  (unscaled, shown in console)

Year-boundary safety
────────────────────
Val is evaluated as two separate episodes (2022 and 2025) to prevent
phantom trades that span a year gap in the concatenated val DataFrame.
The saving criterion uses the sum of both years' realised P/L.

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
STATE_DIM  = (len(FEATURES) * 2 * len(PACES)) + 2
ACTION_DIM = 4

# ── Regime-balanced split — must match diagnostic.py exactly ─────────────────
VAL_YEARS   = [2022, 2025]   # bear + recent out-of-sample
BEAR_YEAR   = 2022

# Training hyperparameters
NUM_EPOCHS       = 100
WARMUP_IDX       = 128          # aggregator warm-up rows (= max_pace × max_history)

PATIENCE         = 3            # epochs without improvement before stopping
WARMUP_EPOCHS    = 2
MIN_IMPROVE      = 0.005        # val PnL must improve by 0.5 pp to reset patience

BUFFER_CAPACITY  = 30_000
BATCH_SIZE       = 128
UPDATE_EVERY     = 4
UPDATES_PER_STEP = 4
LR               = 1e-4
GAMMA            = 0.97
TAU              = 0.005

# Logging
LOG_EVERY_TICKS  = 500
SAVE_EVERY_EPOCH = 5

# ── Reward shaping ────────────────────────────────────────────────────────────
# REWARD_SCALE: rewards pushed to buffer are multiplied by this so Q-values
# settle in the 2–10 range where α (≤1.0) can compete meaningfully.
# realised_pnl display is always unscaled.
REWARD_SCALE    = 100.0
MICRO_HOLD_COST = 0.000005   # 0.5 bp/tick while in position (variance signal)
EPSILON_START   = 0.10
EPSILON_END     = 0.01


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def compute_shaped_reward(realised_pnl: float,
                          is_holding: bool,
                          in_position: bool) -> float:
    """
    Returns unscaled reward (fraction units).
    The caller multiplies by REWARD_SCALE before pushing to the buffer.
    """
    shaped = realised_pnl

    # Small bonus for closing a profitable trade — helps critic learn
    # Q(CLOSE|profitable) > Q(HOLD|profitable) faster.
    if realised_pnl > 0.001:
        shaped += realised_pnl * 0.1

    # Tiny per-tick cost while in position — keeps reward variance non-zero
    # for HOLD transitions so the critic can differentiate Q(HOLD|in-pos)
    # from Q(HOLD|flat).  Small enough not to force premature exits.
    if in_position and is_holding:
        shaped -= MICRO_HOLD_COST

    return shaped


def run_epoch(executor: UnifiedExecutor, df: pd.DataFrame,
              replay_buffer: ReplayBuffer, agent: SACAgent,
              train: bool = True, epsilon: float = 0.0) -> dict:
    """
    Single deterministic or stochastic pass through df.

    train=True  : push scaled rewards to buffer, call agent.update().
    train=False : evaluation only — no buffer writes, no gradient steps.

    Returns a metrics dict with keys:
        realised_pnl, n_trades, action_counts, update_count
    """
    indicators_arr = df[FEATURES].values.astype(np.float32)
    prices_arr     = df["Close"].values.astype(np.float32)
    n              = len(df)

    # ── Reset executor state ──────────────────────────────────────────────────
    executor.total_reward  = 0.0
    executor.inventory.clear()
    executor.current_side  = None
    executor._entry_tick   = 0

    # Warm up aggregator
    executor.aggregator.tick = 0
    executor.aggregator.warm_up_all(indicators_arr, WARMUP_IDX)

    total_realised = 0.0
    n_trades       = 0
    action_counts  = [0, 0, 0, 0]
    update_count   = 0

    # One-step lag: we push (prev_state, prev_action, prev_reward, s_t)
    # so the reward correctly belongs to the action that earned it.
    prev_state  = executor.aggregator.get_state(
        executor.portfolio_info(prices_arr[WARMUP_IDX])
    )
    prev_action = None
    prev_reward = 0.0   # unscaled; scaled at push time

    for i in range(WARMUP_IDX + 1, n):
        indicators = indicators_arr[i]
        price      = prices_arr[i]

        action, probs, realised_pnl, s_t = executor.step(
            indicators, price, tick=i, epsilon=epsilon
        )

        action_counts[action] += 1

        if realised_pnl != 0.0:
            total_realised += realised_pnl
            n_trades       += 1

        reward = compute_shaped_reward(
            realised_pnl,
            is_holding=(executor.current_side is not None and realised_pnl == 0.0),
            in_position=(executor.current_side is not None),
        )

        done = (i == n - 1)

        if train and prev_action is not None:
            replay_buffer.push(
                prev_state, prev_action,
                prev_reward * REWARD_SCALE,   # scaled for critic; Q ≈ 2–10
                s_t, done,
            )
            if (i % UPDATE_EVERY == 0) and replay_buffer.is_ready(BATCH_SIZE):
                for _ in range(UPDATES_PER_STEP):
                    agent.update(replay_buffer, batch_size=BATCH_SIZE)
                update_count += 1

        prev_state  = s_t
        prev_action = action
        prev_reward = reward   # carry unscaled; scaled at next push

        # ── Console heartbeat (training only) ────────────────────────────────
        if train and (i % LOG_EVERY_TICKS == 0):
            total_ticks = max(i - WARMUP_IDX, 1)
            pct = ", ".join(f"{c/total_ticks:.0%}" for c in action_counts)
            print(
                f"  tick {i:>6}  |  realised P/L: {total_realised:+.4%}"
                f"  | trades: {n_trades}"
                f"  | actions [L/S/C/H]: {pct}"
                f"  | α={agent.last_alpha:.4f}"
                f"  H={agent.last_entropy/np.log(2):.2f}b"
            )

    return {
        "realised_pnl":  total_realised,
        "n_trades":      n_trades,
        "action_counts": action_counts,
        "update_count":  update_count,
    }


def _combine_metrics(m1: dict, m2: dict) -> dict:
    """Merge two run_epoch dicts by summing all numeric fields."""
    return {
        "realised_pnl":  m1["realised_pnl"]  + m2["realised_pnl"],
        "n_trades":      m1["n_trades"]       + m2["n_trades"],
        "action_counts": [a + b for a, b in
                          zip(m1["action_counts"], m2["action_counts"])],
        "update_count":  m1["update_count"]   + m2["update_count"],
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    df = update_master_data()

    # ── Load and split data ───────────────────────────────────────────────────
    df = df[["Open_time", "Close"] + FEATURES].dropna().reset_index(drop=True)
    df["year"] = pd.to_datetime(df["Open_time"]).dt.year

    val_mask   = df["year"].isin(VAL_YEARS)
    train_mask = ~val_mask

    train_df   = df[train_mask].reset_index(drop=True)
    val_2022_df = df[df["year"] == 2022].reset_index(drop=True)
    val_2025_df = df[df["year"] == 2025].reset_index(drop=True)
    bear_df     = val_2022_df   # 2022 is both val-bear and the bear monitor

    print(f"Train: {len(train_df):,} rows  |  "
          f"Val 2022: {len(val_2022_df):,}  |  Val 2025: {len(val_2025_df):,}")
    print(f"Train years: {sorted(df[train_mask]['year'].unique().tolist())}")
    print(f"Val years:   {VAL_YEARS}")

    # ── Initialise agent and buffer ───────────────────────────────────────────
    agent = SACAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        hidden_dim=128, lr=LR, gamma=GAMMA, tau=TAU,
    )
    agent.load(CHECKPOINT_PATH)

    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

    train_executor = UnifiedExecutor(
        "Train", agent, paces=PACES,
        deterministic=False, num_indicators=len(FEATURES)
    )

    best_val_pnl = -np.inf
    no_improve   = 0
    EPSILON_DECAY = (EPSILON_END / EPSILON_START) ** (1 / 20)
    epsilon       = EPSILON_START

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # ── Training pass ─────────────────────────────────────────────────────
        train_metrics = run_epoch(
            train_executor, train_df, replay_buffer, agent,
            train=True, epsilon=epsilon,
        )
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # ── Val: two separate episodes to avoid year-boundary phantom trades ──
        val_exec_2022 = UnifiedExecutor(
            "Val2022", agent, paces=PACES,
            deterministic=True, num_indicators=len(FEATURES)
        )
        val_exec_2025 = UnifiedExecutor(
            "Val2025", agent, paces=PACES,
            deterministic=True, num_indicators=len(FEATURES)
        )
        m_2022 = run_epoch(val_exec_2022, val_2022_df, replay_buffer,
                           agent, train=False)
        m_2025 = run_epoch(val_exec_2025, val_2025_df, replay_buffer,
                           agent, train=False)
        val_metrics = _combine_metrics(m_2022, m_2025)

        # ── Bear health monitor (same as val 2022, re-use result) ─────────────
        b_pnl       = m_2022["realised_pnl"]
        b_short_pct = (m_2022["action_counts"][1] /
                       max(sum(m_2022["action_counts"]), 1))

        # ── Epoch reporting ───────────────────────────────────────────────────
        elapsed = time.time() - t0
        t_pnl   = train_metrics["realised_pnl"]
        v_pnl   = val_metrics["realised_pnl"]
        t_ac    = train_metrics["action_counts"]
        v_ac    = val_metrics["action_counts"]

        def pct_str(counts):
            total = max(sum(counts), 1)
            return "/".join(f"{c/total:.0%}" for c in counts)

        print(f"\n  ── Epoch {epoch} ─────────────────────────────────────────")
        print(f"  Train P/L : {t_pnl:+.4%}  |  trades={train_metrics['n_trades']}")
        print(f"  Val 2022  : {m_2022['realised_pnl']:+.4%}"
              f"  |  trades={m_2022['n_trades']}")
        print(f"  Val 2025  : {m_2025['realised_pnl']:+.4%}"
              f"  |  trades={m_2025['n_trades']}")
        print(f"  Val total : {v_pnl:+.4%}  |  trades={val_metrics['n_trades']}")
        print(f"  Bear P/L  : {b_pnl:+.4%}  |  SHORT={b_short_pct:.0%}")
        print(f"  Train actions [L/S/C/H]: {pct_str(t_ac)}")
        print(f"  Val   actions [L/S/C/H]: {pct_str(v_ac)}")
        print(f"  SAC   α={agent.last_alpha:.4f}"
              f"  H={agent.last_entropy/np.log(2):.2f}b"
              f"  critic_loss={agent.last_critic_loss:.4f}"
              f"  actor_loss={agent.last_actor_loss:.4f}")
        print(f"  Time  : {elapsed:.1f}s"
              f"  |  updates this epoch: {train_metrics['update_count']}")

        # ── SHORT collapse hard-stop ──────────────────────────────────────────
        short_pct_train = t_ac[1] / max(sum(t_ac), 1)
        if epoch > WARMUP_EPOCHS and short_pct_train < 0.03:
            print(f"  ⛔ SHORT collapsed to {short_pct_train:.1%} — stopping")
            break
        if epoch > WARMUP_EPOCHS and short_pct_train < 0.05:
            print(f"  ⚠ SHORT at {short_pct_train:.1%} in training — watch")

        # ── Directional saving criterion ──────────────────────────────────────
        short_pct_val = v_ac[1] / max(sum(v_ac), 1)
        long_pct_val  = v_ac[0] / max(sum(v_ac), 1)

        bear_pnl_floor = -0.20

        # Both directions must be present in val to save
        is_directional = short_pct_val >= 0.03 and long_pct_val >= 0.03

        if is_directional and v_pnl > best_val_pnl + MIN_IMPROVE and b_pnl > bear_pnl_floor:
            best_val_pnl = v_pnl
            no_improve = 0
            agent.save(BEST_PATH)
            print(f"  ⭐ New best val P/L: {best_val_pnl:+.4%}  bear={b_pnl:+.4%}"
                  f"  (L={long_pct_val:.0%} S={short_pct_val:.0%})")
        elif not is_directional:
            print(f"  ↷ Skipped save — directional collapse"
                  f" (L={long_pct_val:.0%} S={short_pct_val:.0%})")
            no_improve += 1
        elif b_pnl <= bear_pnl_floor:
            print(f"  ↷ Skipped save — bear floor breached (bear={b_pnl:+.4%})")
            no_improve += 1
        else:
            no_improve += 1

        # ── Early stopping ────────────────────────────────────────────────────
        if epoch >= WARMUP_EPOCHS and no_improve >= PATIENCE:
            print(f"  ⚠ No val improvement for {PATIENCE} epochs — early stop")
            break

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if epoch % SAVE_EVERY_EPOCH == 0:
            agent.save(CHECKPOINT_PATH)

    print(f"\n✅ Training complete.  Best val P/L: {best_val_pnl:+.4%}")
    agent.save(CHECKPOINT_PATH)


if __name__ == "__main__":
    main()

# ── Useful one-liners ─────────────────────────────────────────────────────────
# copy outcomes\sac_agent_best.pt outcomes\sac_agent.pt
# del outcomes\sac_agent_buffer.pkl