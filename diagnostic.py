"""
diagnostic.py
─────────────
Full post-training diagnostic suite for the SAC-Discrete trading agent.

Run from project root:
    python diagnostic.py [--split val|train|both] [--checkpoint outcomes/sac_agent_best.pt]

What it measures and why
────────────────────────
The goal is to understand *why* the policy makes each decision per state,
and whether those decisions are the best achievable given the learned Q-values.

  Section 1  Policy Confidence & Entropy
             Is the policy actually committing to actions, or still diffuse?
             A well-trained policy should show bimodal conviction: high certainty
             on clear setups, moderate certainty on ambiguous ones.

  Section 2  Q-Value Health
             Q1 vs Q2 disagreement reveals critic uncertainty. High disagreement
             means the agent is in state-action regions it hasn't seen enough.
             The chosen action should consistently have the highest min(Q1,Q2).

  Section 3  Action Distribution & Sequencing
             Are LONG/SHORT/CLOSE/HOLD balanced sensibly?
             Pathological patterns: all-HOLD (entropy collapse), alternating
             LONG-SHORT every tick (churning), never CLOSE (holding forever).

  Section 4  Trade Outcome Analysis
             Win rate, average PnL per trade, Sharpe ratio, max drawdown.
             Conditioned on conviction level — are high-confidence trades
             actually more profitable? This is the core signal.

  Section 5  Feature → Action Sensitivity
             For each of the 6 raw indicators, how much does moving it from
             its 10th to 90th percentile shift the action probabilities?
             This reveals what the policy has actually learned to look for.

  Section 6  State-Conditional Probability Maps
             Bin RSI and MACD into grid cells, compute mean action probs
             per cell. Visualises the policy's learned market intuition.

  Section 7  Regime Analysis
             Classifies each tick into a regime based on MeanDev (trend) and ATR
             (volatility), then compares action distributions and conviction per regime.

  Section 8  Timing Analysis
             Average conviction and win rate by hour-of-day and day-of-week.
             Finds structural edges (or weaknesses) in specific sessions.

  Section 9  Critic Disagreement vs Outcome
             Ticks where Q1 and Q2 disagree most should correlate with
             lower win rates. Confirms the uncertainty signal is calibrated.

  Section 10 Action-State Consistency Check
             For every LONG action, was RSI/momentum actually bullish?
             For every SHORT, was it bearish? Catches policy inversion.
             Outputs a confusion matrix of action vs market context.

Output
──────
  outcomes/diagnostics/
    01_confidence_entropy.png
    02_qvalue_health.png
    03_action_distribution.png
    04_trade_outcomes.png
    05_feature_sensitivity.png
    06_state_probability_maps.png
    07_regime_analysis.png
    08_timing_analysis.png
    09_uncertainty_vs_outcome.png
    10_action_consistency.png
    diagnostic_summary.txt
"""

import argparse
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from agents.sac_agent      import SACAgent
from agents.unified_executor import UnifiedExecutor
from data.data_manager     import update_master_data

# ── Configuration (must match main_sac.py exactly) ───────────────────────────
BEST_PATH   = "outcomes/sac_agent_best.pt"
FEATURES    = ["RSI_Scaled", "MACD_Scaled", "BB_Scaled",
               "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled"]
PACES     =  (1, 6, 42, 90)
STATE_DIM = (len(FEATURES) * 2 * len(PACES)) + 2
ACTION_DIM  = 4
ACTION_NAMES = ["LONG", "SHORT", "CLOSE", "HOLD"]
ACTION_COLORS = ["#2ecc71", "#e74c3c", "#f39c12", "#95a5a6"]
TRAIN_SPLIT = 0.8
WARMUP_IDX  = 720
OUT_DIR     = "outcomes/diagnostics"
GAMMA = 0.97

os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data collection pass
# ─────────────────────────────────────────────────────────────────────────────

def collect_episode(agent: SACAgent, df: pd.DataFrame, label: str) -> dict:
    """
    Full deterministic pass through df. Collects everything needed for
    all diagnostic sections. Returns a dict of parallel arrays (one entry
    per tick from WARMUP_IDX+1 onward).
    """
    executor = UnifiedExecutor(
        name=label, agent=agent, paces=PACES,
        deterministic=True, num_indicators=len(FEATURES)
    )

    indicators_arr = df[FEATURES].values.astype(np.float32)
    prices_arr     = df["Close"].values.astype(np.float32)
    n              = len(df)

    executor.aggregator.tick = 0
    executor.aggregator.warm_up_all(indicators_arr, WARMUP_IDX)

    # Per-tick storage
    ticks, prices, actions, entropies         = [], [], [], []
    prob_long, prob_short, prob_close, prob_hold = [], [], [], []
    q1_chosen, q2_chosen                      = [], []
    q1_all, q2_all                            = [], []   # shape (n_ticks, 4)
    q_disagreement                            = []
    raw_features                              = []       # 6 raw indicators
    positions, unrealized_pnl                = [], []
    states_arr                               = []

    # Trade tracking
    trades = []   # list of dicts
    open_tick = None
    open_price = None
    open_side = None

    device = agent.device

    for i in range(WARMUP_IDX + 1, n):
        ind   = indicators_arr[i]
        price = prices_arr[i]

        action, probs, realised_pnl, s_t = executor.step(ind, price, tick=i)

        # ── Q-values via critic ───────────────────────────────────────────
        with torch.no_grad():
            s_tensor = torch.FloatTensor(s_t).unsqueeze(0).to(device)
            q1v, q2v = agent.critic(s_tensor)           # (1, 4) each
            q1v = q1v.squeeze(0).cpu().numpy()
            q2v = q2v.squeeze(0).cpu().numpy()

        min_q = np.minimum(q1v, q2v)

        # ── Entropy (bits) ────────────────────────────────────────────────
        ent = -np.sum(probs * np.log2(probs + 1e-9))

        # ── Record tick ───────────────────────────────────────────────────
        ticks.append(i)
        prices.append(price)
        actions.append(action)
        entropies.append(ent)
        prob_long.append(probs[0])
        prob_short.append(probs[1])
        prob_close.append(probs[2])
        prob_hold.append(probs[3])
        q1_chosen.append(float(q1v[action]))
        q2_chosen.append(float(q2v[action]))
        q1_all.append(q1v.copy())
        q2_all.append(q2v.copy())
        q_disagreement.append(float(np.abs(q1v - q2v).mean()))
        raw_features.append(ind.copy())
        positions.append(executor.current_side)
        unrealized_pnl.append(float(s_t[-1]))
        states_arr.append(s_t.copy())

        # ── Trade tracking ────────────────────────────────────────────────
        if realised_pnl != 0.0:
            if open_tick is not None:
                duration = i - open_tick
                # conviction at entry (prob of directional action taken)
                entry_conviction = float(
                    prob_long[open_tick - WARMUP_IDX - 1]
                    if open_side == "LONG"
                    else prob_short[open_tick - WARMUP_IDX - 1]
                )
                trades.append({
                    "entry_tick":       open_tick,
                    "exit_tick":        i,
                    "duration":         duration,
                    "side":             open_side,
                    "pnl":              realised_pnl,
                    "entry_price":      open_price,
                    "exit_price":       price,
                    "entry_conviction": entry_conviction,
                    "exit_conviction":  float(probs[action]),
                    "q_disagree_entry": q_disagreement[open_tick - WARMUP_IDX - 1]
                                        if (open_tick - WARMUP_IDX - 1) >= 0
                                        else 0.0,
                    "win":              realised_pnl > 0,
                    "rsi_at_entry":     float(raw_features[open_tick - WARMUP_IDX - 1][0])
                                        if (open_tick - WARMUP_IDX - 1) >= 0
                                        else 0.0,
                    "macd_at_entry":    float(raw_features[open_tick - WARMUP_IDX - 1][1])
                                        if (open_tick - WARMUP_IDX - 1) >= 0
                                        else 0.0,
                })
            open_tick = None
            open_price = None
            open_side = None

        # Track new position openings
        if action in (0, 1) and executor.current_side is not None:
            if open_tick is None:
                open_tick  = i
                open_price = price
                open_side  = executor.current_side

    # Convert to arrays
    ticks          = np.array(ticks,          dtype=np.int32)
    prices         = np.array(prices,         dtype=np.float32)
    actions        = np.array(actions,        dtype=np.int32)
    entropies      = np.array(entropies,      dtype=np.float32)
    prob_long      = np.array(prob_long,      dtype=np.float32)
    prob_short     = np.array(prob_short,     dtype=np.float32)
    prob_close     = np.array(prob_close,     dtype=np.float32)
    prob_hold      = np.array(prob_hold,      dtype=np.float32)
    q1_chosen      = np.array(q1_chosen,      dtype=np.float32)
    q2_chosen      = np.array(q2_chosen,      dtype=np.float32)
    q1_all         = np.array(q1_all,         dtype=np.float32)   # (N, 4)
    q2_all         = np.array(q2_all,         dtype=np.float32)   # (N, 4)
    q_disagreement = np.array(q_disagreement, dtype=np.float32)
    raw_features   = np.array(raw_features,   dtype=np.float32)   # (N, 6)
    unrealized_pnl = np.array(unrealized_pnl, dtype=np.float32)
    states_arr     = np.array(states_arr,     dtype=np.float32)   # (N, 92)

    # Max probability (conviction) at every tick
    probs_all = np.stack([prob_long, prob_short, prob_close, prob_hold], axis=1)
    max_prob  = probs_all.max(axis=1)

    # Cumulative PnL
    pnl_curve = np.zeros(len(ticks))
    for t in trades:
        pnl_curve[t["exit_tick"] - WARMUP_IDX - 1:] += t["pnl"]

    # Parse timestamps if present
    timestamps = None
    if "Open_time" in df.columns:
        try:
            ts = pd.to_datetime(df["Open_time"].iloc[WARMUP_IDX + 1:], errors='coerce')
            timestamps = ts.tolist()  # or keep as series depending on downstream requirements

        except Exception as e:
            print(f"⚠ Warning: Timestamp parsing failed during diagnostics collection: {e}")
            timestamps = None

    return {
        "label":          label,
        "ticks":          ticks,
        "prices":         prices,
        "actions":        actions,
        "entropies":      entropies,
        "probs_all":      probs_all,
        "max_prob":       max_prob,
        "q1_chosen":      q1_chosen,
        "q2_chosen":      q2_chosen,
        "q1_all":         q1_all,
        "q2_all":         q2_all,
        "q_disagreement": q_disagreement,
        "raw_features":   raw_features,
        "unrealized_pnl": unrealized_pnl,
        "states_arr":     states_arr,
        "trades":         trades,
        "pnl_curve":      pnl_curve,
        "timestamps":     timestamps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    "axes.facecolor":    "#1a1a2e",
    "figure.facecolor":  "#0f0f1a",
    "axes.edgecolor":    "#444466",
    "axes.labelcolor":   "#ccccee",
    "xtick.color":       "#aaaacc",
    "ytick.color":       "#aaaacc",
    "text.color":        "#ddddff",
    "grid.color":        "#2a2a4a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
}


def make_fig(rows, cols, title, figsize=None):
    with plt.rc_context(STYLE):
        fs = figsize or (cols * 5, rows * 4)
        fig, axes = plt.subplots(rows, cols, figsize=fs, squeeze=False)
        fig.suptitle(title, color="#ffffff", fontsize=14, fontweight="bold", y=0.98)
        fig.patch.set_facecolor(STYLE["figure.facecolor"])
    return fig, axes


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=STYLE["figure.facecolor"])
    plt.close(fig)
    print(f"  ✓  {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Policy Confidence & Entropy
# ─────────────────────────────────────────────────────────────────────────────

def plot_confidence_entropy(ep: dict):
    fig, axes = make_fig(2, 3, "Section 1 — Policy Confidence & Entropy")
    with plt.rc_context(STYLE):
        ax = axes[0][0]
        ax.hist(ep["max_prob"], bins=50, color="#7f5af0", edgecolor="none", alpha=0.85)
        ax.axvline(0.5, color="#ff6b6b", lw=1.5, linestyle="--", label="50% conviction")
        ax.axvline(np.mean(ep["max_prob"]), color="#ffd700", lw=1.5,
                   label=f"mean={np.mean(ep['max_prob']):.2f}")
        ax.set_title("Max-Prob (Conviction) Distribution")
        ax.set_xlabel("max π(a|s)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(True)

        ax = axes[0][1]
        ax.hist(ep["entropies"], bins=50, color="#2cb67d", edgecolor="none", alpha=0.85)
        target_entropy_bits = 0.98 * np.log2(4)
        ax.axvline(target_entropy_bits, color="#ff6b6b", lw=1.5, linestyle="--",
                   label=f"target={target_entropy_bits:.2f}b")
        ax.axvline(np.mean(ep["entropies"]), color="#ffd700", lw=1.5,
                   label=f"mean={np.mean(ep['entropies']):.2f}b")
        ax.set_title("Policy Entropy Distribution (bits)")
        ax.set_xlabel("H[π(·|s)] bits")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(True)

        # Conviction over time (rolling mean)
        ax = axes[0][2]
        window = min(500, len(ep["max_prob"]) // 10)
        rolling = pd.Series(ep["max_prob"]).rolling(window).mean().values
        ax.plot(ep["ticks"], rolling, color="#7f5af0", lw=1.0)
        ax.axhline(0.5, color="#ff6b6b", lw=1, linestyle="--")
        ax.set_title(f"Conviction Over Time (rolling {window})")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Mean max-prob")
        ax.grid(True)

        # Per-action probability distributions
        for j, (name, col) in enumerate(zip(ACTION_NAMES, ACTION_COLORS)):
            ax = axes[1][j] if j < 3 else axes[1][2]
            if j < 3:
                ax.hist(ep["probs_all"][:, j], bins=40, color=col,
                        edgecolor="none", alpha=0.75, label=name)
                ax.set_title(f"π({name}|s) distribution")
                ax.set_xlabel("Probability")
                ax.set_ylabel("Frequency")
                ax.grid(True)

        # HOLD in last panel alongside close
        ax = axes[1][2]
        ax.hist(ep["probs_all"][:, 3], bins=40, color=ACTION_COLORS[3],
                edgecolor="none", alpha=0.55, label="HOLD")
        ax.legend(fontsize=8)

    savefig(fig, "01_confidence_entropy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Q-Value Health
# ─────────────────────────────────────────────────────────────────────────────

def plot_qvalue_health(ep: dict):
    fig, axes = make_fig(2, 3, "Section 2 — Q-Value Health")
    with plt.rc_context(STYLE):
        # Q1 vs Q2 for chosen action — should be tightly correlated
        ax = axes[0][0]
        lim_lo = min(ep["q1_chosen"].min(), ep["q2_chosen"].min())
        lim_hi = max(ep["q1_chosen"].max(), ep["q2_chosen"].max())
        ax.scatter(ep["q1_chosen"], ep["q2_chosen"], s=1, alpha=0.3, color="#7f5af0")
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", lw=1, label="Q1=Q2")
        corr = np.corrcoef(ep["q1_chosen"], ep["q2_chosen"])[0, 1]
        ax.set_title(f"Q1 vs Q2 (chosen action)  r={corr:.3f}")
        ax.set_xlabel("Q1")
        ax.set_ylabel("Q2")
        ax.legend(fontsize=8)
        ax.grid(True)

        # Q1-Q2 disagreement distribution — low is healthy
        ax = axes[0][1]
        ax.hist(ep["q_disagreement"], bins=50, color="#f25f4c", edgecolor="none", alpha=0.8)
        ax.axvline(np.mean(ep["q_disagreement"]), color="#ffd700", lw=1.5,
                   label=f"mean={np.mean(ep['q_disagreement']):.4f}")
        ax.set_title("Mean |Q1-Q2| per Tick (Critic Uncertainty)")
        ax.set_xlabel("|Q1-Q2| mean over actions")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(True)

        # Q-value distribution per action (min_Q)
        ax = axes[0][2]
        min_q_all = np.minimum(ep["q1_all"], ep["q2_all"])   # (N, 4)
        for j, (name, col) in enumerate(zip(ACTION_NAMES, ACTION_COLORS)):
            ax.hist(min_q_all[:, j], bins=40, color=col, alpha=0.55,
                    label=name, edgecolor="none")
        ax.set_title("min(Q1,Q2) Distribution per Action")
        ax.set_xlabel("Q-value")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(True)

        # Does the policy's chosen action have the highest min-Q?
        ax = axes[1][0]
        best_q_action = min_q_all.argmax(axis=1)
        policy_action = ep["actions"]
        match_rate = (best_q_action == policy_action).mean()
        confusion = np.zeros((4, 4), dtype=np.int32)
        for pa, qa in zip(policy_action, best_q_action):
            confusion[pa, qa] += 1
        im = ax.imshow(confusion, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(ACTION_NAMES, fontsize=7)
        ax.set_yticklabels(ACTION_NAMES, fontsize=7)
        ax.set_xlabel("Best-Q Action")
        ax.set_ylabel("Policy Action")
        ax.set_title(f"Policy vs Best-Q Action  (match={match_rate:.1%})")
        for r in range(4):
            for c in range(4):
                ax.text(c, r, str(confusion[r, c]), ha="center",
                        va="center", fontsize=7, color="white")

        # Q-value of chosen action over time (rolling mean)
        ax = axes[1][1]
        window = min(500, len(ep["q1_chosen"]) // 10)
        chosen_minq = np.minimum(ep["q1_chosen"], ep["q2_chosen"])
        rolling = pd.Series(chosen_minq).rolling(window).mean().values
        ax.plot(ep["ticks"], rolling, color="#2cb67d", lw=1.0)
        ax.axhline(0, color="#aaaacc", lw=0.8, linestyle="--")
        ax.set_title(f"min(Q1,Q2) of Chosen Action (rolling {window})")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Q-value")
        ax.grid(True)

        # Q-value vs actual outcome for trades
        ax = axes[1][2]
        if ep["trades"]:
            entry_q = []
            outcomes = []
            for t in ep["trades"]:
                idx = t["entry_tick"] - WARMUP_IDX - 1
                if 0 <= idx < len(ep["q1_all"]):
                    a = 0 if t["side"] == "LONG" else 1
                    q = float(min(ep["q1_all"][idx, a], ep["q2_all"][idx, a]))
                    entry_q.append(q)
                    outcomes.append(t["pnl"])
            if entry_q:
                wins   = [q for q, p in zip(entry_q, outcomes) if p > 0]
                losses = [q for q, p in zip(entry_q, outcomes) if p <= 0]
                ax.hist(wins,   bins=20, color="#2ecc71", alpha=0.7, label="Win",  edgecolor="none")
                ax.hist(losses, bins=20, color="#e74c3c", alpha=0.7, label="Loss", edgecolor="none")
                ax.set_title("Entry Q-value: Wins vs Losses")
                ax.set_xlabel("min(Q1,Q2) at entry")
                ax.set_ylabel("Trade count")
                ax.legend(fontsize=8)
                ax.grid(True)

    savefig(fig, "02_qvalue_health.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Action Distribution & Sequencing
# ─────────────────────────────────────────────────────────────────────────────

def plot_action_distribution(ep: dict):
    fig, axes = make_fig(2, 3, "Section 3 — Action Distribution & Sequencing")
    with plt.rc_context(STYLE):
        # Pie chart of action counts
        ax = axes[0][0]
        counts = [(ep["actions"] == i).sum() for i in range(4)]
        wedges, texts, autotexts = ax.pie(
            counts, labels=ACTION_NAMES, colors=ACTION_COLORS,
            autopct="%1.1f%%", startangle=90,
            textprops={"color": "#ddddff", "fontsize": 9}
        )
        ax.set_title("Action Distribution")

        # Action over time (rolling window shows policy evolution)
        ax = axes[0][1]
        window = min(500, len(ep["actions"]) // 10)
        for j, (name, col) in enumerate(zip(ACTION_NAMES, ACTION_COLORS)):
            mask = (ep["actions"] == j).astype(float)
            rolling = pd.Series(mask).rolling(window).mean().values
            ax.plot(ep["ticks"], rolling, color=col, lw=1.0, alpha=0.8, label=name)
        ax.set_title(f"Action Frequency Over Time (rolling {window})")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Fraction of ticks")
        ax.legend(fontsize=7)
        ax.grid(True)

        # Action transition matrix — what action follows each action
        ax = axes[0][2]
        trans = np.zeros((4, 4), dtype=np.float32)
        for curr, nxt in zip(ep["actions"][:-1], ep["actions"][1:]):
            trans[curr, nxt] += 1
        row_sums = trans.sum(axis=1, keepdims=True) + 1e-9
        trans_pct = trans / row_sums
        im = ax.imshow(trans_pct, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([f"→{n}" for n in ACTION_NAMES], fontsize=7)
        ax.set_yticklabels(ACTION_NAMES, fontsize=7)
        ax.set_title("Action Transition Matrix")
        for r in range(4):
            for c in range(4):
                ax.text(c, r, f"{trans_pct[r,c]:.2f}", ha="center",
                        va="center", fontsize=7,
                        color="black" if trans_pct[r, c] > 0.5 else "white")
        plt.colorbar(im, ax=ax)

        # Run-length distribution — how many consecutive HOLDs?
        ax = axes[1][0]
        runs = []
        cur_run = 1
        for i in range(1, len(ep["actions"])):
            if ep["actions"][i] == 3 and ep["actions"][i-1] == 3:
                cur_run += 1
            else:
                if ep["actions"][i-1] == 3:
                    runs.append(cur_run)
                cur_run = 1
        if runs:
            ax.hist(runs, bins=min(50, max(runs)), color="#95a5a6",
                    edgecolor="none", alpha=0.8)
            ax.set_title(f"Consecutive HOLD Run Lengths  (median={np.median(runs):.0f})")
            ax.set_xlabel("Run length (ticks)")
            ax.set_ylabel("Count")
            ax.set_xlim(0, np.percentile(runs, 95))
            ax.grid(True)

        # Trade positions on price chart (last 2000 ticks for readability)
        ax = axes[1][1]
        N = min(2000, len(ep["ticks"]))
        slice_t = ep["ticks"][-N:]
        slice_p = ep["prices"][-N:]
        slice_a = ep["actions"][-N:]
        ax.plot(slice_t, slice_p, color="#aaaacc", lw=0.8, alpha=0.7)
        for action_idx, col, marker in zip([0,1,2], ["#2ecc71","#e74c3c","#f39c12"],
                                           ["^","v","x"]):
            mask = slice_a == action_idx
            ax.scatter(slice_t[mask], slice_p[mask], color=col,
                       s=15, marker=marker, zorder=3,
                       label=ACTION_NAMES[action_idx], alpha=0.8)
        ax.set_title(f"Signals on Price (last {N} ticks)")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Price")
        ax.legend(fontsize=7)
        ax.grid(True)

        # Conviction at each action type
        ax = axes[1][2]
        for j, (name, col) in enumerate(zip(ACTION_NAMES, ACTION_COLORS)):
            mask = ep["actions"] == j
            if mask.sum() > 0:
                ax.hist(ep["max_prob"][mask], bins=30, color=col, alpha=0.6,
                        label=f"{name} (n={mask.sum()})", edgecolor="none")
        ax.set_title("Conviction (max-prob) by Action Type")
        ax.set_xlabel("max π(a|s)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=7)
        ax.grid(True)

    savefig(fig, "03_action_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Trade Outcome Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_trade_outcomes(ep: dict):
    trades = ep["trades"]
    if not trades:
        print("  ⚠  No trades completed — skipping Section 4.")
        return None

    fig, axes = make_fig(2, 3, "Section 4 — Trade Outcome Analysis")
    pnls       = np.array([t["pnl"] for t in trades])
    wins       = pnls > 0
    durations  = np.array([t["duration"] for t in trades])
    conviction = np.array([t["entry_conviction"] for t in trades])

    with plt.rc_context(STYLE):
        # PnL distribution
        ax = axes[0][0]
        ax.hist(pnls[wins],  bins=30, color="#2ecc71", alpha=0.8, label="Win",  edgecolor="none")
        ax.hist(pnls[~wins], bins=30, color="#e74c3c", alpha=0.8, label="Loss", edgecolor="none")
        ax.axvline(0, color="white", lw=1, linestyle="--")
        ax.axvline(pnls.mean(), color="#ffd700", lw=1.5,
                   label=f"mean={pnls.mean():.4%}")
        win_rate = wins.mean()
        ax.set_title(f"Trade PnL Distribution  (WR={win_rate:.1%}, n={len(trades)})")
        ax.set_xlabel("PnL (fraction)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(True)

        # Cumulative PnL curve
        ax = axes[0][1]
        cum = np.cumsum(pnls)
        ax.plot(cum, color="#7f5af0", lw=1.5)
        ax.fill_between(range(len(cum)), 0, cum,
                        where=cum >= 0, color="#2ecc71", alpha=0.2)
        ax.fill_between(range(len(cum)), 0, cum,
                        where=cum < 0, color="#e74c3c", alpha=0.2)
        ax.axhline(0, color="#aaaacc", lw=0.8, linestyle="--")
        ax.set_title(f"Cumulative PnL  (total={cum[-1]:.4%})")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative PnL")
        ax.grid(True)

        # Holding duration histogram
        ax = axes[0][2]
        ax.hist(durations, bins=40, color="#f39c12", edgecolor="none", alpha=0.8)
        ax.axvline(np.median(durations), color="#ffd700", lw=1.5,
                   label=f"median={np.median(durations):.0f} ticks")
        ax.set_title("Holding Duration Distribution")
        ax.set_xlabel("Duration (ticks) — 1 tick = 4 h")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True)

        # Conviction vs PnL scatter
        ax = axes[1][0]
        ax.scatter(conviction[wins],  pnls[wins],  s=8, color="#2ecc71",
                   alpha=0.5, label="Win")
        ax.scatter(conviction[~wins], pnls[~wins], s=8, color="#e74c3c",
                   alpha=0.5, label="Loss")
        ax.axhline(0, color="white", lw=0.8, linestyle="--")
        # Add conviction quintile win rates
        for q_lo, q_hi in [(0, 0.2),(0.2, 0.4),(0.4, 0.6),(0.6, 0.8),(0.8, 1.0)]:
            mask = (conviction >= q_lo) & (conviction < q_hi)
            if mask.sum() > 0:
                wr = wins[mask].mean()
                ax.text((q_lo + q_hi) / 2, pnls.min() * 0.9,
                        f"{wr:.0%}", ha="center", fontsize=7, color="#ffd700")
        ax.set_title("Entry Conviction vs PnL (Win rate by quintile in yellow)")
        ax.set_xlabel("Entry conviction (prob of chosen direction)")
        ax.set_ylabel("PnL")
        ax.legend(fontsize=7)
        ax.grid(True)

        # Duration vs PnL
        ax = axes[1][1]
        ax.scatter(durations[wins],  pnls[wins],  s=8, color="#2ecc71", alpha=0.5)
        ax.scatter(durations[~wins], pnls[~wins], s=8, color="#e74c3c", alpha=0.5)
        ax.axhline(0, color="white", lw=0.8, linestyle="--")
        ax.set_title("Holding Duration vs PnL")
        ax.set_xlabel("Duration (ticks)")
        ax.set_ylabel("PnL")
        ax.set_xlim(0, np.percentile(durations, 95))
        ax.grid(True)

        # LONG vs SHORT performance
        ax = axes[1][2]
        for side, col, label in [("LONG","#2ecc71","Long"), ("SHORT","#e74c3c","Short")]:
            side_pnls = [t["pnl"] for t in trades if t["side"] == side]
            if side_pnls:
                side_pnls = np.array(side_pnls)
                wr = (side_pnls > 0).mean()
                ax.hist(side_pnls, bins=25, color=col, alpha=0.7, edgecolor="none",
                        label=f"{label}: WR={wr:.1%} n={len(side_pnls)}")
        ax.axvline(0, color="white", lw=0.8, linestyle="--")
        ax.set_title("PnL by Trade Side")
        ax.set_xlabel("PnL")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(True)

    savefig(fig, "04_trade_outcomes.png")
    return pnls


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Feature → Action Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_sensitivity(agent: SACAgent, ep: dict):
    """
    Measures how much each raw indicator shifts action probabilities.
    Uses the p10→p90 perturbation: for each feature, clamp all states
    to the 10th and 90th percentile of that feature, re-run the actor,
    and compute the mean probability shift for each action.
    This reveals what the policy has actually learned to respond to.
    """
    states = torch.FloatTensor(ep["states_arr"]).to(agent.device)
    device = agent.device

    # Baseline probabilities
    with torch.no_grad():
        base_probs = agent.actor(states).cpu().numpy()   # (N, 4)

    # Feature indices in the 92-d state:
    # The state is [pace1_cur(6), pace1_slope(6), pace1_std(6),
    #               pace2_cur(6), ..., pace12_std(6),  position, unrealized_pnl]
    # Raw indicator i appears at positions: i, i+6, i+12 for pace1,
    # and i+18k, i+18k+6, i+18k+12 for pace k.
    # But the simplest sensitivity is to perturb the raw indicator at pace1
    # (indices 0-5 = current values of pace-1 agent) which updates every tick.

    fig, axes = make_fig(2, 3, "Section 5 — Feature → Action Sensitivity (p10→p90)")
    with plt.rc_context(STYLE):
        for feat_idx, feat_name in enumerate(FEATURES):
            ax = axes[feat_idx // 3][feat_idx % 3]

            # Perturb: set feature to p10 and p90 for the pace-1 current value
            state_col_idx = feat_idx   # pace-1 current block starts at 0

            feat_vals  = ep["states_arr"][:, state_col_idx]
            p10        = np.percentile(feat_vals, 10)
            p90        = np.percentile(feat_vals, 90)

            states_lo  = states.clone()
            states_hi  = states.clone()
            states_lo[:, state_col_idx] = torch.tensor(p10, dtype=torch.float32, device=device)
            states_hi[:, state_col_idx] = torch.tensor(p90, dtype=torch.float32, device=device)

            with torch.no_grad():
                probs_lo = agent.actor(states_lo).cpu().numpy()
                probs_hi = agent.actor(states_hi).cpu().numpy()

            delta = probs_hi - probs_lo   # (N, 4) — mean shift from p10→p90

            means  = delta.mean(axis=0)
            stds   = delta.std(axis=0)
            x      = np.arange(4)

            bars = ax.bar(x, means, color=ACTION_COLORS, alpha=0.85, width=0.6)
            ax.errorbar(x, means, yerr=stds, fmt="none", color="white",
                        capsize=4, lw=1.5)
            ax.axhline(0, color="#aaaacc", lw=0.8, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels(ACTION_NAMES, fontsize=8)
            ax.set_title(f"{feat_name}\n(p10={p10:.2f} → p90={p90:.2f})")
            ax.set_ylabel("Δ probability")
            ax.grid(True, axis="y")

    savefig(fig, "05_feature_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — State-Conditional Probability Maps
# ─────────────────────────────────────────────────────────────────────────────

def plot_state_probability_maps(ep: dict):
    """
    Bins RSI × MACD into a 10×10 grid and shows mean action probabilities
    per cell. Reveals the learned policy surface on the two most
    interpretable momentum indicators.
    """
    rsi  = ep["raw_features"][:, 0]   # RSI_Scaled ∈ [-1, 1]
    macd = ep["raw_features"][:, 1]   # MACD_Scaled ∈ [-1, 1]
    probs = ep["probs_all"]            # (N, 4)

    bins = 10
    rsi_edges  = np.linspace(rsi.min(),  rsi.max(),  bins + 1)
    macd_edges = np.linspace(macd.min(), macd.max(), bins + 1)

    fig, axes = make_fig(2, 2, "Section 6 — Policy Surface: RSI × MACD Grid",
                         figsize=(12, 10))
    with plt.rc_context(STYLE):
        for j, (name, col) in enumerate(zip(ACTION_NAMES, ACTION_COLORS)):
            ax = axes[j // 2][j % 2]
            grid = np.full((bins, bins), np.nan)
            for ri in range(bins):
                for mi in range(bins):
                    mask = (
                        (rsi  >= rsi_edges[ri])  & (rsi  < rsi_edges[ri + 1]) &
                        (macd >= macd_edges[mi]) & (macd < macd_edges[mi + 1])
                    )
                    if mask.sum() > 5:
                        grid[ri, mi] = probs[mask, j].mean()

            im = ax.imshow(
                grid, origin="lower", aspect="auto", cmap="plasma",
                vmin=0, vmax=probs[:, j].quantile_if_possible
                if hasattr(probs[:, j], "quantile_if_possible")
                else min(0.9, np.nanpercentile(grid[~np.isnan(grid)], 95))
                   if not np.all(np.isnan(grid)) else 0.5
            )
            # Cleaner vmax
            vmax_val = min(0.9, np.nanpercentile(grid[~np.isnan(grid)], 95)) \
                       if not np.all(np.isnan(grid)) else 0.5
            im.set_clim(0, vmax_val)

            ax.set_title(f"Mean π({name}|s)  — RSI × MACD grid")
            ax.set_xlabel("MACD_Scaled bins")
            ax.set_ylabel("RSI_Scaled bins")
            # Axis tick labels
            rsi_centers  = (rsi_edges[:-1]  + rsi_edges[1:])  / 2
            macd_centers = (macd_edges[:-1] + macd_edges[1:]) / 2
            ax.set_xticks([0, bins//2, bins-1])
            ax.set_xticklabels([f"{macd_centers[0]:.1f}",
                                 f"{macd_centers[bins//2]:.1f}",
                                 f"{macd_centers[-1]:.1f}"], fontsize=7)
            ax.set_yticks([0, bins//2, bins-1])
            ax.set_yticklabels([f"{rsi_centers[0]:.1f}",
                                 f"{rsi_centers[bins//2]:.1f}",
                                 f"{rsi_centers[-1]:.1f}"], fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    savefig(fig, "06_state_probability_maps.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Regime Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_analysis(ep: dict):
    """
    Classifies each tick into a regime based on MeanDev (trend) and ATR
    (volatility), then compares action distributions and conviction per regime.
    """
    mean_dev = ep["raw_features"][:, 5]   # MeanDev_Scaled
    atr      = ep["raw_features"][:, 4]   # ATR_Scaled

    # Regime labels
    trending_up   = (mean_dev >  0.1) & (atr > 0)
    trending_down = (mean_dev < -0.1) & (atr > 0)
    ranging       = (mean_dev >= -0.1) & (mean_dev <= 0.1)
    high_vol      = atr > np.percentile(atr, 75)
    low_vol       = atr < np.percentile(atr, 25)

    regimes = {
        "Trending Up":   trending_up,
        "Trending Down": trending_down,
        "Ranging":       ranging,
        "High Vol":      high_vol,
        "Low Vol":       low_vol,
    }
    regime_colors = ["#2ecc71","#e74c3c","#95a5a6","#f39c12","#3498db"]

    fig, axes = make_fig(2, 3, "Section 7 — Regime Analysis")
    with plt.rc_context(STYLE):
        # Action distribution per regime
        ax = axes[0][0]
        x = np.arange(4)
        width = 0.15
        for k, (rname, mask) in enumerate(regimes.items()):
            if mask.sum() < 10:
                continue
            counts = np.array([(ep["actions"][mask] == i).mean() for i in range(4)])
            ax.bar(x + k * width, counts, width=width,
                   color=regime_colors[k], alpha=0.8, label=rname)
        ax.set_xticks(x + 2 * width)
        ax.set_xticklabels(ACTION_NAMES, fontsize=8)
        ax.set_title("Action Distribution by Regime")
        ax.set_ylabel("Fraction of ticks")
        ax.legend(fontsize=6)
        ax.grid(True, axis="y")

        # Conviction per regime
        ax = axes[0][1]
        conv_data  = []
        conv_labels = []
        for rname, mask in regimes.items():
            if mask.sum() > 10:
                conv_data.append(ep["max_prob"][mask])
                conv_labels.append(f"{rname}\n(n={mask.sum():,})")
        bp = ax.boxplot(conv_data, patch_artist=True,
                        medianprops={"color": "white", "lw": 2})
        for patch, col in zip(bp["boxes"], regime_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(conv_labels) + 1))
        ax.set_xticklabels(conv_labels, fontsize=6)
        ax.set_title("Conviction by Regime")
        ax.set_ylabel("max π(a|s)")
        ax.grid(True, axis="y")

        # Entropy per regime
        ax = axes[0][2]
        ent_data = []
        for rname, mask in regimes.items():
            if mask.sum() > 10:
                ent_data.append(ep["entropies"][mask])
        bp2 = ax.boxplot(ent_data, patch_artist=True,
                         medianprops={"color": "white", "lw": 2})
        for patch, col in zip(bp2["boxes"], regime_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(conv_labels) + 1))
        ax.set_xticklabels(conv_labels, fontsize=6)
        ax.set_title("Policy Entropy by Regime")
        ax.set_ylabel("H[π] bits")
        ax.grid(True, axis="y")

        # Trade win rate per regime
        ax = axes[1][0]
        if ep["trades"]:
            wr_vals  = []
            wr_names = []
            for rname, mask in regimes.items():
                regime_ticks = set(ep["ticks"][mask].tolist())
                regime_trades = [t for t in ep["trades"]
                                 if t["entry_tick"] in regime_ticks]
                if len(regime_trades) >= 3:
                    wr = np.mean([t["win"] for t in regime_trades])
                    wr_vals.append(wr)
                    wr_names.append(f"{rname}\n(n={len(regime_trades)})")
            if wr_vals:
                bars = ax.bar(range(len(wr_vals)), wr_vals,
                              color=[regime_colors[i] for i in range(len(wr_vals))],
                              alpha=0.8)
                ax.axhline(0.5, color="white", lw=1, linestyle="--")
                ax.set_xticks(range(len(wr_names)))
                ax.set_xticklabels(wr_names, fontsize=6)
                ax.set_title("Trade Win Rate by Entry Regime")
                ax.set_ylabel("Win rate")
                ax.set_ylim(0, 1)
                ax.grid(True, axis="y")

        # MeanDev vs action scatter
        ax = axes[1][1]
        for j, (name, col) in enumerate(zip(ACTION_NAMES[:2], ACTION_COLORS[:2])):
            mask = ep["actions"] == j
            ax.scatter(mean_dev[mask], atr[mask], s=4, color=col,
                       alpha=0.3, label=name)
        ax.set_title("MeanDev vs ATR at LONG/SHORT Ticks")
        ax.set_xlabel("MeanDev_Scaled (trend)")
        ax.set_ylabel("ATR_Scaled (volatility)")
        ax.legend(fontsize=8)
        ax.grid(True)

        # Conviction during HOLD — is HOLD high-confidence?
        ax = axes[1][2]
        hold_mask = ep["actions"] == 3
        nonhold_mask = ep["actions"] != 3
        ax.hist(ep["max_prob"][nonhold_mask], bins=40, color="#7f5af0",
                alpha=0.7, label="Active actions", edgecolor="none", density=True)
        ax.hist(ep["max_prob"][hold_mask], bins=40, color="#95a5a6",
                alpha=0.7, label="HOLD", edgecolor="none", density=True)
        ax.set_title("Conviction: HOLD vs Active Actions")
        ax.set_xlabel("max π(a|s)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True)

    savefig(fig, "07_regime_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Timing Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_timing_analysis(ep: dict):
    if ep["timestamps"] is None:
        print("  ⚠  No timestamps — skipping Section 8.")
        return

    ts = ep["timestamps"]
    hours    = np.array([t.hour     for t in ts])
    weekdays = np.array([t.dayofweek for t in ts])  # 0=Mon, 4=Fri

    fig, axes = make_fig(2, 2, "Section 8 — Timing Analysis")
    with plt.rc_context(STYLE):
        # Conviction by hour
        ax = axes[0][0]
        hour_conv = [ep["max_prob"][hours == h] for h in range(24)]
        means = [v.mean() if len(v) > 0 else 0 for v in hour_conv]
        ax.bar(range(24), means, color="#7f5af0", alpha=0.8)
        ax.axhline(ep["max_prob"].mean(), color="#ffd700", lw=1.5, linestyle="--",
                   label=f"overall mean={ep['max_prob'].mean():.2f}")
        ax.set_title("Mean Conviction by Hour (UTC)")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Mean max-prob")
        ax.set_xticks(range(24))
        ax.legend(fontsize=7)
        ax.grid(True, axis="y")

        # Win rate by hour (for completed trades)
        ax = axes[0][1]
        if ep["trades"]:
            hour_of_entry = {}
            for t in ep["trades"]:
                idx = t["entry_tick"] - WARMUP_IDX - 1
                if 0 <= idx < len(ts):
                    h = ts[idx].hour
                    hour_of_entry.setdefault(h, []).append(t["win"])
            hours_list   = sorted(hour_of_entry.keys())
            wr_by_hour   = [np.mean(hour_of_entry[h]) for h in hours_list]
            trade_counts = [len(hour_of_entry[h]) for h in hours_list]
            bars = ax.bar(hours_list, wr_by_hour,
                          color=["#2ecc71" if w >= 0.5 else "#e74c3c"
                                 for w in wr_by_hour], alpha=0.8)
            ax.axhline(0.5, color="white", lw=1, linestyle="--")
            for h, wr, cnt in zip(hours_list, wr_by_hour, trade_counts):
                ax.text(h, wr + 0.02, str(cnt), ha="center",
                        fontsize=6, color="#aaaacc")
            ax.set_title("Trade Win Rate by Hour of Day")
            ax.set_xlabel("Hour (UTC)")
            ax.set_ylabel("Win rate")
            ax.set_ylim(0, 1.1)
            ax.grid(True, axis="y")

        # Conviction by weekday
        ax = axes[1][0]
        day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        day_conv  = [ep["max_prob"][weekdays == d] for d in range(7)]
        means_d   = [v.mean() if len(v) > 0 else 0 for v in day_conv]
        ax.bar(range(7), means_d, color="#2cb67d", alpha=0.8)
        ax.axhline(ep["max_prob"].mean(), color="#ffd700", lw=1.5, linestyle="--")
        ax.set_xticks(range(7))
        ax.set_xticklabels(day_names)
        ax.set_title("Mean Conviction by Day of Week")
        ax.set_ylabel("Mean max-prob")
        ax.grid(True, axis="y")

        # LONG vs SHORT frequency by hour
        ax = axes[1][1]
        long_by_hour  = [(ep["actions"][hours == h] == 0).sum() for h in range(24)]
        short_by_hour = [(ep["actions"][hours == h] == 1).sum() for h in range(24)]
        x = np.arange(24)
        ax.bar(x - 0.2, long_by_hour,  0.4, color="#2ecc71", alpha=0.8, label="LONG")
        ax.bar(x + 0.2, short_by_hour, 0.4, color="#e74c3c", alpha=0.8, label="SHORT")
        ax.set_title("LONG vs SHORT Trade Openings by Hour")
        ax.set_xlabel("Hour (UTC)")
        ax.set_ylabel("Count")
        ax.set_xticks(range(24))
        ax.legend(fontsize=8)
        ax.grid(True, axis="y")

    savefig(fig, "08_timing_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Critic Uncertainty vs Trade Outcome
# ─────────────────────────────────────────────────────────────────────────────

def plot_uncertainty_vs_outcome(ep: dict):
    trades = ep["trades"]
    if not trades:
        print("  ⚠  No trades — skipping Section 9.")
        return

    fig, axes = make_fig(2, 2, "Section 9 — Critic Uncertainty vs Outcome")
    with plt.rc_context(STYLE):
        # Q-disagreement at entry vs trade PnL
        ax = axes[0][0]
        entry_disagree = []
        pnls = []
        for t in trades:
            idx = t["entry_tick"] - WARMUP_IDX - 1
            if 0 <= idx < len(ep["q_disagreement"]):
                entry_disagree.append(ep["q_disagreement"][idx])
                pnls.append(t["pnl"])
        if entry_disagree:
            ed = np.array(entry_disagree)
            pl = np.array(pnls)
            wins = pl > 0
            ax.scatter(ed[wins],  pl[wins],  s=10, color="#2ecc71", alpha=0.5, label="Win")
            ax.scatter(ed[~wins], pl[~wins], s=10, color="#e74c3c", alpha=0.5, label="Loss")
            ax.axhline(0, color="white", lw=0.8, linestyle="--")
            corr = np.corrcoef(ed, pl)[0, 1]
            ax.set_title(f"Q-Disagreement at Entry vs PnL  (r={corr:.3f})")
            ax.set_xlabel("|Q1-Q2| mean at entry")
            ax.set_ylabel("PnL")
            ax.legend(fontsize=8)
            ax.grid(True)

        # Disagreement quintile win rates
        ax = axes[0][1]
        if entry_disagree:
            quintile_wr = []
            quintile_labels = []
            for q in range(5):
                lo = np.percentile(ed, q * 20)
                hi = np.percentile(ed, (q + 1) * 20)
                mask = (ed >= lo) & (ed <= hi)
                if mask.sum() > 0:
                    wr = wins[mask].mean()
                    quintile_wr.append(wr)
                    quintile_labels.append(f"Q{q+1}\n({lo:.3f}-{hi:.3f})")
            bars = ax.bar(range(len(quintile_wr)), quintile_wr,
                          color=["#2ecc71" if w >= 0.5 else "#e74c3c"
                                 for w in quintile_wr], alpha=0.8)
            ax.axhline(0.5, color="white", lw=1, linestyle="--")
            ax.set_xticks(range(len(quintile_labels)))
            ax.set_xticklabels(quintile_labels, fontsize=7)
            ax.set_title("Win Rate by Q-Disagreement Quintile at Entry")
            ax.set_ylabel("Win rate")
            ax.set_ylim(0, 1)
            ax.grid(True, axis="y")

        # Disagreement over time (rolling mean)
        ax = axes[1][0]
        window = min(500, len(ep["q_disagreement"]) // 10)
        rolling = pd.Series(ep["q_disagreement"]).rolling(window).mean().values
        ax.plot(ep["ticks"], rolling, color="#f39c12", lw=1.0)
        ax.set_title(f"Critic Disagreement Over Time (rolling {window})")
        ax.set_xlabel("Tick")
        ax.set_ylabel("|Q1-Q2| mean")
        ax.grid(True)

        # Disagreement vs conviction — should be negatively correlated
        ax = axes[1][1]
        ax.scatter(ep["q_disagreement"], ep["max_prob"], s=2, alpha=0.15,
                   color="#7f5af0")
        corr = np.corrcoef(ep["q_disagreement"], ep["max_prob"])[0, 1]
        ax.set_title(f"Critic Disagreement vs Actor Conviction  (r={corr:.3f})")
        ax.set_xlabel("|Q1-Q2| mean")
        ax.set_ylabel("max π(a|s)")
        ax.grid(True)

    savefig(fig, "09_uncertainty_vs_outcome.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — Action-State Consistency Check
# ─────────────────────────────────────────────────────────────────────────────

def plot_action_consistency(ep: dict):
    """
    For each directional action, checks whether the market indicators
    at that tick are aligned with the expected direction.
    LONG should correlate with: RSI > 0, MACD > 0, MeanDev > 0
    SHORT should correlate with: RSI < 0, MACD < 0, MeanDev < 0
    A well-trained policy shows strong separation here.
    """
    fig, axes = make_fig(2, 3, "Section 10 — Action vs Market Context Consistency")
    with plt.rc_context(STYLE):
        feat_labels = FEATURES

        action_masks = {
            "LONG":  ep["actions"] == 0,
            "SHORT": ep["actions"] == 1,
            "HOLD":  ep["actions"] == 3
        }
        action_colors = {
            "LONG":  ACTION_COLORS[0],
            "SHORT": ACTION_COLORS[1],
            "HOLD":  ACTION_COLORS[3]
        }

        # Violin plots: each feature distribution per action
        for fi, fname in enumerate(feat_labels):
            ax = axes[fi // 3][fi % 3]

            plot_data = []
            plot_labels = []
            plot_positions = []
            plot_colors = []

            current_position = 0
            for action_name in ["LONG", "SHORT", "HOLD"]:
                mask = action_masks[action_name]
                if mask.sum() > 0:  # Only add data if there are samples for this action
                    plot_data.append(ep["raw_features"][mask, fi])
                    plot_labels.append(f"{action_name}\n(n={mask.sum():,})")
                    plot_positions.append(current_position)
                    plot_colors.append(action_colors[action_name])
                    current_position += 1

            if plot_data: # Only plot if there is data to plot
                parts = ax.violinplot(plot_data, positions=plot_positions,
                                      showmedians=True, showextrema=False)
                for pc, col in zip(parts["bodies"], plot_colors):
                    pc.set_facecolor(col)
                    pc.set_alpha(0.6)
                parts["cmedians"].set_color("white")

                ax.axhline(0, color="#aaaacc", lw=0.8, linestyle="--", alpha=0.7)
                ax.set_xticks(plot_positions)
                ax.set_xticklabels(plot_labels, fontsize=7)
                ax.set_title(f"{fname}")
                ax.set_ylabel("Scaled value")
                ax.grid(True, axis="y")
            else:
                ax.set_title(f"{fname}\n(No data for actions)")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.5, 0.5, "No action data to plot",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, color='gray', fontsize=10)


    savefig(fig, "10_action_consistency.png")


# ─────────────────────────────────────────────────────────────────────────────
# Text summary
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(ep: dict, agent: SACAgent, pnls):
    trades = ep["trades"]
    n      = len(ep["ticks"])

    lines = []
    lines.append("=" * 62)
    lines.append(f"  SAC DIAGNOSTIC SUMMARY — {ep['label'].upper()}")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Checkpoint training steps: {agent.training_steps:,}")
    lines.append(f"  Target entropy: {agent.target_entropy:.3f} nats "
                 f"({agent.target_entropy/np.log(2):.2f} bits)")
    lines.append("=" * 62)

    lines.append("\n── POLICY HEALTH ──────────────────────────────────────────")
    lines.append(f"  Mean conviction (max-prob):  {ep['max_prob'].mean():.3f}")
    lines.append(f"  Conviction > 0.50:           "
                 f"{(ep['max_prob'] > 0.50).mean():.1%} of ticks")
    lines.append(f"  Conviction > 0.70:           "
                 f"{(ep['max_prob'] > 0.70).mean():.1%} of ticks")
    lines.append(f"  Mean entropy:                {ep['entropies'].mean():.3f} bits")
    lines.append(f"  Policy match to best-Q:      "
                 f"{(np.minimum(ep['q1_all'], ep['q2_all']).argmax(axis=1) == ep['actions']).mean():.1%}")

    lines.append("\n── ACTION DISTRIBUTION ────────────────────────────────────")
    for j, name in enumerate(ACTION_NAMES):
        count = (ep["actions"] == j).sum()
        lines.append(f"  {name:<6}: {count:>6,}  ({count/n:.1%})")

    lines.append("\n── CRITIC HEALTH ──────────────────────────────────────────")
    lines.append(f"  Mean Q1-Q2 disagreement:     "
                 f"{ep['q_disagreement'].mean():.5f}")
    lines.append(f"  Q1/Q2 correlation:           "
                 f"{np.corrcoef(ep['q1_chosen'], ep['q2_chosen'])[0,1]:.4f}")
    lines.append(f"  Mean min(Q1,Q2) chosen:      "
                 f"{np.minimum(ep['q1_chosen'], ep['q2_chosen']).mean():.4f}")

    if trades:
        pnl_arr = np.array([t["pnl"] for t in trades])
        wins    = pnl_arr > 0
        dur_arr = np.array([t["duration"] for t in trades])
        longs   = [t for t in trades if t["side"] == "LONG"]
        shorts  = [t for t in trades if t["side"] == "SHORT"]

        lines.append("\n── TRADE OUTCOMES ─────────────────────────────────────────")
        lines.append(f"  Total trades:                {len(trades):,}")
        lines.append(f"  Win rate:                    {wins.mean():.1%}")
        lines.append(f"  Mean PnL per trade:          {pnl_arr.mean():.4%}")
        lines.append(f"  Median PnL per trade:        {np.median(pnl_arr):.4%}")
        lines.append(f"  Std PnL per trade:           {pnl_arr.std():.4%}")
        lines.append(f"  Total cumulative PnL:        {pnl_arr.sum():.4%}")
        lines.append(f"  Best trade:                  {pnl_arr.max():.4%}")
        lines.append(f"  Worst trade:                 {pnl_arr.min():.4%}")
        cum = np.cumsum(pnl_arr)
        roll_max = np.maximum.accumulate(cum)
        drawdown = cum - roll_max
        lines.append(f"  Max drawdown:                {drawdown.min():.4%}")
        if pnl_arr.std() > 0:
            sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(len(trades))
            lines.append(f"  Sharpe (simplified):         {sharpe:.3f}")
        lines.append(f"  Median hold duration:        {np.median(dur_arr):.0f} ticks "
                     f"({np.median(dur_arr) * 4:.0f} hrs)")
        if longs:
            lp = np.array([t["pnl"] for t in longs])
            lines.append(f"  LONG  win rate:              {(lp > 0).mean():.1%} "
                         f"(n={len(longs)}, mean={lp.mean():.4%})")
        if shorts:
            sp = np.array([t["pnl"] for t in shorts])
            lines.append(f"  SHORT win rate:              {(sp > 0).mean():.1%} "
                         f"(n={len(shorts)}, mean={sp.mean():.4%})")

        # Conviction edge: are high-conviction trades more profitable?
        conv = np.array([t["entry_conviction"] for t in trades])
        hi_conv = conv >= np.percentile(conv, 66)
        lo_conv = conv <  np.percentile(conv, 33)
        lines.append(f"\n── CONVICTION EDGE ────────────────────────────────────────")
        lines.append(f"  Top-33% conviction WR:       {wins[hi_conv].mean():.1%}  "
                     f"(mean PnL {pnl_arr[hi_conv].mean():.4%})")
        lines.append(f"  Bot-33% conviction WR:       {wins[lo_conv].mean():.1%}  "
                     f"(mean PnL {pnl_arr[lo_conv].mean():.4%})")
        edge = wins[hi_conv].mean() - wins[lo_conv].mean()
        lines.append(f"  Conviction edge (WR delta):  {edge:+.1%}")
        lines.append(f"  → {'Conviction is predictive ✓' if edge > 0.03 else 'Conviction not yet predictive — policy may need more training'}")

        if trades and wins.mean() > 0.80:
            lines.append("\n⚠ WARNING: Win rate >80% strongly suggests regime overfitting.")
            lines.append("  Validate on a bear market period before trusting these results.")
        if len(trades) > 0:
            ann_sharpe = pnl_arr.mean() / (pnl_arr.std() + 1e-9) * np.sqrt(252 * 6)  # ~6 trades/day at 4 h
            lines.append(f"  Annualized Sharpe (realistic): {ann_sharpe:.2f}")
            if ann_sharpe > 5:
                lines.append("  ⚠ Sharpe >5 is implausible — check for data leakage or regime bias.")

    lines.append("\n" + "=" * 62)
    text = "\n".join(lines)

    summary_path = os.path.join(OUT_DIR, "diagnostic_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n  ✓  {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAC Diagnostic Suite")
    parser.add_argument("--split",      default="val",
                        choices=["train", "val", "both"],
                        help="Which data split to diagnose")
    parser.add_argument("--checkpoint", default=BEST_PATH,
                        help="Path to .pt checkpoint")
    args = parser.parse_args()

    print(f"\n{'═'*62}")
    print(f"  SAC DIAGNOSTIC SUITE")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Split       : {args.split}")
    print(f"  Output dir : {OUT_DIR}")
    print(f"{'═'*62}\n")

    # ── Load agent ────────────────────────────────────────────────────────────
    agent = SACAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=128, gamma=GAMMA)
    agent.load(args.checkpoint)
    agent.actor.eval()
    agent.critic.eval()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    df = update_master_data()
    df = df[["Open_time", "Close"] + FEATURES].dropna().reset_index(drop=True)
    split_idx  = int(len(df) * TRAIN_SPLIT)
    train_df   = df.iloc[:split_idx].reset_index(drop=True)
    val_df     = df.iloc[split_idx:].reset_index(drop=True)
    print(f"  Train: {len(train_df):,} rows  |  Val: {len(val_df):,} rows\n")

    splits = []
    if args.split in ("train", "both"):
        splits.append(("train", train_df))
    if args.split in ("val", "both"):
        splits.append(("val", val_df))

    for label, split_df in splits:
        print(f"\n{'─'*62}")
        print(f"  Collecting episode: {label.upper()}")
        print(f"{'─'*62}")
        ep = collect_episode(agent, split_df, label)

        print(f"  Ticks collected : {len(ep['ticks']):,}")
        print(f"  Trades completed: {len(ep['trades']):,}")
        print(f"\n  Generating plots...")

        plot_confidence_entropy(ep)
        plot_qvalue_health(ep)
        plot_action_distribution(ep)
        pnls = plot_trade_outcomes(ep)
        plot_feature_sensitivity(agent, ep)
        plot_state_probability_maps(ep)
        plot_regime_analysis(ep)
        plot_timing_analysis(ep)
        plot_uncertainty_vs_outcome(ep)
        plot_action_consistency(ep)

        print()
        write_summary(ep, agent, pnls)

    print(f"\n{'═'*62}")
    print(f"  Diagnostic complete.  All outputs in: {OUT_DIR}/")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()

# Add to diagnostic.py or run standalone:
import numpy as np, pandas as pd
from data.data_manager import update_master_data

df = update_master_data()
split = int(len(df) * 0.8)
val_df = df.iloc[split:].reset_index(drop=True)
prices = val_df['Close'].values
COMMISSION = 0.00015

n_trials, n_trades = 1000, 500
pnls = []
for _ in range(n_trials):
    pnl = 0.0
    for _ in range(n_trades):
        entry_idx = np.random.randint(0, len(prices) - 33)
        hold = np.random.randint(1, 33)
        side = np.random.choice([-1, 1])
        entry = prices[entry_idx] * (1 + side * COMMISSION)
        exit_ = prices[entry_idx + hold] * (1 - side * COMMISSION)
        pnl += side * (exit_ - entry) / entry
    pnls.append(pnl)

print(f"Random policy: mean={np.mean(pnls):.4%}  "
      f"std={np.std(pnls):.4%}  "
      f"win_rate={(np.array(pnls) > 0).mean():.1%}")