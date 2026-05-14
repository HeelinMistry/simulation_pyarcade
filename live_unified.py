"""
live_unified.py
───────────────
Real-world live trading loop for the Unified World Model.

Architecture
------------
1.  Startup: fetch the last 1000 closed candles, warm up the StateAggregator
    so slope/std features are fully populated before the first live decision.
2.  Loop:    sleep until the next 15-minute candle closes, fetch and process
             only the latest closed candle, pass it to the executor, log the
             decision and any realised P/L to disk.
3.  Safety:  incomplete (forming) candle is always excluded; commission is
             applied on both entry and exit matching training exactly.

Key parameters are kept identical to training to avoid train/inference mismatch:
    lookahead_depth : 10   (matches main_headless.py)
    min_conviction  : 0.45 (matches training conviction filter)
    live_temp       : 0.08 (marginally above training temp of 0.05)
"""

import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import deque # Import deque

from agents.MCTSPlanner import MCTSPlanner
from agents.UnifiedWorldModel import UnifiedWorldModel
from agents.unified_executor import UnifiedExecutor
from preprocessing import preprocess_indicators

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BINANCE_API_URL  = "https://api.binance.com/api/"
CANDLE_INTERVAL  = "15m"
CANDLE_SECONDS   = 15 * 60          # 900 seconds per candle
WARMUP_CANDLES   = 1000             # candles fetched on startup for aggregator warm-up
WARMUP_IDX       = 200              # aggregator warm_up_all index within the fetched batch
FEATURES         = ["RSI_Scaled", "MACD_Scaled", "BB_Scaled",
                    "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled"]
PACES             = (1, 2, 4, 8, 12)
NUM_INDICATORS   = len(FEATURES)
INPUT_SIZE       = (NUM_INDICATORS * 3 * len(PACES)) + 2  # 92
MODEL_PATH       = "outcomes/best_world_model.pkl"
LOG_DIR          = "outcomes/live_logs"
ACTION_NAMES     = {0: "LONG", 1: "SHORT", 2: "CLOSE", 3: "HOLD"}

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "live_session.log"), encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Binance helpers
# ─────────────────────────────────────────────
def _parse_klines(data: list) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=[
        "Open_time", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "Quote_volume", "Trades",
        "Taker_buy_base", "Taker_buy_quote", "Ignore"
    ])
    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[num_cols] = df[num_cols].astype(float)
    # Binance REST API returns millisecond timestamps
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms")
    return df


def get_candles(symbol: str, limit: int = WARMUP_CANDLES) -> pd.DataFrame | None:
    """Fetch the latest `limit` candles. Always excludes the last (forming) candle."""
    url = (f"{BINANCE_API_URL}v3/klines"
           f"?symbol={symbol}USDT&interval={CANDLE_INTERVAL}&limit={limit}")
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = _parse_klines(r.json())
        # Drop the last row — it is the currently forming, incomplete candle
        return df.iloc[:-1].reset_index(drop=True)
    except Exception as e:
        log.error(f"Binance API error: {e}")
        return None


# ─────────────────────────────────────────────
# Trade log persistence
# ─────────────────────────────────────────────
class TradeLog:
    """Persists every decision and realised trade to a JSON-lines file."""

    def __init__(self, symbol: str):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(LOG_DIR, f"trades_{symbol}_{timestamp}.jsonl")
        # State file: survives restarts so we don't lose open position context
        self.state_path = os.path.join(LOG_DIR, f"position_state_{symbol}.json")

    def record(self, candle_time, action: int, price: float,
               probs: np.ndarray, realised_pnl: float | None = None):
        entry = {
            "utc":          candle_time.isoformat() if hasattr(candle_time, "isoformat") else str(candle_time),
            "action":       ACTION_NAMES[action],
            "price":        round(float(price), 6),
            "probs":        {k: round(float(v), 4) for k, v in zip(ACTION_NAMES.values(), probs)},
            "realised_pnl": round(float(realised_pnl), 6) if realised_pnl is not None else None
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_position(self, executor: UnifiedExecutor):
        """Persist current position so a restart can warn the operator."""
        state = {
            "current_side": executor.current_side,
            "inventory":    list(executor.inventory), # Convert deque to list
            "total_reward": executor.total_reward,
            "saved_at":     datetime.now(timezone.utc).isoformat()
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def load_position(self) -> dict | None:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return None


# ─────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────
def seconds_until_next_candle() -> float:
    """Returns seconds until the next 15-minute candle closes on Binance."""
    now_ts = time.time()
    elapsed_in_period = now_ts % CANDLE_SECONDS
    return CANDLE_SECONDS - elapsed_in_period + 2  # +2s buffer for Binance propagation


# ─────────────────────────────────────────────
# Main live loop
# ─────────────────────────────────────────────
def run_live(symbol: str = "XRP"):
    log.info(f"{'='*55}")
    log.info(f"  🚀  World Model Live Trading  |  {symbol}/USDT  |  {CANDLE_INTERVAL}")
    log.info(f"{'='*55}")

    # ── 1. Load model ────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        log.error(f"No model found at {MODEL_PATH}. Train first.")
        return

    model = UnifiedWorldModel(input_size=INPUT_SIZE)
    model.load(MODEL_PATH)
    log.info(f"🧠 World Model loaded  (input_size={INPUT_SIZE}, LR={model.lr:.2e})")

    # lookahead_depth matches training (main_headless uses 10)
    planner = MCTSPlanner(model, lookahead_depth=10)

    # min_conviction and live_temp are set inside UnifiedExecutor.__init__
    # to match training values — do not override them here.
    executor = UnifiedExecutor(name=f"Live_{symbol}", planner=planner, paces=PACES)
    log.info(f"   conviction filter : {executor.min_conviction:.0%}")
    log.info(f"   live temperature  : {executor.live_temp}")

    trade_log = TradeLog(symbol)

    # ── 2. Warn if a position was open when we last shut down ────
    prior_state = trade_log.load_position()
    if prior_state and prior_state.get("current_side"):
        log.warning(
            f"⚠️  Prior session had an OPEN {prior_state['current_side']} "
            f"position at entry {prior_state['inventory']}. "
            f"Restoring executor state."
        )
        executor.current_side = prior_state["current_side"]
        # Convert list back to deque with maxlen=100
        executor.inventory    = deque(prior_state["inventory"], maxlen=100)
        executor.total_reward = prior_state["total_reward"]

    # ── 3. Warm-up: fetch history and fully populate aggregator ──
    log.info(f"📡 Fetching {WARMUP_CANDLES} candles for warm-up...")
    warmup_df = get_candles(symbol, limit=WARMUP_CANDLES)
    if warmup_df is None:
        log.error("Failed to fetch warm-up candles. Aborting.")
        return

    warmup_df = preprocess_indicators(warmup_df)
    warmup_df.dropna(inplace=True)
    warmup_df.reset_index(drop=True, inplace=True)

    indicators_history = warmup_df[FEATURES].values.astype(np.float32)

    # warm_up_all pre-fills each pace agent's history buffer looking
    # BACKWARDS from WARMUP_IDX so the very first live decision has a
    # fully populated slope/std context — no zero-padding artefacts.
    safe_warmup_idx = min(WARMUP_IDX, len(indicators_history) - 1)
    executor.aggregator.warm_up_all(indicators_history, safe_warmup_idx)
    log.info(f"✅ Aggregator warmed up on {len(indicators_history)} candles "
             f"(warm_up_all idx={safe_warmup_idx})")

    # ── 4. Live polling loop ─────────────────────────────────────
    tick = 0
    log.info("⏳ Waiting for next candle close...\n")

    while True:
        # Sleep precisely until the next candle closes
        wait = seconds_until_next_candle()
        log.info(f"   next candle in {wait:.0f}s  ({wait/60:.1f} min)")
        time.sleep(wait)

        # Preprocess: need a small DataFrame for rolling indicators
        # We append the new candle to recent history for accurate indicator calc
        recent_raw = get_candles(symbol, limit=300)
        if recent_raw is None:
            log.warning("⚠️  Failed to fetch recent candles for indicator calc — skipping tick.")
            continue

        processed = preprocess_indicators(recent_raw)
        processed.dropna(inplace=True)
        if len(processed) == 0:
            log.warning("⚠️  Preprocessing returned no rows — skipping tick.")
            continue

        latest = processed.iloc[-1]
        indicators = np.array([latest[f] for f in FEATURES], dtype=np.float32)
        price      = float(latest["Close"])
        candle_time = latest["Open_time"]

        # Execute decision
        prev_reward = executor.total_reward
        action, probs = executor.step(indicators=indicators, price=price, tick=tick)

        realised_pnl = None
        delta = executor.total_reward - prev_reward
        if abs(delta) > 1e-9:
            realised_pnl = delta

        # Log to file and console
        trade_log.record(candle_time, action, price, probs, realised_pnl)
        trade_log.save_position(executor)

        status = executor.get_status()
        conviction = float(probs[action])
        entropy    = -np.sum(probs * np.log2(probs + 1e-9))

        log.info(
            f"Tick {tick:04d} | {candle_time} | ${price:.5f} | "
            f"Action: {ACTION_NAMES[action]:<5} ({conviction:.1%}) | "
            f"Position: {status['position']:<5} | "
            f"Session P/L: {status['pnl']:+.4%} | "
            f"Entropy: {entropy:.2f}b"
            + (f" | ✅ CLOSED  PnL: {realised_pnl:+.4%}" if realised_pnl is not None else "")
        )

        tick += 1


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_live("XRP")