import arcade
import pandas as pd
import requests
import numpy as np
import cupy as cp

from agents.MCTSPlanner import MCTSPlanner
from agents.UnifiedWorldModel import UnifiedWorldModel
from agents.unified_executor import UnifiedExecutor
from simulation.environment import SimulationEnv
from preprocessing import preprocess_indicators

BINANCE_API_URL = "https://api.binance.com/api/"


def get_live_candles(symbol="XRP"):
    """Fetches the latest 1000 candles from Binance."""
    interval = "15m"
    limit = 1000
    url = f"{BINANCE_API_URL}v3/klines?symbol={symbol}USDT&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"❌ Binance API Error: {e}")
        return None

    df = pd.DataFrame(data, columns=[
        "Open_time", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "Quote_volume", "Trades",
        "Taker_buy_base", "Taker_buy_quote", "Ignore"
    ])

    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[num_cols] = df[num_cols].astype(float)
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms")
    return df


def run_live_sim(symbol="XRP"):
    # 1. Data Acquisition
    print(f"📡 Fetching live {symbol}/USDT data...")
    raw = get_live_candles(symbol)
    if raw is None: return

    # 2. Preprocessing
    df = preprocess_indicators(raw)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 3. World Model Setup
    paces = (1, 2, 4, 8, 12)
    # UPDATED: 6 indicators * 3 states (raw, slope, std) * 5 paces + 2 portfolio = 92
    num_indicators = 6
    input_size = (num_indicators * 3 * len(paces)) + 2

    model = UnifiedWorldModel(input_size=input_size)
    model.load("outcomes/best_world_model.pkl")
    print(f"🧠 Unified World Model loaded (Input Size: {input_size})")

    planner = MCTSPlanner(model, lookahead_depth=100)

    # 4. Unified Executor
    executor = UnifiedExecutor(
        name=f"Live_WM_{symbol}",
        planner=planner,
        paces=paces
    )
    
    executor.live_temp = 0.20
    executor.min_conviction = 0.30

    # 5. Launch Simulation Window
    print(f"🚀 Launching World Model UI for {symbol}...")
    window = SimulationEnv(
        data_df=df,
        executor=executor,
        show_chart=True,
        title=f"World Model Live Strategy: {symbol}USDT"
    )

    arcade.run()


if __name__ == "__main__":
    run_live_sim("XRP")
