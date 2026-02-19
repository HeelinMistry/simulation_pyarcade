import arcade
import pandas as pd
import requests
import numpy as np

from agents.unified_brain import UnifiedBrain
from agents.unified_executor import UnifiedExecutor
from simulation.environment import SimulationEnv
from preprocessing import process_live_indicators

BINANCE_API_URL = "https://api.binance.com/api/"


# -------------------------------------------------------
# Binance Fetch
# -------------------------------------------------------
def get_live_candles(symbol="XRP"):
    """Fetches the latest 1000 candles from Binance."""
    interval = "15m"
    limit = 1000
    url = (
        f"{BINANCE_API_URL}v3/klines"
        f"?symbol={symbol}USDT"
        f"&interval={interval}"
        f"&limit={limit}"
    )
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


# -------------------------------------------------------
# Live / Replay Runner
# -------------------------------------------------------
def run_live_sim(symbol="XRP"):
    # 1. Data Acquisition
    print(f"📡 Fetching live {symbol}/USDT data...")
    raw = get_live_candles(symbol)
    if raw is None: return

    # 2. Preprocessing (Indicators & Scaling)
    df = process_live_indicators(raw)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 3. Brain Setup
    # Configuration matches the architecture: 5 paces * 8 features (4 raw + 4 slopes)
    paces = (1, 2, 4, 8, 16)
    input_size = len(paces) * 8

    brain = UnifiedBrain(input_size=input_size)
    brain.load()  # Loads from outcomes/unified_brain.pkl
    print(f"🧠 UnifiedBrain initialized (Input Size: {input_size})")

    # 4. Unified Executor
    # This now handles the Aggregator and MultiPaceAgents internally
    executor = UnifiedExecutor(
        name=f"Live_{symbol}",
        brain=brain,
        paces=paces
    )

    # 5. Launch Simulation Window
    # The Environment will call executor.step() on every frame
    print(f"🚀 Launching UI for {symbol}...")
    window = SimulationEnv(
        data_df=df,
        executor=executor,
        show_chart=True,
        title=f"Live Replay: {symbol}USDT"
    )

    arcade.run()
    return brain


if __name__ == "__main__":
    run_live_sim("XRP")