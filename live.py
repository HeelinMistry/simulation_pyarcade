import arcade
import pandas as pd
import requests

from agents.agent_master import MasterObserver
from agents.probabilistic_brain import ProbabilisticBrain
from environment import SimulationEnv
from preprocessing import process_live_indicators

BINANCE_API_URL = "https://api.binance.com/api/"

def get_live_candles(symbol="BTC"):
    interval = '15m'
    limit = 1000  # We only need enough to calculate the longest SMA (e.g., 200)

    url = f"{BINANCE_API_URL}v3/klines?symbol={symbol}USDT&interval={interval}&limit={limit}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_time', 'Quote_volume', 'Trades', 'Taker_buy_base', 'Taker_buy_quote', 'Ignore'
        ])

        # CRITICAL: Convert strings to numeric (Binance returns strings)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

        # Convert Open_time to readable format if needed for UI
        df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')

        return df
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

def run_live_sim():
    # 1. Fetch
    raw_df = get_live_candles("XRP")
    if raw_df is None: return

    # 2. Process
    df = process_live_indicators(raw_df)
    df.dropna(inplace=True)  # Remove the "warm-up" period for indicators
    df.reset_index(drop=True, inplace=True)  # Crucial for tick alignment

    if df.isnull().values.any():
        print("Warning: NaN values detected in indicators. Filling with 0.")
        df = df.fillna(0)

    # 3. Brain & Agents (Same as before)
    brain = ProbabilisticBrain(alpha=0.0005)
    agents = [
        MasterObserver("BRAIN_MASTER", pace=1, brain=brain)
    ]

    # 4. Background training on the 500 recent candles
    print("Adapting brain to recent market conditions...")
    for _ in range(10):  # 10 Epochs
        for idx, row in df.iterrows():
            for agent in agents:
                if idx % agent.pace == 0:
                    state, action = agent.act(row)
                    agent.handle_reward(state, action, row['Close'])

    # 5. UI Launch
    # Reset stats so UI starts at 0.0%
    for agent in agents:
        agent.total_reward = 0.0
        agent.inventory = []

    window = SimulationEnv(df, agents, brain, show_chart=True)
    arcade.run()

    return brain

if __name__ == "__main__":
    trained_brain = run_live_sim()
