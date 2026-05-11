import numpy as np
import pandas as pd
import zipfile
import os

def preprocess_binance_data(zip_path, output_csv):
    # 1. Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/raw/")
        extracted_file = zip_ref.namelist()[0]

    raw_csv_path = os.path.join("data/raw", extracted_file)

    raw_columns = [
        "Open_time", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "Quote_asset_volume", "Num_trades",
        "Taker_buy_base_vol", "Taker_buy_quote_vol", "Ignore"
    ]

    # Load with robustness
    first_line = pd.read_csv(raw_csv_path, nrows=1)
    has_header = not str(first_line.iloc[0, 0]).isdigit()

    df = pd.read_csv(
        raw_csv_path,
        names=raw_columns if not has_header else None,
        header=0 if has_header else None
    )

    # 2. Time Alignment (FIXED: Binance uses 'us', not 'ms')
    print(pd.to_datetime(df['Open_time'].iloc[0], unit='us'))

    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='us')
    df = df.sort_values('Open_time').reset_index(drop=True)

    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Open_time'] + num_cols, inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"✨ Consistency Check: {len(df)} ticks processed from {df['Open_time'].min()} to {df['Open_time'].max()}")
    return df


def preprocess_indicators(df):
    # --- MACD (Trend Momentum) ---
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    macd_diff = macd - signal
    macd_std = macd_diff.rolling(100).std() + 1e-9
    df['MACD_Scaled'] = (macd_diff / macd_std).clip(-3,3)/3

    # --- RSI (Overbought/Oversold) ---
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean() + 1e-9
    rsi = 100 - (100 / (1 + gain / loss))
    df['RSI_Scaled'] = (rsi - 50) / 50

    # --- Bollinger (Volatility) ---
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std() + 1e-9
    upper = sma + 2 * std
    lower = sma - 2 * std
    bb = (df['Close'] - lower) / (upper - lower + 1e-9)
    df['BB_Scaled'] = bb - 0.5

    # --- OBV (Net Volume Flow Velocity) ---
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    obv_velocity = obv.diff(13)
    v_mean = obv_velocity.rolling(200).mean()
    v_std = obv_velocity.rolling(200).std() + 1e-9
    df['OBV_Scaled'] = ((obv_velocity - v_mean) / v_std).clip(-3, 3) / 3

    # --- NEW: Volatility (ATR-like) ---
    atr_raw = df['Close'].diff().abs().rolling(14).mean()
    atr_pct = atr_raw / df['Close'] * 100
    atr_mean = atr_pct.rolling(200).mean()
    atr_std  = atr_pct.rolling(200).std() + 1e-9
    df['ATR_Scaled'] = ((atr_pct - atr_mean) / atr_std).clip(-3, 3) / 3

    # --- NEW: Mean Deviation (Distance from Trend) ---
    mean_dev = (df['Close'] - sma) / std
    df['MeanDev_Scaled'] = mean_dev.clip(-3, 3) / 3

    # Final cleanup (Total 6 indicators + Price/Time)
    cols = ['Open_time', 'Close', 'RSI_Scaled', 'MACD_Scaled', 'BB_Scaled', 'OBV_Scaled', 'ATR_Scaled', 'MeanDev_Scaled']
    df = df[cols].dropna().reset_index(drop=True)
    return df
