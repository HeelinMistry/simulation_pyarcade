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

    # 1. Load with robustness:
    # Check if file has a header already to prevent shifting
    first_line = pd.read_csv(raw_csv_path, nrows=1)
    has_header = not str(first_line.iloc[0, 0]).isdigit()

    df = pd.read_csv(
        raw_csv_path,
        names=raw_columns if not has_header else None,
        header=0 if has_header else None
    )

    # 2. Time Alignment (CRITICAL)
    # Ensure unit is 'ms' and sort to maintain time-series consistency
    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='us')
    df = df.sort_values('Open_time').reset_index(drop=True)

    # 3. Type Casting
    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Cleanup
    df.dropna(subset=['Open_time'] + num_cols, inplace=True)

    # Save a clean version
    df.to_csv(output_csv, index=False)
    print(f"✨ Consistency Check: {len(df)} ticks processed from {df['Open_time'].min()} to {df['Open_time'].max()}")
    return df


def preprocess_indicators(df):

    # --- MACD ---
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    macd_diff = macd - signal

    macd_std = macd_diff.rolling(100).std() + 1e-9
    df['MACD_Scaled'] = (macd_diff / macd_std).clip(-3,3)/3


    # --- RSI ---
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean() + 1e-9

    rsi = 100 - (100 / (1 + gain / loss))
    df['RSI_Scaled'] = (rsi - 50) / 50


    # --- Bollinger ---
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()

    upper = sma + 2 * std
    lower = sma - 2 * std

    bb = (df['Close'] - lower) / (upper - lower + 1e-9)
    df['BB_Scaled'] = bb - 0.5


    # --- OBV ---
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    obv_sma = obv.rolling(20).mean()
    obv_std = obv.rolling(50).std() + 1e-9

    df['OBV_Scaled'] = ((obv - obv_sma) / obv_std).clip(-3,3)/3

    df = df[['Open_time', 'Close', 'RSI_Scaled', 'MACD_Scaled', 'BB_Scaled', 'OBV_Scaled']].dropna().reset_index(drop=True)
    return df
