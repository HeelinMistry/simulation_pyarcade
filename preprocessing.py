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

    # 2. Load data with Binance Kline headers
    columns = [
        "Open_time", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "Quote_asset_volume", "Num_trades",
        "Taker_buy_base_vol", "Taker_buy_quote_vol", "Ignore"
    ]
    df = pd.read_csv(raw_csv_path, names=columns)

    # 3. Preprocessing: Convert timestamps and cast to float
    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    # Drop NaNs and save
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_csv, index=False)

    print(f"Data preprocessed and saved to {output_csv}. Total ticks: {len(df)}")
    return df


def preprocess_indicators_data(input_file, output_file):
    df = pd.read_csv(input_file)

    # Technical Indicators
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df['MACD_Diff'] = macd - signal

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # Center RSI around 0.0 (Range: -0.5 to 0.5)
    df['RSI_Scaled'] = (df['RSI'] / 100.0) - 0.5

    # Use a slightly more aggressive clip for MACD to make signals louder
    limit = df['MACD_Diff'].std()
    df['MACD_Scaled'] = (df['MACD_Diff'] / limit).clip(-1, 1)

    # --- Bollinger Band %B ---
    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    # %B = (Price - Lower) / (Upper - Lower)
    df['BB_pct'] = (df['Close'] - lower_band) / (upper_band - lower_band + 1e-9)
    df['BB_Scaled'] = df['BB_pct'] - 0.5  # Centered -0.5 to 0.5

    # --- OBV (On-Balance Volume) ---
    # We use the change in OBV over a window to keep the signal local
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    obv_sma = obv.rolling(window=20).mean()
    # Normalize OBV relative to its own volatility
    df['OBV_Scaled'] = ((obv - obv_sma) / (obv.rolling(window=20).std() + 1e-9)).clip(-1, 1)

    # Strip everything except what's needed for UI and SARS
    df = df[['Open_time', 'Close', 'RSI_Scaled', 'MACD_Scaled', 'BB_Scaled', 'OBV_Scaled']].dropna().reset_index(drop=True)
    df.to_csv(output_file, index=False)


def process_live_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['RSI_Scaled'] = (df['RSI'] / 100.0) - 0.5

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd_diff = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
    df['MACD_Scaled'] = (macd_diff / (macd_diff.std() * 2)).clip(-1, 1)


    # --- Bollinger Band %B ---
    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    # %B = (Price - Lower) / (Upper - Lower)
    df['BB_pct'] = (df['Close'] - lower_band) / (upper_band - lower_band + 1e-9)
    df['BB_Scaled'] = df['BB_pct'] - 0.5  # Centered -0.5 to 0.5

    # --- OBV (On-Balance Volume) ---
    # We use the change in OBV over a window to keep the signal local
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    obv_sma = obv.rolling(window=20).mean()
    # Normalize OBV relative to its own volatility
    df['OBV_Scaled'] = ((obv - obv_sma) / (obv.rolling(window=20).std() + 1e-9)).clip(-1, 1)

    return df.dropna().reset_index(drop=True)