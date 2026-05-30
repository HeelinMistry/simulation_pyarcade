import numpy as np
import pandas as pd
import zipfile
import os

# ── Single canonical format used everywhere in the pipeline ──────────────────
OPEN_TIME_FMT = "%Y-%m-%d %H:%M:%S"


def _detect_unit(first_timestamp: int) -> str:
    """
    Binance changed the precision of Open_time across data exports:
      pre-2025  : milliseconds  (13 digits, ~1.6e12)
      2025+     : microseconds  (16 digits, ~1.7e15)
      (nanoseconds / 19 digits included for future-proofing)

    Thresholds have a 1000x safety margin — no overlap is possible
    between units for any real date in the range 2000–2099.
    """
    if first_timestamp > 10_000_000_000_000_000:   # > 1e16 → nanoseconds
        return "ns"
    elif first_timestamp > 10_000_000_000_000:     # > 1e13 → microseconds
        return "us"
    else:                                           # ≤ 1e13 → milliseconds
        return "ms"


def preprocess_binance_data(zip_path: str, output_csv: str) -> pd.DataFrame:
    """
    Extract a Binance kline zip, parse timestamps correctly regardless of
    which precision Binance used, and write a clean CSV whose Open_time
    column is always the string format OPEN_TIME_FMT ('%Y-%m-%d %H:%M:%S').

    The extracted raw CSV is removed after processing so data/raw/ stays clean.
    """
    raw_columns = [
        "Open_time", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "Quote_asset_volume", "Num_trades",
        "Taker_buy_base_vol", "Taker_buy_quote_vol", "Ignore",
    ]

    # ── 1. Extract ────────────────────────────────────────────────────────────
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("data/raw/")
        extracted_file = zf.namelist()[0]

    raw_csv_path = os.path.join("data/raw", extracted_file)

    try:
        # ── 2. Load — detect whether file has a header row ───────────────────
        first_line = pd.read_csv(raw_csv_path, nrows=1, header=None)
        has_header = not str(first_line.iloc[0, 0]).lstrip("-").isdigit()

        df = pd.read_csv(
            raw_csv_path,
            names=raw_columns if not has_header else None,
            header=0 if has_header else None,
        )

        # ── 3. Detect timestamp unit and parse ───────────────────────────────
        first_ts = int(df["Open_time"].iloc[0])
        unit = _detect_unit(first_ts)
        df["Open_time"] = pd.to_datetime(df["Open_time"], unit=unit, utc=False)

        # ── 4. Normalise Open_time to the canonical string format ────────────
        df["Open_time"] = df["Open_time"].dt.strftime(OPEN_TIME_FMT)

        # ── 5. Numeric columns ───────────────────────────────────────────────
        num_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["Open_time"] + num_cols, inplace=True)
        df.sort_values("Open_time", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # ── 6. Save ──────────────────────────────────────────────────────────
        df.to_csv(output_csv, index=False)

        print(
            f"  ✓  {os.path.basename(zip_path)}  "
            f"unit={unit}  {len(df):,} rows  "
            f"{df['Open_time'].iloc[0]}  →  {df['Open_time'].iloc[-1]}"
        )
        return df

    finally:
        # ── 7. Always remove the extracted raw CSV ────────────────────────────
        if os.path.exists(raw_csv_path):
            os.remove(raw_csv_path)


def preprocess_indicators_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Compute the six scaled technical indicators over a clean OHLCV CSV and
    write the result back to output_path.  Open_time is passed through as-is
    (the canonical OPEN_TIME_FMT string) so no re-parsing is needed.
    """
    df = pd.read_csv(input_path)

    # ── MACD (Trend Momentum) ─────────────────────────────────────────────────
    ema12      = df["Close"].ewm(span=12).mean()
    ema26      = df["Close"].ewm(span=26).mean()
    macd       = ema12 - ema26
    signal     = macd.ewm(span=9).mean()
    macd_diff  = macd - signal
    macd_std   = macd_diff.rolling(100).std() + 1e-9
    df["MACD_Scaled"] = (macd_diff / macd_std).clip(-3, 3) / 3

    # ── RSI (Overbought / Oversold) ───────────────────────────────────────────
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean() + 1e-9
    rsi   = 100 - (100 / (1 + gain / loss))
    df["RSI_Scaled"] = (rsi - 50) / 50

    # ── Bollinger Bands (Volatility position) ─────────────────────────────────
    sma   = df["Close"].rolling(20).mean()
    std   = df["Close"].rolling(20).std() + 1e-9
    upper = sma + 2 * std
    lower = sma - 2 * std
    bb    = (df["Close"] - lower) / (upper - lower + 1e-9)
    df["BB_Scaled"] = bb - 0.5

    # ── OBV velocity (Net Volume Flow) ────────────────────────────────────────
    # NOTE: rolling(200) requires ≥ 200 rows of history.  The master CSV easily
    # satisfies this; live inference must fetch ≥ 250 candles (see live.py).
    obv          = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    obv_velocity = obv.diff(13)
    v_mean       = obv_velocity.rolling(200).mean()
    v_std        = obv_velocity.rolling(200).std() + 1e-9
    df["OBV_Scaled"] = ((obv_velocity - v_mean) / v_std).clip(-3, 3) / 3

    # ── ATR-like Volatility ───────────────────────────────────────────────────
    # NOTE: same rolling(200) constraint as OBV above.
    atr_raw  = df["Close"].diff().abs().rolling(14).mean()
    atr_pct  = atr_raw / df["Close"] * 100
    atr_mean = atr_pct.rolling(200).mean()
    atr_std  = atr_pct.rolling(200).std() + 1e-9
    df["ATR_Scaled"] = ((atr_pct - atr_mean) / atr_std).clip(-3, 3) / 3

    # ── Mean Deviation (Distance from Trend) ──────────────────────────────────
    mean_dev = (df["Close"] - sma) / std
    df["MeanDev_Scaled"] = mean_dev.clip(-3, 3) / 3

    # ── Final output ──────────────────────────────────────────────────────────
    cols = [
        "Open_time", "Close", "Volume",
        "RSI_Scaled", "MACD_Scaled", "BB_Scaled",
        "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled",
    ]
    df = df[cols].dropna().reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df