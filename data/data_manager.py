import os
import glob
import pandas as pd
from .preprocessing import preprocess_binance_data, preprocess_indicators_data

# --- 1. Configuration ---
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MASTER_CSV = os.path.join(PROCESSED_DIR, "XRPUSDT_master_processed.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)


def update_master_data():
    # Find all zip files in the raw folder
    zip_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.zip")))

    if not zip_files:
        if os.path.exists(MASTER_CSV):
            print("No new zips found. Loading existing master data...")
            return pd.read_csv(MASTER_CSV)
        else:
            raise FileNotFoundError("No raw data (.zip) or processed data found!")

    print(f"Found {len(zip_files)} zip files. Processing...")

    all_dfs = []

    # If master already exists, load it first to append new data to it
    if os.path.exists(MASTER_CSV):
        all_dfs.append(pd.read_csv(MASTER_CSV))

    for zp in zip_files:
        temp_csv = zp.replace(".zip", "_temp.csv")

        print(f"--> Extracting and cleaning: {os.path.basename(zp)}")
        # 1. Extract/Clean raw Binance format
        preprocess_binance_data(zp, temp_csv)

        # 2. Load into memory
        new_data = pd.read_csv(temp_csv)
        all_dfs.append(new_data)

        # 3. Clean up: Remove the zip and the temp csv as requested
        os.remove(zp)
        os.remove(temp_csv)
        print(f"Done. {os.path.basename(zp)} deleted.")

    # Combine all data sequentially
    full_df = pd.concat(all_dfs).drop_duplicates(subset=['Open_time']).sort_values('Open_time')
    full_df.reset_index(drop=True, inplace=True)

    # We save to a temp file because your preprocess_indicators_data likely expects a path
    full_df.to_csv(MASTER_CSV, index=False)

    print("Recalculating indicators across the full dataset timeline...")
    # This ensures RSI/MACD flow perfectly from Dec into Jan
    preprocess_indicators_data(MASTER_CSV, MASTER_CSV)

    return pd.read_csv(MASTER_CSV)
