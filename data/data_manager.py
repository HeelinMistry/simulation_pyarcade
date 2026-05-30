import os
import glob
import shutil
import pandas as pd
from .preprocessing import preprocess_binance_data, preprocess_indicators_data, OPEN_TIME_FMT

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
MASTER_CSV    = os.path.join(PROCESSED_DIR, "XRPUSDT_master_processed.csv")
MASTER_TMP    = os.path.join(PROCESSED_DIR, "XRPUSDT_master_processed.tmp.csv")

# Any row whose year falls outside this range is a corrupt ghost from a
# previous wrong-unit parse and is stripped before merging.
VALID_YEAR_MIN = 2020
VALID_YEAR_MAX = 2030

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_open_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Open_time (whatever its current type) and rewrite it as the
    canonical OPEN_TIME_FMT string.  This is the single point where all
    frames — whether freshly extracted from a zip or loaded from the
    existing master — are brought to the same format before any merge or
    deduplication step.
    """
    df = df.copy()
    df["Open_time"] = (
        pd.to_datetime(df["Open_time"], errors="coerce")
          .dt.strftime(OPEN_TIME_FMT)
    )
    return df


def _strip_corrupt_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove rows whose Open_time parses to a year outside VALID_YEAR_MIN–MAX.
    Returns (clean_df, n_purged).
    """
    ts    = pd.to_datetime(df["Open_time"], errors="coerce")
    valid = ts.dt.year.between(VALID_YEAR_MIN, VALID_YEAR_MAX)
    bad   = (~valid | ts.isna()).sum()
    return df[valid].copy(), int(bad)


def _classify_zip(path: str) -> str:
    """
    Return 'monthly' (YYYY-MM.zip) or 'daily' (YYYY-MM-DD.zip).
    Counts the number of numeric segments in the filename after the interval
    field: 2 numeric → monthly, 3 numeric → daily.
    """
    base  = os.path.basename(path).replace(".zip", "")
    parts = base.split("-")
    return "daily" if sum(1 for p in parts if p.isdigit()) >= 3 else "monthly"


def _log_coverage(df: pd.DataFrame, label: str) -> None:
    """Print row count, date range, years present, and any gaps > 1 h."""
    if df.empty:
        print(f"  [{label}] empty — nothing to report.")
        return
    ts    = pd.to_datetime(df["Open_time"], errors="coerce").sort_values()
    years = sorted(ts.dt.year.dropna().unique().tolist())
    print(
        f"  [{label}]  {len(df):,} rows  |  "
        f"{ts.iloc[0].strftime('%Y-%m-%d %H:%M')} → "
        f"{ts.iloc[-1].strftime('%Y-%m-%d %H:%M')}  |  "
        f"years: {years}"
    )
    gaps = ts.diff().dropna()
    big  = gaps[gaps > pd.Timedelta(hours=1)]
    if not big.empty:
        print(
            f"  [{label}]  ⚠  {len(big)} gap(s) > 1 h — "
            f"largest: {gaps.max()}.  Check for missing zips."
        )


# ── Public API ────────────────────────────────────────────────────────────────

def update_master_data() -> pd.DataFrame:
    """
    Merge every zip in RAW_DIR into the master processed CSV.

    Safety guarantees
    ─────────────────
    1. Zips are deleted only AFTER the master CSV is confirmed saved.
    2. Master is written atomically (.tmp → rename) so a mid-write failure
       never corrupts the existing file.
    3. All Open_time values are normalised to '%Y-%m-%d %H:%M:%S' strings
       via a single shared function before any concat or deduplication, so
       ms, us, and ns source files always merge cleanly.
    4. Corrupt ghost rows (years outside 2020–2030) are stripped from the
       existing master on load so they cannot survive deduplication.
    5. Coverage is logged before and after every run.
    """
    zip_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.zip")))

    if not zip_files:
        if os.path.exists(MASTER_CSV):
            print("No new zips found — loading existing master.")
            df = pd.read_csv(MASTER_CSV)
            _log_coverage(df, "master")
            return df
        raise FileNotFoundError(
            "No zip files in data/raw/ and no existing master CSV."
        )

    # ── Report what we found ──────────────────────────────────────────────────
    monthly = [z for z in zip_files if _classify_zip(z) == "monthly"]
    daily   = [z for z in zip_files if _classify_zip(z) == "daily"]
    print(
        f"Found {len(zip_files)} zip(s)  "
        f"({len(monthly)} monthly, {len(daily)} daily)"
    )
    if monthly and daily:
        print("  ⚠  Mixed monthly + daily zips — duplicates removed via dedup.")

    all_dfs: list[pd.DataFrame] = []

    # ── Load and sanitise existing master ─────────────────────────────────────
    if os.path.exists(MASTER_CSV):
        existing = pd.read_csv(MASTER_CSV)
        existing = _normalise_open_time(existing)
        existing, purged = _strip_corrupt_rows(existing)
        if purged:
            print(
                f"  ⚠  Purged {purged:,} corrupt row(s) from master "
                f"(year outside {VALID_YEAR_MIN}–{VALID_YEAR_MAX})."
            )
        _log_coverage(existing, "existing master")
        all_dfs.append(existing)

    # ── Extract and normalise each zip ────────────────────────────────────────
    processed_zips: list[str] = []
    temp_csvs:      list[str] = []

    try:
        for zp in zip_files:
            temp_csv = zp.replace(".zip", "_temp.csv")
            preprocess_binance_data(zp, temp_csv)   # writes canonical format

            new_data = pd.read_csv(temp_csv)
            # preprocess_binance_data already formats Open_time; call
            # _normalise_open_time as a defensive re-parse to guarantee type
            # and format consistency regardless of pandas CSV round-trip quirks.
            new_data = _normalise_open_time(new_data)

            all_dfs.append(new_data)
            processed_zips.append(zp)
            temp_csvs.append(temp_csv)

        # ── Merge: dedup and sort on the normalised string column ─────────────
        full_df = (
            pd.concat(all_dfs, ignore_index=True)
              .drop_duplicates(subset=["Open_time"])
              .sort_values("Open_time")
              .reset_index(drop=True)
        )
        _log_coverage(full_df, "merged (pre-indicators)")

        # ── Compute indicators and save atomically ────────────────────────────
        print("Recalculating indicators …")
        full_df.to_csv(MASTER_TMP, index=False)
        preprocess_indicators_data(MASTER_TMP, MASTER_TMP)
        shutil.move(MASTER_TMP, MASTER_CSV)
        print(f"  ✓  Master saved → {MASTER_CSV}")

    except Exception as exc:
        print(f"\n  ❌  {exc}")
        print("  Master CSV is unchanged.  Zips have NOT been deleted.")
        for tc in temp_csvs:
            if os.path.exists(tc):
                os.remove(tc)
        if os.path.exists(MASTER_TMP):
            os.remove(MASTER_TMP)
        raise

    # ── Delete zips only after confirmed save ─────────────────────────────────
    for zp, tc in zip(processed_zips, temp_csvs):
        if os.path.exists(tc):
            os.remove(tc)
        os.remove(zp)
        print(f"  Deleted  {os.path.basename(zp)}")

    # ── Final report ──────────────────────────────────────────────────────────
    final = pd.read_csv(MASTER_CSV)
    _log_coverage(final, "final master")
    return final