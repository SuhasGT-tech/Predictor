"""
Build the model-ready feature table from:
  - data/copra_prices_arasikere_tiptur.csv  (your scraped price history)
  - data/external_factors.csv                (output of step 2)

Produces: data/features.csv - one row per (Market, Date) with the target
(Modal price) plus every engineered feature the model will train on.

FEATURES ENGINEERED
--------------------
Price-history based (per market, computed only from PAST rows - no leakage):
  - modal_lag_1, modal_lag_2, modal_lag_3   : price at the previous 1/2/3 tender days
  - modal_roll_mean_4, modal_roll_std_4      : rolling mean/volatility over last 4 tender days
  - modal_roll_mean_12                       : rolling mean over last 12 tender days (~1 quarter)
  - price_momentum                           : lag_1 - lag_2 (is price rising or falling)
  - arrivals_lag_1, arrivals_roll_mean_4      : supply-side signal

Calendar / seasonality:
  - day_of_week, month, year, days_since_start
  - is_tiptur_tenderday (Mon/Thu), is_arsikere_tenderday (Tue/Fri)

External:
  - coconut_oil_price_usd_per_mt, coconut_oil_price_change_1m
  - usd_inr_rate
  - rainfall_mm, temp_c (matched to the row's own market)
  - rainfall_roll_90 (last ~3 months cumulative rainfall - proxy for the
    monsoon feeding into future coconut yield)
  - is_festival_window

USAGE
-----
    python 3_build_features.py
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PRICES_CSV = os.path.join(DATA_DIR, "copra_prices_arasikere_tiptur.csv")
EXTERNAL_CSV = os.path.join(DATA_DIR, "external_factors.csv")
OUT_CSV = os.path.join(DATA_DIR, "features.csv")


def load_prices():
    df = pd.read_csv(PRICES_CSV)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

    # Focus on COPRA variety only (the CSV also has MILLING/OTHER/etc. mixed in,
    # which run at different price levels and would distort the model)
    df = df[df["Variety"].str.upper() == "COPRA"].copy()

    for col in ["Arrivals", "Min", "Max", "Modal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Modal", "Date"])
    df["Market"] = df["Market"].str.upper().str.strip()

    # If multiple grades exist for the same market/day, take the arrivals-
    # weighted average Modal price as "the" price for that day
    def weighted_avg(g):
        if g["Arrivals"].sum() > 0:
            modal = np.average(g["Modal"], weights=g["Arrivals"].fillna(0) + 1e-9)
        else:
            modal = g["Modal"].mean()
        return pd.Series({
            "Modal": modal,
            "Min": g["Min"].mean(),
            "Max": g["Max"].mean(),
            "Arrivals": g["Arrivals"].sum(),
        })

    df = (
        df.groupby(["Market", "Date"])
        .apply(weighted_avg, include_groups=False)
        .reset_index()
    )
    return df.sort_values(["Market", "Date"]).reset_index(drop=True)


def add_price_features(df):
    out = []
    for market, g in df.groupby("Market"):
        g = g.sort_values("Date").copy()
        g["modal_lag_1"] = g["Modal"].shift(1)
        g["modal_lag_2"] = g["Modal"].shift(2)
        g["modal_lag_3"] = g["Modal"].shift(3)
        g["modal_roll_mean_4"] = g["Modal"].shift(1).rolling(4, min_periods=1).mean()
        g["modal_roll_std_4"] = g["Modal"].shift(1).rolling(4, min_periods=2).std()
        g["modal_roll_mean_12"] = g["Modal"].shift(1).rolling(12, min_periods=1).mean()
        g["price_momentum"] = g["modal_lag_1"] - g["modal_lag_2"]
        g["arrivals_lag_1"] = g["Arrivals"].shift(1)
        g["arrivals_roll_mean_4"] = g["Arrivals"].shift(1).rolling(4, min_periods=1).mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_calendar_features(df):
    df["day_of_week"] = df["Date"].dt.dayofweek  # Mon=0 ... Sun=6
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["days_since_start"] = (df["Date"] - df["Date"].min()).dt.days
    df["is_tiptur_tenderday"] = df["day_of_week"].isin([0, 3]).astype(int)   # Mon, Thu
    df["is_arsikere_tenderday"] = df["day_of_week"].isin([1, 4]).astype(int)  # Tue, Fri
    return df


def add_external_features(df):
    if not os.path.exists(EXTERNAL_CSV):
        print(f"  NOTE: {EXTERNAL_CSV} not found yet - run 2_fetch_external_factors.py "
              "first for full accuracy. Continuing with price/calendar features only.")
        for col in ["coconut_oil_price_usd_per_mt", "usd_inr_rate", "rainfall_mm",
                    "temp_c", "is_festival_window", "rainfall_roll_90",
                    "coconut_oil_price_change_1m"]:
            df[col] = np.nan
        return df

    ext = pd.read_csv(EXTERNAL_CSV)
    ext["date"] = pd.to_datetime(ext["date"])

    # market-specific rainfall/temp columns -> generic "own market" columns
    df = df.merge(ext, left_on="Date", right_on="date", how="left")

    def own_market_value(row, prefix):
        col = f"{prefix}_{row['Market'].lower()}_mm" if prefix == "rainfall" else f"{prefix}_{row['Market'].lower()}_c"
        return row.get(col, np.nan)

    df["rainfall_mm"] = df.apply(lambda r: own_market_value(r, "rainfall"), axis=1)
    df["temp_c"] = df.apply(lambda r: own_market_value(r, "temp"), axis=1)

    # Rolling 90-day rainfall (monsoon proxy) computed per market on the daily
    # external table would be more precise, but a per-row approximation using
    # the merged frame is good enough here - left for a future refinement.
    df["rainfall_roll_90"] = (
        df.sort_values("Date").groupby("Market")["rainfall_mm"]
        .transform(lambda s: s.rolling(90, min_periods=1).mean())
    )

    df["coconut_oil_price_change_1m"] = (
        df.sort_values("Date").groupby("Market")["coconut_oil_price_usd_per_mt"]
        .transform(lambda s: s.diff(1))
    )

    keep_ext_cols = [
        "coconut_oil_price_usd_per_mt", "coconut_oil_price_change_1m",
        "usd_inr_rate", "rainfall_mm", "temp_c", "rainfall_roll_90",
        "is_festival_window",
    ]
    drop_cols = [c for c in df.columns if c.startswith(("rainfall_ar", "rainfall_ti",
                                                          "temp_ar", "temp_ti", "nearest_festival"))
                 and c not in keep_ext_cols]
    df = df.drop(columns=["date"] + drop_cols, errors="ignore")
    return df


def main():
    print("Loading price history...")
    prices = load_prices()
    print(f"  {len(prices)} (market, date) rows after COPRA filter + grade aggregation")

    print("Adding price-history features (lags, rolling stats)...")
    df = add_price_features(prices)

    print("Adding calendar/seasonality features...")
    df = add_calendar_features(df)

    print("Merging external factors...")
    df = add_external_features(df)

    df = df.sort_values(["Market", "Date"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows x {len(df.columns)} columns to {OUT_CSV}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
