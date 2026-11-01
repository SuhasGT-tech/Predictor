"""
Build the model-ready feature table from:
  - data/copra_prices_arasikere_tiptur.csv  (APMC copra price history)
  - data/external_factors.csv                (global oil, FX, weather, festival)

Produces: data/features.csv - one row per (Market, Date) with:
  - Target: log-return into the tender (log(Modal / modal_lag_1))
  - Leak-free, stationary, scale-invariant features for machine learning
  - Contextual metadata columns required by dashboard and downstream reports

STATIONARY & SCALE-INVARIANT DESIGN:
All predictive price features are expressed as returns, percentage trend gaps,
or spreads rather than raw rupee levels. This allows the model to generalize
accurately across different price regimes (e.g. Rs 3,000 in early history to
Rs 30,000+ modern levels) without out-of-range tree distortions.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PRICES_CSV = os.path.join(DATA_DIR, "copra_prices_arasikere_tiptur.csv")
EXTERNAL_CSV = os.path.join(DATA_DIR, "external_factors.csv")
OUT_CSV = os.path.join(DATA_DIR, "features.csv")

START_YEAR = 2010
TENDER_DOW = {
    "TIPTUR": [0, 3],     # Monday, Thursday
    "ARSIKERE": [1, 4],   # Tuesday, Friday
}


def load_prices():
    """
    Loads raw APMC price history, filters to COPRA variety, aggregates
    any multi-grade tenders on the same day weighted by arrivals, restricts
    to official tender days, and filters recording anomalies.
    """
    df = pd.read_csv(PRICES_CSV)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["Market"] = df["Market"].str.upper().str.strip()

    # Focus on COPRA variety only (excluding MILLING and other sub-varieties)
    df = df[df["Variety"].str.upper() == "COPRA"].copy()

    for col in ["Arrivals", "Min", "Max", "Modal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Modal", "Date"])
    df = df[(df["Modal"] > 0) & (df["Date"].dt.year >= START_YEAR)]

    # Arrivals-weighted average Modal price for the day
    def weighted_avg(g):
        w = g["Arrivals"].fillna(0).values + 1e-9
        return pd.Series({
            "Modal": float(np.average(g["Modal"], weights=w)),
            "Min": float(g["Min"].min()) if g["Min"].notna().any() else np.nan,
            "Max": float(g["Max"].max()) if g["Max"].notna().any() else np.nan,
            "Arrivals": float(g["Arrivals"].sum()),
        })

    df = (
        df.groupby(["Market", "Date"])
        .apply(weighted_avg, include_groups=False)
        .reset_index()
    )

    # Filter to official APMC tender days and apply Hampel outlier cleaning
    cleaned = []
    for market, g in df.groupby("Market"):
        valid_days = TENDER_DOW.get(market, [0, 1, 2, 3, 4])
        g = g[g["Date"].dt.dayofweek.isin(valid_days)].sort_values("Date").reset_index(drop=True)

        # Hampel-style outlier filter: point deviates >12% from median of its
        # 2 neighbors on each side AND those neighbors agree (<10% spread)
        med = g["Modal"].rolling(5, center=True, min_periods=3).median()
        dev = (g["Modal"] / med - 1).abs()
        nb = pd.concat([g["Modal"].shift(2), g["Modal"].shift(1),
                        g["Modal"].shift(-1), g["Modal"].shift(-2)], axis=1)
        nb_spread = (nb.max(axis=1) / nb.min(axis=1) - 1)
        bad = (dev > 0.12) & (nb_spread < 0.10)
        n_bad = int(bad.sum())
        if n_bad > 0:
            print(f"  {market}: removed {n_bad} recording outlier spikes")
        g = g[~bad].reset_index(drop=True)
        cleaned.append(g)

    return pd.concat(cleaned, ignore_index=True).sort_values(["Market", "Date"]).reset_index(drop=True)


def add_price_features(df):
    """
    Constructs stationary return, momentum, volatility, and mean-reversion features.
    """
    out = []
    for market, g in df.groupby("Market"):
        g = g.sort_values("Date").copy()
        lp = np.log(g["Modal"])
        r = lp.diff()  # log-return between consecutive tenders

        g["modal_lag_1"] = g["Modal"].shift(1)
        g["modal_lag_2"] = g["Modal"].shift(2)
        g["modal_lag_3"] = g["Modal"].shift(3)
        g["modal_lag_4"] = g["Modal"].shift(4)
        g["modal_lag_5"] = g["Modal"].shift(5)

        # Target: log return to predict
        g["target"] = r

        # Past returns
        for k in range(1, 6):
            g[f"ret_lag_{k}"] = r.shift(k)

        # Short & medium momentum (sum of log returns)
        g["ret_sum_4"] = r.shift(1).rolling(4, min_periods=2).sum()
        g["ret_sum_12"] = r.shift(1).rolling(12, min_periods=4).sum()

        # Volatility
        g["vol_4"] = r.shift(1).rolling(4, min_periods=2).std()
        g["vol_12"] = r.shift(1).rolling(12, min_periods=4).std()

        # Mean-reversion trend gaps (log price vs log rolling means)
        roll_mean_12 = g["Modal"].shift(1).rolling(12, min_periods=4).mean()
        roll_mean_26 = g["Modal"].shift(1).rolling(26, min_periods=8).mean()
        g["trend_gap_12"] = np.log(g["modal_lag_1"]) - np.log(roll_mean_12)
        g["trend_gap_26"] = np.log(g["modal_lag_1"]) - np.log(roll_mean_26)

        # Supply signal: arrivals shock relative to 12-tender average
        a = np.log1p(g["Arrivals"].shift(1).clip(lower=0))
        g["arr_ratio"] = a - a.rolling(12, min_periods=4).mean()
        g["arrivals_lag_1"] = g["Arrivals"].shift(1)
        g["arrivals_roll_mean_4"] = g["Arrivals"].shift(1).rolling(4, min_periods=1).mean()

        # Days between consecutive tenders
        g["gap_days"] = g["Date"].diff().dt.days

        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_cross_market_features(df):
    """
    Computes spread and relative returns between Arasikere and Tiptur.
    Uses point-in-time merge_asof (strictly past date, no leakage).
    """
    out = []
    both = df.pivot(index="Date", columns="Market", values="Modal").sort_index().ffill()

    for market, g in df.groupby("Market"):
        g = g.sort_values("Date").copy()
        other = [c for c in both.columns if c != market][0]
        
        # Other market's most recent known price strictly before this date
        o = both[other].shift(1).reindex(g["Date"], method="ffill").values
        g["other_market_last_price"] = o
        
        # Log-spread: is current market cheap or expensive relative to neighbour?
        g["spread_other"] = np.log(g["modal_lag_1"].values) - np.log(o)
        
        # Neighbour market's 2-tender return
        other_ret = np.log(both[other].shift(1)).diff(2).reindex(g["Date"], method="ffill").values
        g["other_ret_4d"] = other_ret

        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_calendar_features(df):
    """
    Adds cyclical annual seasonality and calendar context.
    """
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["month_sin"] = np.sin(2.0 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * df["month"] / 12.0)
    df["is_tiptur_tenderday"] = df["day_of_week"].isin(TENDER_DOW["TIPTUR"]).astype(int)
    df["is_arsikere_tenderday"] = df["day_of_week"].isin(TENDER_DOW["ARSIKERE"]).astype(int)
    return df


def add_external_features(df):
    """
    Merges external factors (oil, FX, weather, festival) with a 7-day publication lag
    to guarantee strict leak-free predictive integrity.
    """
    if not os.path.exists(EXTERNAL_CSV):
        print(f"  NOTE: {EXTERNAL_CSV} not found - continuing with price/calendar features.")
        for col in ["oil_chg_30d", "oil_chg_90d", "fx_chg_30d", "rain_90d", "rain_30d",
                    "is_festival_window", "coconut_oil_price_usd_per_mt", "usd_inr_rate",
                    "rainfall_mm", "temp_c", "rainfall_roll_90"]:
            df[col] = 0.0
        return df

    ext = pd.read_csv(EXTERNAL_CSV, parse_dates=["date"])
    ext = ext.set_index("date").sort_index().asfreq("D").ffill()

    # 7-day publication delay: on tender day t we only know data published up to t-7
    ext_lagged = ext.shift(7)

    oil = ext_lagged["coconut_oil_price_usd_per_mt"]
    fx = ext_lagged["usd_inr_rate"]

    ext_feats = pd.DataFrame(index=ext_lagged.index)
    ext_feats["oil_chg_30d"] = np.log(oil / oil.shift(30))
    ext_feats["oil_chg_90d"] = np.log(oil / oil.shift(90))
    ext_feats["fx_chg_30d"] = np.log(fx / fx.shift(30))

    # Raw display fields (for dashboard UI cards)
    ext_feats["coconut_oil_price_usd_per_mt"] = oil
    ext_feats["usd_inr_rate"] = fx
    ext_feats["is_festival_window"] = ext["is_festival_window"].reindex(ext_feats.index, method="ffill").fillna(0)

    for m in ["arsikere", "tiptur"]:
        rain_col = f"rainfall_{m}_mm"
        temp_col = f"temp_{m}_c"
        if rain_col in ext_lagged:
            ext_feats[f"rain_90d_{m}"] = ext_lagged[rain_col].rolling(90, min_periods=30).sum()
            ext_feats[f"rain_30d_{m}"] = ext_lagged[rain_col].rolling(30, min_periods=10).sum()
            ext_feats[f"rainfall_{m}_mm"] = ext_lagged[rain_col]
        else:
            ext_feats[f"rain_90d_{m}"] = 0.0
            ext_feats[f"rain_30d_{m}"] = 0.0
            ext_feats[f"rainfall_{m}_mm"] = 0.0

        if temp_col in ext_lagged:
            ext_feats[f"temp_{m}_c"] = ext_lagged[temp_col]
        else:
            ext_feats[f"temp_{m}_c"] = 25.0

    merged_rows = []
    for market, g in df.groupby("Market"):
        g = g.sort_values("Date").copy()
        m_lower = market.lower()
        sub_ext = ext_feats.reindex(g["Date"], method="ffill")
        
        g["oil_chg_30d"] = sub_ext["oil_chg_30d"].values
        g["oil_chg_90d"] = sub_ext["oil_chg_90d"].values
        g["fx_chg_30d"] = sub_ext["fx_chg_30d"].values
        g["rain_90d"] = sub_ext.get(f"rain_90d_{m_lower}", pd.Series(0.0, index=g.index)).values
        g["rain_30d"] = sub_ext.get(f"rain_30d_{m_lower}", pd.Series(0.0, index=g.index)).values
        
        # Display columns for dashboard
        g["coconut_oil_price_usd_per_mt"] = sub_ext["coconut_oil_price_usd_per_mt"].values
        g["coconut_oil_price_change_1m"] = sub_ext["oil_chg_30d"].values
        g["usd_inr_rate"] = sub_ext["usd_inr_rate"].values
        g["is_festival_window"] = sub_ext["is_festival_window"].values
        g["rainfall_mm"] = sub_ext.get(f"rainfall_{m_lower}_mm", pd.Series(0.0, index=g.index)).values
        g["temp_c"] = sub_ext.get(f"temp_{m_lower}_c", pd.Series(25.0, index=g.index)).values
        g["rainfall_roll_90"] = g["rain_90d"]

        merged_rows.append(g)

    return pd.concat(merged_rows, ignore_index=True)


def main():
    print("Loading and cleaning APMC tender price history (COPRA only)...")
    prices = load_prices()
    print(f"  {len(prices)} clean tender rows across markets since {START_YEAR}")

    print("Adding stationary price return and momentum features...")
    df = add_price_features(prices)

    print("Adding cross-market spread and relative returns...")
    df = add_cross_market_features(df)

    print("Adding cyclical calendar features...")
    df = add_calendar_features(df)

    print("Merging leak-free external factors...")
    df = add_external_features(df)

    # Require minimum lookback so trend features are fully formed
    clean_valid = df.dropna(subset=["target", "ret_lag_4", "ret_sum_12", "vol_12", "trend_gap_26"]).reset_index(drop=True)
    clean_valid = clean_valid.sort_values(["Market", "Date"]).reset_index(drop=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    clean_valid.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(clean_valid)} rows x {len(clean_valid.columns)} columns to {OUT_CSV}")
    print("Market row counts:\n", clean_valid.groupby("Market").size().to_string())


if __name__ == "__main__":
    main()
