"""
Reliable copra price forecasting pipeline (rebuild).

What changed vs. the original 3/4 scripts
-----------------------------------------
1. DATA CLEANING: COPRA only, drop zero/negative prices, one consistent tender
   calendar per market (Tiptur Mon/Thu, Arasikere Tue/Fri), outlier rows
   (typos / tiny-arrival spikes that revert) removed, history starts 2010.
2. LEAK-FREE FEATURES: every feature is known BEFORE the tender. External data
   is lagged 7 days (publication delay) and rolling windows are in real days,
   not rows.
3. BASELINES FIRST: "no change" and a simple trend baseline are reported next
   to every model. A model must beat naive to be used.
4. HONEST EVALUATION: hyperparameters are chosen ONLY on pre-2020 data; 2020+
   is a pure walk-forward test (refit every 26 tenders). Bootstrap CI on the
   improvement over naive. Empirical prediction intervals + coverage check.

Run:  python reliable_pipeline.py
Out:  models/reliable_report.json, data/clean_features.csv
"""
import os, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "models")
os.makedirs(OUT, exist_ok=True)

START_YEAR = 2010
TENDER_DOW = {"TIPTUR": [0, 3], "ARSIKERE": [1, 4]}
TEST_START = pd.Timestamp("2020-01-01")
REFIT_EVERY = 26


# ----------------------------------------------------------------- cleaning
def load_clean_prices():
    r = pd.read_csv(os.path.join(DATA, "copra_prices_arasikere_tiptur.csv"))
    r["Date"] = pd.to_datetime(r["Date"], format="%d/%m/%Y")
    r["Market"] = r["Market"].str.upper().str.strip()
    r = r[r["Variety"].str.upper() == "COPRA"].copy()
    for c in ["Arrivals", "Min", "Max", "Modal"]:
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r.dropna(subset=["Modal", "Date"])
    r = r[(r["Modal"] > 0) & (r["Date"].dt.year >= START_YEAR)]

    def agg(g):
        w = g["Arrivals"].fillna(0).values + 1e-9
        return pd.Series({"Modal": np.average(g["Modal"], weights=w),
                          "Arrivals": g["Arrivals"].sum()})

    d = r.groupby(["Market", "Date"]).apply(agg, include_groups=False).reset_index()

    out = []
    for m, g in d.groupby("Market"):
        g = g[g["Date"].dt.dayofweek.isin(TENDER_DOW[m])].sort_values("Date").reset_index(drop=True)
        # Hampel-style outlier rule: a point is bad if it deviates >12% from the
        # median of its 2 neighbours on each side AND those neighbours agree.
        med = g["Modal"].rolling(5, center=True, min_periods=3).median()
        dev = (g["Modal"] / med - 1).abs()
        nb = pd.concat([g["Modal"].shift(2), g["Modal"].shift(1),
                        g["Modal"].shift(-1), g["Modal"].shift(-2)], axis=1)
        nb_spread = (nb.max(axis=1) / nb.min(axis=1) - 1)
        bad = (dev > 0.12) & (nb_spread < 0.10)
        g = g[~bad].reset_index(drop=True)
        g["removed_outliers"] = int(bad.sum())
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------- features
def load_external():
    e = pd.read_csv(os.path.join(DATA, "external_factors.csv"), parse_dates=["date"])
    e = e.set_index("date").sort_index().asfreq("D").ffill()
    # 7-day publication lag: on day t we only know data up to t-7
    e = e.shift(7)
    oil = e["coconut_oil_price_usd_per_mt"]
    fx = e["usd_inr_rate"]
    feats = pd.DataFrame({
        "oil_chg_30d": np.log(oil / oil.shift(30)),
        "oil_chg_90d": np.log(oil / oil.shift(90)),
        "fx_chg_30d": np.log(fx / fx.shift(30)),
    }, index=e.index)
    for m in ["arsikere", "tiptur"]:
        feats[f"rain_90d_{m}"] = e[f"rainfall_{m}_mm"].rolling(90, min_periods=30).sum()
        feats[f"rain_30d_{m}"] = e[f"rainfall_{m}_mm"].rolling(30, min_periods=10).sum()
    return feats


def build_features(prices, ext):
    frames = []
    both = prices.pivot(index="Date", columns="Market", values="Modal").sort_index().ffill()
    for m, g in prices.groupby("Market"):
        g = g.sort_values("Date").reset_index(drop=True)
        lp = np.log(g["Modal"])
        r = lp.diff()                                   # log-return into this tender
        f = pd.DataFrame({"Market": m, "Date": g["Date"], "Modal": g["Modal"]})
        f["last_price"] = g["Modal"].shift(1)
        f["target"] = lp - lp.shift(1)                  # what we predict
        for k in range(1, 5):
            f[f"ret_lag_{k}"] = r.shift(k)
        f["ret_sum_4"] = r.shift(1).rolling(4).sum()
        f["ret_sum_12"] = r.shift(1).rolling(12).sum()
        f["vol_12"] = r.shift(1).rolling(12).std()
        f["trend_gap_12"] = lp.shift(1) - lp.shift(1).rolling(12).mean()
        f["trend_gap_26"] = lp.shift(1) - lp.shift(1).rolling(26).mean()
        a = np.log1p(g["Arrivals"].clip(lower=0))
        f["arr_ratio"] = a.shift(1) - a.shift(1).rolling(12).mean()
        f["gap_days"] = g["Date"].diff().dt.days
        f["month_sin"] = np.sin(2 * np.pi * g["Date"].dt.month / 12)
        f["month_cos"] = np.cos(2 * np.pi * g["Date"].dt.month / 12)
        # other market: latest price STRICTLY before this date, as a spread
        other = [c for c in both.columns if c != m][0]
        o = both[other].shift(1)                         # strictly past
        o = o.reindex(g["Date"], method="ffill").values
        f["spread_other"] = np.log(g["Modal"].shift(1).values) - np.log(o)
        f["other_ret_4d"] = np.log(both[other].shift(1)).diff(2).reindex(g["Date"], method="ffill").values
        e = ext.reindex(g["Date"], method="ffill")
        e.index = f.index
        f["oil_chg_30d"] = e["oil_chg_30d"]
        f["oil_chg_90d"] = e["oil_chg_90d"]
        f["fx_chg_30d"] = e["fx_chg_30d"]
        f["rain_90d"] = e[f"rain_90d_{m.lower()}"]
        f["rain_30d"] = e[f"rain_30d_{m.lower()}"]
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=["target", "ret_lag_4", "ret_sum_12", "vol_12", "trend_gap_26"]).reset_index(drop=True)


FEATS = ["ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_4", "ret_sum_4", "ret_sum_12",
         "vol_12", "trend_gap_12", "trend_gap_26", "arr_ratio", "gap_days",
         "month_sin", "month_cos", "spread_other", "other_ret_4d",
         "oil_chg_30d", "oil_chg_90d", "fx_chg_30d", "rain_90d", "rain_30d"]
TREND_FEATS = ["ret_lag_1", "ret_lag_2", "ret_sum_4", "trend_gap_12", "spread_other", "other_ret_4d"]


# ------------------------------------------------------------------- models
def fit_predict(kind, Xtr, ytr, Xte, params=None):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from xgboost import XGBRegressor
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    ytr_c = ytr.clip(ytr.quantile(0.01), ytr.quantile(0.99))  # winsorise noisy targets
    if kind == "zero":
        return np.zeros(len(Xte))
    if kind == "ridge":
        m = make_pipeline(StandardScaler(), Ridge(alpha=(params or {}).get("alpha", 300)))
        m.fit(Xtr[TREND_FEATS], ytr_c)
        return m.predict(Xte[TREND_FEATS])
    if kind == "xgb":
        p = dict(n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.7,
                 colsample_bytree=0.7, min_child_weight=20, reg_lambda=10, random_state=42)
        p.update(params or {})
        m = XGBRegressor(**p)
        m.fit(Xtr[FEATS], ytr_c)
        return m.predict(Xte[FEATS])
    raise ValueError(kind)


def walk_forward(df, kind, start, end=None, params=None, refit_every=REFIT_EVERY):
    """Expanding-window, refit every N tenders. Returns predicted log-returns."""
    df = df.sort_values("Date").reset_index(drop=True)
    test_idx = np.where((df["Date"] >= start) & ((df["Date"] < end) if end is not None else True))[0]
    preds = pd.Series(np.nan, index=df.index)
    for s in range(0, len(test_idx), refit_every):
        chunk = test_idx[s:s + refit_every]
        tr = df.iloc[:chunk[0]]
        preds.iloc[chunk] = fit_predict(kind, tr[FEATS], tr["target"], df.iloc[chunk][FEATS], params)
    return df.loc[test_idx], preds.loc[test_idx].values


def score(test, pred):
    act = test["target"].values
    actual_p = test["last_price"].values * np.exp(act)
    pred_p = test["last_price"].values * np.exp(pred)
    mape = np.abs(pred_p - actual_p) / actual_p * 100
    nz = act != 0
    return {"mape": float(mape.mean()),
            "dir_acc": float((np.sign(pred[nz]) == np.sign(act[nz])).mean() * 100),
            "n": int(len(act))}, mape


def boot_ci(err_a, err_b, n=2000, block=8, seed=0):
    """Block bootstrap CI of mean(err_a - err_b); negative = a better than b."""
    rng = np.random.default_rng(seed)
    d = err_a - err_b
    nb = int(np.ceil(len(d) / block))
    stats = []
    for _ in range(n):
        starts = rng.integers(0, len(d) - block + 1, nb)
        stats.append(np.concatenate([d[s:s + block] for s in starts])[:len(d)].mean())
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


# --------------------------------------------------------------------- main
def main():
    prices = load_clean_prices()
    print("Clean price rows per market:\n", prices.groupby("Market").size().to_string())
    print("Outlier rows removed:", prices.groupby("Market")["removed_outliers"].first().to_dict())
    ext = load_external()
    df = build_features(prices, ext)
    df.to_csv(os.path.join(DATA, "clean_features.csv"), index=False)

    report = {}
    xgb_grid = [{"max_depth": 2, "n_estimators": 100}, {"max_depth": 2, "n_estimators": 250},
                {"max_depth": 3, "n_estimators": 150, "min_child_weight": 30}]
    ridge_grid = [{"alpha": 30}, {"alpha": 300}, {"alpha": 3000}]

    for m in sorted(df["Market"].unique()):
        g = df[df["Market"] == m].sort_values("Date").reset_index(drop=True)
        # ---- tune ONLY on 2015-2019 (never touches the 2020+ test period)
        tune_start, tune_end = pd.Timestamp("2015-01-01"), TEST_START
        best = {}
        for kind, grid in [("xgb", xgb_grid), ("ridge", ridge_grid)]:
            sc = []
            for p in grid:
                t, pr = walk_forward(g[g["Date"] < tune_end], kind, tune_start, params=p)
                sc.append((score(t, pr)[0]["mape"], p))
            best[kind] = min(sc, key=lambda x: x[0])[1]
        # ---- final out-of-sample test 2020+
        res, errs = {}, {}
        for kind, p in [("naive", None), ("ridge", best["ridge"]), ("xgb", best["xgb"])]:
            k = "zero" if kind == "naive" else kind
            test, pr = walk_forward(g, k, TEST_START, params=p)
            res[kind], errs[kind] = score(test, pr)
            if kind == "xgb":
                test_x, pred_x = test, pr
            if kind == "ridge":
                test_r, pred_r = test, pr
        # blend: half-shrunk average of ridge+xgb
        blend = 0.5 * (pred_r + pred_x) * 0.5
        res["blend"], errs["blend"] = score(test_x, blend)

        # ---- empirical interval: residual quantiles from the PRIOR 2 years only
        resid = test_x["target"].values - blend
        cover, w = [], []
        for i in range(len(resid)):
            hist = resid[max(0, i - 200):i]
            if len(hist) < 40:
                continue
            lo, hi = np.percentile(hist, [10, 90])
            cover.append(lo <= resid[i] <= hi)
            w.append(np.exp(hi) - np.exp(lo))
        interval = {"nominal": 80, "empirical_coverage": float(np.mean(cover) * 100),
                    "avg_width_pct": float(np.mean(w) * 100)}

        out = {"test_period": f"{TEST_START.date()} → {g['Date'].max().date()}",
               "tuned_params": best, "results": res,
               "skill_vs_naive_pct": {k: float((1 - res[k]["mape"] / res["naive"]["mape"]) * 100)
                                      for k in ["ridge", "xgb", "blend"]},
               "ci95_mape_diff_vs_naive_pctpts": {k: boot_ci(errs[k], errs["naive"])
                                                  for k in ["ridge", "xgb", "blend"]},
               "interval_80": interval}
        report[m] = out
        print(f"\n=== {m} (test {out['test_period']}, n={res['naive']['n']}) ===")
        for k in ["naive", "ridge", "xgb", "blend"]:
            r = res[k]
            extra = "" if k == "naive" else (f"  skill {out['skill_vs_naive_pct'][k]:+.1f}%  "
                                             f"CI95 diff {out['ci95_mape_diff_vs_naive_pctpts'][k][0]:+.3f}..{out['ci95_mape_diff_vs_naive_pctpts'][k][1]:+.3f}")
            print(f"  {k:6s} MAPE {r['mape']:.2f}%  dir {r['dir_acc']:.1f}%{extra}")
        print(f"  80% interval: coverage {interval['empirical_coverage']:.0f}%, avg width ±{interval['avg_width_pct']/2:.1f}%")

    with open(os.path.join(OUT, "reliable_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
