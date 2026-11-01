"""
Generate dashboard.html - a single static HTML file you open in your
browser after each pipeline run. Shows:
  - Predicted Modal price for the NEXT upcoming tender day, per market
    (Tiptur: Mon/Thu, Arasikere: Tue/Fri)
  - Recent price history chart with the prediction plotted alongside
  - What's driving the prediction (feature importances)
  - The external factors currently feeding the model
  - Model validation accuracy (MAE / MAPE) from the last training run

USAGE
-----
    python 5_generate_dashboard.py

Then open dashboard.html in any browser (double-click it).
"""

import os
import sys
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib

# Ensure model_utils can be imported both when run directly and when imported
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import model_utils

try:
    import holidays
except ImportError:
    holidays = None

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")
OUT_HTML = os.path.join(BASE_DIR, "dashboard.html")

# Tender days: Monday=0 ... Sunday=6
TENDER_DAYS = {
    "TIPTUR": [0, 3],     # Mon, Thu
    "ARSIKERE": [1, 4],   # Tue, Fri
}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def next_tender_date(market, from_date):
    days = TENDER_DAYS[market]
    for offset in range(1, 8):
        candidate = from_date + timedelta(days=offset)
        if candidate.weekday() in days:
            return candidate
    raise RuntimeError("no tender day found in the next week (shouldn't happen)")


def is_festival_window_for(date):
    if holidays is None:
        return 0, ""
    try:
        in_holidays = holidays.India(years=[date.year, date.year + 1])
        for hdate, name in in_holidays.items():
            hdate = pd.Timestamp(hdate)
            if hdate - timedelta(days=10) <= pd.Timestamp(date) <= hdate:
                return 1, name
    except Exception:
        pass
    return 0, ""


def build_future_row(market_hist, feature_cols, target_date, metrics, all_df=None):
    """
    Constructs the stationary feature row for the next tender date, using
    the latest actual tender prices, returns, momentum, spreads, and macro factors.
    """
    g = market_hist.sort_values("Date").reset_index(drop=True)
    last_p = float(g["Modal"].iloc[-1])
    last_dt = g["Date"].iloc[-1]
    
    lp = np.log(g["Modal"])
    r = lp.diff()
    
    # Identify neighbour market price and return for spread features
    current_market = g["Market"].iloc[-1]
    other_market = "TIPTUR" if current_market == "ARSIKERE" else "ARSIKERE"
    if all_df is not None and other_market in all_df["Market"].values:
        other_g = all_df[all_df["Market"] == other_market].sort_values("Date")
    else:
        other_g = g
    other_p = float(other_g["Modal"].iloc[-1]) if len(other_g) else last_p
    other_ret = float(np.log(other_g["Modal"]).diff(2).iloc[-1]) if len(other_g) >= 3 else 0.0

    festival_flag, festival_name = is_festival_window_for(target_date)

    row = {
        "ret_lag_1": float(r.iloc[-1]) if len(r) >= 1 else 0.0,
        "ret_lag_2": float(r.iloc[-2]) if len(r) >= 2 else 0.0,
        "ret_lag_3": float(r.iloc[-3]) if len(r) >= 3 else 0.0,
        "ret_lag_4": float(r.iloc[-4]) if len(r) >= 4 else 0.0,
        "ret_lag_5": float(r.iloc[-5]) if len(r) >= 5 else 0.0,
        "ret_sum_4": float(r.iloc[-4:].sum()) if len(r) >= 4 else 0.0,
        "ret_sum_12": float(r.iloc[-12:].sum()) if len(r) >= 12 else 0.0,
        "vol_4": float(r.iloc[-4:].std()) if len(r) >= 4 and not pd.isna(r.iloc[-4:].std()) else 0.02,
        "vol_12": float(r.iloc[-12:].std()) if len(r) >= 12 and not pd.isna(r.iloc[-12:].std()) else 0.03,
        "trend_gap_12": float(lp.iloc[-1] - np.log(g["Modal"].iloc[-12:].mean())) if len(g) >= 12 else 0.0,
        "trend_gap_26": float(lp.iloc[-1] - np.log(g["Modal"].iloc[-26:].mean())) if len(g) >= 26 else 0.0,
        "arr_ratio": float(np.log1p(g["Arrivals"].clip(lower=0)).iloc[-1] - np.log1p(g["Arrivals"].clip(lower=0)).iloc[-12:].mean()) if len(g) >= 12 else 0.0,
        "gap_days": float((target_date - last_dt.to_pydatetime()).days),
        "spread_other": float(lp.iloc[-1] - np.log(other_p)),
        "other_ret_4d": other_ret,
        "month_sin": float(np.sin(2.0 * np.pi * target_date.month / 12.0)),
        "month_cos": float(np.cos(2.0 * np.pi * target_date.month / 12.0)),
        "oil_chg_30d": float(g["oil_chg_30d"].dropna().iloc[-1]) if "oil_chg_30d" in g and g["oil_chg_30d"].notna().any() else 0.0,
        "oil_chg_90d": float(g["oil_chg_90d"].dropna().iloc[-1]) if "oil_chg_90d" in g and g["oil_chg_90d"].notna().any() else 0.0,
        "fx_chg_30d": float(g["fx_chg_30d"].dropna().iloc[-1]) if "fx_chg_30d" in g and g["fx_chg_30d"].notna().any() else 0.0,
        "rain_90d": float(g["rain_90d"].dropna().iloc[-1]) if "rain_90d" in g and g["rain_90d"].notna().any() else 0.0,
        "rain_30d": float(g["rain_30d"].dropna().iloc[-1]) if "rain_30d" in g and g["rain_30d"].notna().any() else 0.0,
        "is_festival_window": float(festival_flag),
    }

    medians = metrics.get("feature_medians", {})
    for col in feature_cols:
        if col not in row or pd.isna(row[col]):
            row[col] = float(medians.get(col, 0.0))

    return pd.DataFrame([{c: row[c] for c in feature_cols}]), festival_name


MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


def compute_seasonality(g):
    """
    For each historical row, express its price as a ratio to that YEAR's
    average price (detrending 15+ years of growth to compare pure calendar-month
    seasonality).
    """
    g = g.copy()
    g["year"] = g["Date"].dt.year
    yearly_mean = g.groupby("year")["Modal"].transform("mean")
    g["seasonal_ratio"] = g["Modal"] / yearly_mean

    by_month = g.groupby(g["Date"].dt.month)["seasonal_ratio"].agg(["mean", "count"])
    by_month = by_month[by_month["count"] >= 3]

    result = [
        {"month": int(m), "month_name": MONTH_NAMES[int(m)],
         "index": round((row["mean"] - 1.0) * 100.0, 1)}
        for m, row in by_month.iterrows()
    ]
    result.sort(key=lambda r: r["month"])

    if result:
        best = max(result, key=lambda r: r["index"])
        worst = min(result, key=lambda r: r["index"])
    else:
        best = worst = None

    return result, best, worst


def forecast_horizon(market, g, model, feature_cols, metrics, n_tenders=10, all_df=None):
    """
    Recursively forecasts the next `n_tenders` tender days for this market,
    dynamically rolling forward returns and trend gaps.
    """
    g = g.sort_values("Date").reset_index(drop=True)
    medians = metrics.get("feature_medians", {})

    recent_prices = [float(x) for x in g["Modal"].tail(30).tolist()]
    recent_arrivals = [float(x) for x in g["Arrivals"].tail(12).tolist()]
    current_date = datetime.now()

    other_market = "TIPTUR" if market == "ARSIKERE" else "ARSIKERE"
    if all_df is not None and other_market in all_df["Market"].values:
        other_g = all_df[all_df["Market"] == other_market].sort_values("Date")
    else:
        other_g = g
    other_price = float(other_g["Modal"].iloc[-1]) if len(other_g) else recent_prices[-1]
    other_ret = float(np.log(other_g["Modal"]).diff(2).iloc[-1]) if len(other_g) >= 3 else 0.0

    static_ext = {
        "oil_chg_30d": float(g["oil_chg_30d"].dropna().iloc[-1]) if "oil_chg_30d" in g and g["oil_chg_30d"].notna().any() else float(medians.get("oil_chg_30d", 0.0)),
        "oil_chg_90d": float(g["oil_chg_90d"].dropna().iloc[-1]) if "oil_chg_90d" in g and g["oil_chg_90d"].notna().any() else float(medians.get("oil_chg_90d", 0.0)),
        "fx_chg_30d": float(g["fx_chg_30d"].dropna().iloc[-1]) if "fx_chg_30d" in g and g["fx_chg_30d"].notna().any() else float(medians.get("fx_chg_30d", 0.0)),
        "rain_90d": float(g["rain_90d"].dropna().iloc[-1]) if "rain_90d" in g and g["rain_90d"].notna().any() else float(medians.get("rain_90d", 0.0)),
        "rain_30d": float(g["rain_30d"].dropna().iloc[-1]) if "rain_30d" in g and g["rain_30d"].notna().any() else float(medians.get("rain_30d", 0.0)),
    }

    forecast_points = []
    festival_hits = []

    for _ in range(n_tenders):
        target_date = next_tender_date(market, current_date)
        festival_flag, festival_name = is_festival_window_for(target_date)

        rets = np.diff(np.log(recent_prices))
        lag_rets = rets[-5:][::-1] if len(rets) >= 5 else [0.0]*5

        roll4_ret = float(rets[-4:].sum()) if len(rets) >= 4 else 0.0
        roll12_ret = float(rets[-12:].sum()) if len(rets) >= 12 else 0.0
        vol4 = float(np.std(rets[-4:])) if len(rets) >= 4 else 0.02
        vol12 = float(np.std(rets[-12:])) if len(rets) >= 12 else 0.03

        last_price = recent_prices[-1]
        mean12 = float(np.mean(recent_prices[-12:]))
        mean26 = float(np.mean(recent_prices[-26:]))
        trend_gap12 = float(np.log(last_price) - np.log(mean12))
        trend_gap26 = float(np.log(last_price) - np.log(mean26))

        arr_series = pd.Series(recent_arrivals)
        arr_ratio = float(np.log1p(arr_series.iloc[-1]) - np.log1p(arr_series).mean())

        row = {
            "ret_lag_1": float(lag_rets[0]),
            "ret_lag_2": float(lag_rets[1]) if len(lag_rets) > 1 else 0.0,
            "ret_lag_3": float(lag_rets[2]) if len(lag_rets) > 2 else 0.0,
            "ret_lag_4": float(lag_rets[3]) if len(lag_rets) > 3 else 0.0,
            "ret_lag_5": float(lag_rets[4]) if len(lag_rets) > 4 else 0.0,
            "ret_sum_4": roll4_ret,
            "ret_sum_12": roll12_ret,
            "vol_4": vol4,
            "vol_12": vol12,
            "trend_gap_12": trend_gap12,
            "trend_gap_26": trend_gap26,
            "arr_ratio": arr_ratio,
            "gap_days": float((target_date - current_date).days),
            "spread_other": float(np.log(last_price) - np.log(other_price)),
            "other_ret_4d": other_ret,
            "month_sin": float(np.sin(2.0 * np.pi * target_date.month / 12.0)),
            "month_cos": float(np.cos(2.0 * np.pi * target_date.month / 12.0)),
            "is_festival_window": float(festival_flag),
            **static_ext,
        }

        for col in feature_cols:
            if col not in row or pd.isna(row[col]):
                row[col] = float(medians.get(col, 0.0))

        X = pd.DataFrame([{c: row[c] for c in feature_cols}])
        pred_log_return = float(model.predict(X)[0])
        # Smooth bounds to prevent compound extrapolation explosion
        pred_log_return = float(np.clip(pred_log_return, -0.04, 0.04))
        pred = last_price * np.exp(pred_log_return)

        forecast_points.append({
            "date": target_date.strftime("%Y-%m-%d"),
            "modal": round(pred),
            "festival": festival_name if festival_flag else None,
        })
        if festival_flag:
            festival_hits.append({"date": target_date.strftime("%Y-%m-%d"), "name": festival_name})

        recent_prices.append(pred)
        recent_arrivals.append(recent_arrivals[-1] if recent_arrivals else 0.0)
        current_date = target_date

    return forecast_points, festival_hits


def predict_market(market, df):
    model_path = os.path.join(MODELS_DIR, f"model_{market.lower()}.joblib")
    cols_path = os.path.join(MODELS_DIR, f"feature_cols_{market.lower()}.json")
    metrics_path = os.path.join(MODELS_DIR, f"metrics_{market.lower()}.json")

    if not (os.path.exists(model_path) and os.path.exists(cols_path)):
        return None

    model = joblib.load(model_path)
    with open(cols_path) as f:
        feature_cols = json.load(f)
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    g = df[df["Market"] == market].copy().sort_values("Date").reset_index(drop=True)
    target_date = next_tender_date(market, datetime.now())

    X_future, festival_name = build_future_row(g, feature_cols, target_date, metrics, all_df=df)
    lag1_for_pred = float(g["Modal"].iloc[-1])
    pred_log_return = float(model.predict(X_future)[0])
    pred = lag1_for_pred * np.exp(pred_log_return)

    recent = g.sort_values("Date").tail(60)[["Date", "Modal"]]
    history = [{"date": d.strftime("%Y-%m-%d"), "modal": float(m)} for d, m in
               zip(recent["Date"], recent["Modal"])]

    seasonality, best_month, worst_month = compute_seasonality(g)
    forecast_points, festival_hits = forecast_horizon(market, g, model, feature_cols, metrics, n_tenders=10, all_df=df)

    last_actual = float(g["Modal"].iloc[-1])
    last_date = g["Date"].iloc[-1]

    return {
        "market": market,
        "target_date": target_date.strftime("%Y-%m-%d"),
        "target_day_name": DAY_NAMES[target_date.weekday()],
        "prediction": round(pred),
        "last_actual": round(float(last_actual)),
        "last_date": last_date.strftime("%Y-%m-%d"),
        "change": round(pred - float(last_actual)),
        "change_pct": round((pred - float(last_actual)) / float(last_actual) * 100.0, 1) if last_actual else 0.0,
        "mae": metrics.get("mae"),
        "mape": metrics.get("mape"),
        "model_type": metrics.get("model_type", "model"),
        "top_features": metrics.get("top_features", []),
        "history": history,
        "festival_note": festival_name,
        "seasonality": seasonality,
        "best_month": best_month,
        "worst_month": worst_month,
        "forecast_points": forecast_points,
        "festival_hits": festival_hits,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Copra Price Board -- Arasikere &amp; Tiptur</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
<style>
  :root {
    --husk-brown: #3d2b1f;
    --copra-cream: #f2e8d5;
    --coir-gold: #b8862b;
    --leaf-green: #4a5d3a;
    --board-black: #1c1712;
    --up-red: #a63d2f;
    --down-green: #3f6b4a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--board-black);
    color: var(--copra-cream);
    font-family: 'Courier New', ui-monospace, monospace;
    padding: 24px 16px 60px;
  }
  .wrap { max-width: 1080px; margin: 0 auto; }
  header {
    text-align: center;
    border-bottom: 3px double var(--coir-gold);
    padding-bottom: 16px;
    margin-bottom: 28px;
  }
  header h1 {
    font-family: Georgia, 'Times New Roman', serif;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-size: 28px;
    margin: 0 0 4px;
    color: var(--coir-gold);
  }
  header p { margin: 0; color: #a99a80; font-size: 13px; letter-spacing: 1px; }

  .boards { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 28px; }
  @media (max-width: 760px) { .boards { grid-template-columns: 1fr; } }

  .board {
    background: linear-gradient(180deg, #2a1f16, #1f170f);
    border: 1px solid #5a4530;
    border-radius: 6px;
    padding: 20px;
  }
  .board-head {
    display: flex; justify-content: space-between; align-items: baseline;
    border-bottom: 1px dashed #5a4530; padding-bottom: 10px; margin-bottom: 14px;
  }
  .board-head h2 {
    margin: 0; font-family: Georgia, serif; font-size: 20px;
    letter-spacing: 2px; color: var(--copra-cream);
  }
  .tender-tag {
    font-size: 11px; color: var(--board-black); background: var(--coir-gold);
    padding: 3px 8px; border-radius: 3px; letter-spacing: 1px;
  }
  .flap {
    display: flex; gap: 6px; justify-content: center; margin: 14px 0 10px;
  }
  .flap span {
    display: inline-block; background: #0e0b08; color: var(--coir-gold);
    font-size: 40px; font-weight: bold; padding: 6px 4px; border-radius: 4px;
    min-width: 30px; text-align: center; border: 1px solid #4a3826;
    text-shadow: 0 0 6px rgba(184,134,43,0.5);
  }
  .price-label { text-align: center; font-size: 12px; color: #a99a80; letter-spacing: 2px; margin-bottom: 2px; }
  .predicted-for { text-align: center; font-size: 13px; color: var(--copra-cream); margin-bottom: 16px; }
  .predicted-for b { color: var(--coir-gold); }
  .delta { text-align: center; font-size: 15px; margin-bottom: 6px; }
  .delta.up { color: var(--up-red); }
  .delta.down { color: var(--down-green); }
  .meta-row { display: flex; justify-content: space-between; font-size: 12px; color: #a99a80; margin-top: 14px; }
  .accuracy { font-size: 11px; color: #7d715e; text-align: center; margin-top: 10px; }
  .festival-note {
    margin-top: 10px; font-size: 12px; text-align: center; color: var(--leaf-green);
    background: rgba(74,93,58,0.15); border: 1px solid var(--leaf-green); border-radius: 4px; padding: 6px;
  }

  .outlook-list { display: flex; flex-direction: column; gap: 10px; font-size: 13px; }
  .outlook-item { display: flex; gap: 10px; align-items: flex-start; line-height: 1.4; }
  .outlook-icon { flex-shrink: 0; font-size: 15px; }
  .outlook-item b { color: var(--coir-gold); }
  .outlook-item.upcoming-festival { color: var(--leaf-green); }
  .outlook-item.upcoming-festival b { color: var(--leaf-green); }

  canvas { background: #241b12; border-radius: 6px; }
  .chart-wrap { background: linear-gradient(180deg, #2a1f16, #1f170f); border: 1px solid #5a4530;
    border-radius: 6px; padding: 16px; margin-bottom: 20px; }
  .chart-wrap h3 { margin: 0 0 10px; font-family: Georgia, serif; font-size: 15px; color: var(--coir-gold); letter-spacing: 1px; }

  .features { display: flex; flex-direction: column; gap: 6px; }
  .feature-bar-row { display: grid; grid-template-columns: 160px 1fr 50px; align-items: center; gap: 8px; font-size: 12px; }
  .feature-bar-bg { background: #14100b; border-radius: 3px; height: 12px; overflow: hidden; }
  .feature-bar-fill { background: var(--coir-gold); height: 100%; }

  footer { text-align: center; color: #675a48; font-size: 11px; margin-top: 30px; letter-spacing: 1px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Copra Tender Price Board</h1>
    <p>Arasikere (Tue &amp; Fri) &middot; Tiptur (Mon &amp; Thu) &middot; generated __GENERATED_AT__</p>
  </header>

  <div class="boards">
    __BOARD_CARDS__
  </div>

  <div class="chart-wrap">
    <h3>Price Trend + 10-Tender Outlook (dashed = forecast)</h3>
    <canvas id="trendChart" height="90"></canvas>
  </div>

  <div class="boards">
    __OUTLOOK_CARDS__
  </div>

  <div class="chart-wrap">
    <h3>Seasonality — average price vs. that year's own average, by month</h3>
    <canvas id="seasonChart" height="80"></canvas>
  </div>

  <div class="boards">
    __FEATURE_CARDS__
  </div>

  <footer>
    Predictions are a statistical estimate from historical APMC patterns &mdash;
    treat as one input alongside your own market knowledge. The 10-tender outlook assumes
    weather/global-price/fx stay near their current levels, so treat it as directional,
    not a precise multi-week quote.<br>
    Re-run 1_scrape_incremental.py &rarr; 2_fetch_external_factors.py &rarr; 3_build_features.py &rarr;
    4_train_model.py &rarr; 5_generate_dashboard.py to refresh.
  </footer>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-adapter-date-fns/3.0.0/chartjs-adapter-date-fns.bundle.min.js"></script>
<script>
const trendData = __TREND_DATA__;
const seasonData = __SEASON_DATA__;

const ctx = document.getElementById('trendChart');
const trendDatasets = [];
trendData.forEach(d => {
  const color = d.market === 'TIPTUR' ? '#b8862b' : '#4a5d3a';
  trendDatasets.push({
    label: d.market + ' (actual)',
    data: d.history.map(h => ({x: h.date, y: h.modal})),
    borderColor: color, backgroundColor: 'transparent', tension: 0.2, pointRadius: 2,
  });
  const bridge = d.history.length ? [d.history[d.history.length - 1]] : [];
  trendDatasets.push({
    label: d.market + ' (forecast)',
    data: bridge.concat(d.forecast).map(h => ({x: h.date, y: h.modal})),
    borderColor: color, backgroundColor: 'transparent', borderDash: [6, 4],
    tension: 0.2, pointRadius: 2,
  });
});

new Chart(ctx, {
  type: 'line',
  data: { datasets: trendDatasets },
  options: {
    responsive: true,
    scales: {
      x: { type: 'time', time: { unit: 'month' }, ticks: { color: '#a99a80' }, grid: { color: '#3a2c1d' } },
      y: { ticks: { color: '#a99a80' }, grid: { color: '#3a2c1d' }, title: { display: true, text: 'Rs / Quintal', color: '#a99a80' } }
    },
    plugins: { legend: { labels: { color: '#f2e8d5', font: { size: 10 } } } }
  }
});

new Chart(document.getElementById('seasonChart'), {
  type: 'bar',
  data: {
    labels: seasonData.labels,
    datasets: seasonData.datasets,
  },
  options: {
    responsive: true,
    scales: {
      x: { ticks: { color: '#a99a80' }, grid: { color: '#3a2c1d' } },
      y: { ticks: { color: '#a99a80' }, grid: { color: '#3a2c1d' },
           title: { display: true, text: '% vs. that year\\'s average', color: '#a99a80' } }
    },
    plugins: { legend: { labels: { color: '#f2e8d5' } } }
  }
});
</script>
</body>
</html>
"""

BOARD_CARD_TEMPLATE = """
<div class="board">
  <div class="board-head">
    <h2>{market}</h2>
    <span class="tender-tag">NEXT TENDER: {target_day_name} {target_date}</span>
  </div>
  <div class="price-label">PREDICTED MODAL PRICE</div>
  <div class="flap">{digit_spans}</div>
  <div class="predicted-for">for <b>{target_day_name}, {target_date}</b></div>
  <div class="delta {delta_class}">{delta_arrow} Rs {delta_abs:,} ({change_pct}%) vs last actual (Rs {last_actual:,} on {last_date})</div>
  {festival_html}
  <div class="meta-row">
    <span>Model: {model_type}</span>
    <span>Walk-forward MAPE: {mape}</span>
  </div>
  <div class="accuracy">Typical error: &plusmn; Rs {mae:,} / quintal (averaged across walk-forward validation folds)</div>
</div>
"""

FEATURE_CARD_TEMPLATE = """
<div class="board">
  <div class="board-head"><h2 style="font-size:15px;">{market} -- what's driving it</h2></div>
  <div class="features">{feature_rows}</div>
</div>
"""

OUTLOOK_CARD_TEMPLATE = """
<div class="board">
  <div class="board-head"><h2 style="font-size:15px;">{market} -- outlook</h2></div>
  <div class="outlook-list">{items}</div>
</div>
"""


def digit_spans(value):
    return "".join(f"<span>{ch}</span>" for ch in f"{value:,}")


def render_board_card(res):
    if res is None:
        return "<div class='board'><p>No model available yet -- run steps 1-4 first.</p></div>"

    up = res["change"] >= 0
    festival_html = ""
    if res.get("festival_note"):
        festival_html = f"<div class='festival-note'>Festival demand window: {res['festival_note']}</div>"

    return BOARD_CARD_TEMPLATE.format(
        market=res["market"],
        target_day_name=res["target_day_name"],
        target_date=res["target_date"],
        digit_spans=digit_spans(res["prediction"]),
        delta_class="up" if up else "down",
        delta_arrow="&#9650;" if up else "&#9660;",
        delta_abs=abs(res["change"]),
        change_pct=res["change_pct"],
        last_actual=res["last_actual"],
        last_date=res["last_date"],
        model_type=res["model_type"],
        mape=f"{res['mape']:.1f}%" if res.get("mape") is not None else "n/a",
        mae=int(res["mae"]) if res.get("mae") is not None else 0,
        festival_html=festival_html,
    )


FEATURE_LABELS = {
    "spread_other": "Spread vs other market",
    "other_ret_4d": "Other market price momentum",
    "trend_gap_12": "6-week trend gap (mean reversion)",
    "trend_gap_26": "Quarterly trend gap",
    "vol_12": "Recent price volatility",
    "ret_lag_1": "Last tender return",
    "ret_lag_2": "2nd last tender return",
    "ret_sum_4": "2-week price momentum",
    "ret_sum_12": "6-week price momentum",
    "arr_ratio": "Arrival volume shock",
    "oil_chg_30d": "Global coconut oil 30d change",
    "oil_chg_90d": "Global coconut oil 90d change",
    "fx_chg_30d": "USD/INR exchange rate change",
    "rain_90d": "90-day cumulative rainfall",
    "rain_30d": "30-day cumulative rainfall",
    "gap_days": "Days since previous tender",
    "month_sin": "Annual seasonal cycle",
    "month_cos": "Annual seasonal cycle",
    "is_festival_window": "Festival demand window",
}


def render_feature_card(res):
    if res is None or not res.get("top_features"):
        return ""
    max_imp = max(f["importance"] for f in res["top_features"]) or 1.0
    rows = ""
    for f in res["top_features"][:6]:
        pct = f["importance"] / max_imp * 100.0
        label = FEATURE_LABELS.get(f["feature"], f["feature"])
        rows += (
            f"<div class='feature-bar-row' title='{f['feature']}'><span>{label}</span>"
            f"<div class='feature-bar-bg'><div class='feature-bar-fill' style='width:{pct:.0f}%'></div></div>"
            f"<span>{f['importance']:.2f}</span></div>"
        )
    return FEATURE_CARD_TEMPLATE.format(market=res["market"], feature_rows=rows)


def render_outlook_card(res):
    if res is None:
        return ""
    items = []

    if res.get("best_month"):
        bm = res["best_month"]
        sign = "+" if bm["index"] >= 0 else ""
        items.append(
            f"<div class='outlook-item'><span class='outlook-icon'>&#128200;</span>"
            f"<span>Historically, <b>{bm['month_name']}</b> tends to run <b>{sign}{bm['index']}%</b> "
            f"above that year's average price — the strongest month on record for {res['market'].title()}.</span></div>"
        )
    if res.get("worst_month"):
        wm = res["worst_month"]
        items.append(
            f"<div class='outlook-item'><span class='outlook-icon'>&#128201;</span>"
            f"<span>Historically weakest: <b>{wm['month_name']}</b> ({wm['index']}% vs. yearly average) — "
            f"selling then has usually meant a lower price.</span></div>"
        )

    if res.get("forecast_points"):
        last = res["forecast_points"][-1]
        drift = last["modal"] - res["last_actual"]
        direction = "up" if drift > 0 else "down" if drift < 0 else "flat"
        items.append(
            f"<div class='outlook-item'><span class='outlook-icon'>&#128336;</span>"
            f"<span>Over the next {len(res['forecast_points'])} tenders (through {last['date']}), "
            f"the model's directional read is <b>trending {direction}</b> "
            f"(Rs {abs(drift):,} {'higher' if drift > 0 else 'lower' if drift < 0 else 'change'} by then) — "
            f"treat this as a rough trend, not a locked-in price.</span></div>"
        )

    seen_festivals = set()
    for hit in (res.get("festival_hits") or []):
        if hit["name"] in seen_festivals:
            continue
        seen_festivals.add(hit["name"])
        items.append(
            f"<div class='outlook-item upcoming-festival'><span class='outlook-icon'>&#127881;</span>"
            f"<span><b>{hit['name']}</b> falls around {hit['date']} — festival demand windows have "
            f"historically coincided with firmer coconut/copra prices.</span></div>"
        )
        if len(seen_festivals) >= 3:
            break

    if not items:
        items.append("<div class='outlook-item'><span>Not enough history yet to build an outlook.</span></div>")

    return OUTLOOK_CARD_TEMPLATE.format(market=res["market"], items="".join(items))


def main():
    if not os.path.exists(FEATURES_CSV):
        print(f"{FEATURES_CSV} not found -- run 3_build_features.py first.")
        return

    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"])
    results = []
    for market in ["TIPTUR", "ARSIKERE"]:
        if market in df["Market"].unique():
            res = predict_market(market, df)
            results.append(res)
            if res:
                print(f"{market}: predicted Rs {res['prediction']:,} for "
                      f"{res['target_day_name']} {res['target_date']} "
                      f"(last actual Rs {res['last_actual']:,} on {res['last_date']})")
        else:
            results.append(None)

    board_cards = "".join(render_board_card(r) for r in results)
    feature_cards = "".join(render_feature_card(r) for r in results)
    outlook_cards = "".join(render_outlook_card(r) for r in results)
    trend_data = [
        {"market": r["market"], "history": r["history"], "forecast": r["forecast_points"]}
        for r in results if r is not None
    ]

    all_months = list(range(1, 13))
    season_datasets = []
    colors = {"TIPTUR": "#b8862b", "ARSIKERE": "#4a5d3a"}
    for r in results:
        if r is None:
            continue
        by_month = {s["month"]: s["index"] for s in r["seasonality"]}
        season_datasets.append({
            "label": r["market"],
            "data": [by_month.get(m, 0) for m in all_months],
            "backgroundColor": colors.get(r["market"], "#888"),
        })
    season_data = {
        "labels": [MONTH_NAMES[m][:3] for m in all_months],
        "datasets": season_datasets,
    }

    html = (
        HTML_TEMPLATE
        .replace("__GENERATED_AT__", datetime.now().strftime("%Y-%m-%d %H:%M"))
        .replace("__BOARD_CARDS__", board_cards)
        .replace("__FEATURE_CARDS__", feature_cards)
        .replace("__OUTLOOK_CARDS__", outlook_cards)
        .replace("__TREND_DATA__", json.dumps(trend_data))
        .replace("__SEASON_DATA__", json.dumps(season_data))
    )

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    # Also sync docs/index.html if docs directory exists for GitHub Pages
    docs_dir = os.path.join(BASE_DIR, "docs")
    if os.path.exists(docs_dir):
        with open(os.path.join(docs_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(html)

    print(f"\nDashboard saved to {OUT_HTML} -- open it in your browser.")


if __name__ == "__main__":
    main()
