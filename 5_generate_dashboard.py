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
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib

try:
    import holidays
except ImportError:
    holidays = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    in_holidays = holidays.India(years=[date.year, date.year + 1])
    for hdate, name in in_holidays.items():
        hdate = pd.Timestamp(hdate)
        if hdate - timedelta(days=10) <= pd.Timestamp(date) <= hdate:
            return 1, name
    return 0, ""


def build_future_row(market_hist, feature_cols, target_date, metrics):
    """Construct the single feature row for the next tender date, using the
    most recent known values as the best available estimate for anything
    we can't know in advance (weather, global oil price, fx rate)."""
    g = market_hist.sort_values("Date")
    last3 = g["Modal"].tail(3).tolist()[::-1]  # most recent first
    while len(last3) < 3:
        last3.append(np.nan)
    lag1, lag2, lag3 = last3[0], last3[1], last3[2]

    roll4 = g["Modal"].tail(4)
    roll12 = g["Modal"].tail(12)
    arrivals_tail4 = g["Arrivals"].tail(4)

    festival_flag, festival_name = is_festival_window_for(target_date)

    row = {
        "modal_lag_1": lag1,
        "modal_lag_2": lag2,
        "modal_lag_3": lag3,
        "modal_roll_mean_4": roll4.mean(),
        "modal_roll_std_4": roll4.std(),
        "modal_roll_mean_12": roll12.mean(),
        "price_momentum": (lag1 - lag2) if pd.notna(lag1) and pd.notna(lag2) else np.nan,
        "arrivals_lag_1": g["Arrivals"].iloc[-1] if len(g) else np.nan,
        "arrivals_roll_mean_4": arrivals_tail4.mean(),
        "day_of_week": target_date.weekday(),
        "month": target_date.month,
        "year": target_date.year,
        "days_since_start": (target_date - g["Date"].min().to_pydatetime()).days,
        "is_tiptur_tenderday": int(target_date.weekday() in TENDER_DAYS["TIPTUR"]),
        "is_arsikere_tenderday": int(target_date.weekday() in TENDER_DAYS["ARSIKERE"]),
        "coconut_oil_price_usd_per_mt": g["coconut_oil_price_usd_per_mt"].dropna().iloc[-1]
            if "coconut_oil_price_usd_per_mt" in g and g["coconut_oil_price_usd_per_mt"].notna().any() else np.nan,
        "coconut_oil_price_change_1m": g["coconut_oil_price_change_1m"].dropna().iloc[-1]
            if "coconut_oil_price_change_1m" in g and g["coconut_oil_price_change_1m"].notna().any() else np.nan,
        "usd_inr_rate": g["usd_inr_rate"].dropna().iloc[-1]
            if "usd_inr_rate" in g and g["usd_inr_rate"].notna().any() else np.nan,
        "rainfall_mm": g["rainfall_mm"].tail(30).mean() if "rainfall_mm" in g else np.nan,
        "temp_c": g["temp_c"].tail(30).mean() if "temp_c" in g else np.nan,
        "rainfall_roll_90": g["rainfall_roll_90"].dropna().iloc[-1]
            if "rainfall_roll_90" in g and g["rainfall_roll_90"].notna().any() else np.nan,
        "other_market_last_price": g["other_market_last_price"].dropna().iloc[-1]
            if "other_market_last_price" in g and g["other_market_last_price"].notna().any() else np.nan,
        "is_festival_window": festival_flag,
    }

    # Fill anything still missing with the training medians so the model
    # never sees a NaN it wasn't trained to expect
    medians = metrics.get("feature_medians", {})
    for col in feature_cols:
        if col not in row or pd.isna(row[col]):
            row[col] = medians.get(col, 0)

    return pd.DataFrame([{c: row[c] for c in feature_cols}]), festival_name


MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


def compute_seasonality(g):
    """
    For each historical row, express its price as a ratio to that YEAR's
    average price (this detrends 24 years of inflation/growth so we're
    comparing pure calendar-month seasonality, not "prices were higher
    recently because prices are higher recently").
    Returns a list of {month, month_name, index} sorted by month, plus
    the single best month to sell (highest average ratio).
    """
    g = g.copy()
    g["year"] = g["Date"].dt.year
    yearly_mean = g.groupby("year")["Modal"].transform("mean")
    g["seasonal_ratio"] = g["Modal"] / yearly_mean

    by_month = g.groupby(g["Date"].dt.month)["seasonal_ratio"].agg(["mean", "count"])
    by_month = by_month[by_month["count"] >= 3]  # need a few observations to trust it

    result = [
        {"month": int(m), "month_name": MONTH_NAMES[int(m)],
         "index": round((row["mean"] - 1) * 100, 1)}
        for m, row in by_month.iterrows()
    ]
    result.sort(key=lambda r: r["month"])

    if result:
        best = max(result, key=lambda r: r["index"])
        worst = min(result, key=lambda r: r["index"])
    else:
        best = worst = None

    return result, best, worst


def forecast_horizon(market, g, model, feature_cols, metrics, n_tenders=10):
    """
    Recursively forecast the next `n_tenders` tender days for this market,
    feeding each prediction back in as if it were the "actual" price for
    the next step's lag features. This is how you get a multi-week outlook
    instead of just a single next-day number.

    CAVEAT (also shown in the dashboard): external factors (weather, global
    oil price, fx rate) are held at their most recent known value for every
    future step, since we can't know them in advance. Accuracy naturally
    degrades the further out you look - treat this as a directional guide
    ("prices look like they're drifting up/down, and here's roughly why"),
    not a precise multi-week price quote.
    """
    g = g.sort_values("Date")
    medians = metrics.get("feature_medians", {})

    # Rolling state we'll update as we "generate" future prices
    recent_prices = g["Modal"].tail(12).tolist()  # most recent last
    recent_arrivals = g["Arrivals"].tail(4).tolist()
    current_date = datetime.now()
    start_of_history = g["Date"].min().to_pydatetime()

    static_ext = {
        "coconut_oil_price_usd_per_mt": g["coconut_oil_price_usd_per_mt"].dropna().iloc[-1]
            if "coconut_oil_price_usd_per_mt" in g and g["coconut_oil_price_usd_per_mt"].notna().any() else medians.get("coconut_oil_price_usd_per_mt", 0),
        "coconut_oil_price_change_1m": g["coconut_oil_price_change_1m"].dropna().iloc[-1]
            if "coconut_oil_price_change_1m" in g and g["coconut_oil_price_change_1m"].notna().any() else medians.get("coconut_oil_price_change_1m", 0),
        "usd_inr_rate": g["usd_inr_rate"].dropna().iloc[-1]
            if "usd_inr_rate" in g and g["usd_inr_rate"].notna().any() else medians.get("usd_inr_rate", 0),
        "rainfall_mm": g["rainfall_mm"].tail(30).mean() if "rainfall_mm" in g else medians.get("rainfall_mm", 0),
        "temp_c": g["temp_c"].tail(30).mean() if "temp_c" in g else medians.get("temp_c", 0),
        "rainfall_roll_90": g["rainfall_roll_90"].dropna().iloc[-1]
            if "rainfall_roll_90" in g and g["rainfall_roll_90"].notna().any() else medians.get("rainfall_roll_90", 0),
        "other_market_last_price": g["other_market_last_price"].dropna().iloc[-1]
            if "other_market_last_price" in g and g["other_market_last_price"].notna().any() else medians.get("other_market_last_price", 0),
    }

    forecast_points = []
    festival_hits = []

    for _ in range(n_tenders):
        target_date = next_tender_date(market, current_date)
        festival_flag, festival_name = is_festival_window_for(target_date)

        lag1 = recent_prices[-1] if recent_prices else medians.get("modal_lag_1", 0)
        lag2 = recent_prices[-2] if len(recent_prices) >= 2 else lag1
        lag3 = recent_prices[-3] if len(recent_prices) >= 3 else lag2
        roll4 = pd.Series(recent_prices[-4:])
        roll12 = pd.Series(recent_prices[-12:])

        row = {
            "modal_lag_1": lag1, "modal_lag_2": lag2, "modal_lag_3": lag3,
            "modal_roll_mean_4": roll4.mean(), "modal_roll_std_4": roll4.std(),
            "modal_roll_mean_12": roll12.mean(),
            "price_momentum": lag1 - lag2,
            "arrivals_lag_1": recent_arrivals[-1] if recent_arrivals else medians.get("arrivals_lag_1", 0),
            "arrivals_roll_mean_4": (pd.Series(recent_arrivals).mean() if recent_arrivals else medians.get("arrivals_roll_mean_4", 0)),
            "day_of_week": target_date.weekday(), "month": target_date.month, "year": target_date.year,
            "days_since_start": (target_date - start_of_history).days,
            "is_tiptur_tenderday": int(target_date.weekday() in TENDER_DAYS["TIPTUR"]),
            "is_arsikere_tenderday": int(target_date.weekday() in TENDER_DAYS["ARSIKERE"]),
            "is_festival_window": festival_flag,
            **static_ext,
        }
        for col in feature_cols:
            if col not in row or pd.isna(row[col]):
                row[col] = medians.get(col, 0)

        X = pd.DataFrame([{c: row[c] for c in feature_cols}])
        pred_log_return = float(model.predict(X)[0])
        pred = lag1 * np.exp(pred_log_return)

        forecast_points.append({
            "date": target_date.strftime("%Y-%m-%d"),
            "modal": round(pred),
            "festival": festival_name if festival_flag else None,
        })
        if festival_flag:
            festival_hits.append({"date": target_date.strftime("%Y-%m-%d"), "name": festival_name})

        recent_prices.append(pred)
        recent_arrivals.append(recent_arrivals[-1] if recent_arrivals else 0)
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

    g = df[df["Market"] == market].copy()
    target_date = next_tender_date(market, datetime.now())

    X_future, festival_name = build_future_row(g, feature_cols, target_date, metrics)
    lag1_for_pred = g.sort_values("Date")["Modal"].iloc[-1]
    pred_log_return = float(model.predict(X_future)[0])
    pred = lag1_for_pred * np.exp(pred_log_return)

    recent = g.sort_values("Date").tail(60)[["Date", "Modal"]]
    history = [{"date": d.strftime("%Y-%m-%d"), "modal": float(m)} for d, m in
               zip(recent["Date"], recent["Modal"])]

    seasonality, best_month, worst_month = compute_seasonality(g)
    forecast_points, festival_hits = forecast_horizon(market, g, model, feature_cols, metrics, n_tenders=10)

    last_actual = g.sort_values("Date")["Modal"].iloc[-1]
    last_date = g.sort_values("Date")["Date"].iloc[-1]

    return {
        "market": market,
        "target_date": target_date.strftime("%Y-%m-%d"),
        "target_day_name": DAY_NAMES[target_date.weekday()],
        "prediction": round(pred),
        "last_actual": round(float(last_actual)),
        "last_date": last_date.strftime("%Y-%m-%d"),
        "change": round(pred - float(last_actual)),
        "change_pct": round((pred - float(last_actual)) / float(last_actual) * 100, 1) if last_actual else 0,
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
    <h3>Seasonality — average price vs. that year's own average, by month (24-yr history)</h3>
    <canvas id="seasonChart" height="80"></canvas>
  </div>

  <div class="boards">
    __FEATURE_CARDS__
  </div>

  <footer>
    Predictions are a statistical estimate from historical patterns, not a guarantee &mdash;
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
  <div class="accuracy">Typical error: &plusmn; Rs {mae:,} / quintal (averaged across 5 walk-forward validation folds)</div>
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


def render_feature_card(res):
    if res is None or not res.get("top_features"):
        return ""
    max_imp = max(f["importance"] for f in res["top_features"]) or 1
    rows = ""
    for f in res["top_features"][:6]:
        pct = f["importance"] / max_imp * 100
        rows += (
            f"<div class='feature-bar-row'><span>{f['feature']}</span>"
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
        first = res["forecast_points"][0]
        drift = last["modal"] - res["last_actual"]
        direction = "up" if drift > 0 else "down" if drift < 0 else "flat"
        items.append(
            f"<div class='outlook-item'><span class='outlook-icon'>&#128336;</span>"
            f"<span>Over the next {len(res['forecast_points'])} tenders (through {last['date']}), "
            f"the model's directional read is <b>trending {direction}</b> "
            f"(Rs {abs(drift):,} {'higher' if drift > 0 else 'lower' if drift < 0 else 'change'} by then) — "
            f"treat this as a rough trend, not a locked-in price.</span></div>"
        )

    for hit in (res.get("festival_hits") or [])[:3]:
        items.append(
            f"<div class='outlook-item upcoming-festival'><span class='outlook-icon'>&#127881;</span>"
            f"<span><b>{hit['name']}</b> falls around {hit['date']} — festival demand windows have "
            f"historically coincided with firmer coconut/copra prices.</span></div>"
        )

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

    print(f"\nDashboard saved to {OUT_HTML} -- open it in your browser.")


if __name__ == "__main__":
    main()
