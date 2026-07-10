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
        "is_festival_window": festival_flag,
    }

    # Fill anything still missing with the training medians so the model
    # never sees a NaN it wasn't trained to expect
    medians = metrics.get("feature_medians", {})
    for col in feature_cols:
        if col not in row or pd.isna(row[col]):
            row[col] = medians.get(col, 0)

    return pd.DataFrame([{c: row[c] for c in feature_cols}]), festival_name


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
    pred = float(model.predict(X_future)[0])

    recent = g.sort_values("Date").tail(60)[["Date", "Modal"]]
    history = [{"date": d.strftime("%Y-%m-%d"), "modal": float(m)} for d, m in
               zip(recent["Date"], recent["Modal"])]

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
    <h3>Recent Price Trend (last 60 tender days, both markets)</h3>
    <canvas id="trendChart" height="90"></canvas>
  </div>

  <div class="boards">
    __FEATURE_CARDS__
  </div>

  <footer>
    Predictions are a statistical estimate from historical patterns, not a guarantee &mdash;
    treat as one input alongside your own market knowledge.<br>
    Re-run 1_scrape_incremental.py &rarr; 2_fetch_external_factors.py &rarr; 3_build_features.py &rarr;
    4_train_model.py &rarr; 5_generate_dashboard.py to refresh.
  </footer>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-adapter-date-fns/3.0.0/chartjs-adapter-date-fns.bundle.min.js"></script>
<script>
const trendData = __TREND_DATA__;

const ctx = document.getElementById('trendChart');
new Chart(ctx, {
  type: 'line',
  data: {
    datasets: trendData.map(d => ({
      label: d.market,
      data: d.history.map(h => ({x: h.date, y: h.modal})),
      borderColor: d.market === 'TIPTUR' ? '#b8862b' : '#4a5d3a',
      backgroundColor: 'transparent',
      tension: 0.2,
      pointRadius: 2,
    }))
  },
  options: {
    responsive: true,
    scales: {
      x: { type: 'time', time: { unit: 'month' }, ticks: { color: '#a99a80' }, grid: { color: '#3a2c1d' } },
      y: { ticks: { color: '#a99a80' }, grid: { color: '#3a2c1d' }, title: { display: true, text: 'Rs / Quintal', color: '#a99a80' } }
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
    <span>Validation MAPE: {mape}</span>
  </div>
  <div class="accuracy">Typical error: &plusmn; Rs {mae:,} / quintal (from backtest on held-out recent data)</div>
</div>
"""

FEATURE_CARD_TEMPLATE = """
<div class="board">
  <div class="board-head"><h2 style="font-size:15px;">{market} -- what's driving it</h2></div>
  <div class="features">{feature_rows}</div>
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
    trend_data = [
        {"market": r["market"], "history": r["history"]} for r in results if r is not None
    ]

    html = (
        HTML_TEMPLATE
        .replace("__GENERATED_AT__", datetime.now().strftime("%Y-%m-%d %H:%M"))
        .replace("__BOARD_CARDS__", board_cards)
        .replace("__FEATURE_CARDS__", feature_cards)
        .replace("__TREND_DATA__", json.dumps(trend_data))
    )

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDashboard saved to {OUT_HTML} -- open it in your browser.")


if __name__ == "__main__":
    main()
