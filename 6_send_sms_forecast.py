"""
Sends a Kannada-language SMS with:
  1. The predicted Modal price for the NEXT Tiptur tender (Mon/Thu)
  2. Which month has historically been best to hold/sell in

Sent via "SMS Gateway for Android" (https://sms-gate.app) - a free, open-source
app that turns your own Android phone into an SMS sender via API. This means
the message goes out as a completely normal SMS from your own number, so none
of India's DLT/bulk-SMS registration rules apply (those target commercial bulk
senders, not a personal message from your own phone).

ONE-TIME SETUP
--------------
1. Install the app on an Android phone you control:
   https://github.com/capcom6/android-sms-gateway/releases (latest APK)
2. Open the app -> toggle "Cloud Server" -> tap "Online".
3. It will show a username and password - keep the app open/logged in
   (it works in the background; just don't uninstall it or clear its data).
4. In your GitHub repo: Settings -> Secrets and variables -> Actions -> add:
     SMS_GATEWAY_USER      = the username from step 3
     SMS_GATEWAY_PASS      = the password from step 3
     FATHER_PHONE_NUMBER   = father's number with country code, e.g. +919xxxxxxxxx

USAGE
-----
    python 6_send_sms_forecast.py

Run this 2 days before each Tiptur tender (Saturday for Monday's tender,
Tuesday for Thursday's tender) - see .github/workflows/send_sms.yml for the
automated schedule.
"""

import os
import re
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import model_utils  # noqa: F401 - required so joblib can unpickle SeedEnsembleRegressor models
import requests

try:
    import holidays
except ImportError:
    holidays = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")

MARKET = "TIPTUR"
TENDER_DAYS = [0, 3]  # Monday, Thursday - fallback default, overwritten in main()

SMS_GATEWAY_URL = "https://api.sms-gate.app/3rdparty/v1/message"

DAY_NAMES_KN = {0: "ಸೋಮವಾರ", 1: "ಮಂಗಳವಾರ", 2: "ಬುಧವಾರ", 3: "ಗುರುವಾರ",
                4: "ಶುಕ್ರವಾರ", 5: "ಶನಿವಾರ", 6: "ಭಾನುವಾರ"}
MONTH_NAMES_KN = {1: "ಜನವರಿ", 2: "ಫೆಬ್ರವರಿ", 3: "ಮಾರ್ಚ್", 4: "ಏಪ್ರಿಲ್", 5: "ಮೇ",
                   6: "ಜೂನ್", 7: "ಜುಲೈ", 8: "ಆಗಸ್ಟ್", 9: "ಸೆಪ್ಟೆಂಬರ್",
                   10: "ಅಕ್ಟೋಬರ್", 11: "ನವೆಂಬರ್", 12: "ಡಿಸೆಂಬರ್"}


def detect_tender_days(dates, months_back=12, fallback=(0, 3)):
    """Same logic as in 3_build_features.py / 5_generate_dashboard.py - detects
    the market's actual recent tender-day pattern instead of assuming it forever."""
    if len(dates) == 0:
        return sorted(fallback)
    cutoff = dates.max() - pd.DateOffset(months=months_back)
    recent = dates[dates >= cutoff]
    if len(recent) < 10:
        return sorted(fallback)
    top2 = recent.dt.dayofweek.value_counts().head(2).index.tolist()
    return sorted(top2)


def next_tender_date(from_date):
    for offset in range(1, 8):
        candidate = from_date + timedelta(days=offset)
        if candidate.weekday() in TENDER_DAYS:
            return candidate
    raise RuntimeError("no tender day found in the next week")


def is_festival_window_for(date):
    if holidays is None:
        return False, ""
    in_holidays = holidays.India(years=[date.year, date.year + 1])
    for hdate, name in in_holidays.items():
        hdate = pd.Timestamp(hdate)
        if hdate - timedelta(days=10) <= pd.Timestamp(date) <= hdate:
            return True, name
    return False, ""


def compute_best_month(g):
    g = g.copy()
    g["year"] = g["Date"].dt.year
    yearly_mean = g.groupby("year")["Modal"].transform("mean")
    g["seasonal_ratio"] = g["Modal"] / yearly_mean
    by_month = g.groupby(g["Date"].dt.month)["seasonal_ratio"].agg(["mean", "count"])
    by_month = by_month[by_month["count"] >= 3]
    if by_month.empty:
        return None, None
    best_month = int(by_month["mean"].idxmax())
    best_pct = round((by_month["mean"].max() - 1) * 100, 1)
    return best_month, best_pct


def predict_next_tender():
    model = joblib.load(os.path.join(MODELS_DIR, f"model_{MARKET.lower()}.joblib"))
    with open(os.path.join(MODELS_DIR, f"feature_cols_{MARKET.lower()}.json")) as f:
        feature_cols = json.load(f)
    metrics = {}
    metrics_path = os.path.join(MODELS_DIR, f"metrics_{MARKET.lower()}.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    medians = metrics.get("feature_medians", {})

    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"])
    g = df[df["Market"] == MARKET].sort_values("Date")

    global TENDER_DAYS
    TENDER_DAYS = detect_tender_days(g["Date"], fallback=(0, 3))

    target_date = next_tender_date(datetime.now())
    festival_flag, festival_name = is_festival_window_for(target_date)

    last3 = g["Modal"].tail(3).tolist()[::-1]
    while len(last3) < 3:
        last3.append(np.nan)
    lag1, lag2, lag3 = last3[0], last3[1], last3[2]
    roll4, roll12 = g["Modal"].tail(4), g["Modal"].tail(12)

    row = {
        "modal_lag_1": lag1, "modal_lag_2": lag2, "modal_lag_3": lag3,
        "modal_roll_mean_4": roll4.mean(), "modal_roll_std_4": roll4.std(),
        "modal_roll_mean_12": roll12.mean(),
        "price_momentum": (lag1 - lag2) if pd.notna(lag1) and pd.notna(lag2) else np.nan,
        "arrivals_lag_1": g["Arrivals"].iloc[-1] if len(g) else np.nan,
        "arrivals_roll_mean_4": g["Arrivals"].tail(4).mean(),
        "day_of_week": target_date.weekday(), "month": target_date.month, "year": target_date.year,
        "days_since_start": (target_date - g["Date"].min().to_pydatetime()).days,
        "is_tiptur_tenderday": 1, "is_arsikere_tenderday": 0,
        "is_festival_window": int(festival_flag),
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
    }
    for col in feature_cols:
        if col not in row or pd.isna(row[col]):
            row[col] = medians.get(col, 0)

    X = pd.DataFrame([{c: row[c] for c in feature_cols}])
    pred_log_return = float(model.predict(X)[0])
    pred = round(lag1 * np.exp(pred_log_return))

    residual_std = metrics.get("residual_std_log_return")
    range_low = range_high = None
    if residual_std:
        range_low = round(lag1 * np.exp(pred_log_return - residual_std))
        range_high = round(lag1 * np.exp(pred_log_return + residual_std))

    best_month, best_pct = compute_best_month(g)
    best_month_kn = MONTH_NAMES_KN.get(best_month, "")

    return {
        "target_date": target_date,
        "prediction": pred,
        "range_low": range_low,
        "range_high": range_high,
        "festival_flag": festival_flag,
        "festival_name": festival_name,
        "best_month_kn": best_month_kn,
        "best_pct": best_pct,
    }


def compose_message(info):
    date_str = info["target_date"].strftime("%d-%m-%Y")
    day_kn = DAY_NAMES_KN[info["target_date"].weekday()]

    lines = [
        "ಟಿಪ್ಟೂರು ಕೊಪ್ರಾ ಬೆಲೆ ಮುನ್ಸೂಚನೆ",
        f"ಮುಂದಿನ ಟೆಂಡರ್: {day_kn}, {date_str}",
        f"ನಿರೀಕ್ಷಿತ ಬೆಲೆ: ಸುಮಾರು ರೂ. {info['prediction']:,}/ಕ್ವಿಂಟಲ್",
    ]
    if info.get("range_low") and info.get("range_high"):
        lines.append(f"ಸಾಧ್ಯತೆ ಇರುವ ವ್ಯಾಪ್ತಿ: ರೂ. {info['range_low']:,} - {info['range_high']:,}")
    if info["best_month_kn"]:
        lines.append(
            f"ಮಾರಾಟಕ್ಕೆ/ಸಂಗ್ರಹಕ್ಕೆ ಉತ್ತಮ ತಿಂಗಳು: {info['best_month_kn']} "
            f"(ಸರಾಸರಿಗಿಂತ +{info['best_pct']}%)"
        )
    if info["festival_flag"]:
        lines.append(f"ಸೂಚನೆ: {info['festival_name']} ಹಬ್ಬದ ಬೇಡಿಕೆ ಅವಧಿ ಹತ್ತಿರದಲ್ಲಿದೆ")
    lines.append("(ಇದು ಅಂದಾಜು ಬೆಲೆ, ಖಚಿತವಲ್ಲ)")

    return "\n".join(lines)


def send_sms(message):
    user = os.environ.get("SMS_GATEWAY_USER")
    password = os.environ.get("SMS_GATEWAY_PASS")
    father_phone = os.environ.get("FATHER_PHONE_NUMBER")
    extra_phones = os.environ.get("EXTRA_PHONE_NUMBERS", "")  # comma-separated, e.g. your own number

    extra_list = [p.strip() for p in re.split(r"[,\n\s]+", extra_phones) if p.strip()]
    phones = [p.strip() for p in [father_phone] + extra_list if p and p.strip()]

    if not (user and password and phones):
        print("Missing SMS_GATEWAY_USER / SMS_GATEWAY_PASS / phone number "
              "environment variables - skipping actual send. Message would have been:")
        print(message)
        return

    resp = requests.post(
        SMS_GATEWAY_URL,
        auth=(user, password),
        json={"textMessage": {"text": message}, "phoneNumbers": phones},
        timeout=30,
    )
    print(f"SMS gateway response: {resp.status_code} {resp.text} (sent to {phones})")
    resp.raise_for_status()


def main():
    if not os.path.exists(FEATURES_CSV):
        print(f"{FEATURES_CSV} not found - run the earlier pipeline steps first.")
        return

    info = predict_next_tender()
    message = compose_message(info)
    print("Composed message:\n" + message + "\n")
    send_sms(message)


if __name__ == "__main__":
    main()
