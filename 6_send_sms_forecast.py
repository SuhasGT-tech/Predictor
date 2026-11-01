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
import sys
import re
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import model_utils

# Ensure console handles Kannada unicode characters cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import holidays
except ImportError:
    holidays = None

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")

MARKET = "TIPTUR"
TENDER_DAYS = [0, 3]  # Monday, Thursday

SMS_GATEWAY_URL = "https://api.sms-gate.app/3rdparty/v1/message"

DAY_NAMES_KN = {0: "ಸೋಮವಾರ", 1: "ಮಂಗಳವಾರ", 2: "ಬುಧವಾರ", 3: "ಗುರುವಾರ",
                4: "ಶುಕ್ರವಾರ", 5: "ಶನಿವಾರ", 6: "ಭಾನುವಾರ"}
MONTH_NAMES_KN = {1: "ಜನವರಿ", 2: "ಫೆಬ್ರವರಿ", 3: "ಮಾರ್ಚ್", 4: "ಏಪ್ರಿಲ್", 5: "ಮೇ",
                   6: "ಜೂನ್", 7: "ಜುಲೈ", 8: "ಆಗಸ್ಟ್", 9: "ಸೆಪ್ಟೆಂಬರ್",
                   10: "ಅಕ್ಟೋಬರ್", 11: "ನವೆಂಬರ್", 12: "ಡಿಸೆಂಬರ್"}


def next_tender_date(from_date):
    for offset in range(1, 8):
        candidate = from_date + timedelta(days=offset)
        if candidate.weekday() in TENDER_DAYS:
            return candidate
    raise RuntimeError("no tender day found in the next week")


def is_festival_window_for(date):
    if holidays is None:
        return False, ""
    try:
        in_holidays = holidays.India(years=[date.year, date.year + 1])
        for hdate, name in in_holidays.items():
            hdate = pd.Timestamp(hdate)
            if hdate - timedelta(days=10) <= pd.Timestamp(date) <= hdate:
                return True, name
    except Exception:
        pass
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
    best_pct = round((by_month["mean"].max() - 1.0) * 100.0, 1)
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
    g = df[df["Market"] == MARKET].sort_values("Date").reset_index(drop=True)

    target_date = next_tender_date(datetime.now())
    festival_flag, festival_name = is_festival_window_for(target_date)

    last_p = float(g["Modal"].iloc[-1])
    last_dt = g["Date"].iloc[-1]
    lp = np.log(g["Modal"])
    r = lp.diff()

    other_m = "ARSIKERE" if MARKET == "TIPTUR" else "TIPTUR"
    other_g = df[df["Market"] == other_m].sort_values("Date")
    other_p = float(other_g["Modal"].iloc[-1]) if len(other_g) else last_p
    other_ret = float(np.log(other_g["Modal"]).diff(2).iloc[-1]) if len(other_g) >= 3 else 0.0

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

    for col in feature_cols:
        if col not in row or pd.isna(row[col]):
            row[col] = float(medians.get(col, 0.0))

    X = pd.DataFrame([{c: row[c] for c in feature_cols}])
    pred_log_return = float(model.predict(X)[0])
    pred = round(last_p * np.exp(pred_log_return))

    best_month, best_pct = compute_best_month(g)
    best_month_kn = MONTH_NAMES_KN.get(best_month, "")

    return {
        "target_date": target_date,
        "prediction": pred,
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
    extra_phones = os.environ.get("EXTRA_PHONE_NUMBERS", "")

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
