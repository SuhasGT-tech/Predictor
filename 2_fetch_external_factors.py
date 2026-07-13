"""
Fetch external factors that plausibly influence copra prices, and save
them all to data/external_factors.csv (one row per calendar day, from
2002-01-01 to today).

FACTORS FETCHED
---------------
1. Global edible oil prices (USD/metric ton) - World Bank "Pink Sheet"
   monthly commodity data, forward-filled to daily resolution:
     - Coconut oil (the direct reference price)
     - Palm oil, Soybean oil (competing edible oils - coconut oil buyers
       can substitute toward these, so their relative prices matter, not
       just coconut oil's price level in isolation)
2. USD/INR exchange rate (daily) - Frankfurter API (ECB-sourced, free,
   no key). A weaker rupee makes Indian copra/coconut-oil exports more
   competitive, which can support domestic prices.
3. Weather (daily rainfall + mean temperature) for Hassan district
   (Arasikere market) and Tumakuru district (Tiptur market) - Open-Meteo
   historical archive API, free, no key. A 12-month-lagged rainfall
   feature is also derived in 3_build_features.py, matching the coconut
   palm's roughly year-long flowering-to-harvest cycle.
4. Festival calendar - India's major holidays (Diwali, Ganesh Chaturthi,
   Onam, Navratri, etc. push up coconut/coconut-oil demand for cooking,
   offerings, and hair-oil use). Uses the `holidays` python package, then
   flags a +/-10 day window around each as a "festival demand window".

WHY FORWARD-FILL / MERGE TO DAILY
The market data has irregular dates (only tender days), so we build one
daily external-factors table and let 3_build_features.py join it onto
whatever price dates actually exist.

SETUP
-----
    pip install requests pandas openpyxl holidays --break-system-packages

USAGE
-----
    python 2_fetch_external_factors.py
"""

import os
import re
import io
from datetime import datetime, timedelta

import requests
import pandas as pd

try:
    import holidays
except ImportError:
    holidays = None

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_CSV = os.path.join(DATA_DIR, "external_factors.csv")

START_DATE = "2002-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Market coordinates (approx market-town centroids)
LOCATIONS = {
    "arsikere": {"lat": 13.3086, "lon": 76.2577},   # Hassan district
    "tiptur":   {"lat": 13.2568, "lon": 76.4784},   # Tumakuru district
}

WORLDBANK_LANDING_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
# Fallback in case the landing-page scrape fails (link rotates periodically,
# so the scrape-the-landing-page approach is more robust long-term than
# hardcoding this, but we keep a fallback for resilience)
WORLDBANK_XLSX_FALLBACK = (
    "https://thedocs.worldbank.org/en/doc/"
    "74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/"
    "CMO-Historical-Data-Monthly.xlsx"
)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; copra-price-model/1.0)"}


# ----------------------------------------------------------------------
# 1. Global coconut oil price (World Bank Pink Sheet)
# ----------------------------------------------------------------------

def find_worldbank_xlsx_url():
    """The World Bank rotates the exact download URL periodically (the guid
    in the path changes), so we scrape the landing page for the current
    link instead of trusting a hardcoded one."""
    try:
        resp = requests.get(WORLDBANK_LANDING_PAGE, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        match = re.search(
            r'href="([^"]*CMO-Historical-Data-Monthly\.xlsx)"', resp.text
        )
        if match:
            url = match.group(1)
            if url.startswith("//"):
                url = "https:" + url
            elif url.startswith("/"):
                url = "https://www.worldbank.org" + url
            return url
    except Exception as e:
        print(f"  Could not scrape World Bank landing page ({e}), using fallback URL.")
    return WORLDBANK_XLSX_FALLBACK


def fetch_commodity_prices():
    """
    Pulls coconut oil AND two competing edible oils (palm oil, soybean oil)
    from the same World Bank sheet. These aren't just "more data" - coconut
    oil, palm oil, and soybean oil are substitutable in most of their uses
    (cooking, industrial), so buyers switch between them based on relative
    price. A coconut-oil-only view misses that a copra price move might
    really be "palm oil got expensive so buyers shifted to coconut oil,"
    not something intrinsic to coconut supply/demand at all.
    """
    print("Fetching global edible oil prices (World Bank Pink Sheet)...")
    url = find_worldbank_xlsx_url()
    print(f"  Using: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    xl = pd.ExcelFile(io.BytesIO(resp.content))
    # The "Monthly Prices" sheet has commodities as columns, dates as rows,
    # with a few header rows before the real table starts.
    sheet_name = "Monthly Prices" if "Monthly Prices" in xl.sheet_names else xl.sheet_names[0]
    raw = xl.parse(sheet_name, header=None)

    # Find the header row (contains "Coconut oil") and column indices for
    # each commodity we want, all read off that same header row
    header_row_idx = None
    col_indices = {}
    targets = {
        "coconut_oil_price_usd_per_mt": "coconut oil",
        "palm_oil_price_usd_per_mt": "palm oil",
        "soybean_oil_price_usd_per_mt": "soybean oil",
    }
    for i, row in raw.iterrows():
        for j, val in enumerate(row):
            if not isinstance(val, str):
                continue
            val_lower = val.lower()
            for col_name, keyword in targets.items():
                if keyword in val_lower and col_name not in col_indices:
                    col_indices[col_name] = j
        if "coconut_oil_price_usd_per_mt" in col_indices:
            header_row_idx = i
            break

    if header_row_idx is None or "coconut_oil_price_usd_per_mt" not in col_indices:
        raise RuntimeError(
            "Could not find a 'Coconut oil' column in the World Bank sheet - "
            "the file layout may have changed. Open the xlsx manually to check."
        )
    missing = [k for k in targets if k not in col_indices]
    if missing:
        print(f"  NOTE: couldn't find columns for {missing} - continuing without them")

    cols = [0] + list(col_indices.values())
    names = ["period"] + list(col_indices.keys())
    data = raw.iloc[header_row_idx + 1:, cols].copy()
    data.columns = names
    data = data.dropna(subset=["period"])

    def parse_period(p):
        p = str(p).strip()
        m = re.match(r"(\d{4})M(\d{1,2})", p)
        if m:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
        return pd.NaT

    data["month"] = data["period"].apply(parse_period)
    data = data.dropna(subset=["month"])
    price_cols = [c for c in data.columns if c not in ("period", "month")]
    for col in price_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data[["month"] + price_cols].sort_values("month")
    print(f"  Got {len(data)} months of edible oil price data "
          f"({data['month'].min().date()} to {data['month'].max().date()}), "
          f"columns: {price_cols}")
    return data


# ----------------------------------------------------------------------
# 2. USD/INR exchange rate (Frankfurter API)
# ----------------------------------------------------------------------

def fetch_usd_inr():
    print("Fetching USD/INR exchange rates (Frankfurter API)...")
    # Frankfurter's INR data starts a bit after 1999; our START_DATE (2002)
    # is safely within range.
    url = f"https://api.frankfurter.dev/v1/{START_DATE}..{END_DATE}?base=USD&symbols=INR"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    rates = payload.get("rates", {})
    rows = [{"date": pd.Timestamp(d), "usd_inr_rate": v.get("INR")} for d, v in rates.items()]
    df = pd.DataFrame(rows).sort_values("date")
    print(f"  Got {len(df)} days of USD/INR rates")
    return df


# ----------------------------------------------------------------------
# 3. Weather (Open-Meteo archive API)
# ----------------------------------------------------------------------

def fetch_weather(market_key, lat, lon):
    print(f"Fetching weather for {market_key} ({lat}, {lon})...")
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        "&daily=precipitation_sum,temperature_2m_mean"
        "&timezone=Asia%2FKolkata"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    daily = payload.get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])),
        f"rainfall_{market_key}_mm": daily.get("precipitation_sum", []),
        f"temp_{market_key}_c": daily.get("temperature_2m_mean", []),
    })
    print(f"  Got {len(df)} days of weather data for {market_key}")
    return df


# ----------------------------------------------------------------------
# 4. Festival calendar
# ----------------------------------------------------------------------

def build_festival_calendar():
    print("Building festival calendar...")
    date_range = pd.date_range(START_DATE, END_DATE, freq="D")
    df = pd.DataFrame({"date": date_range})
    df["is_festival_window"] = 0
    df["nearest_festival"] = ""

    if holidays is None:
        print("  `holidays` package not installed - skipping festival calendar. "
              "Run: pip install holidays --break-system-packages")
        return df

    years = range(int(START_DATE[:4]), int(END_DATE[:4]) + 1)
    in_holidays = holidays.India(years=years)  # includes major national + regional observances

    WINDOW_DAYS = 10  # demand for coconut/coconut oil ramps up in the days before a festival
    for hdate, name in in_holidays.items():
        hdate = pd.Timestamp(hdate)
        mask = (df["date"] >= hdate - timedelta(days=WINDOW_DAYS)) & (df["date"] <= hdate)
        df.loc[mask, "is_festival_window"] = 1
        df.loc[mask & (df["nearest_festival"] == ""), "nearest_festival"] = name

    print(f"  Flagged {df['is_festival_window'].sum()} days as festival-demand windows "
          f"across {len(in_holidays)} holidays")
    print("  NOTE: `holidays.India()` covers national holidays well but may miss some "
          "regional/Kerala-specific festivals like Onam. Edit this function to add exact "
          "dates manually if you track additional festivals important to your markets.")
    return df


# ----------------------------------------------------------------------
# MAIN - merge everything into one daily table
# ----------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    daily = pd.DataFrame({"date": pd.date_range(START_DATE, END_DATE, freq="D")})

    # Edible oil prices (monthly -> forward-fill to daily): coconut oil plus
    # competing oils (palm, soybean) that copra buyers can substitute toward
    oil_cols = ["coconut_oil_price_usd_per_mt", "palm_oil_price_usd_per_mt",
                "soybean_oil_price_usd_per_mt"]
    try:
        oil = fetch_commodity_prices()
        oil_daily = daily.merge(
            oil.rename(columns={"month": "date"}), on="date", how="left"
        )
        present_cols = [c for c in oil_cols if c in oil_daily.columns]
        for col in present_cols:
            oil_daily[col] = oil_daily[col].ffill()
        daily = daily.merge(oil_daily[["date"] + present_cols], on="date", how="left")
        for col in oil_cols:
            if col not in daily.columns:
                daily[col] = None
    except Exception as e:
        print(f"  !! Edible oil price fetch failed: {e}")
        for col in oil_cols:
            daily[col] = None

    # USD/INR
    try:
        fx = fetch_usd_inr()
        daily = daily.merge(fx, on="date", how="left")
        daily["usd_inr_rate"] = daily["usd_inr_rate"].ffill()
    except Exception as e:
        print(f"  !! USD/INR fetch failed: {e}")
        daily["usd_inr_rate"] = None

    # Weather per market
    for market_key, coords in LOCATIONS.items():
        try:
            w = fetch_weather(market_key, coords["lat"], coords["lon"])
            daily = daily.merge(w, on="date", how="left")
        except Exception as e:
            print(f"  !! Weather fetch failed for {market_key}: {e}")
            daily[f"rainfall_{market_key}_mm"] = None
            daily[f"temp_{market_key}_c"] = None

    # Festivals
    fest = build_festival_calendar()
    daily = daily.merge(fest, on="date", how="left")

    daily.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(daily)} days of external factors to {OUT_CSV}")


if __name__ == "__main__":
    main()
