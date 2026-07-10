"""
INCREMENTAL scraper for Copra prices (Arasikere & Tiptur) from Karnataka's
Krama portal: https://krama.karnataka.gov.in/reports/DateWiseReport

This builds on your original scrape_copra_prices_requests.py, but instead of
re-scraping all ~600 month/market combos every time, it:

  1. Reads your existing CSV to find the last date already saved, per market.
  2. Re-fetches only the LOOKBACK_MONTHS most recent months (in case the portal
     added/late-corrected rows for the current or previous month) plus any
     months since your last run.
  3. Merges the new rows into the existing CSV, de-duplicating so nothing
     doubles up, and writes it back out.

WHY RE-FETCH A FEW RECENT MONTHS INSTEAD OF JUST "NEW" DATES?
Government portals often add rows for a tender day a few days late, or
correct a Modal price after the fact. Re-checking the last LOOKBACK_MONTHS
months on every run catches those without redoing the full 24-year history.

USAGE
-----
    python 1_scrape_incremental.py

Run this after each tender day (Mon/Thu for Tiptur, Tue/Fri for Arasikere),
or just run it daily/weekly - it's cheap now (a handful of months, not 600).
"""

import csv
import os
import random
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

BASE_URL = "https://krama.karnataka.gov.in/reports/DateWiseReport"

MARKETS = {
    "ARSIKERE": "55",
    "TIPTUR": "34",
}
COMMODITY_COPRA = "129"

MONTHS = {
    "JANUARY": "1", "FEBRUARY": "2", "MARCH": "3", "APRIL": "4",
    "MAY": "5", "JUNE": "6", "JULY": "7", "AUGUST": "8",
    "SEPTEMBER": "9", "OCTOBER": "10", "NOVEMBER": "11", "DECEMBER": "12",
}
MONTH_NUM_TO_NAME = {v: k for k, v in MONTHS.items()}

# How many recent months to always re-check (catches late corrections)
LOOKBACK_MONTHS = 3

DELAY_SECONDS = 2.5
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "copra_prices_arasikere_tiptur.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Referer": BASE_URL,
}

FIELD_MONTH = "_ctl0:MainContent:ddlmonth"
FIELD_YEAR = "_ctl0:MainContent:ddlyear"
FIELD_COMMODITY = "_ctl0:MainContent:ddlcommodity"
FIELD_MARKET = "_ctl0:MainContent:ddlmarket"
FIELD_VIEWREPORT = "_ctl0:MainContent:viewreport"
TABLE_ID = "_ctl0_MainContent_gv"

# The full set of columns we want in the final CSV (order matters for readability)
FIELDNAMES = [
    "Date", "Market", "District", "Variety", "Grade",
    "Arrivals", "Min", "Max", "Modal", "Unit",
    "QueryMarket", "QueryMonth", "QueryYear",
]

# ----------------------------------------------------------------------
# SCRAPER (same core logic as the original script)
# ----------------------------------------------------------------------

def get_asp_tokens(session):
    resp = session.get(BASE_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    def val(field_id):
        el = soup.find(id=field_id)
        return el["value"] if el and el.has_attr("value") else ""

    return {
        "__VIEWSTATE": val("__VIEWSTATE"),
        "__VIEWSTATEGENERATOR": val("__VIEWSTATEGENERATOR"),
        "__VIEWSTATEENCRYPTED": val("__VIEWSTATEENCRYPTED"),
        "__EVENTVALIDATION": val("__EVENTVALIDATION"),
        "__PREVIOUSPAGE": val("__PREVIOUSPAGE"),
    }


def fetch_report_html(session, month_val, year, market_val):
    tokens = get_asp_tokens(session)
    payload = {
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "__LASTFOCUS": "",
        **tokens,
        FIELD_MONTH: month_val,
        FIELD_YEAR: str(year),
        FIELD_COMMODITY: COMMODITY_COPRA,
        FIELD_MARKET: market_val,
        FIELD_VIEWREPORT: "View Report",
    }
    resp = session.post(BASE_URL, data=payload, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_table(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find(id=TABLE_ID)
    if not table:
        return []

    trs = table.find_all("tr")
    if not trs:
        return []

    headers = [th.get_text(strip=True) for th in trs[0].find_all("th")]
    if not headers:
        return []

    rows_out = []
    last_market = None
    for tr in trs[1:]:
        cells = tr.find_all("td")
        texts = [c.get_text(strip=True) for c in cells]
        if not texts or not any(texts):
            continue
        joined = " ".join(texts).lower()
        if "total" in joined:
            continue

        row = dict(zip(headers, texts))
        if row.get("Market"):
            last_market = row["Market"]
        else:
            row["Market"] = last_market
        rows_out.append(row)

    return rows_out


# ----------------------------------------------------------------------
# INCREMENTAL LOGIC
# ----------------------------------------------------------------------

def load_existing_rows():
    """Returns (list_of_row_dicts, dict of market -> last datetime)."""
    if not os.path.exists(OUTPUT_CSV):
        return [], {}

    rows = []
    last_date = {}
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            try:
                d = datetime.strptime(row["Date"], "%d/%m/%Y")
            except (ValueError, KeyError):
                continue
            mkt = row.get("Market")
            if mkt and (mkt not in last_date or d > last_date[mkt]):
                last_date[mkt] = d
    return rows, last_date


def months_to_check(last_date_for_market):
    """
    Build the list of (year, month_num_str) to (re-)fetch for one market:
    every month from last_date_for_market (or the last LOOKBACK_MONTHS
    months if we have no data yet) through the current month.
    """
    now = datetime.now()

    if last_date_for_market is None:
        # No existing data for this market - caller should use the full
        # historical scraper instead. Fall back to just recent months.
        start_year, start_month = now.year, max(1, now.month - LOOKBACK_MONTHS)
    else:
        # Step back LOOKBACK_MONTHS-1 extra months from the last known date
        # to catch late corrections, then walk forward to "now".
        y, m = last_date_for_market.year, last_date_for_market.month
        m -= (LOOKBACK_MONTHS - 1)
        while m < 1:
            m += 12
            y -= 1
        start_year, start_month = y, m

    combos = []
    y, m = start_year, start_month
    while (y, m) <= (now.year, now.month):
        combos.append((y, str(m)))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return combos


def normalize_row(row):
    """Map the site's raw column names onto our FIELDNAMES schema."""
    out = {k: row.get(k, "") for k in FIELDNAMES}
    return out


def dedupe_key(row):
    return (row.get("Date"), row.get("Market"), row.get("Variety"), row.get("Grade"))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    existing_rows, last_date = load_existing_rows()
    print(f"Loaded {len(existing_rows)} existing rows.")
    for mkt, d in last_date.items():
        print(f"  {mkt}: last date on file = {d.strftime('%d/%m/%Y')}")

    session = requests.Session()
    new_rows = []

    for market_name, market_val in MARKETS.items():
        combos = months_to_check(last_date.get(market_name))
        print(f"\n{market_name}: checking {len(combos)} month(s): "
              f"{combos[0][0]}-{combos[0][1]} .. {combos[-1][0]}-{combos[-1][1]}")

        for year, month_val in combos:
            month_name = MONTH_NUM_TO_NAME[month_val]
            print(f"  Fetching {market_name} - {month_name} {year} ...")
            try:
                html = fetch_report_html(session, month_val, year, market_val)
                rows = parse_table(html)
                for r in rows:
                    r["QueryMarket"] = market_name
                    r["QueryYear"] = year
                    r["QueryMonth"] = month_name
                    new_rows.append(normalize_row(r))
                print(f"    -> {len(rows)} rows")
            except Exception as e:
                print(f"    !! failed: {e}")

            time.sleep(DELAY_SECONDS + random.uniform(0, 1.0))

    # Merge + de-duplicate (new rows win over old ones with the same key,
    # so late corrections replace whatever was there before)
    merged = {dedupe_key(normalize_row(r)): normalize_row(r) for r in existing_rows}
    added, updated = 0, 0
    for r in new_rows:
        key = dedupe_key(r)
        if key in merged:
            if merged[key] != r:
                updated += 1
            else:
                continue
        else:
            added += 1
        merged[key] = r

    final_rows = sorted(
        merged.values(),
        key=lambda r: (r.get("Market", ""), datetime.strptime(r["Date"], "%d/%m/%Y"))
        if r.get("Date") else (r.get("Market", ""), datetime.min),
    )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\nDone. {added} new rows added, {updated} existing rows corrected.")
    print(f"Total rows now: {len(final_rows)}. Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
