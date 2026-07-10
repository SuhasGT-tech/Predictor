# Copra Price Predictor — Arasikere & Tiptur

Predicts the Modal (most common) copra price for the **next tender day**:
- **Tiptur** — Monday & Thursday
- **Arasikere** — Tuesday & Friday

## How to run it

```bash
pip install -r requirements.txt --break-system-packages
python run_all.py
```

Then open `dashboard.html` in your browser. That's it.

Run this **after each tender day** (or daily/weekly — it's cheap now) to keep
the model current. Each run re-scrapes only the last few months, refetches
external data, retrains, and regenerates the dashboard — this is your
"incremental learning" loop.

## Making it auto-refresh every 2 days — Option A: GitHub Actions (recommended)

This runs on GitHub's servers instead of your PC — no need to keep your
computer on, and the dashboard is reachable from any device via a URL.

**One-time setup:**

1. **Create a GitHub account** if you don't have one (github.com — free).
2. **Create a new repository** — click the "+" top-right → "New repository".
   Name it e.g. `copra-price-predictor`. Choose **Public** (required for
   free GitHub Pages hosting — see note below if you need it private).
3. **Upload this whole folder** to that repo. Easiest way without git
   command-line experience: on the repo page, click "Add file" → "Upload
   files", drag the entire contents of this folder in, commit.
4. **Enable GitHub Pages:** repo → Settings → Pages (left sidebar) →
   under "Build and deployment" → Source: "Deploy from a branch" →
   Branch: `main`, folder: `/docs` → Save.
5. **Turn on Actions if prompted:** repo → Actions tab → if it asks you to
   enable workflows, click "I understand my workflows, go ahead and
   enable them."
6. **Trigger the first run manually** to test it: Actions tab → click
   "Update Copra Price Dashboard" (left side) → "Run workflow" button → Run.
   Wait ~2-3 minutes, refresh the page — you should see a green checkmark.
7. **Find your dashboard URL:** Settings → Pages will show something like
   `https://<your-username>.github.io/copra-price-predictor/` — bookmark
   that. It'll always show the latest run's result.

From here on, `.github/workflows/update_dashboard.yml` runs the whole
pipeline automatically every 2 days and updates that URL — no PC required,
no manual steps.

**Note on private repos:** GitHub Pages needs a paid plan to work on
private repos. If your price data is sensitive and you don't want it
public, keep the repo private and skip step 4 — the workflow will still
run and commit the refreshed `dashboard.html` to the repo every 2 days;
you'd just open it by pulling the repo instead of via a public URL.

## Making it auto-refresh every 2 days — Option B: Windows Task Scheduler (local)

There's no cloud server here — `dashboard.html` is a file that lives on
your computer permanently and stays exactly as it was until something
regenerates it. "Always available" + "refreshes every 2 days" means:
the file is always there to open, and a scheduled task keeps its contents
current.

**One-time setup:**
1. Double-click `setup_schedule.bat` (if Windows complains, right-click →
   "Run as administrator").
2. That's it — a Windows Task Scheduler job called **CopraPricePredictor**
   now runs every 2 days at 7:00 AM: it re-scrapes, refetches external data,
   retrains, regenerates `dashboard.html`, and **opens it automatically**
   in your browser.

**What each file does:**
- `run_and_open.bat` — runs the full pipeline, logs output to
  `pipeline_log.txt`, then opens the refreshed dashboard
- `setup_schedule.bat` — registers the recurring task (run once)

**To change the schedule** (e.g. every day instead of every 2, or a
different time): open Task Scheduler from the Start menu → Task Scheduler
Library → right-click **CopraPricePredictor** → Properties → Triggers tab.

**Caveats:**
- Your PC needs to be on (not asleep) at the scheduled time. If it's
  usually asleep, open the task's Properties → Conditions tab → check
  "Wake the computer to run this task."
- If you want the dashboard reachable from your phone too, you'd need to
  actually host it somewhere (e.g. a free static site host, or a small
  local web server left running) — let me know if you want that set up
  instead of/alongside the local file.

## The 5 steps (you can also run these one at a time)

| Script | What it does |
|---|---|
| `1_scrape_incremental.py` | Pulls new/recent price data from the Karnataka Krama portal, merges into `data/copra_prices_arasikere_tiptur.csv` |
| `2_fetch_external_factors.py` | Fetches global coconut oil price, USD/INR rate, weather, festival calendar → `data/external_factors.csv` |
| `3_build_features.py` | Combines everything into model-ready features → `data/features.csv` |
| `4_train_model.py` | Trains one model per market, saves to `models/` |
| `5_generate_dashboard.py` | Predicts the next tender price, builds `dashboard.html` |

## How the ML actually works (the short version)

**This is not "AI thinking about copra prices."** It's pattern-matching on
history. Concretely, for every past tender day we know the actual Modal
price, and we build a table where each row is one tender day with columns
like:

```
modal_lag_1 (yesterday's-tender price), modal_roll_mean_4 (avg of last 4
tenders), arrivals, day_of_week, coconut_oil_price, rainfall, ...  ->  Modal (target)
```

We train a model (XGBoost — an ensemble of decision trees) to learn the
function `f(lags, seasonality, external factors) → next price`. It works
by fitting hundreds of small decision trees, each correcting the errors of
the ones before it (this is what "boosting" means). For a next-tender
prediction, we don't know that day's actual weather/exchange
rate/global price yet — so the script uses the most recent known values as
the best available estimate. This is the single biggest source of
uncertainty in the forecast; it's a reasonable approximation but not
omniscient.

**Why time-series validation, not random shuffling:** `4_train_model.py`
always tests on the most *recent* 15% of history, never a random sample.
Randomly shuffling would let the model "peek" at future prices while
training — a classic beginner mistake that makes accuracy look great on
paper and terrible in practice.

**Why Modal price weighted by grade/arrivals:** the raw site data has
multiple grades (FAQ etc.) per market per day sometimes. We collapse these
to one arrivals-weighted price per market/day so "price" means one
consistent thing.

## Current accuracy (from the last training run)

Check `models/metrics_tiptur.json` and `models/metrics_arsikere.json` after
each run — they contain MAE (average error in Rs) and MAPE (average error
in %). On the initial backtest with this data:

- Tiptur: ~7% typical error
- Arasikere: ~18% typical error (Arasikere's data has more volatility/gaps
  historically — as you accumulate more recent data this should improve)

These will change (hopefully improve) once `xgboost` is installed (falls
back to RandomForest otherwise) and once `2_fetch_external_factors.py` has
pulled real weather/oil-price/fx data rather than running on price history
alone.

## Where "other factors" live, and how to add more

Currently included: global coconut oil price, USD/INR rate, Hassan/Tumakuru
rainfall & temperature, India festival calendar (via the `holidays` package).

To add more factors (e.g. Kerala/Karnataka-specific festival dates the
`holidays` package misses, palm oil prices as a substitute-good signal,
diesel/transport cost, or a specific mandi's arrival forecast):

1. Add a fetch function in `2_fetch_external_factors.py` that returns a
   `(date, value)` dataframe.
2. Merge it into the `daily` dataframe in that script's `main()`.
3. Reference the new column in `3_build_features.py`'s
   `add_external_features()`.
4. Re-run `run_all.py` — the model automatically picks up the new column
   as a feature (XGBoost handles new numeric features with no other code
   changes needed).

## Honest limitations

- **This predicts a single number, not a guarantee.** Agricultural
  commodity prices respond to things no model captures well — a sudden
  government policy change, a local crop disease outbreak, a trader
  cartel move, etc. Treat the prediction as one input, not the final word.
- **The "next tender day" features for weather/fx/oil price are
  estimates** (most recent known value), since we can't know the future.
  If you want to get fancier later, you could plug in a weather *forecast*
  API (e.g. Open-Meteo also has a forecast endpoint) instead of persistence.
- **No live/scheduled automation** — nothing runs on its own in the
  background. You (or a cron job / Windows Task Scheduler entry you set up
  on your own machine) need to trigger `run_all.py`. If you want, I can
  also show you how to set up a scheduled task so this runs automatically
  every tender-day morning.
