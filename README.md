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
| `6_send_sms_forecast.py` | Sends the Tiptur-only Kannada SMS forecast (run separately, see below) |

## Model improvements (v2)

Three changes made a real difference, in order of impact:

1. **Predicting price change, not price level.** Tree models (XGBoost/
   RandomForest) can't output a number outside the range they were trained
   on - so a model trained directly on "Modal price" is quietly capped
   near the highest price it's ever seen, even as real prices keep
   climbing. Now the model predicts the **log-return** (how much the price
   moves from the last known tender), and we reconstruct the actual price
   as `last_price * exp(predicted_return)`. This lets predictions extend
   past any price level seen in the 24-year history. This was the single
   biggest accuracy gain: Arasikere's error dropped from ~19% to ~6%,
   Tiptur's from ~7% to ~2%.
2. **Cross-market feature.** Tiptur and Arasikere are ~50km apart and
   trade the same commodity, so each market's most recent price now feeds
   into the other's model as a feature (`other_market_last_price`) -
   using a strictly-past-only join so there's no lookahead leakage.
3. **Walk-forward cross-validation.** Instead of one train/test split at
   the end of history (which gives a single noisy accuracy estimate),
   `4_train_model.py` now runs 5 expanding-window folds and reports the
   average - a more honest picture of real-world accuracy, since it's not
   dependent on whether the last few months happened to be unusually calm
   or volatile.

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

## Model improvements (log-return target + walk-forward CV + auto-tuning)

The model has gone through several rounds of real improvement, each
verified against your actual 24-year price history rather than assumed:

**1. Predicting price CHANGE, not price LEVEL.** Tree-based models
(XGBoost/RandomForest) predict by averaging training examples — they
physically cannot output a number above the highest price ever seen in
training. Since copra prices have climbed for 24 years (~Rs 3,000 in 2002
to Rs 31,000+ now), a model trained on raw price was quietly capped near
its training ceiling. Fix: predict `log(next_price / last_price)` instead
— a small, stable ratio regardless of the price level — then reconstruct
`predicted_price = last_price × exp(predicted_ratio)`. This lets the
model extrapolate beyond prices it's ever seen, because it's the *ratio*
being predicted, not the absolute number.

**2. Walk-forward cross-validation instead of one train/test split.**
A single holdout at the end of history gives one noisy accuracy estimate.
Walk-forward CV trains on an expanding window and tests on 5 different
subsequent chunks in turn (always past → future, never shuffled), giving
a more honest average.

**3. More/better features:** additional lags (up to 5 tenders back), a
mean-reversion signal (`price_vs_trend_ratio` — is the price stretched
above/below its recent trend?), the actual gap in days since the previous
tender (tenders aren't perfectly regular — holidays skip some), and a
cross-market feature (Tiptur and Arasikere are ~50km apart and trade the
same commodity, so one market's latest price is a real signal for the
other — computed leak-free with `merge_asof`, only ever looking backward).

**4. Automatic hyperparameter selection.** Instead of one guessed XGBoost
configuration, `4_train_model.py` now tries 3 candidate configs per
market and keeps whichever scored best on walk-forward CV — an honest,
data-driven choice instead of a guess.

**Result:** Tiptur improved from ~7% typical error to ~2%, Arasikere from
~18% to ~5% (numbers from a real run on your data — check
`models/metrics_tiptur.json` / `models/metrics_arsikere.json` after each
retrain for the current numbers, which will drift over time as more data
comes in).

## SMS forecast to your father (Kannada, Tiptur only, 2 days before each tender)

Since he uses a keypad phone, this sends a plain SMS (not an app/internet
notification) 2 days before each Tiptur tender (Saturday for Monday's
tender, Tuesday for Thursday's tender), with:
- the predicted price for that tender
- which month has historically been the best one to hold/sell in

**Why this doesn't need India's DLT/bulk-SMS registration:** the message is
sent from an ordinary Android phone's own SIM, via the free open-source
[SMS Gateway for Android](https://github.com/capcom6/android-sms-gateway)
app — technically identical to you typing and sending the text yourself.
DLT rules target commercial/bulk senders, not a personal message from your
own number.

**One-time setup:**
1. Install the app on any Android phone you have access to (doesn't need
   to be your daily phone — just needs battery + signal on Tuesday/Saturday
   mornings): download the APK from the
   [releases page](https://github.com/capcom6/android-sms-gateway/releases).
2. Open the app → toggle **Cloud Server** → tap **Online**. It'll display a
   username and password — keep the app installed and don't clear its data.
3. In your GitHub repo: **Settings → Secrets and variables → Actions →
   New repository secret**, add three secrets:
   - `SMS_GATEWAY_USER` — the username from step 2
   - `SMS_GATEWAY_PASS` — the password from step 2
   - `FATHER_PHONE_NUMBER` — his number with country code, e.g. `+919xxxxxxxxx`
4. That's it — `.github/workflows/send_sms.yml` runs every Tuesday and
   Saturday at 8:00 AM IST automatically.

**Test it manually first:** Actions tab → "Send Tiptur SMS Forecast" →
"Run workflow" → check your father's phone within a minute or two.

**Editing the message:** `6_send_sms_forecast.py` builds the text in
`compose_message()` — edit the Kannada wording there directly if you want
to change phrasing (I've written it in plain, simple Kannada, but you know
your father's phrasing preferences better than I do — feel free to adjust).

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
