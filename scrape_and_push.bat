@echo off
REM Scrapes fresh price data (works from your PC's Indian IP, unlike GitHub's
REM cloud servers which the government portal blocks) and pushes the result
REM to GitHub, where the scheduled Actions workflows pick it up to retrain
REM the model, refresh the dashboard, and send the SMS.

cd /d "%~dp0"

echo ============================================== >> pipeline_log.txt
echo Local scrape+push started: %date% %time% >> pipeline_log.txt
echo ============================================== >> pipeline_log.txt

python 1_scrape_incremental.py >> pipeline_log.txt 2>&1

git add data/copra_prices_arasikere_tiptur.csv >> pipeline_log.txt 2>&1
git commit -m "Local scrape update %date% %time%" >> pipeline_log.txt 2>&1
git push >> pipeline_log.txt 2>&1

echo Local scrape+push finished: %date% %time% >> pipeline_log.txt
echo. >> pipeline_log.txt
