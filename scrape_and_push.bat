@echo off
REM Scrapes fresh price data (works from your PC's Indian IP, unlike GitHub's
REM cloud servers which the government portal blocks) and pushes the result
REM to GitHub, where the scheduled Actions workflows pick it up to retrain
REM the model, refresh the dashboard, and send the SMS.
REM
REM Guard: only actually scrapes once per calendar day, even if this gets
REM triggered multiple times (e.g. logging in and out several times) - so
REM we're not hammering the government site unnecessarily.

cd /d "%~dp0"

set TODAY=%date%
set MARKER=last_run_date.txt

if exist "%MARKER%" (
    set /p LAST_RUN=<"%MARKER%"
    if "%LAST_RUN%"=="%TODAY%" (
        echo Already ran today ^(%TODAY%^) - skipping. >> pipeline_log.txt
        exit /b 0
    )
)

echo ============================================== >> pipeline_log.txt
echo Local scrape+push started: %date% %time% >> pipeline_log.txt
echo ============================================== >> pipeline_log.txt

python 1_scrape_incremental.py >> pipeline_log.txt 2>&1

git add data/copra_prices_arasikere_tiptur.csv >> pipeline_log.txt 2>&1
git commit -m "Local scrape update %date% %time%" >> pipeline_log.txt 2>&1
git push >> pipeline_log.txt 2>&1

echo %TODAY% > "%MARKER%"

echo Local scrape+push finished: %date% %time% >> pipeline_log.txt
echo. >> pipeline_log.txt
