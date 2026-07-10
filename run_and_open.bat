@echo off
REM Runs the full copra price prediction pipeline, then opens the dashboard.
REM Designed to be triggered by Windows Task Scheduler every 2 days.

cd /d "%~dp0"

echo ============================================== >> pipeline_log.txt
echo Run started: %date% %time% >> pipeline_log.txt
echo ============================================== >> pipeline_log.txt

python run_all.py >> pipeline_log.txt 2>&1

echo Run finished: %date% %time% >> pipeline_log.txt
echo. >> pipeline_log.txt

REM Auto-open the refreshed dashboard in your default browser
start "" "%~dp0dashboard.html"
