@echo off
REM Run this ONCE to set up automatic daily scraping + pushing to GitHub.
REM It registers a Windows Task Scheduler job that runs scrape_and_push.bat
REM in this same folder, every morning.
REM
REM Why daily, even though the dashboard only refreshes every 2 days and the
REM SMS only goes out twice a week: scraping is cheap and this guarantees
REM fresh data is already sitting in GitHub *before* those less-frequent
REM Actions workflows run and need it.

cd /d "%~dp0"

schtasks /create ^
  /tn "CopraPricePredictor" ^
  /tr "\"%~dp0scrape_and_push.bat\"" ^
  /sc daily ^
  /mo 1 ^
  /st 06:00 ^
  /f

if %errorlevel%==0 (
    echo.
    echo Success! "CopraPricePredictor" task created.
    echo It will run every day at 6:00 AM: scrape the latest price data
    echo and push it to your GitHub repo. From there, GitHub Actions takes
    echo over automatically - retraining the model, refreshing the
    echo dashboard every 2 days, and sending the Tiptur SMS every
    echo Tuesday/Saturday.
    echo.
    echo To check it any time: open Task Scheduler ^(search "Task Scheduler"
    echo in the Start menu^) and look under Task Scheduler Library.
    echo To change the time or run it manually, right-click the task there.
) else (
    echo.
    echo Something went wrong creating the task. Try running this file
    echo as Administrator ^(right-click -^> "Run as administrator"^).
)

pause
