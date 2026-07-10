@echo off
REM Run this ONCE to set up automatic refresh every 2 days.
REM It registers a Windows Task Scheduler job that runs run_and_open.bat
REM in this same folder, on a repeating 2-day schedule.

cd /d "%~dp0"

schtasks /create ^
  /tn "CopraPricePredictor" ^
  /tr "\"%~dp0run_and_open.bat\"" ^
  /sc daily ^
  /mo 2 ^
  /st 07:00 ^
  /f

if %errorlevel%==0 (
    echo.
    echo Success! "CopraPricePredictor" task created.
    echo It will run every 2 days at 7:00 AM, refresh the data/model,
    echo and open the updated dashboard in your browser.
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
