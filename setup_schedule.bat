@echo off
REM Run this ONCE to set up automatic scraping + pushing to GitHub.
REM
REM Instead of a fixed clock time (which is useless if you don't know when
REM your PC will be on), this triggers "whenever you log into Windows" -
REM so it runs shortly after you turn your PC on, whatever time that is.
REM It also keeps a daily 6 AM trigger as a backup, in case your PC is
REM already on and logged in (no fresh "logon" event) at that time.
REM
REM scrape_and_push.bat guards against running more than once per day, so
REM having two triggers is safe - it just won't double-scrape.

cd /d "%~dp0"

REM Remove the old single-trigger task if it exists, so it's not left
REM behind duplicating work alongside the new setup below
schtasks /delete /tn "CopraPricePredictor" /f >nul 2>&1

schtasks /create ^
  /tn "CopraPricePredictor_OnLogon" ^
  /tr "\"%~dp0scrape_and_push.bat\"" ^
  /sc onlogon ^
  /delay 0002:00 ^
  /f

schtasks /create ^
  /tn "CopraPricePredictor_Daily" ^
  /tr "\"%~dp0scrape_and_push.bat\"" ^
  /sc daily ^
  /mo 1 ^
  /st 06:00 ^
  /f

if %errorlevel%==0 (
    echo.
    echo Success! Two tasks created:
    echo   - CopraPricePredictor_OnLogon: runs ~2 minutes after you log into
    echo     Windows each day ^(catches you whenever you turn your PC on^)
    echo   - CopraPricePredictor_Daily: also tries at 6:00 AM as a backup
    echo Either one running is enough - scrape_and_push.bat only actually
    echo scrapes once per day even if both fire.
    echo.
    echo To check them any time: open Task Scheduler ^(search "Task Scheduler"
    echo in the Start menu^) and look under Task Scheduler Library.
) else (
    echo.
    echo Something went wrong creating the tasks. Try running this file
    echo as Administrator ^(right-click -^> "Run as administrator"^).
)

pause
