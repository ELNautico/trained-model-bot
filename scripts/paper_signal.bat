@echo off
:: TrainedModel – Paper Trading Morning Signal
:: Runs the paper-trading signal engine for all watchlist tickers.
:: Scheduled daily Mon-Fri at 15:45 MEZ/MESZ (= 09:45 ET).

set REPO=C:\Users\Paul\Coding\TrainedModel
set PYTHON=%REPO%\.venv\Scripts\python.exe
set LOGFILE=%REPO%\logs\paper_signal.log

cd /d "%REPO%"
echo. >> "%LOGFILE%"
echo ===== %DATE% %TIME% – paper_signal START ===== >> "%LOGFILE%"
"%PYTHON%" -m apps.jobs forecast_paper >> "%LOGFILE%" 2>&1
echo ===== %DATE% %TIME% – paper_signal END (exit %ERRORLEVEL%) ===== >> "%LOGFILE%"
