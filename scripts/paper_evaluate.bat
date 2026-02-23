@echo off
:: TrainedModel – Paper Trading EOD Evaluation
:: Runs stop/target/time-stop exit checks for all paper positions.
:: Scheduled daily Mon-Fri at 22:15 MEZ/MESZ (= 16:15 ET).

set REPO=C:\Users\Paul\Coding\TrainedModel
set PYTHON=%REPO%\.venv\Scripts\python.exe
set LOGFILE=%REPO%\logs\paper_evaluate.log

cd /d "%REPO%"
echo. >> "%LOGFILE%"
echo ===== %DATE% %TIME% – paper_evaluate START ===== >> "%LOGFILE%"
"%PYTHON%" -m apps.jobs evaluate_paper >> "%LOGFILE%" 2>&1
echo ===== %DATE% %TIME% – paper_evaluate END (exit %ERRORLEVEL%) ===== >> "%LOGFILE%"
