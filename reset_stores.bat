@echo off
REM Reset Qdrant + OpenSearch + Postgres to a pristine state, then re-ingest.
REM
REM Usage:
REM   reset_stores.bat              preview only (dry run, nothing deleted)
REM   reset_stores.bat --yes        delete, then re-ingest the sample corpus
REM   reset_stores.bat --yes --keep-memory   keep the semantic-memory collection
REM
REM The file is reset_stores.bat and these lines said reset.bat, including
REM the echo below -- which told an operator to re-run a command that does
REM not exist. The FILENAME is the source of truth; a comment that
REM disagrees with the file it sits in is the one kind that cannot be
REM right. Matches scripts/reset_stores.py, which it launches.
REM
REM Any arguments are passed straight through to scripts\reset_stores.py,
REM which since D-157 is a launcher for research_agent.ops.reset_stores.

setlocal

if not exist .venv (
    echo No .venv found. Run run.bat once first.
    exit /b 1
)

call .venv\Scripts\activate.bat
set PYTHONPATH=src

if "%~1"=="" (
    python scripts\reset_stores.py --dry-run
    echo.
    echo Nothing was deleted. Re-run as:  reset_stores.bat --yes
    endlocal
    exit /b 0
)

python scripts\reset_stores.py %*
if errorlevel 1 (
    echo Reset reported an unreachable store. Fix it, then re-run.
    endlocal
    exit /b 1
)

echo.
echo Re-ingesting sample corpus...
python scripts\ingest_sample_data.py

endlocal