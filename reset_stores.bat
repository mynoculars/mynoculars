@echo off
REM Reset Qdrant + OpenSearch + Postgres to a pristine state, then re-ingest.
REM
REM Usage:
REM   reset.bat              preview only (dry run, nothing is deleted)
REM   reset.bat --yes        delete, then re-ingest the sample corpus
REM   reset.bat --yes --keep-memory   keep the semantic-memory collection
REM
REM Any arguments are passed straight through to scripts\reset_stores.py.

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
    echo Nothing was deleted. Re-run as:  reset.bat --yes
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