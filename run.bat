@echo off
REM One-command demo on Windows: venv, install, offline stub run.
REM Usage: run.bat            (or: run.bat "your research question")

setlocal

if not exist .venv (
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create venv. Is Python 3.11+ on PATH?
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

pip install -q -r requirements.txt
if errorlevel 1 (
    echo pip install failed.
    exit /b 1
)

set PYTHONPATH=src

if "%~1"=="" (
    python -m research_agent.cli "Compare Redis and Memcached for session caching"
) else (
    python -m research_agent.cli %1
)

endlocal
