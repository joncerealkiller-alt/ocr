# Starts the WSL agent/chat compute backend (api/agent_main.py) that
# model_console talks to over HTTP when Backend=remote is selected.
# See docs/WSL_COMPUTE_BACKEND_BASELINE.md for the full architecture.
#
# Double-click via a desktop shortcut targeting:
#   powershell.exe -NoExit -ExecutionPolicy Bypass -File "J:\Genealogy\genealogy_pipeline\start_backend.ps1"

$ErrorActionPreference = "Stop"

Write-Host "Checking for an already-running backend..." -ForegroundColor Cyan
try {
    $status = Invoke-RestMethod -Uri "http://localhost:8001/model/status" -TimeoutSec 3
    Write-Host "Backend is already running (resident model: $($status.resident_model_name))." -ForegroundColor Yellow
    Write-Host "Nothing to do - close this window or Ctrl+C to exit."
    Read-Host "Press Enter to exit"
    exit 0
} catch {
    Write-Host "No backend running yet - starting it now." -ForegroundColor Cyan
}

wsl -d Ubuntu-24.04 -- bash -c "source ~/venv_backend/bin/activate && cd /mnt/j/Genealogy/genealogy_pipeline && uvicorn api.agent_main:app --host 0.0.0.0 --port 8001"
