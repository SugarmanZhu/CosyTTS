#Requires -RunAsAdministrator
<#
Install CosyTTS as an auto-starting Windows service via NSSM, so it survives
reboots and restarts on crash.

Prereqs:
  winget install NSSM.NSSM
  (and the project set up per README — conda env at .venv with the model downloaded)

Run from an ELEVATED PowerShell:
  powershell -ExecutionPolicy Bypass -File scripts\install_service.ps1
#>
$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot          # repo root
$py   = Join-Path $proj ".venv\python.exe"        # conda env python lives at the ROOT
if (-not (Test-Path $py)) {
    throw "python not found at $py — set up the conda env first (see README)."
}

# Locate nssm.exe (PATH, or the winget package dir)
$nssm = (Get-Command nssm -ErrorAction SilentlyContinue).Source
if (-not $nssm) {
    $nssm = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\NSSM.NSSM*" `
                -Recurse -Filter nssm.exe -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match 'win64' } |
            Select-Object -First 1 -ExpandProperty FullName
}
if (-not $nssm) { throw "nssm.exe not found. Install it: winget install NSSM.NSSM" }

$svc = "CosyTTS"
& $nssm stop   $svc 2>$null
& $nssm remove $svc confirm 2>$null

New-Item -ItemType Directory -Force -Path (Join-Path $proj "logs") | Out-Null

& $nssm install $svc $py "-m uvicorn app.main:app --host 0.0.0.0 --port 8765 --log-level warning"
& $nssm set $svc AppDirectory   $proj
& $nssm set $svc DisplayName     "CosyTTS (CosyVoice 2 TTS service)"
& $nssm set $svc Description      "Local CosyVoice 2 HTTP TTS service on 0.0.0.0:8765"
& $nssm set $svc Start            SERVICE_AUTO_START
& $nssm set $svc AppStdout       (Join-Path $proj "logs\service.log")
& $nssm set $svc AppStderr       (Join-Path $proj "logs\service.log")
& $nssm set $svc AppRotateFiles  1
& $nssm set $svc AppRotateBytes  10485760
& $nssm set $svc AppExit Default Restart
& $nssm set $svc AppRestartDelay 5000
# The service runs as LocalSystem; point cache lookups (Whisper ~3GB, HuggingFace)
# at the installing user's profile so they're reused instead of re-downloaded.
& $nssm set $svc AppEnvironmentExtra "USERPROFILE=$env:USERPROFILE" "HOME=$env:USERPROFILE"

& $nssm start $svc
Start-Sleep -Seconds 3
Write-Host ("Service '{0}' status: {1}" -f $svc, (& $nssm status $svc))
Write-Host "First /tts call after boot is slow (~40s, CUDA + Whisper warmup); steady state ~2s."
Write-Host "Manage with: services.msc  |  nssm start/stop/restart $svc  |  logs in .\logs\service.log"
