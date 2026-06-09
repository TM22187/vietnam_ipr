$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

python -m PyInstaller `
  --noconfirm `
  --clean `
  --onedir `
  --windowed `
  --name VietnamLPR `
  --add-data "weights;weights" `
  --add-data "config;config" `
  --add-data "scripts;scripts" `
  app.py

Write-Host ""
Write-Host "Build complete: dist\VietnamLPR\VietnamLPR.exe"
