param([string]$PythonVersion = "-3.13")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VenvPython = Join-Path $Root ".venv-build\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & py $PythonVersion -m venv (Join-Path $Root ".venv-build")
}

& $VenvPython -m pip install -r (Join-Path $Root "requirements-build.txt")
& $VenvPython -m PyInstaller --noconfirm --clean --workpath (Join-Path $Root ".build") (Join-Path $Root "VietnamLPR.spec")

$InnoCandidates = @(
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe"
)
$InnoCompiler = $InnoCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($InnoCompiler) {
    & $InnoCompiler (Join-Path $Root "installer.iss")
    Write-Host "Installer: $Root\dist\installer\VietnamLPR-Setup-1.0.0.exe"
} else {
    Write-Warning "Chưa có Inno Setup 6. Bản portable đã sẵn sàng tại dist\VietnamLPR."
    Write-Warning "Cài Inno Setup 6 rồi chạy lại script để tạo file Setup.exe."
}
