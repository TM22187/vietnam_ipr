param(
    [string]$PythonVersion = "-3.13",
    [switch]$SkipDependencyInstall
)

$ErrorActionPreference = "Stop"
$Version = "1.1.0"
$ExpectedModelHash = "8893A6333E6FDA86A47CC36294A40A9C26422E71F42DE75F0046D9F0C7A986E4"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VenvPython = Join-Path $Root ".venv-build\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & py $PythonVersion -m venv (Join-Path $Root ".venv-build")
}
if (-not $SkipDependencyInstall) {
    & $VenvPython -m pip install -r (Join-Path $Root "requirements-lock.txt")
}

$ModelPath = Join-Path $Root "models\best_vietnam_lpr.onnx"
if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Missing model: $ModelPath"
}
$ModelHash = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash
if ($ModelHash -ne $ExpectedModelHash) {
    throw "Model checksum mismatch. Expected $ExpectedModelHash, got $ModelHash"
}

& $VenvPython (Join-Path $Root "tools\generate_icon.py")
& $VenvPython -m unittest discover -s (Join-Path $Root "tests") -v
& $VenvPython -m PyInstaller --noconfirm --clean --workpath (Join-Path $Root ".build") (Join-Path $Root "VietnamLPR.spec")

$PortableExe = Join-Path $Root "dist\VietnamLPR\VietnamLPR.exe"

function Sign-Binary([string]$Path) {
    if (-not $env:CODE_SIGN_CERT_SHA1) {
        return
    }
    $SignTool = if ($env:SIGNTOOL_PATH) { $env:SIGNTOOL_PATH } else { "signtool.exe" }
    $TimestampUrl = if ($env:CODE_SIGN_TIMESTAMP_URL) { $env:CODE_SIGN_TIMESTAMP_URL } else { "http://timestamp.digicert.com" }
    & $SignTool sign /sha1 $env:CODE_SIGN_CERT_SHA1 /fd SHA256 /tr $TimestampUrl /td SHA256 $Path
}

Sign-Binary $PortableExe
$Smoke = Start-Process -FilePath $PortableExe -ArgumentList "--smoke-test" -Wait -PassThru -WindowStyle Hidden
if ($Smoke.ExitCode -ne 0) {
    throw "Packaged smoke test failed with exit code $($Smoke.ExitCode)"
}

$Commit = (& git rev-parse --short HEAD 2>$null)
$BuildInfo = [ordered]@{
    product = "Vietnam LPR"
    version = $Version
    commit = $Commit
    built_at_utc = [DateTime]::UtcNow.ToString("o")
    model_sha256 = $ModelHash.ToLowerInvariant()
    architecture = "x64"
}
$BuildInfoJson = $BuildInfo | ConvertTo-Json
[System.IO.File]::WriteAllText(
    (Join-Path $Root "dist\VietnamLPR\build-info.json"),
    $BuildInfoJson + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)

$InnoCandidates = @(
    $env:INNO_SETUP_COMPILER,
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe",
    "C:\tmp\InnoSetup\ISCC.exe"
) | Where-Object { $_ }
$InnoCompiler = $InnoCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($InnoCompiler) {
    & $InnoCompiler (Join-Path $Root "installer.iss")
    $Installer = Join-Path $Root "dist\installer\VietnamLPR-Setup-$Version.exe"
    Sign-Binary $Installer
    $InstallerHash = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash
    Write-Host "Installer: $Installer"
    Write-Host "SHA-256:   $InstallerHash"
} else {
    Write-Warning "Inno Setup 6 not found. Portable build is ready at dist\VietnamLPR."
}
