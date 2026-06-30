param(
    [string]$OnnxRuntimeRoot = $env:ONNXRUNTIME_ROOT,
    [string]$OpenCvDir = $env:OpenCV_DIR,
    [string]$BuildDir = "",
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $BuildDir) { $BuildDir = Join-Path $Root "build" }
if (-not $OnnxRuntimeRoot) { throw "Set ONNXRUNTIME_ROOT or pass -OnnxRuntimeRoot." }
if (-not $OpenCvDir) { throw "Set OpenCV_DIR or pass -OpenCvDir." }

cmake -S $Root -B $BuildDir `
    -DONNXRUNTIME_ROOT=$OnnxRuntimeRoot `
    -DOpenCV_DIR=$OpenCvDir `
    -DVLPR_BUILD_TESTS=ON
cmake --build $BuildDir --config $Configuration --parallel
ctest --test-dir $BuildDir -C $Configuration --output-on-failure

Write-Host "Built $BuildDir\$Configuration\vlpr_edge.exe"
