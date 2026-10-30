Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$analysisDir = Join-Path $PSScriptRoot "beauty_and_BMI_analysis"
Set-Location $analysisDir
# Clear a prior actionable result even when Python cannot start.
Set-Content -Path (Join-Path $analysisDir "beauty_and_BMI_analysis_result.txt") -Value "error" -Encoding ascii -NoNewline
$python = Join-Path $analysisDir "analyze_beauty.venv\Scripts\python.exe"
& $python (Join-Path $analysisDir "evaluate_images.py")
exit $LASTEXITCODE
