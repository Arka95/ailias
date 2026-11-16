param(
    [string]$SourcePython
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
$SharedEnvironment = Join-Path $WorkspaceRoot '.venv'
$Python = Join-Path $SharedEnvironment 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    if (-not $SourcePython) { throw 'Supply an existing working environment with -SourcePython; no new venv will be created.' }
    if (Test-Path -LiteralPath $SharedEnvironment) { throw 'Shared environment exists but has no interpreter; refusing to replace it.' }
    $SourcePython = (Resolve-Path -LiteralPath $SourcePython).Path
    & $SourcePython -c 'import cv2; assert cv2.dnn.DNN_TARGET_CUDA in cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CUDA)'
    if ($LASTEXITCODE -ne 0) { throw 'Existing CUDA environment validation failed' }
    $SourceEnvironment = Split-Path -Parent (Split-Path -Parent $SourcePython)
    New-Item -ItemType Junction -Path $SharedEnvironment -Target $SourceEnvironment | Out-Null
}

& $Python -c 'import cv2; assert cv2.dnn.DNN_TARGET_CUDA in cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CUDA)'
if ($LASTEXITCODE -ne 0) { throw 'Shared workspace environment does not expose CUDA DNN' }
& $Python -m pip install -e "$ProjectRoot[gpu]"
if ($LASTEXITCODE -ne 0) { throw 'Shared workspace package installation failed' }
Write-Output "Use $Python for workspace applications."
Write-Output 'Existing CUDA packages were reused; no environment or CUDA stack was copied.'
