param(
    [ValidateSet('colmap', 'blender', 'all')]
    [string]$Tool = 'all'
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ToolsRoot = Join-Path (Split-Path -Parent $ProjectRoot) 'tools'
$Specs = Get-Content -LiteralPath (Join-Path $ProjectRoot 'config\generation-tools.json') -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Path $ToolsRoot -Force | Out-Null
foreach ($Name in @('colmap', 'blender')) {
    if ($Tool -ne 'all' -and $Name -ne $Tool) { continue }
    $Spec = $Specs.$Name
    $Executable = Join-Path $ToolsRoot $Spec.executable
    if (-not (Test-Path -LiteralPath $Executable)) {
        $Archive = Join-Path $ToolsRoot "$Name-$($Spec.version).zip"
        $Destination = Join-Path $ToolsRoot $Spec.extract_to
        try {
            if (-not (Test-Path -LiteralPath $Archive)) {
                & curl.exe --fail --location --retry 2 --output $Archive $Spec.url
                if ($LASTEXITCODE -ne 0) { throw "$Name download failed" }
            }
            if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Spec.sha256) {
                throw "$Name archive failed the pinned SHA256 check"
            }
            Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
            if (-not (Test-Path -LiteralPath $Executable)) { throw "$Name archive layout differs from the verified configuration" }
        } finally {
            if (Test-Path -LiteralPath $Archive) { Remove-Item -LiteralPath $Archive -Force }
        }
    }
    if ($Name -eq 'blender') { & $Executable --version } else { & $Executable -h }
    if ($LASTEXITCODE -ne 0) { throw "$Name executable verification failed" }
    Write-Output "Shared $Name ready: $Executable"
}
