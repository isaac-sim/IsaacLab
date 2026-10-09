# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# TEMP diag: report the folders the job wrote to since the last mark, to locate the shader caches.
# Usage: ci_cache_probe.ps1 mark | ci_cache_probe.ps1 report <label>
param([string]$Mode, [string]$Label = "")

$markFile = Join-Path $env:RUNNER_TEMP "cache-probe-mark.txt"
if ($Mode -eq "mark") {
  (Get-Date).ToUniversalTime().ToString("o") | Set-Content -LiteralPath $markFile
  exit 0
}

$since = [datetime]::Parse((Get-Content -LiteralPath $markFile), $null, [Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
$uvCache = "$env:LOCALAPPDATA\uv\"
Write-Host "TEMP=$env:TEMP LOCALAPPDATA=$env:LOCALAPPDATA USERPROFILE=$env:USERPROFILE RUNNER_TEMP=$env:RUNNER_TEMP"
$roots = @($env:GITHUB_WORKSPACE, $env:USERPROFILE, $env:RUNNER_TEMP, "$env:ProgramData\NVIDIA Corporation", "$env:ProgramData\NVIDIA")
foreach ($root in $roots) {
  if (-not (Test-Path -LiteralPath $root)) { Write-Host "=== [$Label] $root : missing"; continue }
  $files = @(Get-ChildItem -LiteralPath $root -File -Recurse -Force -ErrorAction SilentlyContinue |
      Where-Object { $_.LastWriteTimeUtc -gt $since -and -not $_.FullName.StartsWith($uvCache, "OrdinalIgnoreCase") })
  $mb = [math]::Round((($files | Measure-Object Length -Sum).Sum) / 1MB, 1)
  Write-Host "=== [$Label] $root : $($files.Count) files, $mb MB written since $($since.ToString('o'))"
  $files |
    Group-Object { (($_.DirectoryName.Substring($root.Length).TrimStart('\') -split '\\') | Select-Object -First 6) -join '\' } |
    ForEach-Object { [pscustomobject]@{ MB = [math]::Round((($_.Group | Measure-Object Length -Sum).Sum) / 1MB, 1); Files = $_.Count; Dir = $_.Name } } |
    Sort-Object MB -Descending | Select-Object -First 20 | Format-Table -AutoSize | Out-String -Width 300 | Write-Host
}
