# TEMP (revert before review): print what this runner holds in each cache.
param([string]$Label)
"[$Label] runner=$env:RUNNER_NAME booted=$((Get-CimInstance Win32_OperatingSystem).LastBootUpTime)"
function Show-Size($path) {
  if (Test-Path $path) {
    $m = Get-ChildItem -Recurse -File -Force $path -ErrorAction SilentlyContinue | Measure-Object Length -Sum
    '{0} : {1} files, {2} MB' -f $path, $m.Count, [int]($m.Sum / 1MB)
  } else { "$path : missing" }
}
$site = "$env:UV_PROJECT_ENVIRONMENT\Lib\site-packages"
$paths = @((uv cache dir), $env:UV_PROJECT_ENVIRONMENT, "$env:LOCALAPPDATA\NVIDIA\warp\Cache",
  "$env:RUNNER_TEMP\isaaclab-warp-cache", "$env:RUNNER_TEMP\isaaclab-ovrtx-shader-cache\kit",
  "$env:RUNNER_TEMP\isaaclab-ovrtx-shader-cache\kitless")
foreach ($path in $paths) { Show-Size $path }
foreach ($dir in "$site\isaacsim\kit\cache", "$site\ovrtx\bin\cache") {
  if (Test-Path $dir) {
    Get-ChildItem -Force $dir | ForEach-Object {
      $tag = if ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) { ' (junction)' } else { '' }
      (Show-Size $_.FullName) + $tag
    }
  }
}
if (Test-Path "$site\isaacsim\kit\logs") {
  Get-ChildItem -Recurse "$site\isaacsim\kit\logs" -Filter *.log |
    Select-String -Pattern 'driverShaderCachePath|driverAppShaderCachePath' |
    Select-Object -First 4 | ForEach-Object { $_.Line }
}
