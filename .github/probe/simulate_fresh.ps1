# TEMP (revert before review): return this runner to the state of a freshly launched instance, so a
# job measures the fresh-instance path whichever instance it lands on.
$sw = [Diagnostics.Stopwatch]::StartNew()
uv cache clean
foreach ($dir in @(
    "$env:RUNNER_TOOL_CACHE\isaaclab-windows-ci",
    "$env:RUNNER_TOOL_CACHE\isaaclab-windows-combos",
    "$env:LOCALAPPDATA\NVIDIA\warp",
    "$env:LOCALAPPDATA\NVIDIA\DXCache",
    "$env:LOCALAPPDATA\NVIDIA\GLCache",
    "$env:LOCALAPPDATA\ov")) {
  # rmdir removes a junction without following it, which Remove-Item does not guarantee.
  if (Test-Path $dir) { cmd /c rmdir /s /q "$dir"; "removed $dir" }
}
"simulated a fresh instance in $([int]$sw.Elapsed.TotalSeconds) s"
