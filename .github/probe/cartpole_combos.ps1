# TEMP (revert before review): time Cartpole camera training for every physics x renderer pair.
$ErrorActionPreference = 'Continue'
$py = "$env:UV_PROJECT_ENVIRONMENT\Scripts\python.exe"
New-Item -ItemType Directory -Force tests | Out-Null
$rows = @()
foreach ($r in 'newton_renderer', 'ovrtx', 'isaacsim_rtx') {
  foreach ($p in 'newton_mjwarp', 'newton_kamino', 'isaacsim_physx', 'ovphysx') {
    $log = "tests\combo-$p-$r.log"
    $argList = "-m isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera-Direct --num_envs 16 presets=$p,$r --max_iterations 2"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $null = $proc.Handle
    if ($proc.WaitForExit(900000)) { $rc = $proc.ExitCode } else { taskkill /T /F /PID $proc.Id | Out-Null; $rc = 'timeout' }
    $sw.Stop()
    $trained = ($rc -eq 0) -and (Select-String -Quiet -Path $log -Pattern 'Training time:')
    $rows += [pscustomobject]@{ physics = $p; renderer = $r; seconds = [int]$sw.Elapsed.TotalSeconds; rc = $rc; trained = $trained }
    '{0,-15} {1,-16} {2,5} s  rc={3}  trained={4}' -f $p, $r, [int]$sw.Elapsed.TotalSeconds, $rc, $trained
  }
}
($rows | Format-Table -AutoSize | Out-String).Trim()
"total: $(($rows | Measure-Object seconds -Sum).Sum) s, trained $(@($rows | Where-Object trained).Count)/$($rows.Count)"
