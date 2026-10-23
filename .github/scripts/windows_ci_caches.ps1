# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# The Warp kernel and OVRTX shader caches of the Linux test jobs (.github/actions/warp-cache-key,
# .github/actions/ovrtx-shader-cache), each in its own Windows collection. Those actions run bash,
# which resolves to WSL on these runners.
#
# Usage: windows_ci_caches.ps1 keys|restored|growth
#
#   keys      resolves each cache's key and directory and exports the paths the tests read
#   restored  reports each restore and checks the paths took effect; a save also fingerprints them
#   growth    reports each cache's size; a save also reports whether the run changed it
#
# The directories survive on a reused runner, so a pull request keeps what earlier jobs compiled.
# CACHE_MODE=save clears them before the restore, so a save publishes only the restored snapshot
# and what that run compiled.

param([Parameter(Mandatory)][ValidateSet('keys', 'restored', 'growth')][string]$Mode)

$saving = $env:CACHE_MODE -eq 'save'
# warp: Warp kernels. kit: driver shaders Kit's RTX renderer compiles into its default directory.
# kitless: driver shaders the OVRTX renderer compiles, redirected by OVRTX_SHADER_CACHE_PATH.
$caches = [ordered]@{
    warp    = "$env:RUNNER_TOOL_CACHE\isaaclab-warp-cache"
    kit     = "$env:UV_PROJECT_ENVIRONMENT\Lib\site-packages\isaacsim\kit\cache\nv_shadercache"
    kitless = "$env:RUNNER_TOOL_CACHE\isaaclab-ovrtx-shader-cache"
}

function Write-StepOutput([string[]]$Lines) {
    $Lines | Out-File -FilePath $env:GITHUB_OUTPUT -Encoding utf8 -Append
}

function Get-Fingerprint([string]$Dir) {
    uv run --frozen --no-sync python .github/actions/run-package-tests/warp_cache_inventory.py $Dir --fingerprint
}

function Get-Megabytes([string]$Dir) {
    [int]((Get-ChildItem -Recurse -File -Force $Dir | Measure-Object Length -Sum).Sum / 1MB)
}

function ConvertTo-KeyPart([string]$Value) {
    $Value -replace '[^A-Za-z0-9._-]', '_'
}

switch ($Mode) {
    'keys' {
        $warpVersions = uv run --frozen --no-sync python .github/actions/warp-cache-key/collection_id.py
        if ($LASTEXITCODE -ne 0) { exit 1 }
        $driver = (nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim()
        $arch = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim().Replace('.', '')
        # The installed versions stand in for the image tag and the ovrtx pin the Linux keys use.
        $isaacsim, $ovrtx = (uv run --frozen --no-sync python -c "from importlib.metadata import version; print(version('isaacsim'), version('ovrtx'))") -split ' '
        if (-not $driver -or -not $arch -or -not $ovrtx) {
            "::error::Could not read the GPU driver, compute capability, or Isaac Sim and ovrtx versions"
            exit 1
        }
        $gpu = "v1-$env:RUNNER_OS-$env:RUNNER_ARCH-sm$arch-drv$(ConvertTo-KeyPart $driver)"
        $collections = @{
            warp    = "warp-v2-$env:RUNNER_OS-$env:RUNNER_ARCH-$warpVersions"
            kit     = "ovrtx-kit-$gpu-isaacsim$(ConvertTo-KeyPart $isaacsim)"
            kitless = "ovrtx-kitless-$gpu-ovrtx$(ConvertTo-KeyPart $ovrtx)"
        }
        foreach ($name in $caches.Keys) {
            $dir = $caches[$name]
            if ($saving -and (Test-Path $dir)) { Remove-Item -Recurse -Force $dir }
            New-Item -ItemType Directory -Force $dir | Out-Null
            $collection = $collections[$name]
            "${name}: $collection"
            Write-StepOutput @(
                "$name-collection=$collection"
                "$name-key=$collection-$env:GITHUB_SHA-$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT"
                "$name-restore-keys=$collection-"
                "$name-dir=$dir"
            )
        }
        @(
            "WARP_CACHE_PATH=$($caches.warp)"
            "OVRTX_SHADER_CACHE_PATH=$($caches.kitless)"
            "OVRTX_KIT_SHADER_CACHE_PATH=$($caches.kit)"
        ) | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    }
    'restored' {
        foreach ($name in $caches.Keys) {
            $hit = [Environment]::GetEnvironmentVariable("$($name.ToUpper())_HIT")
            if ($hit) { "$name cache hit: $hit" } else { "::warning::$name cache miss" }
            "$name cache: $(Get-Megabytes $caches[$name]) MB"
        }
        # tools/verify_warp_cache.py's path check, which fails on the \\?\ prefix Warp reports on
        # Windows; its version check always passes on an environment synced from uv.lock.
        $reported = uv run --frozen --no-sync python -c "import warp as wp; wp.config.log_level = wp.LOG_WARNING; wp.init(); print(wp.config.kernel_cache_dir)" | Select-Object -Last 1
        if (-not "$reported".Replace('\\?\', '').StartsWith($caches.warp, [StringComparison]::OrdinalIgnoreCase)) {
            "::error::Warp resolved its kernel cache to $reported, not $($caches.warp)"
            exit 1
        }
        uv run --frozen --no-sync python tools/verify_ovrtx_shader_cache.py
        if ($LASTEXITCODE -ne 0) { exit 1 }
        if ($saving) {
            foreach ($name in $caches.Keys) { Write-StepOutput "$name-fingerprint=$(Get-Fingerprint $caches[$name])" }
        }
    }
    'growth' {
        foreach ($name in $caches.Keys) {
            $dir = $caches[$name]
            # Every snapshot is a new immutable cache entry, so a save that only read back what it
            # restored, or found nothing, publishes nothing.
            $changed = 'false'
            if ($saving -and (Get-ChildItem -Recurse -File $dir | Select-Object -First 1)) {
                $before = [Environment]::GetEnvironmentVariable("$($name.ToUpper())_BEFORE")
                if ((Get-Fingerprint $dir) -ne $before) { $changed = 'true' }
            }
            "$name cache: $(Get-Megabytes $dir) MB, changed=$changed"
            Write-StepOutput "$name-changed=$changed"
        }
    }
}
