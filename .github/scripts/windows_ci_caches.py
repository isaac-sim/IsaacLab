# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Warp kernel and OVRTX shader caches of the Linux test jobs, each in its own Windows collection.

The Linux actions (.github/actions/warp-cache-key, .github/actions/ovrtx-shader-cache) run bash,
which resolves to WSL on the Windows runners.

Usage: windows_ci_caches.py keys|restored|growth

  keys      resolves each cache's key and directory and exports the paths the tests read
  restored  reports each restore and checks Warp uses its path; a save also fingerprints them
  growth    reports each cache's size; a save also reports whether the run changed it

The directories survive on a reused runner, so a pull request keeps what earlier jobs compiled.
CACHE_MODE=save clears them before the restore, so a save publishes only the restored snapshot
and what that run compiled.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAVING = os.environ.get("CACHE_MODE") == "save"
# warp: Warp kernels. kit: driver shaders Kit's RTX renderer compiles into its default directory.
# kitless: driver shaders the OVRTX renderer compiles, redirected by OVRTX_SHADER_CACHE_PATH.
CACHES = {
    "warp": Path(os.environ["RUNNER_TOOL_CACHE"], "isaaclab-warp-cache"),
    "kit": Path(
        os.environ["UV_PROJECT_ENVIRONMENT"], "Lib", "site-packages", "isaacsim", "kit", "cache", "nv_shadercache"
    ),
    "kitless": Path(os.environ["RUNNER_TOOL_CACHE"], "isaaclab-ovrtx-shader-cache"),
}


def _write_lines(variable: str, lines: list[str]) -> None:
    with open(os.environ[variable], "a", encoding="utf-8") as handle:
        handle.writelines(f"{line}\n" for line in lines)


def _run_script(path: str) -> str:
    """Return the stdout of one of the repository's Python scripts, failing the step if it fails."""
    result = subprocess.run([sys.executable, str(REPO_ROOT / path)], capture_output=True, text=True, check=False)
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        sys.exit(1)
    return result.stdout.strip()


def _fingerprint(path: Path) -> str:
    spec = importlib.util.spec_from_file_location(
        "warp_cache_inventory", REPO_ROOT / ".github/actions/run-package-tests/warp_cache_inventory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fingerprint(path)


def _megabytes(path: Path) -> int:
    return round(sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / 2**20)


def _gpu_query(field: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"], capture_output=True, text=True, check=False
    )
    return next(iter(result.stdout.splitlines()), "").strip()


def _key_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def keys() -> None:
    warp_versions = _run_script(".github/actions/warp-cache-key/collection_id.py")
    driver = _gpu_query("driver_version")
    arch = _gpu_query("compute_cap").replace(".", "")
    if not driver or not arch:
        print("::error::Could not read the GPU driver or compute capability")
        sys.exit(1)
    # The installed versions stand in for the image tag and the ovrtx pin the Linux keys use.
    gpu = f"v1-{os.environ['RUNNER_OS']}-{os.environ['RUNNER_ARCH']}-sm{arch}-drv{_key_part(driver)}"
    collections = {
        "warp": f"warp-v2-{os.environ['RUNNER_OS']}-{os.environ['RUNNER_ARCH']}-{warp_versions}",
        "kit": f"ovrtx-kit-{gpu}-isaacsim{_key_part(version('isaacsim'))}",
        "kitless": f"ovrtx-kitless-{gpu}-ovrtx{_key_part(version('ovrtx'))}",
    }
    run = f"{os.environ['GITHUB_SHA']}-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    for name, path in CACHES.items():
        if SAVING and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        collection = collections[name]
        print(f"{name}: {collection}")
        _write_lines(
            "GITHUB_OUTPUT",
            [
                f"{name}-collection={collection}",
                f"{name}-key={collection}-{run}",
                f"{name}-restore-keys={collection}-",
                f"{name}-dir={path}",
            ],
        )
    _write_lines(
        "GITHUB_ENV",
        [
            f"WARP_CACHE_PATH={CACHES['warp']}",
            f"OVRTX_SHADER_CACHE_PATH={CACHES['kitless']}",
            f"OVRTX_KIT_SHADER_CACHE_PATH={CACHES['kit']}",
        ],
    )


def restored() -> None:
    import warp as wp

    for name, path in CACHES.items():
        hit = os.environ.get(f"{name.upper()}_HIT")
        print(f"{name} cache hit: {hit}" if hit else f"::warning::{name} cache miss")
        print(f"{name} cache: {_megabytes(path)} MB")
    # tools/verify_warp_cache.py's path check, which fails on the \\?\ prefix Warp reports on
    # Windows; its version check always passes on an environment synced from uv.lock.
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    reported = wp.config.kernel_cache_dir.removeprefix("\\\\?\\")
    if not os.path.normcase(reported).startswith(os.path.normcase(str(CACHES["warp"]))):
        print(f"::error::Warp resolved its kernel cache to {reported}, not {CACHES['warp']}")
        sys.exit(1)
    if SAVING:
        _write_lines("GITHUB_ENV", [f"{name.upper()}_BEFORE={_fingerprint(path)}" for name, path in CACHES.items()])


def growth() -> None:
    for name, path in CACHES.items():
        # Every snapshot is a new immutable cache entry, so a save that only read back what it
        # restored, or found nothing, publishes nothing.
        changed = "false"
        if SAVING and any(file.is_file() for file in path.rglob("*")):
            if _fingerprint(path) != os.environ.get(f"{name.upper()}_BEFORE", ""):
                changed = "true"
        print(f"{name} cache: {_megabytes(path)} MB, changed={changed}")
        _write_lines("GITHUB_OUTPUT", [f"{name}-changed={changed}"])


if __name__ == "__main__":
    {"keys": keys, "restored": restored, "growth": growth}[sys.argv[1]]()
