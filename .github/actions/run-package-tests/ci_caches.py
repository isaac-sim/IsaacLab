# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Key and report the Warp kernel and OVRTX shader caches the CI test jobs restore and publish.

The Linux jobs run this through .github/actions/warp-cache-key, .github/actions/ovrtx-shader-cache
and this directory's action, and Windows CI runs it on the runner, so the collection a job restores
from and the one a job saves to cannot drift.

Usage:
    ci_caches.py keys [--isaacsim-version VERSION] [--installed] NAME[=DIR]...
    ci_caches.py restored [--publishes] NAME[=DIR]...
    ci_caches.py growth [--publishes] [--require-files] NAME[=DIR]...

    keys      writes each cache's collection, write key, restore prefix and directory to the step outputs
    restored  reports the restore that <NAME>_MATCHED_KEY names and records the baseline for the growth pass
    growth    reports how far the run grew each cache and writes <NAME>-files and <NAME>-changed

NAME is ``warp`` for Warp kernels, ``kit`` for the driver shaders Kit's RTX renderer compiles, or
``kitless`` for the driver shaders the ovrtx wheel compiles. A bare NAME means the directory the tests
read from that cache's variable in ``TEST_ENV``, as on Windows, where the tests run on the runner.
Only the standard library is used: the Linux steps run the runner's system Python, which has no
``tomllib``.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import os
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[3]
LABELS = {"warp": "Warp cache", "kit": "OVRTX kit shader cache", "kitless": "OVRTX kitless shader cache"}
# The variable that points the tests at each cache; see .github/actions/run-tests/run_tests.sh.
TEST_ENV = {"warp": "WARP_CACHE_PATH", "kit": "OVRTX_KIT_SHADER_CACHE_PATH", "kitless": "OVRTX_SHADER_CACHE_PATH"}
SM_PATTERN = re.compile(r"\.(sm\d+[a-z]?)\.")


def _fail(message: str) -> NoReturn:
    print(f"::error::{message}")
    raise SystemExit(1)


def _append(variable: str, *lines: str) -> None:
    """Append lines to the file a GitHub Actions variable such as GITHUB_OUTPUT names."""
    with open(os.environ[variable], "a", encoding="utf-8") as handle:
        handle.writelines(f"{line}\n" for line in lines)


def _state(name: str, field: str) -> str:
    """Name the job environment variable that carries one cache's restore baseline to the growth pass."""
    return f"CI_CACHE_{name.upper()}_{field}"


def _sanitize(value: str) -> str:
    """Replace the characters a cache key may not contain, such as commas."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def _lock_packages(text: str) -> dict[str, str]:
    """Map each ``[[package]]`` name in uv.lock to the raw text of its block.

    uv writes ``name``/``version``/``source`` as plain single-line pairs, so they are read with
    regexes instead of a TOML parser.
    """
    packages = {}
    for block in re.split(r"^\[\[package\]\]$", text, flags=re.MULTILINE)[1:]:
        # The block ends at the next table header starting at column zero.
        block = re.split(r"^\[", block, flags=re.MULTILINE)[0]
        name = re.search(r'^name = "(.*)"$', block, flags=re.MULTILINE)
        if name:
            packages[name.group(1)] = block
    return packages


def _warp_versions() -> str:
    """Identify the Warp, Newton and MuJoCo Warp builds that uv.lock pins.

    Only these decide whether a restored tree is usable: Warp's module hash already separates
    changed Isaac Lab kernels while leaving unchanged modules reusable. GPU architecture stays
    out because Warp puts the target in the artifact filename (``<module>.sm120.ptx``), so one
    collection accumulates every variant the runner fleet needs.
    """
    packages = _lock_packages((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    try:
        warp = re.search(r'^version = "(.*)"$', packages["warp-lang"], flags=re.MULTILINE).group(1)
        mjwarp = re.search(r'^version = "(.*)"$', packages["mujoco-warp"], flags=re.MULTILINE).group(1)
        newton = packages["newton"]
        newton_version = re.search(r'^version = "(.*)"$', newton, flags=re.MULTILINE).group(1)
    except (AttributeError, KeyError) as exc:
        _fail(f"uv.lock has no version entry for {exc}")

    # Git builds can share a placeholder version across revisions, so identify them by commit.
    # Registry releases are immutable and use their version.
    source = re.search(r"^source = (.*)$", newton, flags=re.MULTILINE)
    source = source.group(1) if source else ""
    match = re.search(r"(?:rev=|#)([0-9a-f]{7,40})", source)
    if match:
        newton_id = match.group(1)[:8]
    elif "registry" in source:
        newton_id = newton_version
    else:
        _fail(f"Could not identify the newton package from {source!r}")
    return f"wp{warp}-newton{newton_id}-mjwarp{mjwarp}"


def _pinned_ovrtx() -> str:
    """Return [tool.isaaclab.versions].ovrtx from pyproject.toml.

    It is the pin resolve-ov-pins installs into the Linux test images (an internal release, not the
    public wheel uv.lock resolves), so the key stays a function of the commit and an upstream
    release cannot re-key every open pull request onto a cold collection.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = re.search(r"^\[tool\.isaaclab\.versions\][^\n]*\n(.*?)(?=^\[|\Z)", text, flags=re.MULTILINE | re.DOTALL)
    pin = section and re.search(r'^ovrtx\s*=\s*"([^"]*)"', section.group(1), flags=re.MULTILINE)
    if not pin:
        _fail("pyproject.toml [tool.isaaclab.versions] has no entry for ovrtx")
    return pin.group(1)


def _query_gpu(field: str) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"], capture_output=True, text=True
        )
    except OSError:
        return ""
    lines = result.stdout.splitlines()
    return lines[0].strip() if lines else ""


def _shader_base() -> str:
    """Return the key part every shader cache shares.

    nv_shadercache holds NVIDIA driver PSO blobs, which are only valid for the GPU architecture and
    driver version that compiled them, so both gate every entry.
    """
    driver = _query_gpu("driver_version")
    arch = _query_gpu("compute_cap").replace(".", "")
    if not driver or not arch:
        _fail("nvidia-smi reported no driver version or compute capability; cannot key the OVRTX shader cache")
    return f"v1-{os.environ['RUNNER_OS']}-{os.environ['RUNNER_ARCH']}-sm{arch}-drv{_sanitize(driver)}"


def _installed(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        _fail(f"{package} is not installed in {sys.executable}")


def keys(caches: dict[str, Path], isaacsim_version: str | None, installed: bool) -> None:
    """Write each cache's collection, write key, restore prefix and directory to the step outputs.

    Only values that stay constant for the lifetime of a job may be keyed on. The write key is
    unique per run, so each cumulative snapshot is its own immutable entry, including on reruns
    and docs-only merges, where the source is unchanged but the cache may still have grown; a
    restore matches the collection prefix, newest first.
    """
    run = f"{os.environ['GITHUB_SHA']}-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    for name, directory in caches.items():
        if name == "warp":
            collection = f"warp-v2-{os.environ['RUNNER_OS']}-{os.environ['RUNNER_ARCH']}-{_warp_versions()}"
        elif name == "kit":
            # The Isaac Sim build that compiles the blobs: the image tag on Linux, the installed wheel on Windows.
            isaacsim = isaacsim_version or (_installed("isaacsim") if installed else None)
            if not isaacsim:
                _fail("the kit cache needs --isaacsim-version or --installed")
            collection = f"ovrtx-kit-{_shader_base()}-isaacsim{_sanitize(isaacsim)}"
        else:
            ovrtx = _installed("ovrtx") if installed else _pinned_ovrtx()
            collection = f"ovrtx-kitless-{_shader_base()}-ovrtx{_sanitize(ovrtx)}"
        print(f"{LABELS[name]} collection: {collection}")
        _append(
            "GITHUB_OUTPUT",
            f"{name}-collection={collection}",
            f"{name}-key={collection}-{run}",
            f"{name}-restore-keys={collection}-",
            f"{name}-dir={directory}",
        )


def _files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file()) if directory.is_dir() else []


def _bytes(files: list[Path]) -> int:
    total = 0
    for path in files:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def fingerprint(directory: Path) -> str:
    """Return a digest of every file's relative path and contents.

    Contents rather than size or file count, because Warp and the driver rewrite files in place as
    well as adding them, and a run that only read the restored files must not publish another
    immutable entry.
    """
    digest = hashlib.sha256()
    for path in _files(directory):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(b"\0")
        file_digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    file_digest.update(chunk)
        except OSError:
            continue
        digest.update(file_digest.digest())
    return digest.hexdigest()


def _summarize(line: str) -> None:
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        _append("GITHUB_STEP_SUMMARY", line)


def _warp_inventory(directory: Path, files: list[Path], label: str) -> None:
    """Print the module count, artifact mix and GPU targets a Warp cache has accumulated."""
    extensions: collections.Counter[str] = collections.Counter(path.suffix.lstrip(".") or "(none)" for path in files)
    sm_targets: collections.Counter[str] = collections.Counter(
        found.group(1) for found in (SM_PATTERN.search(path.name) for path in files) if found
    )
    # Warp namespaces by its own version, so each child of the root is a version and each
    # grandchild is one module's compiled artifacts.
    versions = sorted(path for path in directory.iterdir() if path.is_dir()) if directory.is_dir() else []
    module_dirs = sum(1 for version_dir in versions for path in version_dir.iterdir() if path.is_dir())
    mix = ", ".join(f"{ext}={count}" for ext, count in extensions.most_common(6)) or "empty"
    targets = ", ".join(f"{sm}={count}" for sm, count in sorted(sm_targets.items())) or "none"

    print(f"{label}: {_bytes(files) / 1e6:.0f} MB across {module_dirs} module dirs")
    print(f"  warp versions: {', '.join(path.name for path in versions) or 'none'}")
    print(f"  artifacts:     {mix}")
    print(f"  gpu targets:   {targets}")
    _summarize(f"🔵 {label}: {_bytes(files) / 1e6:.0f} MB, {module_dirs} modules, targets [{targets}]")


def restored(caches: dict[str, Path], publishes: bool) -> None:
    """Report each restore and record the baseline the growth pass compares with."""
    for name, directory in caches.items():
        label = LABELS[name]
        directory.mkdir(parents=True, exist_ok=True)
        if not os.access(directory, os.W_OK):
            _fail(f"{label} directory is not writable: {directory}")

        # cache-matched-key is authoritative. Inferring a hit from the directory is not: an empty
        # directory still has a size on disk.
        hit = os.environ.get(f"{name.upper()}_MATCHED_KEY", "")
        if hit:
            print(f"{label} hit: {hit}")
        else:
            print(f"::warning::{label} miss - no entry in its collection")
        files = _files(directory)
        _append("GITHUB_ENV", f"{_state(name, 'HIT')}={hit or 'miss'}", f"{_state(name, 'BYTES')}={_bytes(files)}")
        if name == "warp":
            _warp_inventory(directory, files, "Warp cache restored")
        # Only a job that publishes compares fingerprints, and hashing is the one part of this
        # pass that scales with the restored snapshot.
        if publishes and hit:
            _append("GITHUB_ENV", f"{_state(name, 'FINGERPRINT')}={fingerprint(directory)}")


def growth(caches: dict[str, Path], publishes: bool, require_files: bool) -> int:
    """Report how far the run grew each cache and whether a publishing job should save it."""
    status = 0
    for name, directory in caches.items():
        label = LABELS[name]
        files = _files(directory)
        before = int(os.environ.get(_state(name, "BYTES")) or 0)
        after = _bytes(files)
        grew = after - before
        if os.environ.get(_state(name, "HIT"), "miss") == "miss":
            verdict = "cold run, nothing restored"
        elif grew <= 0:
            verdict = "no net growth"
        else:
            verdict = f"compiled {grew * 100 // max(before, 1)}% beyond the restored cache"
        before_mb, after_mb, grew_mb = (round(value / 1e6) for value in (before, after, grew))
        print(f"{label}: restored {before_mb} MB, ended at {after_mb} MB (+{grew_mb} MB) - {verdict}")
        _summarize(f"🔵 {label}: {before_mb} -> {after_mb} MB (+{grew_mb}) - {verdict}")
        if name == "warp":
            _warp_inventory(directory, files, "Warp cache final")

        # Every save is a new immutable entry, so an empty tree must not replace a good snapshot,
        # and a run that only read back what it restored publishes nothing. Without a baseline,
        # which a cold restore never records, any file is new.
        changed = False
        if publishes and not files:
            print(f"::warning::{label} contains no files; skipping publish")
        elif publishes:
            baseline = os.environ.get(_state(name, "FINGERPRINT"))
            changed = not baseline or fingerprint(directory) != baseline
            if not changed:
                print(f"{label} contents are unchanged; skipping duplicate snapshot")
        _append("GITHUB_OUTPUT", f"{name}-files={len(files)}", f"{name}-changed={str(changed).lower()}")

        if require_files and not files:
            print(f"::error::{label} is empty - the selected tests did not exercise it")
            status = 1
    return status


def _cache(spec: str) -> tuple[str, Path]:
    name, separator, directory = spec.partition("=")
    if name in TEST_ENV and not separator:
        directory = os.environ.get(TEST_ENV[name], "")
    if name not in LABELS or not directory:
        variable = TEST_ENV.get(name, "its variable")
        raise argparse.ArgumentTypeError(f"expected NAME=DIR, or NAME with {variable} set; got {spec!r}")
    return name, Path(directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    keys_parser = commands.add_parser("keys")
    keys_parser.add_argument("--isaacsim-version", help="Isaac Sim build that compiles the kit cache")
    keys_parser.add_argument(
        "--installed", action="store_true", help="key on the isaacsim and ovrtx this Python has installed"
    )
    commands.add_parser("restored").add_argument(
        "--publishes", action="store_true", help="the job saves the caches later, so fingerprint the restore"
    )
    growth_parser = commands.add_parser("growth")
    growth_parser.add_argument("--publishes", action="store_true", help="decide whether each cache changed")
    # An empty tree means the tests a cache warmer selected never exercised that path, leaving it to
    # compile cold on every consumer; fail rather than let a green check imply it was covered.
    growth_parser.add_argument("--require-files", action="store_true", help="fail when a cache is empty")
    for command in commands.choices.values():
        command.add_argument("caches", nargs="+", type=_cache, metavar="NAME[=DIR]")
    args = parser.parse_args()

    caches = dict(args.caches)
    if args.command == "keys":
        keys(caches, args.isaacsim_version, args.installed)
    elif args.command == "restored":
        restored(caches, args.publishes)
    else:
        return growth(caches, args.publishes, args.require_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
