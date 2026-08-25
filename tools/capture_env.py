#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture an Isaac Lab installation into a portable bundle, and diff two captures.

Reproducing a report from another machine means reproducing four things that no single file
records: the resolved Python environment, the host's GPU and driver, the process environment,
and the local filesystem wiring -- symlinks and ``.pth`` files that inject import paths a
lockfile knows nothing about. ``uv.lock`` pins the first and only the first, and an
installation drifts away from it the moment anything is installed by hand.

``capture`` writes all four into a zip: ``pyproject.toml`` and ``uv.lock`` verbatim, the
installed distributions as they exist *on disk* rather than as they were resolved, the subset
of the process environment Isaac Lab is known to read, every symlink and ``.pth`` that
redirects an import, the GPU and driver inventory, and the repository's git state. It then
runs a set of checks over that snapshot and renders ``REPRODUCE.md``: the steps to rebuild the
environment, what the bundle cannot reproduce, and anything that looks wrong.

``diff`` compares two bundles, or one bundle against the machine running it, which is the
question a support case actually asks -- what is different here?

Reading the environment from disk rather than through :mod:`importlib.metadata` is deliberate.
It makes the answer independent of which interpreter runs this script, and it is the only way
to see a distribution whose metadata still claims files that were deleted underneath it. That
failure mode is silent: two wheels that both vendor a complete ``pxr`` overwrite each other,
and uninstalling either leaves the survivor's package hollow, importing as an empty namespace
package rather than raising. :func:`check_record_integrity` is what makes it visible.

The script imports nothing outside the standard library and never imports ``isaaclab``, because
the environment it is asked to describe is frequently one that cannot import anything at all.

Usage:
    python3 tools/capture_env.py capture --command "<the command that failed>"
    python3 tools/capture_env.py diff customer-bundle.zip
    python3 tools/capture_env.py diff customer-bundle.zip our-bundle.zip
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import platform
import re
import socket
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
"""Manifest schema version.

``diff`` refuses to compare across versions rather than silently misreading a renamed key.
Bump this whenever a key changes meaning; adding keys is backwards compatible.
"""

MAX_COMMAND_OUTPUT_BYTES = 1 << 20
"""Cap on a single captured command's output (1 MiB).

``nvidia-smi -q`` on a dense multi-GPU node and ``vulkaninfo`` on a driver with many
extensions both run to hundreds of kilobytes. The cap keeps one pathological host from
turning a support bundle into something nobody can attach to a ticket.
"""

MAX_COPIED_FILE_BYTES = 16 << 20
"""Cap on a single verbatim file copy (16 MiB), sized for ``uv.lock``."""

LOG_TAIL_BYTES = 256 * 1024
"""Bytes kept from the end of each log collected by ``--include-logs``."""

SYMLINK_SCAN_DEPTH = 3
"""Directory depth searched for symlinks under the repository root.

Deep enough to reach ``source/<package>/<module>`` where an editable checkout is usually
wired in, shallow enough that it never walks into an asset tree.
"""

PRUNED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "logs",
        "outputs",
        "_build",
    }
)
"""Directories the symlink scan never descends into.

``.venv`` and ``.git`` are excluded by size -- both run to tens of gigabytes in a working
tree -- not because they are uninteresting. The parts of ``.venv`` that redirect imports are
collected directly by :func:`collect_python`, which reads only the top level of
``site-packages``.
"""

CREDENTIAL_BEARING_URL_SCHEMES = frozenset({"ftp", "ftps", "http", "https"})
"""URL schemes whose userinfo field holds an authentication secret.

An HTTPS remote authenticates with what the URL carries, so a checkout cloned with a token
stores that token in ``.git/config`` and ``git remote -v`` prints it back verbatim. An SSH
remote authenticates with a key instead, and its userinfo is the login name -- ``git`` on
every hosted forge -- which is not a secret and is needed for the clone step to work.
"""

ISAAC_LAB_ENV_VARS = frozenset(
    {
        "CARB_APP_PATH",
        "CI",
        "CI_MARKER",
        "CMAKE_POLICY_VERSION_MINIMUM",
        "CONDA_PREFIX",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "DEBUG",
        "DEBUG_TIMERS",
        "DEBUG_TIMER_RESET",
        "DEBUG_TIMER_STEP",
        "DISPLAY",
        "EXP_PATH",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "HEADLESS",
        "ISAACLAB_CHANGELOG_BASE_REF",
        "ISAACLAB_CXR_SKIP_AUTOLAUNCH",
        "ISAACLAB_DISABLE_LIVE_PLOTS",
        "ISAACLAB_DUMP_ARTICULATION_PARTITION_IMAGES",
        "ISAACLAB_FRANKA_POUR_CUPS_USD_PATH",
        "ISAACLAB_FRANKA_POUR_ROBOT_USD_PATH",
        "ISAACLAB_PATH",
        "ISAACLAB_PPISP_PERF",
        "ISAACLAB_PPISP_PERF_DEVICE",
        "ISAACLAB_PPISP_PERF_MEASURE_ITERS",
        "ISAACLAB_PPISP_PERF_MEMORY_FRACTION",
        "ISAACLAB_PPISP_PERF_NUM_ENVS",
        "ISAACLAB_PPISP_PERF_OUTPUT",
        "ISAACLAB_PPISP_PERF_RESOLUTIONS",
        "ISAACLAB_PPISP_PERF_VARIANTS",
        "ISAACLAB_PPISP_PERF_WARMUP_ITERS",
        "ISAACLAB_RUN_STANDALONE_SCRIPT_TESTS",
        "ISAACLAB_STANDALONE_SCREENSHOT_DELAY",
        "ISAACLAB_STANDALONE_SCREENSHOT_DIR",
        "ISAACLAB_STANDALONE_SCRIPT_RUNTIME_GROUP",
        "ISAACLAB_STANDALONE_SCRIPT_SCOPE",
        "ISAACLAB_STANDALONE_SOAK_TIME",
        "ISAACLAB_STANDALONE_STARTUP_TIMEOUT",
        "ISAACLAB_STANDALONE_VISUALIZER",
        "ISAACLAB_TEST_DEVICES",
        "ISAACLAB_TEST_QUEUE",
        "ISAACLAB_WHEEL",
        "ISAACSIM_ASSET_ROOT",
        "ISAACSIM_CI_SHORT",
        "ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION",
        "ISAAC_LAB_OVRTX_USE_OVSTAGE",
        "ISAAC_LAB_SAVE_STAGES",
        "ISAAC_PATH",
        "JAX_LOCAL_RANK",
        "JAX_RANK",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "LIVESTREAM",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "NEWTON_ASSET_DIR",
        "NO_COLOR",
        "OMNI_KIT_ACCEPT_EULA",
        "OPENBLAS_NUM_THREADS",
        "OVRTX_SKIP_USD_CHECK",
        "PUBLIC_IP",
        "PXR_PLUGINPATH_NAME",
        "PXR_WORK_THREAD_LIMIT",
        "PYTEST_CURRENT_TEST",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "PYTHONUNBUFFERED",
        "RANK",
        "RAY_AIR_NEW_OUTPUT",
        "RLINF_CONFIG_FILE",
        "TERM",
        "TEST_CUROBO_ONLY",
        "TEST_EXCLUDE_PATTERN",
        "TEST_FILTER_PATTERN",
        "TEST_INCLUDE_FILES",
        "TEST_K_EXPR",
        "TEST_NODE_IDS",
        "TEST_NODE_IDS_FILE",
        "TEST_NODE_IDS_KEY",
        "TEST_QUARANTINED_ONLY",
        "TEST_RESULT_FILE",
        "TEST_SHARD_COUNT",
        "TEST_SHARD_INDEX",
        "TMPDIR",
        "TUNE_DISABLE_STRICT_METRIC_CHECKING",
        "USER",
        "USERNAME",
        "USE_RELATIVE_MODE",
        "UV_PYTHON",
        "VIRTUAL_ENV",
        "WANDB_DISABLED",
        "WARP_CACHE_PATH",
        "WAYLAND_DISPLAY",
        "WORLD_SIZE",
        "XR",
    }
)
"""The exact variable names Isaac Lab reads or sets. Nothing outside this list is captured.

Derived by searching ``source/``, ``scripts/``, ``tools/``, ``apps/``, and ``isaaclab.sh``
for ``os.environ``, ``os.getenv``, and shell assignments, so every entry traces to a call
site rather than to a guess about what might matter.

The list is closed and matched by exact name. No prefix or pattern matching is used, so a
variable this project does not read cannot reach a bundle -- neither its value nor its name.
That matters because the process environment is where a machine keeps its credentials,
licence keys, and internal hostnames, and almost none of it affects how Isaac Lab behaves.

Adding a variable here is the only way to widen what a capture collects, which keeps the
decision reviewable in one place. When Isaac Lab starts reading a new variable, add it.
"""

MACHINE_OWNED_ENV_VARS = frozenset(
    {
        "CONDA_PREFIX",
        "ISAACLAB_PATH",
        "TMPDIR",
        "USER",
        "USERNAME",
        "VIRTUAL_ENV",
    }
)
"""Collected variables that must not appear in the document's ``export`` block.

These describe the machine a capture came from, not the run. Recording them is useful -- a
``CONDA_PREFIX`` on a machine that is supposed to be using uv explains a great deal -- but
copying them onto a second machine would point it at directories that do not exist there.
"""

USD_PROVIDER_DISTRIBUTIONS = ("usd-core", "usd-exchange")
"""Distributions that each install a complete ``pxr`` into ``site-packages``.

Two of them in one environment overwrite each other's files, and removing either then
breaks ``pxr`` for the survivor. ``pyproject.toml`` documents this and the dependency set is
arranged to keep it from happening; this check catches an environment where it happened
anyway, by hand.
"""

PIN_TO_DISTRIBUTION = {
    "isaacsim": "isaacsim",
    "ovphysx": "ovphysx",
    "ovrtx": "ovrtx",
    "ovstage": "ovstage",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "usd_exchange": "usd-exchange",
    "warp": "warp-lang",
}
"""``[tool.isaaclab.versions]`` keys mapped to the distribution that should match them.

``newton`` is deliberately absent: it is pinned to a git rev rather than a version, so
there is nothing to compare an installed version string against.
"""

ISAAC_SIM_DOWNLOAD_URL = "https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html"
"""Where a downloaded Isaac Sim package comes from, cited in the reproduction steps."""

ISAAC_SIM_SOURCE_URL = "https://github.com/isaac-sim/IsaacSim.git"
"""Where a locally built Isaac Sim comes from, cited in the reproduction steps."""

_VERSION_LIKE = re.compile(r"^\d+(\.\d+)*")
"""Matches a leading dotted-numeric version, used to skip non-version pins such as git revs."""

_ISAAC_SIM_BUILD = re.compile(r"\+(?P<branch>[^.+]+)\.\d+\.(?P<revision>[0-9a-f]{7,40})")
"""Pulls the branch and revision out of an Isaac Sim ``VERSION`` string.

Both a downloaded package and a local build stamp their provenance into the local version
segment -- ``6.1.0-alpha.59+develop.0.4877ef77.local`` was built from ``develop`` at
``4877ef77`` -- which is the only thing that says *which* Isaac Sim to rebuild.
"""

_LOCK_REQUIREMENT = re.compile(r'\{\s*name\s*=\s*"([^"]*)"((?:[^{}]|\{[^{}]*\})*)\}')
"""Matches one inline requirement entry in ``uv.lock``, capturing its name and the rest of it.

The keys after ``name`` vary -- ``version``, ``source``, ``extra``, ``marker`` -- and
``source`` is itself a nested table, so the tail is captured whole and picked apart by
:data:`_LOCK_REQUIREMENT_EXTRAS` rather than matched key by key.
"""

_LOCK_REQUIREMENT_EXTRAS = re.compile(r"extras?\s*=\s*\[([^\]]*)\]")
"""Pulls the extras out of a requirement entry's tail.

``uv.lock`` spells the key ``extra`` in a resolved ``dependencies`` array and ``extras`` in
``[package.metadata]``, so both are accepted. It cannot match a ``marker`` that mentions
``extra == '...'``, which is a comparison rather than a list.
"""


@dataclass
class Finding:
    """One problem or noteworthy property detected in a captured environment.

    Attributes:
        level: ``"error"``, ``"warning"``, or ``"info"``, in descending severity.
        code: Stable kebab-case identifier, safe to grep for across bundles.
        summary: One line stating what was found.
        detail: Supporting evidence, or an empty list when the summary stands alone.
    """

    level: str
    code: str
    summary: str
    detail: list[str] = field(default_factory=list)


@dataclass
class Distribution:
    """An installed distribution as it exists on disk.

    Attributes:
        name: Name from the distribution metadata, verbatim.
        key: PEP 503 normalized name, used for every lookup and comparison.
        version: Version from the distribution metadata.
        installer: Contents of ``INSTALLER``, e.g. ``"uv"`` or ``"pip"``, or ``None``.
        editable: Whether ``direct_url.json`` marks this as an editable install.
        origin: URL or path recorded in ``direct_url.json``, or ``None`` for an index install.
        location: ``site-packages``-relative name of the ``.dist-info``/``.egg-info``.
    """

    name: str
    key: str
    version: str
    installer: str | None
    editable: bool
    origin: str | None
    location: str


def _utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string with a trailing ``Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_name(name: str) -> str:
    """Return the PEP 503 normalized form of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _run(command: list[str], timeout: int = 30) -> str | None:
    """Run ``command`` and return its combined output, or ``None`` if it could not run.

    A tool that is absent, times out, or exits non-zero is not an error worth aborting a
    capture over -- ``vulkaninfo`` is frequently missing on a headless node, and
    ``nvidia-smi`` on a machine with no NVIDIA driver is exactly the fact being recorded.

    Args:
        command: Argument vector, passed without a shell.
        timeout: Seconds to wait before giving up.

    Returns:
        Combined stdout and stderr, truncated to :data:`MAX_COMMAND_OUTPUT_BYTES`, or ``None``.
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout + result.stderr
    if not output.strip():
        return None
    if len(output) > MAX_COMMAND_OUTPUT_BYTES:
        omitted = len(output) - MAX_COMMAND_OUTPUT_BYTES
        output = output[:MAX_COMMAND_OUTPUT_BYTES] + f"\n... [truncated, {omitted} bytes omitted]\n"
    return output


def _read_text(path: Path, limit: int = MAX_COPIED_FILE_BYTES) -> str | None:
    """Return the contents of ``path``, truncated to ``limit`` bytes, or ``None`` if unreadable."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    truncated = len(data) > limit
    text = data[:limit].decode("utf-8", errors="replace")
    if truncated:
        text += f"\n... [truncated, {len(data) - limit} bytes omitted]\n"
    return text


def _read_tail(path: Path, limit: int = LOG_TAIL_BYTES) -> str | None:
    """Return the last ``limit`` bytes of ``path``, or ``None`` if unreadable."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > limit:
                handle.seek(size - limit)
            data = handle.read()
    except OSError:
        return None
    text = data.decode("utf-8", errors="replace")
    if size > limit:
        text = f"... [tail only, {size - limit} of {size} bytes omitted]\n" + text
    return text


def _git(repo_root: Path, args: list[str], timeout: int = 15) -> str | None:
    """Run a git command inside ``repo_root`` and return its stdout, or ``None`` on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def sanitize_remote_url(url: str) -> str:
    """Return ``url`` with any embedded credential removed.

    Bundles are attached to public issues, so a remote that carries a token or password has
    to be stripped before it is written anywhere. The userinfo field is dropped rather than
    masked: the host and path are what make the clone step in the reproduction document
    actionable, and a URL left with a placeholder where a token was cannot be pasted.

    Args:
        url: Remote URL as ``git remote -v`` reports it.

    Returns:
        The URL without userinfo where that field can hold a secret, otherwise unchanged.
        SCP-style (``git@host:path``) and ``ssh://`` remotes are returned as they are unless
        they carry a password, because their userinfo is a login name.
    """
    scheme, separator, rest = url.partition("://")
    if not separator:
        # SCP-style or a local path. Only the `user:password@host` form holds a secret.
        userinfo, at_sign, tail = url.rpartition("@")
        return tail if at_sign and ":" in userinfo else url
    authority, slash, path = rest.partition("/")
    userinfo, at_sign, host = authority.rpartition("@")
    if not at_sign:
        return url
    if scheme.lower() in CREDENTIAL_BEARING_URL_SCHEMES or ":" in userinfo:
        return f"{scheme}://{host}{slash}{path}"
    return url


def sanitize_remote_listing(remotes: str) -> tuple[str, list[str], bool]:
    """Return ``git remote -v`` output with credentials removed, and the distinct URLs.

    Args:
        remotes: Verbatim ``git remote -v`` output.

    Returns:
        The listing with every URL passed through :func:`sanitize_remote_url`, the sorted set
        of sanitized URLs, and whether anything was removed. The flag is reported so the
        redaction is visible to whoever follows the clone step rather than silent.
    """
    lines: list[str] = []
    urls: set[str] = set()
    redacted = False
    for line in remotes.splitlines():
        fields = line.split()
        if len(fields) < 2:
            lines.append(line)
            continue
        url = sanitize_remote_url(fields[1])
        redacted = redacted or url != fields[1]
        urls.add(url)
        lines.append("\t".join([fields[0], " ".join([url, *fields[2:]])]))
    return "\n".join(lines), sorted(urls), redacted


def find_repo_root(start: Path) -> Path:
    """Return the Isaac Lab repository root containing ``start``.

    Args:
        start: Directory to search upwards from.

    Returns:
        The first ancestor holding both ``pyproject.toml`` and ``source/``, or ``start``
        itself when no ancestor qualifies. Falling back rather than raising keeps the tool
        usable when it has been copied out of the tree onto a customer machine.
    """
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "source").is_dir():
            return candidate
    return start


def find_site_packages(venv: Path) -> Path | None:
    """Return the ``site-packages`` directory of ``venv``, or ``None`` if there is none."""
    for candidate in sorted(venv.glob("lib/python*/site-packages")):
        if candidate.is_dir():
            return candidate
    windows_candidate = venv / "Lib" / "site-packages"
    return windows_candidate if windows_candidate.is_dir() else None


def parse_version_pins(pyproject: str) -> dict[str, str]:
    """Return the ``[tool.isaaclab.versions]`` table parsed out of ``pyproject``.

    Hand-parsed rather than read with :mod:`tomllib` so the script keeps working on an
    interpreter older than 3.11, which is a real possibility for the system Python that runs
    it when the project's own environment is broken. The table holds only quoted strings, so
    a full TOML parser buys nothing here.

    Args:
        pyproject: Contents of ``pyproject.toml``.

    Returns:
        Mapping of pin name to pinned version.
    """
    pins: dict[str, str] = {}
    in_table = False
    for line in pyproject.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_table = stripped == "[tool.isaaclab.versions]"
            continue
        if not in_table or not stripped or stripped.startswith("#"):
            continue
        match = re.match(r'^([A-Za-z0-9_-]+)\s*=\s*"([^"]*)"', stripped)
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


def _requirement_nodes(entry_name: str, entry_tail: str) -> list[str]:
    """Return the graph nodes one ``uv.lock`` requirement entry points at.

    A requirement on ``cuda-toolkit[cufft]`` is two edges, not one: the package itself and the
    extra it asks for. They are kept apart because a package's extras are stored apart in the
    lock, under ``[package.optional-dependencies]``.

    Args:
        entry_name: Name captured from the requirement entry.
        entry_tail: The rest of the entry, which may declare extras.

    Returns:
        The normalized package name, followed by one ``name[extra]`` node per extra.
    """
    name = _normalize_name(entry_name)
    nodes = [name]
    for group in _LOCK_REQUIREMENT_EXTRAS.findall(entry_tail):
        nodes.extend(
            f"{name}[{_normalize_name(part.strip().strip(chr(34)))}]" for part in group.split(",") if part.strip()
        )
    return nodes


def parse_lock(lock: str) -> dict:
    """Return ``uv.lock`` as a dependency graph.

    Hand-parsed for the same reason as :func:`parse_version_pins`: this script has to keep
    running on whatever interpreter is available when the project's own environment is
    broken. Only the parts of the format that decide what ``uv sync`` installs are read --
    each package's name, version, ``dependencies``, and ``[package.optional-dependencies]``.
    Markers are deliberately ignored; they narrow a dependency to some platforms, and a
    package that is installed here is by definition one whose markers matched.

    Args:
        lock: Contents of ``uv.lock``.

    Returns:
        A dict with ``root``, the normalized name of the project the lock is written for, and
        ``packages``, mapping each normalized name to its ``versions``, its ``requires``
        edges, and its ``extras``. Edges are node names as produced by
        :func:`_requirement_nodes`.
    """
    packages: dict[str, dict] = {}
    root: str | None = None
    current: dict | None = None
    current_name: str | None = None
    mode = "other"
    collecting: str | None = None

    for line in lock.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            if stripped == "[[package]]":
                mode, current, current_name = "package", None, None
            elif stripped == "[package.optional-dependencies]":
                mode = "optional"
            else:
                mode = "other"
            collecting = None
            continue
        if mode == "other":
            continue

        if mode == "package":
            match = re.match(r'^(name|version)\s*=\s*"([^"]*)"', stripped)
            if match and match.group(1) == "name":
                current_name = _normalize_name(match.group(2))
                current = packages.setdefault(current_name, {"versions": set(), "requires": set(), "extras": {}})
                continue
            if current is None:
                continue
            if match:
                current["versions"].add(match.group(2))
                continue
            # The root project is the one the lockfile places at the checkout itself.
            if re.match(r'^source\s*=\s*\{\s*(virtual|editable)\s*=\s*"\."\s*\}', stripped):
                root = current_name
                continue
            if re.match(r"^dependencies\s*=\s*\[", stripped):
                collecting = "requires"
            if collecting == "requires":
                for entry_name, entry_extras in _LOCK_REQUIREMENT.findall(stripped):
                    current["requires"].update(_requirement_nodes(entry_name, entry_extras))
        elif current is not None:
            header = re.match(r"^([A-Za-z0-9._-]+)\s*=\s*\[", stripped)
            if header:
                collecting = _normalize_name(header.group(1))
                current["extras"].setdefault(collecting, set())
            if collecting is not None:
                for entry_name, entry_extras in _LOCK_REQUIREMENT.findall(stripped):
                    current["extras"][collecting].update(_requirement_nodes(entry_name, entry_extras))

        if stripped.startswith("]"):
            collecting = None

    return {"root": root, "packages": packages}


def lock_extras(graph: dict) -> dict[str, set[str]]:
    """Return the root project's extras, mapped to the distributions each one directly requires.

    An extra that only refers back to the project itself -- ``all`` is defined as
    ``isaaclab-dev[sb3, skrl, ...]`` -- is expanded into the extras it names, so every extra
    is described by the same kind of set and they can be compared against each other.

    Args:
        graph: A graph from :func:`parse_lock`.

    Returns:
        Mapping of extra name to the normalized names it requires directly. Transitive
        dependencies are deliberately absent: an extra is recognized by its own requirements
        being installed, and following the closure would make that test no more accurate.
    """
    root = graph.get("root")
    if not root:
        return {}
    declared = graph["packages"][root]["extras"]
    # An extra is recognized by distributions being installed, so a node's own extras are
    # dropped here: `isaacsim[all]` and `isaacsim` are the same thing to look for on disk.
    resolved = {
        name: {node.partition("[")[0] for node in nodes if not (node == root or node.startswith(f"{root}["))}
        for name, nodes in declared.items()
    }
    aliases = {
        name: [node[len(root) + 1 : -1] for node in nodes if node.startswith(f"{root}[")]
        for name, nodes in declared.items()
    }
    # Repeated to a fixed point so an alias that names another alias still resolves.
    for _ in range(len(aliases) + 1):
        for alias, referenced in aliases.items():
            for target in referenced:
                resolved[alias] |= resolved.get(target, set())
    return resolved


def lock_closure(graph: dict, extras: list[str]) -> set[str]:
    """Return every distribution ``uv sync`` would install for ``extras``.

    This is what makes the difference between "the lockfile mentions it" and "the sync
    installs it". A package can sit in the lockfile under an extra nobody selected, and
    ``uv sync`` will remove it just as readily as one the lockfile never heard of.

    Args:
        graph: A graph from :func:`parse_lock`.
        extras: Extras enabled on the root project.

    Returns:
        Normalized names reachable from the root project with those extras enabled.
    """
    root = graph.get("root")
    if not root:
        return set()
    packages = graph["packages"]
    frontier = [root, *[f"{root}[{name}]" for name in extras]]
    seen: set[str] = set()
    reached: set[str] = set()
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        name, _, extra = node.partition("[")
        entry = packages.get(name)
        if entry is None:
            continue
        reached.add(name)
        if extra:
            # An extra brings the package's own dependencies with it.
            frontier.append(name)
            frontier.extend(entry["extras"].get(extra.rstrip("]"), ()))
        else:
            frontier.extend(entry["requires"])
    return reached


def select_sync_extras(extras: dict[str, set[str]], installed: set[str]) -> list[str]:
    """Return the fewest extras that account for what is installed.

    An extra counts as selected when every distribution it requires directly is present. The
    test errs towards naming an extra that was not asked for -- another extra may have pulled
    the same packages in -- which is the harmless direction: re-selecting it installs nothing
    new, while missing one lets ``uv sync`` delete packages the environment needs.

    Args:
        extras: Extras mapped to the names they require, from :func:`parse_lock_extras`.
        installed: Normalized names of everything installed.

    Returns:
        Extra names, sorted. Any extra whose requirements are wholly covered by another
        selected extra is dropped, so ``all`` appears instead of the eight extras it expands to.
    """
    satisfied = {name: required for name, required in extras.items() if required and required <= installed}
    # Extras that resolve to identical requirements are interchangeable. Keeping the first by
    # name rather than both keeps the emitted command stable across captures.
    canonical: dict[frozenset[str], str] = {}
    for name in sorted(satisfied):
        canonical.setdefault(frozenset(satisfied[name]), name)
    chosen = set(canonical.values())
    return sorted(
        name for name in chosen if not any(satisfied[name] < satisfied[other] for other in chosen if other != name)
    )


def find_unlocked_distributions(distributions: list[dict], graph: dict, closure: set[str]) -> list[dict]:
    """Return the installed distributions a sync of this lockfile would not put back.

    ``uv sync`` installs its closure and removes everything else, so anything here is lost the
    moment a reproduction attempt syncs -- and none of it is visible in the metadata, because
    ``uv pip install`` writes the same ``INSTALLER`` value that ``uv sync`` does. Comparing
    against the lockfile rather than against the installer name is what makes it visible.

    Args:
        distributions: Installed distributions, as stored in the manifest.
        graph: A graph from :func:`parse_lock`.
        closure: What the selected sync installs, from :func:`lock_closure`.

    Returns:
        One record per distribution, naming why the sync does not account for it.
    """
    packages = graph.get("packages", {})
    unlocked: list[dict] = []
    for dist in distributions:
        entry = packages.get(dist["key"])
        locked = sorted(entry["versions"]) if entry else []
        if entry is None:
            reason = "absent from uv.lock"
        elif dist["key"] not in closure:
            reason = "in uv.lock, but no selected extra reaches it"
        # A local version segment (2.11.0+cu128) is a build of the locked version, not a
        # different one, and matches the same way it does in the pin-drift check.
        elif dist["version"] in entry["versions"] or dist["version"].split("+")[0] in entry["versions"]:
            continue
        else:
            reason = "version differs from uv.lock"
        unlocked.append(
            {
                "name": dist["name"],
                "version": dist["version"],
                "installer": dist.get("installer"),
                "locked": locked,
                "reason": reason,
            }
        )
    return sorted(unlocked, key=lambda entry: _normalize_name(entry["name"]))


def resolve_sync_plan(lock: str | None, distributions: list[dict]) -> dict:
    """Return the sync that rebuilds this environment, and what that sync would not restore.

    Args:
        lock: Contents of ``uv.lock``, or ``None`` when the checkout has none.
        distributions: Installed distributions, as stored in the manifest.

    Returns:
        A dict carrying the selected ``extras``, the ``command`` that applies them, the
        distributions that sync ``covers``, and the ``unlocked`` ones it cannot account for.
    """
    if not lock:
        return {"lock_available": False, "extras": [], "covers": [], "unlocked": [], "command": "uv sync --locked"}
    graph = parse_lock(lock)
    extras = select_sync_extras(lock_extras(graph), {dist["key"] for dist in distributions})
    closure = lock_closure(graph, extras)
    return {
        "lock_available": True,
        "extras": extras,
        "covers": sorted(closure),
        "unlocked": find_unlocked_distributions(distributions, graph, closure),
        "command": "uv sync --locked" + "".join(f" --extra {name}" for name in extras),
    }


def scan_distributions(site_packages: Path) -> list[Distribution]:
    """Return every distribution installed in ``site_packages``, read from disk.

    Args:
        site_packages: Directory to scan.

    Returns:
        Distributions sorted by normalized name. Metadata that cannot be parsed is skipped
        rather than raising, so one malformed ``.dist-info`` cannot lose the whole inventory.
    """
    distributions: list[Distribution] = []
    for info in sorted(site_packages.iterdir()):
        if info.suffix not in (".dist-info", ".egg-info") or not info.is_dir():
            continue
        metadata_file = info / ("METADATA" if info.suffix == ".dist-info" else "PKG-INFO")
        name, version = None, None
        for line in (_read_text(metadata_file, limit=64 << 10) or "").splitlines():
            if not line.strip():
                break
            if line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("Version:"):
                version = line.split(":", 1)[1].strip()
        if not name:
            continue

        installer = (_read_text(info / "INSTALLER", limit=1024) or "").strip() or None
        editable, origin = False, None
        direct_url = _read_text(info / "direct_url.json", limit=64 << 10)
        if direct_url:
            try:
                parsed = json.loads(direct_url)
                origin = parsed.get("url")
                editable = bool(parsed.get("dir_info", {}).get("editable"))
            except (ValueError, AttributeError):
                pass

        distributions.append(
            Distribution(
                name=name,
                key=_normalize_name(name),
                version=version or "unknown",
                installer=installer,
                editable=editable,
                origin=origin,
                location=info.name,
            )
        )
    return sorted(distributions, key=lambda dist: dist.key)


def check_record_integrity(site_packages: Path) -> dict:
    """Return which installed distributions are missing files their own ``RECORD`` claims.

    An installer records every file it wrote. When a second wheel overwrites those files and
    is later removed, the removal takes the shared paths with it and the survivor is left
    with metadata describing files that are gone. Nothing raises at import time -- a package
    directory stripped of its ``__init__.py`` becomes an empty namespace package -- so the
    first symptom is an ``AttributeError`` deep in unrelated code.

    ``.pyc`` files are skipped because byte-compiled caches are legitimately absent, and
    ``RECORD`` itself is skipped because it is written after the hash of its own contents.

    Args:
        site_packages: Directory to check.

    Returns:
        A dict with the number of distributions and files checked, and a ``damaged`` list of
        per-distribution records naming a bounded sample of the missing paths.
    """
    damaged: list[dict] = []
    checked_files = 0
    checked_distributions = 0

    for info in sorted(site_packages.iterdir()):
        if info.suffix != ".dist-info" or not info.is_dir():
            continue
        record = info / "RECORD"
        if not record.is_file():
            continue
        checked_distributions += 1
        missing: list[str] = []
        recorded = 0
        try:
            with record.open(newline="", encoding="utf-8", errors="replace") as handle:
                for row in csv.reader(handle):
                    if not row or not row[0]:
                        continue
                    relative = row[0]
                    if relative.endswith(".pyc"):
                        continue
                    recorded += 1
                    if not (site_packages / relative).exists():
                        missing.append(relative)
        except OSError:
            continue
        checked_files += recorded
        if missing:
            damaged.append(
                {
                    "distribution": info.name,
                    "recorded": recorded,
                    "missing": len(missing),
                    "examples": sorted(missing)[:10],
                }
            )

    return {
        "checked_distributions": checked_distributions,
        "checked_files": checked_files,
        "damaged": sorted(damaged, key=lambda entry: -entry["missing"]),
    }


def probe_venv_interpreter(venv: Path) -> dict:
    """Ask the environment's own interpreter to describe itself.

    The interpreter running this script is frequently not the one the environment uses --
    someone following instructions on a broken machine reaches for whatever ``python3``
    resolves to -- and the version that matters for reproduction is the environment's. Asking
    it directly also doubles as a liveness check, and returns the ``sys.path`` that its
    ``.pth`` files actually produce rather than a reconstruction of it.

    Args:
        venv: Virtual environment to probe.

    Returns:
        A dict that always carries ``available``; on success it also carries ``version``,
        ``executable``, ``prefix``, ``base_prefix``, and ``sys_path``, and on failure the
        ``error`` that prevented the interpreter from starting.
    """
    for name in ("bin/python", "bin/python3", "Scripts/python.exe"):
        interpreter = venv / name
        if interpreter.exists():
            break
    else:
        return {"available": False, "error": "no interpreter found in the virtual environment"}

    probe = (
        "import json, sys;"
        "print(json.dumps({'version': sys.version.split()[0], 'executable': sys.executable,"
        " 'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'sys_path': sys.path}))"
    )
    try:
        result = subprocess.run(
            [str(interpreter), "-c", probe],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error": str(error)}
    if result.returncode != 0:
        return {"available": False, "error": (result.stderr or "").strip()[-2000:]}
    try:
        return {"available": True, **json.loads(result.stdout.strip().splitlines()[-1])}
    except (ValueError, IndexError):
        return {"available": False, "error": "the interpreter started but produced unparsable output"}


def venv_python_version(python_section: dict) -> str | None:
    """Return the environment's Python version, preferring the interpreter's own answer.

    Falls back to ``version_info`` in ``pyvenv.cfg``, which is still correct when the
    interpreter itself will not start -- the case where the version matters most.
    """
    venv = python_section.get("venv") or {}
    probed = (venv.get("interpreter") or {}).get("version")
    return probed or venv.get("pyvenv_cfg", {}).get("version_info")


def collect_host() -> tuple[dict, dict[str, str]]:
    """Return the host's OS, kernel, CPU, and memory facts plus the raw files behind them."""
    artifacts: dict[str, str] = {}
    os_release: dict[str, str] = {}
    release_text = _read_text(Path("/etc/os-release"), limit=64 << 10)
    if release_text:
        artifacts["host/os-release.txt"] = release_text
        for line in release_text.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                os_release[key.strip()] = value.strip().strip('"')

    cpu_model, cpu_count = None, os.cpu_count()
    cpuinfo = _read_text(Path("/proc/cpuinfo"), limit=1 << 20)
    if cpuinfo:
        artifacts["host/cpuinfo.txt"] = cpuinfo
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break

    memory_total_kb = None
    meminfo = _read_text(Path("/proc/meminfo"), limit=64 << 10)
    if meminfo:
        artifacts["host/meminfo.txt"] = meminfo
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                digits = re.search(r"(\d+)", line)
                if digits:
                    memory_total_kb = int(digits.group(1))
                break

    uname = _run(["uname", "-a"])
    if uname:
        artifacts["host/uname.txt"] = uname

    # A capture taken inside a container describes the container, not the machine, and the
    # difference decides whether a driver mismatch is even actionable.
    cgroup = _read_text(Path("/proc/1/cgroup"), limit=64 << 10) or ""
    in_container = Path("/.dockerenv").exists() or "docker" in cgroup or "kubepods" in cgroup

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "libc": " ".join(platform.libc_ver()).strip() or None,
        "os_release": os_release,
        "cpu_model": cpu_model,
        "cpu_count": cpu_count,
        "memory_total_kb": memory_total_kb,
        "in_container": in_container,
    }, artifacts


def collect_gpu() -> tuple[dict, dict[str, str]]:
    """Return the GPU inventory, driver version, and graphics stack, plus raw tool output."""
    artifacts: dict[str, str] = {}
    devices: list[dict] = []
    driver_version, cuda_driver_version = None, None

    fields = [
        "index",
        "name",
        "uuid",
        "driver_version",
        "vbios_version",
        "memory.total",
        "compute_mode",
        "persistence_mode",
    ]
    query = _run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"])
    if query:
        artifacts["host/nvidia-smi-query.csv"] = query
        for line in query.strip().splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) == len(fields):
                devices.append(dict(zip(fields, values)))
        if devices:
            driver_version = devices[0].get("driver_version")

    full = _run(["nvidia-smi", "-q"], timeout=60)
    if full:
        artifacts["host/nvidia-smi.txt"] = full
    header = _run(["nvidia-smi"], timeout=60)
    if header:
        artifacts["host/nvidia-smi-summary.txt"] = header
        match = re.search(r"CUDA Version:\s*([0-9.]+)", header)
        if match:
            cuda_driver_version = match.group(1)

    module_version = _read_text(Path("/proc/driver/nvidia/version"), limit=64 << 10)
    if module_version:
        artifacts["host/nvidia-module-version.txt"] = module_version

    # ovrtx initializes through Vulkan, so a bundle without this is missing the first thing
    # to check when a renderer fails to start on a machine whose GPU looks fine.
    vulkan = _run(["vulkaninfo", "--summary"], timeout=60)
    if vulkan:
        artifacts["host/vulkaninfo.txt"] = vulkan
    glx = _run(["glxinfo", "-B"], timeout=30)
    if glx:
        artifacts["host/glxinfo.txt"] = glx
    nvcc = _run(["nvcc", "--version"])
    if nvcc:
        artifacts["host/nvcc.txt"] = nvcc

    return {
        "driver_version": driver_version,
        "cuda_driver_version": cuda_driver_version,
        "device_count": len(devices),
        "devices": devices,
        "nvidia_smi_available": query is not None or header is not None,
        "vulkan_available": vulkan is not None,
        "kernel_module": (module_version or "").strip().splitlines()[0] if module_version else None,
    }, artifacts


def collect_python(venv: Path | None, skip_integrity: bool = False) -> tuple[dict, dict[str, str]]:
    """Return the interpreter, the target virtual environment, and everything installed in it.

    Args:
        venv: Virtual environment to describe, or ``None`` to describe only the interpreter.
        skip_integrity: Skip the ``RECORD``-against-disk check, which is the slow part on a
            large environment.

    Returns:
        The manifest section and the raw files to store alongside it.
    """
    artifacts: dict[str, str] = {}
    section: dict = {
        "interpreter": {
            "version": sys.version.split()[0],
            "version_full": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "implementation": platform.python_implementation(),
        },
        "sys_path": list(sys.path),
        "venv": None,
        "distributions": [],
        "duplicates": [],
        "pth_files": [],
        "integrity": None,
    }
    artifacts["python/capturing-sys-path.txt"] = "\n".join(sys.path) + "\n"

    if venv is None or not venv.is_dir():
        return section, artifacts

    pyvenv_cfg = _read_text(venv / "pyvenv.cfg", limit=64 << 10)
    if pyvenv_cfg:
        artifacts["files/pyvenv.cfg"] = pyvenv_cfg
    venv_info: dict = {"path": str(venv), "pyvenv_cfg": {}, "created_by": None}
    for line in (pyvenv_cfg or "").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            venv_info["pyvenv_cfg"][key.strip()] = value.strip()
    if "uv" in venv_info["pyvenv_cfg"]:
        venv_info["created_by"] = "uv " + venv_info["pyvenv_cfg"]["uv"]
    venv_info["interpreter"] = probe_venv_interpreter(venv)
    section["venv"] = venv_info
    if venv_info["interpreter"].get("sys_path"):
        artifacts["python/venv-sys-path.txt"] = "\n".join(venv_info["interpreter"]["sys_path"]) + "\n"

    site_packages = find_site_packages(venv)
    if site_packages is None:
        return section, artifacts
    venv_info["site_packages"] = str(site_packages)

    distributions = scan_distributions(site_packages)
    section["distributions"] = [asdict(dist) for dist in distributions]

    seen: dict[str, list[str]] = {}
    for dist in distributions:
        seen.setdefault(dist.key, []).append(f"{dist.version} ({dist.location})")
    section["duplicates"] = [{"name": key, "installs": v} for key, v in sorted(seen.items()) if len(v) > 1]

    artifacts["python/distributions.txt"] = "".join(
        f"{dist.name}=={dist.version}"
        f"{'  [editable]' if dist.editable else ''}"
        f"{'  installer=' + dist.installer if dist.installer else ''}"
        f"{'  origin=' + dist.origin if dist.origin else ''}\n"
        for dist in distributions
    )

    # A .pth file is executed at interpreter start and can put anything on sys.path. Two
    # environments with identical lockfiles still import different code if these differ, so
    # they are copied verbatim rather than merely listed.
    pth_files = []
    for pth in sorted(site_packages.glob("*.pth")):
        content = _read_text(pth, limit=256 << 10) or ""
        pth_files.append({"name": pth.name, "lines": [ln for ln in content.splitlines() if ln.strip()]})
        artifacts[f"files/pth/{pth.name}"] = content
    section["pth_files"] = pth_files

    if not skip_integrity:
        integrity = check_record_integrity(site_packages)
        section["integrity"] = integrity
        artifacts["python/integrity.json"] = json.dumps(integrity, indent=2) + "\n"

    return section, artifacts


def is_collected_env_var(name: str) -> bool:
    """Return whether ``name`` is one of the exact variables this tool may capture.

    Membership in :data:`ISAAC_LAB_ENV_VARS` and nothing else. Deliberately not a pattern
    match: the process environment is where a machine keeps its credentials and its internal
    topology, and a rule clever enough to admit an unforeseen name is also clever enough to
    admit a secret.
    """
    return name in ISAAC_LAB_ENV_VARS


def collect_environment() -> tuple[dict, dict[str, str]]:
    """Return the variables Isaac Lab reads, and a count of everything else.

    Returns:
        The manifest section and the rendered listing to store. Variables outside the
        allowlist are counted but never named, because a name alone can disclose what a
        machine is connected to.
    """
    variables = {name: value for name, value in os.environ.items() if is_collected_env_var(name)}
    omitted = len(os.environ) - len(variables)

    header = (
        "# Only the exact variables Isaac Lab reads or sets are captured.\n"
        f"# {omitted} other variable(s) were present and deliberately not collected.\n"
    )
    rendered = header + "".join(f"{name}={variables[name]}\n" for name in sorted(variables))
    return (
        {"variables": variables, "omitted_count": omitted},
        {"env/environment.txt": rendered},
    )


def collect_links(repo_root: Path, venv: Path | None) -> dict:
    """Return the symlinks that decide which files this installation actually imports.

    Walks the repository to :data:`SYMLINK_SCAN_DEPTH`, and additionally reports the top
    level of ``site-packages`` and the virtual environment's ``bin``, where an editable
    install or a relocated interpreter shows up as a link rather than a file.

    Each link records whether git tracks it. A tracked symlink arrives with the checkout and
    needs no reproduction step; an untracked one -- ``_isaac_sim`` above all -- exists only
    because someone created it, and nothing else in the bundle would reveal it. Both are
    recorded, because a tracked link that is broken locally is still worth seeing.

    Args:
        repo_root: Repository to scan.
        venv: Virtual environment to scan, or ``None``.

    Returns:
        A dict with a ``symlinks`` list, each entry naming the link, its raw target, the
        resolved path, whether that path exists, and whether git tracks the link.
    """
    tracked: set[str] = set()
    for line in (_git(repo_root, ["ls-files", "-s"]) or "").splitlines():
        mode, _, relative = line.partition(" ")
        if mode == "120000" and "\t" in relative:
            tracked.add(str(repo_root / relative.split("\t", 1)[1]))

    def describe(path: Path) -> dict:
        try:
            target = os.readlink(path)
        except OSError:
            target = None
        try:
            resolved = str(path.resolve())
            exists = path.exists()
        except OSError:
            resolved, exists = None, False
        # An ancestor holding pyvenv.cfg makes this an interpreter or library link that the
        # environment's own installer created and will recreate. Detected by walking up rather
        # than by comparing against the active venv, so a second, stale environment left in the
        # tree is classified the same way -- and still recorded, because finding one explains a
        # lot about a machine.
        return {
            "path": str(path),
            "target": target,
            "resolved": resolved,
            "exists": exists,
            "is_directory": exists and path.is_dir(),
            "tracked": str(path) in tracked,
            "in_virtualenv": any((parent / "pyvenv.cfg").is_file() for parent in path.parents),
        }

    symlinks: list[dict] = []

    def scan(root: Path, depth: int) -> None:
        if depth < 0 or not root.is_dir():
            return
        try:
            entries = sorted(root.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.is_symlink():
                symlinks.append(describe(entry))
                continue
            if entry.is_dir() and entry.name not in PRUNED_DIRECTORIES:
                scan(entry, depth - 1)

    scan(repo_root, SYMLINK_SCAN_DEPTH)

    if venv is not None and venv.is_dir():
        for directory in [find_site_packages(venv), venv / "bin", venv / "Scripts"]:
            if directory is None or not directory.is_dir():
                continue
            try:
                for entry in sorted(directory.iterdir()):
                    if entry.is_symlink():
                        symlinks.append(describe(entry))
            except OSError:
                continue

    return {"symlinks": symlinks}


def collect_repo(repo_root: Path, include_diff: bool) -> tuple[dict, dict[str, str]]:
    """Return the repository's git state and the pinned versions declared in ``pyproject.toml``.

    Args:
        repo_root: Repository to describe.
        include_diff: Whether to store the working-tree patch. Off by default because a
            dirty tree can hold code the person sending the bundle is not free to share; the
            diffstat is always included so the omission is visible rather than silent.

    Returns:
        The manifest section and the raw files to store alongside it.
    """
    artifacts: dict[str, str] = {}

    pyproject = _read_text(repo_root / "pyproject.toml")
    if pyproject:
        artifacts["files/pyproject.toml"] = pyproject
    lock = _read_text(repo_root / "uv.lock")
    if lock:
        artifacts["files/uv.lock"] = lock
    version = _read_text(repo_root / "VERSION", limit=1024)

    status = _git(repo_root, ["status", "--porcelain"])
    diffstat = _git(repo_root, ["diff", "--stat", "HEAD"])
    # Untracked files are counted apart from modifications because `git diff` does not carry
    # them: a bundle can reproduce a modified file from a patch, but an untracked one has to
    # be sent as itself.
    status_lines = [line for line in (status or "").splitlines() if line.strip()]
    untracked_lines = [line for line in status_lines if line.startswith("??")]
    git_info: dict = {
        "available": status is not None,
        "commit": (_git(repo_root, ["rev-parse", "HEAD"]) or "").strip() or None,
        "branch": (_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]) or "").strip() or None,
        "commit_date": (_git(repo_root, ["log", "-1", "--format=%cI"]) or "").strip() or None,
        "commit_subject": (_git(repo_root, ["log", "-1", "--format=%s"]) or "").strip() or None,
        "dirty": bool(status_lines),
        "modified_files": len(status_lines) - len(untracked_lines),
        "untracked_files": len(untracked_lines),
        "diff_included": False,
    }

    # Remotes are recorded because the clone step needs one, but a checkout cloned over HTTPS
    # with a token holds that token here. Redacting on the way in keeps it out of the manifest,
    # the stored listing, and the clone command derived from them.
    remotes = _git(repo_root, ["remote", "-v"])
    if remotes:
        listing, urls, redacted = sanitize_remote_listing(remotes)
        artifacts["repo/git-remote.txt"] = listing
        git_info["remotes"] = urls
        git_info["remotes_redacted"] = redacted
    if status is not None:
        artifacts["repo/git-status.txt"] = status
    if diffstat:
        artifacts["repo/git-diffstat.txt"] = diffstat
    log = _git(repo_root, ["log", "-20", "--format=%h %cI %s"])
    if log:
        artifacts["repo/git-log.txt"] = log

    # A commit no remote branch contains cannot be fetched from the bundle alone, which
    # changes the reproduction steps from "check out this sha" to "get this patch first".
    contains = _git(repo_root, ["branch", "-r", "--contains", "HEAD"])
    git_info["commit_on_remote"] = bool((contains or "").strip())

    if include_diff and git_info["dirty"]:
        patch = _git(repo_root, ["diff", "HEAD"], timeout=60)
        if patch:
            artifacts["repo/git-diff.patch"] = patch
        untracked = _git(repo_root, ["ls-files", "--others", "--exclude-standard"])
        if untracked and untracked.strip():
            artifacts["repo/git-untracked.txt"] = untracked
        git_info["diff_included"] = True

    return {
        "root": str(repo_root),
        "version": (version or "").strip() or None,
        "git": git_info,
        "pins": parse_version_pins(pyproject or ""),
    }, artifacts


def collect_isaac_sim(repo_root: Path, distributions: list[dict]) -> tuple[dict, dict[str, str]]:
    """Return how this installation reaches Isaac Sim, and how to obtain the same one.

    Isaac Sim arrives one of three ways, and each needs different steps to reproduce: the
    ``isaacsim`` wheel, which the lockfile already describes; a downloaded package, which has
    to be fetched at a matching version; or a local build, which has to be rebuilt from the
    revision it was built at. Recording which one is in use is what lets the document give
    the right instruction instead of a generic one.

    The ``_isaac_sim`` link is the single most load-bearing piece of local wiring in a
    source checkout: it decides which Kit build runs, and ``setup_python_env.sh`` beneath it
    is what prepends Kit's own paths to ``PYTHONPATH``. Both are recorded here because
    neither is derivable from any file under version control.

    Args:
        repo_root: Repository to inspect.
        distributions: Installed distributions, used to detect the ``isaacsim`` wheel.

    Returns:
        The manifest section and the raw files to store alongside it.
    """
    artifacts: dict[str, str] = {}
    wheel = next((dist for dist in distributions if dist["key"] == "isaacsim"), None)
    link = repo_root / "_isaac_sim"
    section: dict = {
        "link_present": link.exists() or link.is_symlink(),
        "link_is_symlink": link.is_symlink(),
        "link_target": None,
        "link_resolves": False,
        "version": None,
        "source_build": False,
        "source_branch": None,
        "source_revision": None,
        "wheel_version": wheel["version"] if wheel else None,
        "install_method": "none",
        "isaac_path_env": os.environ.get("ISAAC_PATH"),
    }

    if link.is_symlink():
        with contextlib.suppress(OSError):
            section["link_target"] = os.readlink(link)

    isaac_path = os.environ.get("ISAAC_PATH")
    candidates = [link] + ([Path(isaac_path)] if isaac_path else [])
    resolved: Path | None = None
    for candidate in candidates:
        if candidate.is_dir():
            resolved = candidate.resolve()
            break

    if resolved is not None:
        section["link_resolves"] = True
        section["resolved_path"] = str(resolved)
        version = _read_text(resolved / "VERSION", limit=4096)
        if version:
            section["version"] = version.strip()
            artifacts["isaacsim/VERSION"] = version
            build = _ISAAC_SIM_BUILD.search(section["version"])
            if build:
                section["source_branch"] = build.group("branch")
                section["source_revision"] = build.group("revision")
        # A path running through a build tree is a locally compiled Kit, and so is a version
        # stamped `.local`. Nothing in the bundle can reconstruct one, so the reproduction
        # steps have to say so and point at the revision it was built from instead.
        section["source_build"] = "_build" in resolved.parts or (section["version"] or "").endswith(".local")
        setup_env = _read_text(resolved / "setup_python_env.sh", limit=256 << 10)
        if setup_env:
            artifacts["isaacsim/setup_python_env.sh"] = setup_env

    # A resolving link is a deliberate act and takes precedence in the description; a wheel
    # installed alongside it is still recorded, because having both is worth seeing.
    if section["link_resolves"]:
        section["install_method"] = "source_build" if section["source_build"] else "binary"
    elif wheel is not None:
        section["install_method"] = "wheel"

    return section, artifacts


def collect_logs(repo_root: Path) -> dict[str, str]:
    """Return tails of the most recent renderer and Omniverse logs.

    Args:
        repo_root: Repository whose ``logs`` directory is searched.

    Returns:
        Archive-relative paths mapped to log tails, empty when no log was found.
    """
    artifacts: dict[str, str] = {}

    ovrtx_log = Path(os.environ.get("TMPDIR", "/tmp")) / "ovrtx_renderer.log"
    tail = _read_tail(ovrtx_log)
    if tail:
        artifacts["logs/ovrtx_renderer.log"] = tail

    omniverse_logs = Path.home() / ".nvidia-omniverse" / "logs"
    if omniverse_logs.is_dir():
        candidates = [path for path in omniverse_logs.rglob("*.log") if path.is_file()]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for path in candidates[:5]:
            tail = _read_tail(path)
            if tail:
                artifacts[f"logs/omniverse/{path.name}"] = tail

    kit_logs = repo_root / "logs"
    if kit_logs.is_dir():
        candidates = [path for path in kit_logs.rglob("*.log") if path.is_file()]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for path in candidates[:5]:
            tail = _read_tail(path)
            if tail:
                artifacts[f"logs/isaaclab/{path.name}"] = tail

    return artifacts


def analyze(manifest: dict) -> list[Finding]:
    """Return everything about a captured environment that would change how it is reproduced.

    The checks are deliberately narrow. Each one names a condition that has actually broken
    an Isaac Lab installation, and each states the evidence rather than a guess, because a
    finding that cannot be acted on is worse than no finding.

    Args:
        manifest: A manifest produced by :func:`build_manifest`.

    Returns:
        Findings ordered by descending severity, then by code.
    """
    findings: list[Finding] = []
    python = manifest.get("python", {})
    distributions = {dist["key"]: dist for dist in python.get("distributions", [])}

    integrity = python.get("integrity") or {}
    for entry in integrity.get("damaged", []):
        findings.append(
            Finding(
                level="error",
                code="missing-package-files",
                summary=(
                    f"{entry['distribution']} is missing {entry['missing']} of {entry['recorded']} files"
                    " its own RECORD claims to have installed"
                ),
                detail=[
                    "The package is present in metadata but gutted on disk. A directory left without",
                    "an __init__.py imports as an empty namespace package instead of raising, so this",
                    "surfaces as an AttributeError on a symbol that should exist.",
                    "Repair: uv pip install --reinstall-package <name> --no-deps '<name>==<version>'",
                    *[f"  missing: {path}" for path in entry["examples"]],
                ],
            )
        )

    providers = [name for name in USD_PROVIDER_DISTRIBUTIONS if name in distributions]
    if len(providers) > 1:
        findings.append(
            Finding(
                level="error",
                code="usd-provider-conflict",
                summary=f"two USD providers are installed together: {', '.join(providers)}",
                detail=[
                    "Each installs a complete pxr into the same directory and overwrites the other's",
                    "files. Removing either one then deletes the shared files and breaks pxr for the",
                    "survivor. Keep only the provider pyproject.toml declares.",
                ],
            )
        )

    for entry in python.get("duplicates", []):
        findings.append(
            Finding(
                level="error",
                code="duplicate-distribution",
                summary=f"{entry['name']} is installed more than once",
                detail=[f"  {install}" for install in entry["installs"]],
            )
        )

    venv = python.get("venv") or {}
    interpreter = venv.get("interpreter") or {}
    if venv and not interpreter.get("available", True):
        findings.append(
            Finding(
                level="error",
                code="venv-interpreter-unusable",
                summary="the virtual environment's own interpreter could not start",
                detail=[
                    "Nothing in this environment can run until this is fixed, and the package",
                    "inventory below was read from disk rather than from a live interpreter.",
                    *[f"  {line}" for line in interpreter.get("error", "").splitlines()[-10:]],
                ],
            )
        )

    unlocked = manifest.get("sync", {}).get("unlocked", [])
    if unlocked:
        findings.append(
            Finding(
                level="warning",
                code="outside-lockfile",
                summary=f"{len(unlocked)} installed distribution(s) are not what uv.lock would produce",
                detail=[
                    "`uv sync` installs the lockfile and removes everything else, so a reproduction",
                    "attempt deletes these unless they are reinstalled explicitly. The installer name",
                    "does not reveal them: `uv pip install` records the same `uv` that `uv sync` does,",
                    "so only a comparison against the lockfile finds them.",
                    *[
                        f"  {entry['name']}=={entry['version']}  {entry['reason']}"
                        + (f" (locked {', '.join(entry['locked'])})" if entry["locked"] else "")
                        for entry in unlocked[:20]
                    ],
                ],
            )
        )

    pins = manifest.get("repo", {}).get("pins", {})
    drift: list[str] = []
    for pin_name, pinned in sorted(pins.items()):
        dist_key = PIN_TO_DISTRIBUTION.get(pin_name)
        if dist_key is None or not _VERSION_LIKE.match(pinned):
            continue
        installed = distributions.get(dist_key)
        # An absent package is not drift: most of these are optional extras, and a kitless
        # install legitimately has no isaacsim at all. Only a version mismatch is reported.
        # A local version segment (2.11.0+cu128) is a build of the pinned version, not drift.
        if installed is not None and installed["version"].split("+")[0] != pinned:
            drift.append(f"  {dist_key}: pinned {pinned}, installed {installed['version']}")
    if drift:
        findings.append(
            Finding(
                level="warning",
                code="pin-drift",
                summary=f"{len(drift)} package(s) differ from [tool.isaaclab.versions] in pyproject.toml",
                detail=drift,
            )
        )

    environment = manifest.get("environment", {}).get("variables", {})
    venv_path = venv.get("path")
    isaac_resolved = manifest.get("isaac_sim", {}).get("resolved_path")
    pythonpath = [entry for entry in environment.get("PYTHONPATH", "").split(os.pathsep) if entry]
    foreign = [
        entry
        for entry in pythonpath
        if not (venv_path and entry.startswith(venv_path)) and not (isaac_resolved and entry.startswith(isaac_resolved))
    ]
    if foreign:
        findings.append(
            Finding(
                level="warning",
                code="pythonpath-outside-environment",
                summary=f"PYTHONPATH adds {len(foreign)} path(s) from neither the venv nor Isaac Sim",
                detail=[
                    "These take precedence over installed packages and are invisible to uv.lock.",
                    *[f"  {entry}" for entry in foreign[:20]],
                ],
            )
        )

    if environment.get("LD_PRELOAD"):
        findings.append(
            Finding(
                level="warning",
                code="ld-preload-set",
                summary="LD_PRELOAD is set and will override shared libraries for every process",
                detail=[f"  LD_PRELOAD={environment['LD_PRELOAD']}"],
            )
        )

    broken = [link for link in manifest.get("links", {}).get("symlinks", []) if not link["exists"]]
    if broken:
        findings.append(
            Finding(
                level="error",
                code="broken-symlink",
                summary=f"{len(broken)} symlink(s) point at paths that do not exist",
                detail=[f"  {link['path']} -> {link['target']}" for link in broken[:20]],
            )
        )

    isaac_sim = manifest.get("isaac_sim", {})
    if isaac_sim.get("source_build"):
        revision = isaac_sim.get("source_revision")
        findings.append(
            Finding(
                level="warning",
                code="isaac-sim-source-build",
                summary="_isaac_sim points into a local build tree rather than a released Isaac Sim",
                detail=[
                    f"  {isaac_sim.get('link_target')}",
                    *([f"  built from {isaac_sim.get('source_branch')} at {revision}"] if revision else []),
                    "Nothing in this bundle can reconstruct a locally compiled Kit. Reproducing this",
                    "environment requires the same build, or a switch to the isaacsim wheel.",
                ],
            )
        )
    if isaac_sim.get("link_resolves") and isaac_sim.get("wheel_version"):
        findings.append(
            Finding(
                level="info",
                code="isaac-sim-wheel-and-link",
                summary="an isaacsim wheel and a resolving _isaac_sim link are both present",
                detail=[
                    f"  wheel:  isaacsim=={isaac_sim['wheel_version']}",
                    f"  link:   {isaac_sim.get('resolved_path')} ({isaac_sim.get('version')})",
                    "Which one is imported depends on how the process was launched, because Kit's own",
                    "setup_python_env.sh prepends the linked build's paths to PYTHONPATH. Reproducing",
                    "this environment means installing both, not choosing between them.",
                ],
            )
        )

    git = manifest.get("repo", {}).get("git", {})
    if git.get("dirty") and not git.get("diff_included"):
        counted = []
        if git.get("modified_files"):
            counted.append(f"{git['modified_files']} modified file(s)")
        if git.get("untracked_files"):
            counted.append(f"{git['untracked_files']} untracked file(s)")
        findings.append(
            Finding(
                level="warning",
                code="uncommitted-changes-omitted",
                summary=f"the working tree has {' and '.join(counted)} not included in this bundle",
                detail=[
                    "The checkout cannot be reproduced from the recorded commit alone.",
                    "Re-run with --include_diff to attach the patch, if the changes can be shared.",
                ],
            )
        )
    if git.get("commit") and not git.get("commit_on_remote"):
        findings.append(
            Finding(
                level="warning",
                code="commit-not-published",
                summary="HEAD is not contained by any remote branch and cannot be fetched",
                detail=[f"  {git.get('commit')} on {git.get('branch')}"],
            )
        )

    for pth in python.get("pth_files", []):
        outside = [
            line for line in pth["lines"] if line.startswith("/") and venv_path and not line.startswith(venv_path)
        ]
        if outside:
            findings.append(
                Finding(
                    level="info",
                    code="pth-outside-venv",
                    summary=f"{pth['name']} adds import paths from outside the virtual environment",
                    detail=[f"  {line}" for line in outside[:10]],
                )
            )

    gpu = manifest.get("gpu", {})
    if not gpu.get("nvidia_smi_available"):
        findings.append(
            Finding(
                level="error",
                code="no-nvidia-driver",
                summary="nvidia-smi did not run, so no GPU or driver information was captured",
            )
        )
    elif not gpu.get("vulkan_available"):
        findings.append(
            Finding(
                level="info",
                code="no-vulkan",
                summary=(
                    "vulkaninfo is unavailable, so the Vulkan stack that ovrtx initializes through was not captured"
                ),
            )
        )

    order = {"error": 0, "warning": 1, "info": 2}
    return sorted(findings, key=lambda finding: (order[finding.level], finding.code))


def build_manifest(
    repo_root: Path,
    venv: Path | None,
    command: str | None = None,
    include_diff: bool = False,
    include_logs: bool = False,
    skip_integrity: bool = False,
) -> tuple[dict, dict[str, str]]:
    """Capture an environment into a manifest and its supporting files.

    Args:
        repo_root: Isaac Lab repository to describe.
        venv: Virtual environment to describe, or ``None`` to describe only the interpreter.
        command: The command whose behaviour prompted the capture, recorded verbatim.
        include_diff: Store the working-tree patch.
        include_logs: Store tails of the most recent renderer and Omniverse logs.
        skip_integrity: Skip the ``RECORD``-against-disk check.

    Returns:
        The manifest, and archive-relative paths mapped to the file contents to store with it.
    """
    artifacts: dict[str, str] = {}
    manifest: dict = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": _utc_now(),
        "capture": {
            "tool": "tools/capture_env.py",
            "hostname": socket.gethostname(),
            "user": os.environ.get("USER") or os.environ.get("USERNAME"),
            "working_directory": os.getcwd(),
            "command_under_test": command,
            "include_diff": include_diff,
            "include_logs": include_logs,
        },
    }

    # Collected before the loop because the Isaac Sim section needs the package inventory to
    # tell a wheel install from a linked one, and the sync plan needs both it and the lockfile.
    python_section, python_artifacts = collect_python(venv, skip_integrity=skip_integrity)
    repo_section, repo_artifacts = collect_repo(repo_root, include_diff=include_diff)
    distributions = python_section["distributions"]

    for key, (section, files) in {
        "host": collect_host(),
        "gpu": collect_gpu(),
        "python": (python_section, python_artifacts),
        "environment": collect_environment(),
        "repo": (repo_section, repo_artifacts),
        "isaac_sim": collect_isaac_sim(repo_root, distributions),
    }.items():
        manifest[key] = section
        artifacts.update(files)

    manifest["links"] = collect_links(repo_root, venv)
    manifest["sync"] = resolve_sync_plan(repo_artifacts.get("files/uv.lock"), distributions)

    # The bundle carries its own reader. `diff` is the step that decides whether a
    # reproduction worked, and the machine running it frequently has no Isaac Lab checkout --
    # or has one older than this script.
    own_source = _read_text(Path(__file__).resolve())
    if own_source:
        artifacts["capture_env.py"] = own_source
    if include_logs:
        artifacts.update(collect_logs(repo_root))

    manifest["findings"] = [asdict(finding) for finding in analyze(manifest)]
    return manifest, artifacts


def _summary_rows(manifest: dict) -> list[tuple[str, str]]:
    """Return the label/value pairs shown at the top of the rendered document."""
    host = manifest.get("host", {})
    gpu = manifest.get("gpu", {})
    python = manifest.get("python", {})
    repo = manifest.get("repo", {})
    isaac_sim = manifest.get("isaac_sim", {})
    git = repo.get("git", {})
    distributions = {dist["key"]: dist["version"] for dist in python.get("distributions", [])}
    devices = gpu.get("devices", [])

    commit = git.get("commit") or "unknown"
    return [
        ("Captured", f"{manifest.get('captured_at', '?')} on {manifest.get('capture', {}).get('hostname', '?')}"),
        ("OS", host.get("os_release", {}).get("PRETTY_NAME") or host.get("platform") or "unknown"),
        ("Kernel", host.get("kernel") or "unknown"),
        ("Container", "yes" if host.get("in_container") else "no"),
        ("CPU", f"{host.get('cpu_model') or 'unknown'} ({host.get('cpu_count')} threads)"),
        ("GPU", f"{gpu.get('device_count', 0)} x {devices[0].get('name')}" if devices else "none detected"),
        ("Driver", gpu.get("driver_version") or "unknown"),
        ("CUDA (driver)", gpu.get("cuda_driver_version") or "unknown"),
        ("Python (environment)", venv_python_version(python) or "unknown"),
        ("Virtual env", (python.get("venv") or {}).get("path") or "none"),
        ("Created by", (python.get("venv") or {}).get("created_by") or "unknown"),
        ("Isaac Lab commit", f"{commit[:12]} on {git.get('branch') or '?'}{' (dirty)' if git.get('dirty') else ''}"),
        ("Isaac Lab release", repo.get("version") or "unknown"),
        ("Isaac Sim", isaac_sim.get("version") or distributions.get("isaacsim") or "not found"),
        ("Isaac Sim install", isaac_sim.get("install_method") or "unknown"),
        ("Sync command", manifest.get("sync", {}).get("command") or "unknown"),
        ("torch", distributions.get("torch", "not installed")),
        ("warp", distributions.get("warp-lang", "not installed")),
        ("newton", distributions.get("newton", "not installed")),
        ("ovrtx", distributions.get("ovrtx", "not installed")),
        ("Packages installed", str(len(python.get("distributions", [])))),
    ]


def _render_checkout_step(git: dict, remote: str) -> list[str]:
    """Return the reproduction step that clones the captured source at the captured commit.

    Args:
        git: The manifest's ``repo.git`` section.
        remote: The remote URL to clone, already sanitized by :func:`sanitize_remote_url`.

    Returns:
        The body of the step. The caller supplies the step number, its title, and the
        surrounding blank lines.
    """
    lines = ["```bash", f"git clone {remote} IsaacLab-repro", "cd IsaacLab-repro"]
    if git.get("commit"):
        if not git.get("commit_on_remote"):
            lines.append(f"# {git['commit']} is not on any remote branch; fetch it from the")
            lines.append(f"# machine that produced this bundle, or ask for branch {git.get('branch')}.")
        lines.append(f"git checkout {git['commit']}")
    lines.append("```")
    if git.get("remotes_redacted"):
        lines.append("")
        lines.append(
            "A credential embedded in one of the captured remote URLs was removed. Supply your own"
            " if the clone asks for one."
        )
    return lines


def _render_isaac_sim_step(isaac_sim: dict, sync: dict) -> list[str]:
    """Return the reproduction step for obtaining Isaac Sim the way the captured machine did.

    Args:
        isaac_sim: The manifest's ``isaac_sim`` section.
        sync: The manifest's ``sync`` section, used to tell whether the sync already covers a
            wheel install.

    Returns:
        The step title followed by its body lines, or an empty list when there is nothing to
        say. The caller supplies the step number and the surrounding blank lines.
    """
    method = isaac_sim.get("install_method", "none")
    version = isaac_sim.get("version")
    wheel_version = isaac_sim.get("wheel_version")
    resolved = isaac_sim.get("resolved_path") or "/path/to/isaacsim"

    if method == "wheel":
        body = [f"There was no `_isaac_sim` link; Isaac Sim was installed as the `isaacsim` wheel at {wheel_version}."]
        if "isaacsim" in sync.get("covers", []):
            body.append("")
            body.append(
                "The sync above already restores it -- the extras selected there pull in `isaacsim`"
                " -- so there is nothing further to do. Verify rather than assume:"
            )
            body.append("")
            body.append("```bash")
            body.append("uv pip show isaacsim")
            body.append("```")
        else:
            body.append("")
            body.append(
                "The extras selected above do not pull it in, so install it explicitly. Take the"
                " extras from `pyproject.toml` rather than a bare install: the wheel is split across"
                " several distributions and `isaacsim` alone is a shell."
            )
            body.append("")
            body.append("```bash")
            body.append(f"uv pip install 'isaacsim[all,extscache]=={wheel_version}'")
            body.append("```")
        return ["Install the same Isaac Sim wheel.", *body]

    if method == "binary":
        body = [
            f"`_isaac_sim` pointed at a downloaded Isaac Sim package, version"
            f" `{version or 'unknown'}`. Download that same version from the"
            f" [Isaac Sim download page]({ISAAC_SIM_DOWNLOAD_URL}), extract it anywhere, and link"
            " it the way the captured machine did. The link is what decides which Kit build runs;"
            " the path it points at is yours to choose.",
            "",
            "```bash",
            f"export ISAACSIM_PATH=/path/to/isaacsim   # was {resolved}",
            "ln -s ${ISAACSIM_PATH} _isaac_sim",
            "```",
        ]
        if wheel_version:
            body.extend(
                [
                    "",
                    f"An `isaacsim` wheel ({wheel_version}) was installed alongside the link. Keep"
                    " both: which one is imported depends on how the process is launched, so"
                    " dropping either changes the environment being reproduced.",
                ]
            )
        return ["Install the same downloaded Isaac Sim package.", *body]

    if method == "source_build":
        revision = isaac_sim.get("source_revision")
        branch = isaac_sim.get("source_branch")
        provenance = f" built from `{branch}` at `{revision}`" if revision else ""
        # The link points at the build output, not the checkout, so the step has to name the
        # same subpath the captured machine used rather than the checkout root.
        parts = Path(resolved).parts
        output = "/".join(parts[parts.index("_build") :]) if "_build" in parts else "_build/linux-x86_64/release"
        body = [
            f"`_isaac_sim` pointed into a local Isaac Sim build tree, version"
            f" `{version or 'unknown'}`{provenance}. No bundle can carry a compiled Kit, so this is"
            " the one step that has to be repeated rather than restored.",
            "",
            "```bash",
            f"git clone {ISAAC_SIM_SOURCE_URL}",
            "cd IsaacSim",
        ]
        if revision:
            body.append(f"git checkout {revision}")
        else:
            body.append("# the VERSION string carried no revision; ask which one this was built from")
        body.extend(
            [
                "./build.sh",
                "cd -",
                f"ln -s /path/to/IsaacSim/{output} _isaac_sim",
                "```",
                "",
                "Substituting the `isaacsim` wheel is a legitimate shortcut when the build is not"
                " the thing under investigation, but it is a change to the environment and belongs"
                " in the report.",
            ]
        )
        if wheel_version:
            body.extend(
                [
                    "",
                    f"An `isaacsim` wheel ({wheel_version}) was installed alongside the build. Keep"
                    " both: which one is imported depends on how the process is launched.",
                ]
            )
        return ["Build the same Isaac Sim revision.", *body]

    return [
        "Isaac Sim was not installed on the captured machine.",
        "No `_isaac_sim` link resolved and no `isaacsim` wheel was installed. If the command under"
        " test needed Isaac Sim, this is the finding rather than a step.",
    ]


def render_document(manifest: dict, artifacts: dict[str, str]) -> str:
    """Render the reproduction document for a captured environment.

    Args:
        manifest: A manifest produced by :func:`build_manifest`.
        artifacts: The files stored alongside it, listed in the contents section.

    Returns:
        Markdown, written both into the bundle and next to it.
    """
    repo = manifest.get("repo", {})
    git = repo.get("git", {})
    isaac_sim = manifest.get("isaac_sim", {})
    environment = manifest.get("environment", {})
    findings = manifest.get("findings", [])
    remotes = git.get("remotes") or ["https://github.com/isaac-sim/IsaacLab.git"]

    lines: list[str] = []
    lines.append("# Isaac Lab environment capture")
    lines.append("")
    lines.append(
        "Everything needed to rebuild the environment this bundle was taken from, and an"
        " explicit list of what it cannot rebuild."
    )
    lines.append("")

    lines.append("## At a glance")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    for label, value in _summary_rows(manifest):
        lines.append(f"| {label} | {value} |")
    lines.append("")

    command = manifest.get("capture", {}).get("command_under_test")
    if command:
        lines.append("## Command under test")
        lines.append("")
        lines.append("```")
        lines.append(command)
        lines.append("```")
        lines.append("")

    lines.append("## Findings")
    lines.append("")
    if not findings:
        lines.append("No problems detected. Nothing below contradicts a clean installation.")
        lines.append("")
    else:
        counts = {level: sum(1 for f in findings if f["level"] == level) for level in ("error", "warning", "info")}
        lines.append(
            f"{counts['error']} error(s), {counts['warning']} warning(s), {counts['info']} note(s)."
            " Errors describe an environment that is already broken; warnings describe one that"
            " will not reproduce identically."
        )
        lines.append("")
        for finding in findings:
            marker = {"error": "ERROR", "warning": "WARNING", "info": "NOTE"}[finding["level"]]
            lines.append(f"### {marker}: {finding['summary']}")
            lines.append("")
            lines.append(f"`{finding['code']}`")
            lines.append("")
            if finding["detail"]:
                lines.append("```")
                lines.extend(finding["detail"])
                lines.append("```")
                lines.append("")

    lines.append("## Reproducing this environment")
    lines.append("")
    lines.append(
        "The steps below rebuild the Python side exactly, from the lockfile carried in this"
        " bundle. The host side -- GPU model and driver -- has to be matched by hand; see"
        " *What this bundle cannot reproduce*."
    )
    lines.append("")

    step = 1
    lines.append(f"**{step}. Unpack this bundle.**")
    lines.append("")
    lines.append("Later steps copy files out of it, so it has to exist on disk as a directory.")
    lines.append("")
    lines.append("```bash")
    lines.append("unzip <this-bundle>.zip -d bundle")
    lines.append("```")
    lines.append("")
    lines.append("Paths written as `/path/to/bundle/...` below refer to that directory.")
    lines.append("")
    step += 1
    lines.append(f"**{step}. Check out the same source.**")
    lines.append("")
    lines.extend(_render_checkout_step(git, remotes[0]))
    lines.append("")
    step += 1

    if git.get("diff_included"):
        lines.append(f"**{step}. Apply the uncommitted changes.**")
        lines.append("")
        lines.append("```bash")
        lines.append("git apply /path/to/bundle/repo/git-diff.patch")
        lines.append("```")
        lines.append("")
        step += 1
    elif git.get("dirty"):
        lines.append(f"**{step}. Uncommitted changes are missing from this bundle.**")
        lines.append("")
        lines.append(
            f"The source tree had {git.get('modified_files', 0)} modified and"
            f" {git.get('untracked_files', 0)} untracked file(s) that were not captured."
            " `repo/git-diffstat.txt` lists the modified ones, so the gap is at least visible."
            " Ask for a bundle taken with `--include_diff`, or for the patch separately, before"
            " treating a reproduction attempt as conclusive."
        )
        lines.append("")
        step += 1

    sync = manifest.get("sync", {})
    extras = sync.get("extras", [])
    lines.append(f"**{step}. Restore the exact dependency set.**")
    lines.append("")
    lines.append(
        "`files/uv.lock` and `files/pyproject.toml` are the captured originals. Copy them over"
        " the checkout so the resolution cannot drift, then sync."
    )
    lines.append("")
    lines.append("```bash")
    lines.append("cp /path/to/bundle/files/pyproject.toml .")
    lines.append("cp /path/to/bundle/files/uv.lock .")
    lines.append(sync.get("command", "uv sync --locked"))
    lines.append("```")
    lines.append("")
    if extras:
        lines.append(
            "The extras are not a guess: "
            + ", ".join(f"`{name}`" for name in extras)
            + (" was" if len(extras) == 1 else " were")
            + " selected because every distribution each one requires was found installed, and no"
            " other extra's was. Leaving them off matters more than it looks -- `uv sync` installs"
            " the lockfile and *removes* everything else, so a bare `uv sync` would strip this"
            " environment down to the default dependency set rather than rebuild it."
        )
    elif sync.get("lock_available"):
        lines.append(
            "No extra had all of its requirements installed, so the captured environment is the default dependency set."
        )
    else:
        lines.append(
            "The checkout had no `uv.lock`, so the extras could not be determined and the command"
            " above is the bare default. Compare `python/distributions.txt` against the extras in"
            " `pyproject.toml` before trusting a reproduction."
        )
    lines.append("")
    lines.append(
        "`--locked` fails rather than re-resolving if the lockfile does not satisfy"
        " `pyproject.toml`, which is what you want here: a silent re-resolution would defeat"
        " the purpose of the capture."
    )
    lines.append("")
    step += 1

    unlocked = sync.get("unlocked", [])
    if unlocked:
        lines.append(f"**{step}. Reinstall what the lockfile does not describe.**")
        lines.append("")
        lines.append(
            "The sync above removes these, because no lockfile entry accounts for them. Something"
            " other than `uv sync` put them on the captured machine -- `uv pip install`, a manual"
            " `pip`, or a later upgrade -- and one of them may be the difference being"
            " investigated. Reinstall them after the sync, not before."
        )
        lines.append("")
        lines.append("```bash")
        for entry in unlocked[:20]:
            note = entry["reason"] + (f", locked {', '.join(entry['locked'])}" if entry["locked"] else "")
            lines.append(f"uv pip install --no-deps '{entry['name']}=={entry['version']}'  # {note}")
        lines.append("```")
        lines.append("")
        if len(unlocked) > 20:
            lines.append(f"{len(unlocked) - 20} further distribution(s) are listed in `manifest.json`.")
            lines.append("")
        step += 1

    isaac_step = _render_isaac_sim_step(isaac_sim, sync)
    if isaac_step:
        lines.append(f"**{step}. {isaac_step[0]}**")
        lines.append("")
        lines.extend(isaac_step[1:])
        lines.append("")
        step += 1

    symlinks = manifest.get("links", {}).get("symlinks", [])
    # Tracked links arrive with the clone, and virtual-environment links are recreated by the
    # sync in the previous step. What is left is what somebody made by hand.
    repo_root = str(repo.get("root", ""))
    # The Isaac Sim step above recreates _isaac_sim with instructions for obtaining what it
    # points at, which a bare `ln -s` to the captured machine's path cannot do.
    isaac_link_covered = isaac_sim.get("install_method") in ("binary", "source_build")
    repo_links = [
        link
        for link in symlinks
        if link["path"].startswith(repo_root)
        and not link.get("tracked")
        and not link.get("in_virtualenv")
        and not (isaac_link_covered and os.path.basename(link["path"]) == "_isaac_sim")
    ]
    if repo_links:
        lines.append(f"**{step}. Recreate the local symlinks.**")
        lines.append("")
        lines.append(
            "Git does not track these and no lockfile mentions them, but they decide which files"
            " actually get imported. Paths are relative to the repository root."
        )
        lines.append("")
        lines.append("```bash")
        for link in repo_links[:20]:
            relative = link["path"][len(repo_root) :].lstrip("/") or link["path"]
            state = "ok" if link["exists"] else "BROKEN on the captured machine"
            lines.append(f"ln -s {link['target']} {relative}   # {state}")
        lines.append("```")
        lines.append("")
        if len(repo_links) > 20:
            lines.append(f"{len(repo_links) - 20} further link(s) are listed in `manifest.json`.")
            lines.append("")
        step += 1

    variables = environment.get("variables", {})
    exportable = {name: value for name, value in variables.items() if name not in MACHINE_OWNED_ENV_VARS}
    if exportable:
        lines.append(f"**{step}. Set the environment variables that change behaviour.**")
        lines.append("")
        lines.append(
            "These are the variables Isaac Lab and its runtime stack are known to read, and they"
            " affect import resolution, device selection, and renderer choice. Nothing else from"
            " the captured environment was collected."
        )
        lines.append("")
        lines.append("```bash")
        for name in sorted(exportable):
            lines.append(f"export {name}={exportable[name]!r}")
        lines.append("```")
        lines.append("")
        machine_owned = sorted(set(variables) & MACHINE_OWNED_ENV_VARS)
        if machine_owned:
            lines.append(
                "Recorded but deliberately not listed above, because they describe the captured"
                " machine rather than the run and copying them would break yours: "
                + ", ".join(f"`{name}`" for name in machine_owned)
                + ". See `env/environment.txt` for their captured values."
            )
            lines.append("")
        step += 1

    lines.append(f"**{step}. Confirm you actually reproduced it.**")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 /path/to/bundle/capture_env.py diff /path/to/bundle.zip")
    lines.append("```")
    lines.append("")
    lines.append(
        "This captures the rebuilt machine and reports every difference against the bundle."
        " An empty report means the two environments agree on everything recorded here. The"
        " bundle carries the script that produced it, so this works against a checkout too old"
        " to contain it and on a machine without the repository at all."
    )
    lines.append("")

    lines.append("## What this bundle cannot reproduce")
    lines.append("")
    blockers: list[str] = []
    gpu = manifest.get("gpu", {})
    devices = gpu.get("devices", [])
    if devices:
        blockers.append(
            f"**Hardware.** Captured on {gpu.get('device_count')} x"
            f" {devices[0].get('name')} with driver {gpu.get('driver_version')}."
            " GPU-specific and driver-specific behaviour will not appear on different hardware."
        )
    if isaac_sim.get("source_build"):
        blockers.append(
            "**A locally built Isaac Sim.** `_isaac_sim` points into a build tree"
            f" (`{isaac_sim.get('link_target')}`). That Kit build is not in this bundle and"
            " cannot be rebuilt from it. Either obtain the same build, or reproduce against"
            " the `isaacsim` wheel and note the change."
        )
    if git.get("dirty") and not git.get("diff_included"):
        blockers.append("**Uncommitted source changes.** Recorded as a diffstat only; see step 2.")
    if environment.get("omitted_count"):
        blockers.append(
            f"**The rest of the process environment.** {environment['omitted_count']} variable(s)"
            " were present but not collected, because Isaac Lab does not read them. If a"
            " reproduction still diverges after everything else matches, an uncollected"
            " variable is one place left to look."
        )
    blockers.append(
        "**Anything outside the repository and the virtual environment.** System packages,"
        " ROS overlays, and files reached through `PYTHONPATH` or `LD_LIBRARY_PATH` are"
        " recorded by path only."
    )
    for blocker in blockers:
        lines.append(f"- {blocker}")
    lines.append("")

    lines.append("## What is in this bundle")
    lines.append("")
    descriptions = {
        "manifest.json": "Everything below in machine-readable form; the input to `diff`.",
        "REPRODUCE.md": "This document.",
        "capture_env.py": "The script that produced this bundle, so `diff` runs without a checkout.",
        "files/pyproject.toml": "Project definition, captured verbatim.",
        "files/uv.lock": "Resolved dependency set, captured verbatim.",
        "files/pyvenv.cfg": "Virtual environment configuration, including its base interpreter.",
        "python/distributions.txt": "Every installed package with its version, installer, and origin.",
        "python/integrity.json": "Packages whose files are missing from disk despite their metadata.",
        "python/venv-sys-path.txt": "Import path the environment's own interpreter resolves, `.pth` files included.",
        "python/capturing-sys-path.txt": "Import path of whichever interpreter ran the capture.",
        "env/environment.txt": "The allowlisted environment variables; everything else was not collected.",
        "repo/git-status.txt": "Working tree status at capture time.",
        "repo/git-diffstat.txt": "Which files were modified, and by how much.",
        "repo/git-diff.patch": "The uncommitted changes themselves.",
        "repo/git-log.txt": "Last 20 commits.",
        "repo/git-remote.txt": "Configured remotes, with any embedded credential removed.",
        "repo/git-untracked.txt": "Untracked files not covered by .gitignore.",
        "host/nvidia-smi.txt": "Full driver and GPU inventory.",
        "host/nvidia-smi-query.csv": "One row per GPU: model, UUID, VBIOS, memory, compute mode.",
        "host/nvidia-smi-summary.txt": "Driver and CUDA version header.",
        "host/vulkaninfo.txt": "Vulkan stack that ovrtx initializes through.",
        "host/glxinfo.txt": "OpenGL renderer strings.",
        "host/nvidia-module-version.txt": "Loaded NVIDIA kernel module version.",
        "host/os-release.txt": "Distribution identification.",
        "host/uname.txt": "Kernel identification.",
        "host/cpuinfo.txt": "CPU model and topology.",
        "host/meminfo.txt": "Installed memory.",
        "host/nvcc.txt": "CUDA toolkit version, when a toolkit is installed.",
        "isaacsim/VERSION": "Version of the Isaac Sim that `_isaac_sim` resolves to.",
        "isaacsim/setup_python_env.sh": "What Kit prepends to PYTHONPATH and LD_LIBRARY_PATH.",
    }
    lines.append("| Path | Contents |")
    lines.append("|---|---|")
    for name in sorted(artifacts):
        if name.startswith("files/pth/"):
            description = "A `.pth` file executed at interpreter start; it adds import paths."
        elif name.startswith("logs/"):
            description = "Tail of a log from the captured machine."
        else:
            description = descriptions.get(name, "Captured file.")
        lines.append(f"| `{name}` | {description} |")
    lines.append("")

    lines.append("## What was deliberately not collected")
    lines.append("")
    lines.append(
        f"The process environment is captured by allowlist, not wholesale."
        f" {len(environment.get('variables', {}))} variable(s) from the fixed list of names"
        f" Isaac Lab reads or sets were collected; {environment.get('omitted_count', 0)} other"
        " variable(s) were present and left out, and are not named anywhere in this bundle."
    )
    lines.append("")
    lines.append(
        "The list is closed and matched by exact name, so a variable this project does not read"
        " cannot appear here however it is spelled. No credential store, SSH key, cloud"
        " configuration, or Docker configuration is read, and nothing outside the repository,"
        " the virtual environment, and the host inventory commands is opened."
    )
    lines.append("")
    lines.append(
        "Paths, hostnames, and usernames *are* captured as-is, because they are frequently the"
        " cause. Review `env/environment.txt` and, if it is present, `repo/git-diff.patch`"
        " before sending this bundle anywhere."
    )
    lines.append("")
    return "\n".join(lines)


def write_bundle(destination: Path, manifest: dict, artifacts: dict[str, str], document: str) -> None:
    """Write the manifest, the document, and every artifact into a zip at ``destination``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        archive.writestr("REPRODUCE.md", document)
        for name, content in sorted(artifacts.items()):
            archive.writestr(name, content)


def load_manifest(path: Path) -> dict:
    """Load a manifest from a bundle zip, an unpacked bundle directory, or a manifest file.

    Args:
        path: What to load.

    Returns:
        The parsed manifest.

    Raises:
        FileNotFoundError: If ``path`` holds no manifest.
        ValueError: If the manifest was written by an incompatible schema version.
    """
    if path.is_dir():
        manifest = json.loads((path / "manifest.json").read_text())
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    elif path.is_file():
        manifest = json.loads(path.read_text())
    else:
        raise FileNotFoundError(f"no bundle, directory, or manifest at {path}")

    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses manifest schema {version}, but this script writes and reads schema"
            f" {SCHEMA_VERSION}. Compare with the version of capture_env.py that produced it."
        )
    return manifest


def _flatten(manifest: dict) -> dict[str, str]:
    """Return the scalar facts of a manifest, keyed by the label ``diff`` reports them under."""
    host = manifest.get("host", {})
    gpu = manifest.get("gpu", {})
    python = manifest.get("python", {})
    repo = manifest.get("repo", {})
    git = repo.get("git", {})
    isaac_sim = manifest.get("isaac_sim", {})
    devices = gpu.get("devices", [])
    return {
        "os": host.get("os_release", {}).get("PRETTY_NAME") or host.get("platform"),
        "kernel": host.get("kernel"),
        "architecture": host.get("architecture"),
        "libc": host.get("libc"),
        "in_container": str(host.get("in_container")),
        "cpu_model": host.get("cpu_model"),
        "gpu_model": devices[0].get("name") if devices else None,
        "gpu_count": str(gpu.get("device_count")),
        "driver_version": gpu.get("driver_version"),
        "cuda_driver_version": gpu.get("cuda_driver_version"),
        "vulkan_available": str(gpu.get("vulkan_available")),
        "python_version": venv_python_version(python),
        "venv_created_by": (python.get("venv") or {}).get("created_by"),
        "isaaclab_commit": git.get("commit"),
        "isaaclab_branch": git.get("branch"),
        "isaaclab_dirty": str(git.get("dirty")),
        "isaaclab_release": repo.get("version"),
        "isaac_sim_version": isaac_sim.get("version"),
        "isaac_sim_link_target": isaac_sim.get("link_target"),
        "isaac_sim_install_method": isaac_sim.get("install_method"),
        "isaac_sim_wheel_version": isaac_sim.get("wheel_version"),
        "isaac_sim_source_revision": isaac_sim.get("source_revision"),
        "sync_extras": " ".join(manifest.get("sync", {}).get("extras", [])) or None,
    }


def _pth_entries(lines: list[str] | None) -> str:
    """Return the lines of one ``.pth`` file as a single readable cell."""
    if lines is None:
        return "*absent*"
    entries = [line for line in lines if line.strip()]
    return ", ".join(f"`{entry}`" for entry in entries) if entries else "*empty*"


def render_diff(baseline: dict, current: dict) -> str:
    """Render every difference between two captured environments.

    Every section of the manifest that can change what an installation imports or how it runs
    is compared, because a reproduction accepted on a partial comparison is worse than one
    that was never attempted. A section that matches says so rather than being dropped, so the
    report doubles as the list of what was checked.

    Args:
        baseline: The manifest being compared against, usually the reported environment.
        current: The manifest being checked, usually the local reproduction attempt.

    Returns:
        Markdown listing what differs, section by section.
    """
    lines: list[str] = ["# Environment diff", ""]
    baseline_label = baseline.get("capture", {}).get("hostname", "baseline")
    current_label = current.get("capture", {}).get("hostname", "current")
    # Comparing a machine against its own earlier capture is the common case, and identical
    # column headings would make the table unreadable in exactly that case.
    if baseline_label == current_label:
        baseline_label, current_label = f"{baseline_label} (baseline)", f"{current_label} (current)"
    lines.append(f"- **baseline**: `{baseline_label}` captured {baseline.get('captured_at')}")
    lines.append(f"- **current**: `{current_label}` captured {current.get('captured_at')}")
    lines.append("")

    differences = 0

    left, right = _flatten(baseline), _flatten(current)
    rows = [(key, left[key], right[key]) for key in left if left[key] != right[key]]
    lines.append("## Host and versions")
    lines.append("")
    if rows:
        differences += len(rows)
        lines.append(f"| | {baseline_label} | {current_label} |")
        lines.append("|---|---|---|")
        for key, before, after in rows:
            lines.append(f"| {key} | {before} | {after} |")
    else:
        lines.append("Identical.")
    lines.append("")

    def distributions(manifest: dict) -> dict[str, str]:
        return {dist["key"]: dist["version"] for dist in manifest.get("python", {}).get("distributions", [])}

    before_dists, after_dists = distributions(baseline), distributions(current)
    only_baseline = sorted(set(before_dists) - set(after_dists))
    only_current = sorted(set(after_dists) - set(before_dists))
    changed = sorted(key for key in set(before_dists) & set(after_dists) if before_dists[key] != after_dists[key])

    lines.append("## Packages")
    lines.append("")
    if only_baseline or only_current or changed:
        differences += len(only_baseline) + len(only_current) + len(changed)
        lines.append(f"| Package | {baseline_label} | {current_label} |")
        lines.append("|---|---|---|")
        for key in changed:
            lines.append(f"| {key} | {before_dists[key]} | {after_dists[key]} |")
        for key in only_baseline:
            lines.append(f"| {key} | {before_dists[key]} | *not installed* |")
        for key in only_current:
            lines.append(f"| {key} | *not installed* | {after_dists[key]} |")
    else:
        lines.append(f"Identical: {len(before_dists)} packages match.")
    lines.append("")

    before_env = baseline.get("environment", {}).get("variables", {})
    after_env = current.get("environment", {}).get("variables", {})
    env_rows = [
        (name, before_env.get(name, "*unset*"), after_env.get(name, "*unset*"))
        for name in sorted(set(before_env) | set(after_env))
        if before_env.get(name) != after_env.get(name)
    ]
    lines.append("## Environment variables")
    lines.append("")
    if env_rows:
        differences += len(env_rows)
        lines.append(f"| Variable | {baseline_label} | {current_label} |")
        lines.append("|---|---|---|")
        for name, before, after in env_rows:
            lines.append(f"| `{name}` | `{before}` | `{after}` |")
    else:
        lines.append("Identical.")
    lines.append("")

    def links(manifest: dict) -> dict[str, str]:
        return {link["path"]: str(link["target"]) for link in manifest.get("links", {}).get("symlinks", [])}

    before_links, after_links = links(baseline), links(current)
    link_names = set(before_links) | set(after_links)
    link_rows = [
        (name, before_links.get(name, "*absent*"), after_links.get(name, "*absent*"))
        for name in sorted(link_names)
        if before_links.get(name) != after_links.get(name)
    ]
    lines.append("## Symlinks")
    lines.append("")
    if link_rows:
        differences += len(link_rows)
        lines.append(f"| Path | {baseline_label} | {current_label} |")
        lines.append("|---|---|---|")
        for name, before, after in link_rows:
            lines.append(f"| `{name}` | `{before}` | `{after}` |")
    else:
        lines.append("Identical.")
    lines.append("")

    # A `.pth` file is executed at interpreter start and can put anything on sys.path, so two
    # environments holding the same package versions still import different code when these
    # differ. Compared line by line rather than as whole files so the added path is named.
    def pth_files(manifest: dict) -> dict[str, list[str]]:
        return {
            entry["name"]: list(entry.get("lines", [])) for entry in manifest.get("python", {}).get("pth_files", [])
        }

    before_pth, after_pth = pth_files(baseline), pth_files(current)
    pth_names = [
        name for name in sorted(set(before_pth) | set(after_pth)) if before_pth.get(name) != after_pth.get(name)
    ]
    lines.append("## Import path files")
    lines.append("")
    if pth_names:
        differences += len(pth_names)
        for name in pth_names:
            lines.append(f"- `{name}`")
            lines.append(f"    - {baseline_label}: {_pth_entries(before_pth.get(name))}")
            lines.append(f"    - {current_label}: {_pth_entries(after_pth.get(name))}")
    else:
        lines.append("Identical.")
    lines.append("")

    # A distribution present at the right version but missing the files its own RECORD claims
    # imports as an empty namespace package instead of raising, so package versions matching is
    # not enough to call two environments equivalent.
    def damaged(manifest: dict) -> dict[str, str]:
        integrity = manifest.get("python", {}).get("integrity") or {}
        return {
            entry["distribution"]: f"{entry['missing']} of {entry['recorded']} files missing"
            for entry in integrity.get("damaged", [])
        }

    lines.append("## Package integrity")
    lines.append("")
    if baseline.get("python", {}).get("integrity") is None or current.get("python", {}).get("integrity") is None:
        lines.append("Not comparable: at least one capture ran with `--skip_integrity`.")
    else:
        before_damaged, after_damaged = damaged(baseline), damaged(current)
        damaged_rows = [
            (name, before_damaged.get(name, "*intact*"), after_damaged.get(name, "*intact*"))
            for name in sorted(set(before_damaged) | set(after_damaged))
            if before_damaged.get(name) != after_damaged.get(name)
        ]
        if damaged_rows:
            differences += len(damaged_rows)
            lines.append(f"| Distribution | {baseline_label} | {current_label} |")
            lines.append("|---|---|---|")
            for name, before, after in damaged_rows:
                lines.append(f"| `{name}` | {before} | {after} |")
        else:
            lines.append("Identical.")
    lines.append("")

    # Keyed by code and summary together: one code covers many findings -- `missing-package-files`
    # is raised once per damaged distribution -- and collapsing by code alone reports two
    # environments broken in different places as agreeing.
    def findings(manifest: dict) -> set[tuple[str, str]]:
        return {(finding["code"], finding["summary"]) for finding in manifest.get("findings", [])}

    before_findings, after_findings = findings(baseline), findings(current)
    lines.append("## Findings")
    lines.append("")
    only_before = sorted(before_findings - after_findings)
    only_after = sorted(after_findings - before_findings)
    if only_before or only_after:
        differences += len(only_before) + len(only_after)
        for code, summary in only_before:
            lines.append(f"- Only on `{baseline_label}`: **{code}** -- {summary}")
        for code, summary in only_after:
            lines.append(f"- Only on `{current_label}`: **{code}** -- {summary}")
    else:
        shared = sorted({code for code, _ in before_findings})
        lines.append(
            "Both environments report the same findings." + (f" ({', '.join(shared)})" if shared else " (none)")
        )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"{differences} difference(s) recorded."
        if differences
        else "No differences recorded. The two environments agree on everything captured."
    )
    lines.append("")
    return "\n".join(lines)


def command_capture(args: argparse.Namespace) -> int:
    """Capture the current environment into a bundle and write the reproduction document."""
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(Path.cwd())
    if args.venv:
        venv: Path | None = Path(args.venv).resolve()
    elif os.environ.get("VIRTUAL_ENV"):
        venv = Path(os.environ["VIRTUAL_ENV"]).resolve()
    elif (repo_root / ".venv").is_dir():
        venv = repo_root / ".venv"
    else:
        venv = None

    print(f"repository: {repo_root}")
    print(f"virtual environment: {venv or 'none found'}")
    print("collecting ...")

    manifest, artifacts = build_manifest(
        repo_root=repo_root,
        venv=venv,
        command=args.command,
        include_diff=args.include_diff,
        include_logs=args.include_logs,
        skip_integrity=args.skip_integrity,
    )
    document = render_document(manifest, artifacts)

    stem = args.name or f"isaaclab-env-{socket.gethostname()}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    output_dir = Path(args.output_dir).resolve()
    bundle_path = output_dir / f"{stem}.zip"
    write_bundle(bundle_path, manifest, artifacts, document)
    document_path = output_dir / f"{stem}.md"
    document_path.write_text(document)

    print("")
    for label, value in _summary_rows(manifest):
        print(f"  {label:<20} {value}")
    print("")

    findings = manifest["findings"]
    if findings:
        counts = {level: sum(1 for f in findings if f["level"] == level) for level in ("error", "warning", "info")}
        print(f"  {counts['error']} error(s), {counts['warning']} warning(s), {counts['info']} note(s):")
        for finding in findings:
            print(f"    [{finding['level']}] {finding['summary']}")
    else:
        print("  no problems detected")
    print("")
    print(f"bundle:   {bundle_path}  ({bundle_path.stat().st_size / 1024:.0f} KiB)")
    print(f"document: {document_path}")
    environment = manifest["environment"]
    print(
        f"env vars: {len(environment['variables'])} collected by allowlist,"
        f" {environment['omitted_count']} present and not collected"
    )
    print("")
    print("Review the bundle before sending it: paths, hostnames, and usernames are captured as-is.")
    return 0


def command_diff(args: argparse.Namespace) -> int:
    """Compare two bundles, or one bundle against the machine running this script."""
    baseline = load_manifest(Path(args.baseline))
    if args.current:
        current = load_manifest(Path(args.current))
    else:
        repo_root = find_repo_root(Path.cwd())
        venv_env = os.environ.get("VIRTUAL_ENV")
        venv = Path(venv_env) if venv_env else (repo_root / ".venv" if (repo_root / ".venv").is_dir() else None)
        current, _ = build_manifest(repo_root=repo_root, venv=venv, skip_integrity=args.skip_integrity)

    report = render_diff(baseline, current)
    if args.output:
        Path(args.output).write_text(report)
        print(f"wrote {args.output}")
    else:
        print(report)
    return 0


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the capture and diff subcommands."""
    parser = argparse.ArgumentParser(
        prog="capture_env.py",
        description="Capture an Isaac Lab installation into a portable bundle, and diff two captures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python3 tools/capture_env.py capture --command 'isaaclab train --task Isaac-Cartpole-v0'\n"
            "  python3 tools/capture_env.py capture --include-diff --include-logs\n"
            "  python3 tools/capture_env.py diff customer-bundle.zip\n"
            "  python3 tools/capture_env.py diff customer-bundle.zip our-bundle.zip\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    capture = subparsers.add_parser("capture", help="Capture this machine's environment into a bundle.")
    capture.add_argument("--repo_root", default=None, help="Isaac Lab repository to describe. Default: auto-detected.")
    capture.add_argument(
        "--venv",
        default=None,
        help="Virtual environment to describe. Default: $VIRTUAL_ENV, else <repo_root>/.venv.",
    )
    capture.add_argument("--output_dir", default=".", help="Where to write the bundle and document. Default: cwd.")
    capture.add_argument("--name", default=None, help="Bundle name without extension. Default: host and timestamp.")
    capture.add_argument(
        "--command",
        default=None,
        help="The command whose behaviour prompted this capture, recorded verbatim in the bundle.",
    )
    capture.add_argument(
        "--include_diff",
        action="store_true",
        help="Attach the uncommitted working-tree patch. Off by default: a dirty tree can hold unshareable code.",
    )
    capture.add_argument(
        "--include_logs",
        action="store_true",
        help="Attach tails of the most recent renderer, Omniverse, and Isaac Lab logs.",
    )
    capture.add_argument(
        "--skip_integrity",
        action="store_true",
        help="Skip checking installed files against each package's RECORD. Faster, but misses gutted packages.",
    )
    capture.set_defaults(handler=command_capture)

    diff = subparsers.add_parser("diff", help="Report the differences between two captures.")
    diff.add_argument("baseline", help="Bundle zip, unpacked bundle directory, or manifest.json.")
    diff.add_argument(
        "current",
        nargs="?",
        default=None,
        help="What to compare against. Default: capture this machine now.",
    )
    diff.add_argument(
        "--skip_integrity", action="store_true", help="Skip the RECORD check when capturing this machine."
    )
    diff.add_argument("--output", default=None, help="Write the report to this path instead of stdout.")
    diff.set_defaults(handler=command_diff)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_cli_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
