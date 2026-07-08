# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Select which CI test jobs to run for a change based on the affected files.

The Docker test jobs in ``.github/workflows/build.yaml`` are expensive. Rather than run
every job on every pull request, this helper maps the set of changed files to a per-job
"run" flag using a conservative, dependency-aware path map derived from the inter-package
import graph.

Design invariants:

* **Conservative by construction.** Any change to a hub package (core, assets, tasks,
  newton, physx), to shared low-level packages, to build tooling / Docker / CI, or to any
  path not covered by an explicit rule falls back to running the *entire* suite. The
  failure mode is always "runs too much", never "misses a break".
* **isaaclab_target packages get a subset.** Packages that few or nothing depend on
  (``isaaclab_mimic``, ``isaaclab_rl``, ``isaaclab_teleop``, ``isaaclab_visualizers``,
  ``isaaclab_ov`` / ``isaaclab_ovphysx``, ``isaaclab_contrib``) map to just the jobs whose
  tests can be affected.
* **Docs-only changes run nothing.** Markdown / reStructuredText / changelog fragments and
  anything under a ``docs/`` directory contribute no jobs.

The module exposes a pure :func:`select_ci_jobs` for unit testing and a ``__main__`` entry
point that reads newline-separated changed paths from stdin and writes ``run_<group>=<bool>``
lines to stdout (append to ``$GITHUB_OUTPUT``).
"""

from __future__ import annotations

import re
import sys

# --- Job groups -------------------------------------------------------------------------
# Each name corresponds to a ``run_<name>`` output consumed by a job's ``if:`` in build.yaml.
# The three sharded tasks/core jobs share a single flag.
_BASE_JOBS: tuple[str, ...] = (
    "core",
    "tasks",
    "rl",
    "mimic",
    "contrib",
    "teleop",
    "visualizers",
    "assets",
    "newton",
    "physx",
    "ov",
    "curobo",
    "skillgen",
    "environments_training",
    "rendering",
    "rendering_kitless",
)

# The tasks package drives its own job plus the rendering / training / curobo / skillgen
# jobs, which all exercise ``source/isaaclab_tasks``.
_TASKS_FAMILY: frozenset[str] = frozenset(
    {"tasks", "rendering", "rendering_kitless", "environments_training", "curobo", "skillgen"}
)

# --- Path classification ----------------------------------------------------------------
# Docs / changelog files never trigger tests.
_IGNORED_SUFFIXES = (".md", ".rst", ".skip")

# Package prefixes that fan out to the whole suite because many packages import them
# (core is imported by every package; assets/newton/physx/tasks are deep hubs) or because
# they are shared low-level packages without a dedicated job.
_GLOBAL_PACKAGE_PREFIXES: tuple[str, ...] = (
    "source/isaaclab/",
    "source/isaaclab_assets/",
    "source/isaaclab_tasks/",
    "source/isaaclab_newton/",
    "source/isaaclab_physx/",
    "source/isaaclab_experimental/",
    "source/isaaclab_ppisp/",
    "source/isaaclab_tasks_experimental/",
)

# Non-source directories/files whose changes can affect any job.
_GLOBAL_DIR_PREFIXES: tuple[str, ...] = ("docker/", "tools/", "apps/", "scripts/")
_GLOBAL_CI_PREFIXES: tuple[str, ...] = (".github/actions/",)
_GLOBAL_CI_FILES: frozenset[str] = frozenset(
    {
        ".github/workflows/build.yaml",
        ".github/workflows/config.yaml",
    }
)

# Repo-root config files (dependency pins, lock files, top-level tooling config).
_ROOT_CONFIG = re.compile(r"^(?:[^/]+\.(?:toml|yaml|yml|json|ini|cfg|conf|lock|sh|bat|ps1)|\.gitmodules)$")

# isaaclab_target packages -> the exact set of job groups whose tests they can affect. Derived from the
# import graph; see the module docstring. ``core`` is included where core lazily imports the
# package (video recorder / markers / teleop shim), so editing it can touch core runtime paths.
_TARGETED: dict[str, frozenset[str]] = {
    "source/isaaclab_mimic/": frozenset({"mimic"}),
    "source/isaaclab_rl/": frozenset({"rl"}) | _TASKS_FAMILY,
    "source/isaaclab_teleop/": frozenset({"teleop", "core"}) | _TASKS_FAMILY,
    "source/isaaclab_visualizers/": frozenset({"visualizers", "core", "rendering", "rendering_kitless"}),
    "source/isaaclab_ov/": frozenset({"ov", "core"}) | _TASKS_FAMILY,
    "source/isaaclab_ovphysx/": frozenset({"ov", "core"}) | _TASKS_FAMILY,
    "source/isaaclab_contrib/": frozenset({"contrib", "core", "newton", "physx", "assets"}) | _TASKS_FAMILY,
}


def _is_ignored(path: str) -> bool:
    """Return whether a docs/changelog path should contribute no test jobs."""
    return path.endswith(_IGNORED_SUFFIXES) or path.startswith("docs/") or "/docs/" in path


def _is_relevant(path: str) -> bool:
    """Return whether a path can affect the Docker test jobs at all.

    Files that are neither relevant nor ignored (e.g. unrelated workflow files, license
    metadata) contribute no jobs, mirroring the previous all-or-nothing gate.
    """
    if path.startswith("source/"):
        return True
    if path.startswith(_GLOBAL_DIR_PREFIXES) or path.startswith(_GLOBAL_CI_PREFIXES):
        return True
    return path in _GLOBAL_CI_FILES or bool(_ROOT_CONFIG.match(path))


def _is_global(path: str) -> bool:
    """Return whether a relevant path forces the full suite to run."""
    if path.startswith(_GLOBAL_PACKAGE_PREFIXES):
        return True
    if path.startswith(_GLOBAL_DIR_PREFIXES) or path.startswith(_GLOBAL_CI_PREFIXES):
        return True
    return path in _GLOBAL_CI_FILES or bool(_ROOT_CONFIG.match(path))


def _all_jobs(value: bool) -> dict[str, bool]:
    """Return a flag dict with every base job set to ``value`` plus derived flags."""
    return _with_derived({job: value for job in _BASE_JOBS})


def _with_derived(flags: dict[str, bool]) -> dict[str, bool]:
    """Add the aggregate ``any`` (build gate) and ``curobo_image`` (build-curobo gate) flags."""
    flags = dict(flags)
    flags["any"] = any(flags[job] for job in _BASE_JOBS)
    flags["curobo_image"] = flags["curobo"] or flags["skillgen"]
    return flags


def select_ci_jobs(paths: list[str]) -> dict[str, bool]:
    """Map changed paths to per-job run flags.

    Args:
        paths: Repo-relative changed file paths (e.g. from the pull request files API).

    Returns:
        A mapping of job group -> whether that job should run, including the aggregate
        ``any`` and ``curobo_image`` gates. An empty ``paths`` list returns all-true as a
        fail-safe; a change touching only ignored docs returns all-false.
    """
    if not paths:
        return _all_jobs(True)

    relevant = [path for path in paths if _is_relevant(path) and not _is_ignored(path)]
    if not relevant:
        return _all_jobs(False)

    flags = {job: False for job in _BASE_JOBS}
    for path in relevant:
        if _is_global(path):
            return _all_jobs(True)
        jobs = next((groups for prefix, groups in _TARGETED.items() if path.startswith(prefix)), None)
        if jobs is None:
            # Relevant but not covered by a targeted rule (e.g. a new package): fail safe.
            return _all_jobs(True)
        for job in jobs:
            flags[job] = True
    return _with_derived(flags)


if __name__ == "__main__":
    selected = select_ci_jobs([line.strip() for line in sys.stdin if line.strip()])
    for key in list(_BASE_JOBS) + ["any", "curobo_image"]:
        print(f"run_{key}={'true' if selected[key] else 'false'}")
