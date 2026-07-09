# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coverage guard for the path-based CI job selector.

``select_ci_jobs.py`` skips test jobs for changes to *isaaclab_target* packages, and a skipped
job reports green to branch protection. That is only safe if :data:`select_ci_jobs._TARGETED`
never *under*-approximates the jobs a package can affect. Everything else in the selector is
conservative by construction (unmapped/hub/tooling paths fall back to the full suite); the
targeted map is the one place a real break could slip through, so it is the one place worth a
machine check.

This module derives the required job set for each targeted package from the *live* inter-package
import graph and asserts the checked-in map is a superset. When someone adds a cross-package
import that widens a package's blast radius, this test fails and forces the map to be updated (or
the new coupling to be reconsidered) instead of silently shipping a false-green PR.

Model (why one hop is the right granularity)
--------------------------------------------
Each job runs the tests of exactly one package (see :data:`_PACKAGE_JOBS`), and a package's own
tests eagerly import that package. So a targeted package ``P`` can affect job ``J`` when:

1. ``J`` is ``P``'s own job, or
2. ``J``'s package directly imports ``P`` at module level (``J``'s tests load ``P``), or
3. ``J``'s test files directly import ``P``.

We deliberately do **not** take the transitive closure. IsaacLab uses ``lazy_loader`` /
function-local / ``importlib`` imports so that importing a hub package does **not** eagerly pull
its optional integrations (e.g. importing ``isaaclab`` does not eagerly import
``isaaclab_visualizers``). A transitive walk over module-level edges would therefore treat every
lazy hub edge as real and demand the full suite for almost any change -- a false alarm. One hop
from each job's own package matches how the map was built and how the code actually loads.

Soundness caveat: because the graph is static, a genuinely *eager* 2+-hop chain that we cannot see
(a dynamically constructed import) could still be missed. That residual risk is covered by the
post-merge full-suite run on protected branches, not by this test. This guard eliminates the
common, silent drift (a newly added module-level import), which is what actually rots the map.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_ci_jobs import _BASE_JOBS, _TARGETED, _TASKS_FAMILY

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE = _REPO_ROOT / "source"

# --- Source of truth: which job runs which package's tests -------------------------------
# Mirrors the ``filter-pattern`` / test-path wiring in .github/workflows/build.yaml. Packages
# with no dedicated test job (shared/experimental packages that force the full suite when
# changed) map to an empty set; the model-completeness test below fails loudly if one of them
# ever imports a targeted package, so this table cannot silently go stale.
_PACKAGE_JOBS: dict[str, frozenset[str]] = {
    "isaaclab": frozenset({"core"}),
    "isaaclab_tasks": _TASKS_FAMILY,  # the whole tasks family runs isaaclab_tasks tests
    "isaaclab_rl": frozenset({"rl"}),
    "isaaclab_mimic": frozenset({"mimic"}),
    "isaaclab_contrib": frozenset({"contrib"}),
    "isaaclab_teleop": frozenset({"teleop"}),
    "isaaclab_visualizers": frozenset({"visualizers"}),
    "isaaclab_assets": frozenset({"assets"}),
    "isaaclab_newton": frozenset({"newton"}),
    "isaaclab_physx": frozenset({"physx"}),
    "isaaclab_ov": frozenset({"ov"}),
    "isaaclab_ovphysx": frozenset({"ov"}),  # the isaaclab_ov job's filter also matches isaaclab_ovphysx
    "isaaclab_experimental": frozenset(),
    "isaaclab_ppisp": frozenset(),
    "isaaclab_tasks_experimental": frozenset(),
}

# Test directories each job scans, keyed by job -> the package tree its test files live in.
# tasks-family jobs all run test files under source/isaaclab_tasks/test; the ov job runs both
# isaaclab_ov and isaaclab_ovphysx tests.
_JOB_TEST_PACKAGES: dict[str, tuple[str, ...]] = {
    **{job: ("isaaclab_tasks",) for job in _TASKS_FAMILY},
    "core": ("isaaclab",),
    "rl": ("isaaclab_rl",),
    "mimic": ("isaaclab_mimic",),
    "contrib": ("isaaclab_contrib",),
    "teleop": ("isaaclab_teleop",),
    "visualizers": ("isaaclab_visualizers",),
    "assets": ("isaaclab_assets",),
    "newton": ("isaaclab_newton",),
    "physx": ("isaaclab_physx",),
    "ov": ("isaaclab_ov", "isaaclab_ovphysx"),
}

_ALL_PACKAGES = frozenset(_PACKAGE_JOBS)


def _package_of_prefix(prefix: str) -> str:
    """Turn a ``_TARGETED`` key like ``source/isaaclab_rl/`` into the package name."""
    return prefix.rstrip("/").split("/")[-1]


def _module_level_imports(py_file: Path) -> set[str]:
    """Return the isaaclab packages imported at module top level (``col_offset == 0``).

    Restricting to column 0 excludes function-local, ``try/except``-guarded, and ``if``-guarded
    imports -- exactly the deferred/optional edges that ``lazy_loader`` and feature flags keep
    from executing eagerly. See the module docstring for why that is the right cut.
    """
    out: set[str] = set()
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError):
        return out
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or getattr(node, "col_offset", 1) != 0:
            continue
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if node.module else [])
        for name in names:
            top = (name or "").split(".")[0]
            if top in _ALL_PACKAGES:
                out.add(top)
    return out


def _py_files(root: Path, *, include_tests: bool) -> list[Path]:
    files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        posix = Path(dirpath).as_posix()
        if ".pytest_cache" in posix:
            continue
        if not include_tests and "/test/" in f"{posix}/":
            continue
        files.extend(Path(dirpath) / f for f in filenames if f.endswith(".py"))
    return files


def _package_source_root(package: str) -> Path:
    inner = _SOURCE / package / package
    return inner if inner.exists() else _SOURCE / package


def _direct_source_importers() -> dict[str, set[str]]:
    """Map each package to the set of packages whose *source* module-level imports it."""
    importers: dict[str, set[str]] = {pkg: set() for pkg in _ALL_PACKAGES}
    for pkg in _ALL_PACKAGES:
        for py_file in _py_files(_package_source_root(pkg), include_tests=False):
            for dep in _module_level_imports(py_file) - {pkg}:
                importers[dep].add(pkg)
    return importers


def _jobs_whose_tests_import(target: str) -> set[str]:
    """Return jobs whose *test files* directly module-level import ``target``."""
    jobs: set[str] = set()
    for job, packages in _JOB_TEST_PACKAGES.items():
        for pkg in packages:
            test_dir = _SOURCE / pkg / "test"
            if not test_dir.exists():
                continue
            if any(target in _module_level_imports(f) for f in _py_files(test_dir, include_tests=True)):
                jobs.add(job)
                break
    return jobs


def _required_jobs(package: str, importers: dict[str, set[str]]) -> set[str]:
    """Compute the jobs a change to ``package`` can affect, via the one-hop model."""
    required: set[str] = set(_PACKAGE_JOBS[package])  # (1) its own job(s)
    for importer in importers[package]:  # (2) packages that import it -> their jobs
        required |= _PACKAGE_JOBS[importer]
    required |= _jobs_whose_tests_import(package)  # (3) test files that import it
    return required


# --- Structural invariants (no graph needed) --------------------------------------------


def test_every_source_package_is_classified():
    """Every source/isaaclab* package is either a targeted rule or a known job package.

    Guards against a brand-new package being added with no selector rule and no job mapping,
    which would leave its blast radius unmodelled.
    """
    on_disk = {p.name for p in _SOURCE.iterdir() if p.is_dir() and p.name.startswith("isaaclab")}
    modelled = set(_ALL_PACKAGES)
    assert on_disk == modelled, (
        f"source packages not modelled here: {sorted(on_disk - modelled)}; "
        f"stale entries: {sorted(modelled - on_disk)}. Update _PACKAGE_JOBS and select_ci_jobs._TARGETED."
    )


def test_targeted_job_names_are_valid():
    for prefix, jobs in _TARGETED.items():
        unknown = set(jobs) - set(_BASE_JOBS)
        assert not unknown, f"{prefix} maps to unknown jobs {sorted(unknown)}"


def test_targeted_package_includes_its_own_job():
    for prefix, jobs in _TARGETED.items():
        pkg = _package_of_prefix(prefix)
        own = _PACKAGE_JOBS[pkg]
        assert own <= set(jobs), f"{prefix} must include its own job(s) {sorted(own)}"


def test_jobless_packages_do_not_import_targeted_packages():
    """Model-completeness: a package with no test job must not import a targeted package.

    If it did, a change to that targeted package could be exercised by tests we have not mapped to
    any job, so the one-hop model would be incomplete. This currently holds; the assertion makes
    that assumption enforced rather than implicit.
    """
    importers = _direct_source_importers()
    jobless = {pkg for pkg, jobs in _PACKAGE_JOBS.items() if not jobs}
    targeted = {_package_of_prefix(p) for p in _TARGETED}
    offenders = {tgt: sorted(importers[tgt] & jobless) for tgt in targeted if importers[tgt] & jobless}
    assert not offenders, f"job-less packages import targeted packages, model is incomplete: {offenders}"


# --- The coverage guarantee --------------------------------------------------------------


def test_targeted_map_covers_live_import_graph():
    """``_TARGETED[P]`` must be a superset of the jobs a change to ``P`` can affect.

    This is the core guarantee: a targeted package never skips a job whose tests load it.
    """
    importers = _direct_source_importers()
    gaps: dict[str, list[str]] = {}
    for prefix, mapped in _TARGETED.items():
        pkg = _package_of_prefix(prefix)
        missing = _required_jobs(pkg, importers) - set(mapped)
        if missing:
            gaps[prefix] = sorted(missing)
    assert not gaps, (
        "select_ci_jobs._TARGETED under-approximates the import graph; add these jobs (a new "
        f"cross-package import likely widened the blast radius): {gaps}"
    )


@pytest.mark.parametrize("prefix", list(_TARGETED))
def test_targeted_test_directory_exists(prefix):
    """The package behind each targeted rule must actually have a test dir the guard can scan."""
    pkg = _package_of_prefix(prefix)
    assert (_SOURCE / pkg / "test").exists(), f"{pkg} has no test/ directory; _JOB_TEST_PACKAGES is stale"
