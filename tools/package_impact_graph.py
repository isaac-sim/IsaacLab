#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a file-level interpackage import graph and compute CI job impact.

Usage
-----
# Print the full impact map (JSON):
python tools/package_impact_graph.py

# Print CI jobs triggered by specific changed files:
python tools/package_impact_graph.py --changed source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py

# Print CI jobs triggered by all files changed vs a git ref:
python tools/package_impact_graph.py --since HEAD~1

The tool walks every .py file under source/, parses its import statements
(skipping TYPE_CHECKING guards), and records which isaaclab* packages each file
imports.  From that file-level graph it derives a package-level reverse-dep map:
for each package P, which other packages Q have at least one file that imports
from P.  When P changes, the tests for P plus every Q that (transitively) imports
P must run.

The output is a JSON object:

  {
    "package_to_ci_jobs": {
      "isaaclab_newton": ["test-isaaclab-newton", "test-isaaclab-physx", ...],
      ...
    },
    "file_imports": {
      "isaaclab_newton/isaaclab_newton/foo.py": ["isaaclab", "isaaclab_physx"],
      ...
    },
    "reverse_deps": {
      "isaaclab_newton": ["isaaclab_physx", "isaaclab_contrib", ...],
      ...
    }
  }
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PREFIX = "isaaclab"
_DEP_NAME_RE = re.compile(r"^([A-Za-z0-9_-]+)")

# Maps each source package name to its CI job label in build.yaml.
# Packages not listed here have no dedicated CI job.
_PACKAGE_TO_CI_JOB: dict[str, str] = {
    "isaaclab": "test-isaaclab-core",
    "isaaclab_assets": "test-isaaclab-assets",
    "isaaclab_contrib": "test-isaaclab-contrib",
    "isaaclab_experimental": "test-isaaclab-core",  # no dedicated job; covered by core
    "isaaclab_mimic": "test-isaaclab-mimic",
    "isaaclab_newton": "test-isaaclab-newton",
    "isaaclab_ov": "test-isaaclab-ov",
    "isaaclab_ovphysx": "test-isaaclab-core",  # no dedicated job
    "isaaclab_physx": "test-isaaclab-physx",
    "isaaclab_ppisp": "test-isaaclab-core",  # no dedicated job
    "isaaclab_rl": "test-isaaclab-rl",
    "isaaclab_tasks": "test-isaaclab-tasks",
    "isaaclab_tasks_experimental": "test-isaaclab-tasks",  # bundled with tasks
    "isaaclab_teleop": "test-isaaclab-teleop",
    "isaaclab_visualizers": "test-isaaclab-visualizers",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "source").is_dir():
            return parent
    raise RuntimeError("Cannot find repo root.")


def _all_packages(source: Path) -> list[str]:
    """Return the name of every subpackage directory that has a pyproject.toml."""
    return sorted(
        p.name for p in source.iterdir() if p.is_dir() and (p / "pyproject.toml").is_file()
    )


def _owner_package(py_file: Path, source: Path) -> str | None:
    """Return the isaaclab* package that owns *py_file*, or None if not under source/."""
    try:
        rel = py_file.relative_to(source)
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def _runtime_imports(py_file: Path) -> set[str]:
    """Parse *py_file* and return the isaaclab* packages it imports at runtime.

    Imports inside ``if TYPE_CHECKING:`` guards are excluded because they
    create no runtime dependency.
    """
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()

    tc_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            t = node.test
            if (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
                isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"
            ):
                for child in ast.walk(node):
                    tc_ids.add(id(child))

    names: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in tc_ids:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top.startswith(_PREFIX):
                    names.add(top)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top.startswith(_PREFIX):
                names.add(top)
    return names


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(source: Path) -> dict:
    """Walk source/ and return the full dependency graph.

    Returns a dict with keys:
      - ``file_imports``: {rel_path: [pkg, ...]} — runtime imports per file
      - ``reverse_deps``: {pkg: [pkg, ...]} — packages that import each pkg
      - ``package_to_ci_jobs``: {pkg: [job, ...]}
    """
    packages = set(_all_packages(source))

    # file_imports[rel_path] = sorted list of isaaclab* packages imported
    file_imports: dict[str, list[str]] = {}

    # For each package P, which packages Q have files that import P?
    reverse_deps: dict[str, set[str]] = defaultdict(set)

    for pkg in packages:
        pkg_dir = source / pkg
        for py_file in sorted(pkg_dir.rglob("*.py")):
            imports = _runtime_imports(py_file)
            # Exclude self-imports
            pkg_imports = sorted(imports - {pkg})
            if pkg_imports:
                rel = str(py_file.relative_to(source))
                file_imports[rel] = pkg_imports
                for imported_pkg in pkg_imports:
                    reverse_deps[imported_pkg].add(pkg)

    # BFS that also records the shortest import chain from start to each affected pkg.
    # Returns {affected_pkg: [start, intermediate..., affected_pkg]}
    def _transitive_affected_with_paths(start: str) -> dict[str, list[str]]:
        paths: dict[str, list[str]] = {start: [start]}
        queue = [start]
        while queue:
            pkg = queue.pop(0)
            for importer in reverse_deps.get(pkg, set()):
                if importer not in paths:
                    paths[importer] = paths[pkg] + [importer]
                    queue.append(importer)
        return paths

    package_to_ci_jobs: dict[str, list[str]] = {}
    # Also store the shortest chain that explains each (pkg, job) pair
    package_job_chains: dict[str, dict[str, list[str]]] = {}

    for pkg in packages:
        affected_paths = _transitive_affected_with_paths(pkg)
        jobs: dict[str, list[str]] = {}  # job -> shortest chain
        for affected_pkg, chain in affected_paths.items():
            job = _PACKAGE_TO_CI_JOB.get(affected_pkg)
            if job and (job not in jobs or len(chain) < len(jobs[job])):
                jobs[job] = chain
        package_to_ci_jobs[pkg] = sorted(jobs)
        package_job_chains[pkg] = {j: c for j, c in jobs.items()}

    return {
        "file_imports": dict(sorted(file_imports.items())),
        "reverse_deps": {k: sorted(v) for k, v in sorted(reverse_deps.items())},
        "package_to_ci_jobs": dict(sorted(package_to_ci_jobs.items())),
        "package_job_chains": package_job_chains,
    }


# ---------------------------------------------------------------------------
# CI job lookup for changed files
# ---------------------------------------------------------------------------


def build_manifest(changed: list[str], source: Path, graph: dict) -> dict:
    """Return a manifest explaining which jobs run and why.

    The manifest has shape::

        {
          "jobs": {
            "test-isaaclab-newton": [
              {
                "file": "source/isaaclab_newton/isaaclab_newton/foo.py",
                "owner": "isaaclab_newton",
                "chain": ["isaaclab_newton"]
              }
            ],
            "test-isaaclab-tasks": [
              {
                "file": "source/isaaclab_newton/isaaclab_newton/foo.py",
                "owner": "isaaclab_newton",
                "chain": ["isaaclab_newton", "isaaclab_tasks"]
              }
            ]
          },
          "changed_files": [...],
          "non_python_files": [...]
        }

    Each entry under a job shows the shortest reverse-import chain from the
    changed file's owning package to the package whose tests that job covers.
    """
    # job -> list of {file, owner, chain} entries
    job_reasons: dict[str, list[dict]] = defaultdict(list)
    non_python: list[str] = []

    for path_str in changed:
        py_file = Path(path_str)
        # Try to resolve; fall back to repo-relative if it doesn't exist yet
        try:
            resolved = py_file.resolve()
        except Exception:
            resolved = py_file

        if py_file.suffix != ".py":
            non_python.append(path_str)
            for job in sorted(set(_PACKAGE_TO_CI_JOB.values())):
                job_reasons[job].append({"file": path_str, "owner": None, "chain": ["(non-python)"]})
            continue

        owner = _owner_package(resolved, source)
        if owner is None:
            for job in sorted(set(_PACKAGE_TO_CI_JOB.values())):
                job_reasons[job].append({"file": path_str, "owner": None, "chain": ["(unknown)"]})
            continue

        chains = graph.get("package_job_chains", {}).get(owner, {})
        for job, chain in chains.items():
            job_reasons[job].append({"file": path_str, "owner": owner, "chain": chain})

    return {
        "jobs": dict(sorted(job_reasons.items())),
        "changed_files": list(changed),
        "non_python_files": non_python,
    }


def _changed_files_since(ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", ref],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--changed",
        nargs="+",
        metavar="FILE",
        help="Print CI jobs triggered by these changed files.",
    )
    group.add_argument(
        "--since",
        metavar="GIT_REF",
        help="Print CI jobs triggered by files changed since GIT_REF (e.g. HEAD~1, main).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full graph as JSON (default when neither --changed nor --since given).",
    )
    args = parser.parse_args()

    root = _repo_root()
    source = root / "source"

    print("Building file-level import graph...", file=sys.stderr)
    graph = build_graph(source)

    if args.changed or args.since:
        changed = args.changed or _changed_files_since(args.since)
        manifest = build_manifest(changed, source, graph)

        if args.json:
            print(json.dumps(manifest, indent=2))
        else:
            jobs = manifest["jobs"]
            if not jobs:
                print("(no CI jobs affected)")
                return

            # Human-readable manifest
            print(f"Changed files ({len(manifest['changed_files'])}):")
            for f in manifest["changed_files"]:
                print(f"  {f}")
            print()
            print(f"CI jobs to run ({len(jobs)}):")
            for job, reasons in sorted(jobs.items()):
                print(f"\n  {job}")
                # Deduplicate reasons by chain string
                seen_chains: set[str] = set()
                for r in reasons:
                    chain_str = " → ".join(r["chain"])
                    key = f"{r['file']}|{chain_str}"
                    if key in seen_chains:
                        continue
                    seen_chains.add(key)
                    print(f"    ← {r['file']}")
                    print(f"       chain: {chain_str}")
    else:
        print(json.dumps(graph, indent=2))


if __name__ == "__main__":
    main()
