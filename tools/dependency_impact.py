# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classify the impact of a change into changed / dependent / transitively-dependent files.

Given the set of files a pull request changed, this script answers "what else could this
change affect?" using the **package dependency graph declared in the ``extension.toml``
files**, not runtime coverage. Each extension's ``config/extension.toml`` lists, under
``[dependencies]``, the other extensions it depends on (by their source directory name).
Reversing those edges tells us which packages depend *on* a changed package, and therefore
which of their files are potentially affected.

The output is a JSON document with three buckets, each grouped by language::

    {
      "changed":    {"python": [...]},   # changed .py files that live in a known package
      "dependents": {"python": [...]},   # .py files in packages that DIRECTLY depend on a
                                         #   changed package (reverse-dependency depth 1)
      "transitive": {"python": [...]}    # .py files in packages that depend on a changed
                                         #   package only INDIRECTLY (reverse depth >= 2)
    }

Every package lands in exactly one bucket, so the three file lists never overlap: a file is
"changed" if it was edited, "dependent" if its package directly depends on a changed
package, and "transitive" otherwise (further out in the reverse-dependency graph).

Usage::

    # Changed files are read from stdin, one repo-relative path per line.
    printf '%s\n' source/isaaclab/isaaclab/foo.py \
        | python tools/dependency_impact.py

The JSON document is written to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib  # type: ignore[no-redef]

# The repository root, inferred from this file's location (``<repo>/tools/dependency_impact.py``).
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories, relative to the repo root, scanned for ``*/config/extension.toml`` packages.
_DEFAULT_SEARCH_DIRS = ("source",)


@dataclass
class Package:
    """A single Isaac Lab extension package discovered from its ``extension.toml``.

    Attributes:
        name: Extension identifier — the directory name under ``source/`` that other
            packages reference in their ``[dependencies]`` table.
        directory: Repo-relative POSIX path to the package root (the parent of ``config/``).
        dependencies: Names of the extensions this package declares a dependency on.
    """

    name: str
    directory: str
    dependencies: set[str] = field(default_factory=set)


def _normalize(path: str) -> str:
    """Normalize a path for matching: forward slashes, no leading ``./``."""
    path = path.replace("\\", "/")
    return path[2:] if path.startswith("./") else path


def discover_packages(repo_root: Path, search_dirs: tuple[str, ...] = _DEFAULT_SEARCH_DIRS) -> dict[str, Package]:
    """Discover extension packages and their declared dependencies.

    Args:
        repo_root: Absolute path to the repository root.
        search_dirs: Repo-relative directories to scan for ``*/config/extension.toml``.

    Returns:
        A mapping from extension name to its :class:`Package`. Dependencies pointing at
        packages that were not discovered are kept (they simply never resolve to files),
        which keeps the graph honest without raising on optional/out-of-tree extensions.
    """
    packages: dict[str, Package] = {}
    for search_dir in search_dirs:
        for toml_path in sorted((repo_root / search_dir).glob("*/config/extension.toml")):
            package_dir = toml_path.parent.parent
            name = package_dir.name
            try:
                data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError) as exc:
                print(f"::warning::Could not read {toml_path}: {exc}", file=sys.stderr)
                continue
            dependencies = set(data.get("dependencies", {}).keys())
            directory = _normalize(str(package_dir.relative_to(repo_root)))
            packages[name] = Package(name=name, directory=directory, dependencies=dependencies)
    return packages


def build_reverse_graph(packages: dict[str, Package]) -> dict[str, set[str]]:
    """Build the reverse-dependency graph: package -> packages that depend on it directly."""
    reverse: dict[str, set[str]] = defaultdict(set)
    for package in packages.values():
        for dependency in package.dependencies:
            reverse[dependency].add(package.name)
    return reverse


def owning_package(path: str, packages: dict[str, Package]) -> str | None:
    """Return the name of the package a repo-relative file belongs to, if any.

    Matching is by directory prefix; when packages nest, the longest (most specific)
    matching directory wins.
    """
    norm = _normalize(path)
    best: str | None = None
    best_len = -1
    for package in packages.values():
        prefix = package.directory + "/"
        if norm.startswith(prefix) and len(package.directory) > best_len:
            best = package.name
            best_len = len(package.directory)
    return best


def _reverse_levels(reverse: dict[str, set[str]], seeds: set[str]) -> tuple[set[str], set[str]]:
    """Split reverse-reachable packages into direct (depth 1) and transitive (depth >= 2).

    Args:
        reverse: The reverse-dependency graph from :func:`build_reverse_graph`.
        seeds: The changed packages to walk out from.

    Returns:
        A tuple ``(direct, transitive)`` of package-name sets, both excluding the seeds and
        excluding each other (a package that is both a direct and an indirect dependent is
        classified as direct — the stronger, closer relationship).
    """
    direct: set[str] = set()
    for seed in seeds:
        direct |= reverse.get(seed, set())
    direct -= seeds

    transitive: set[str] = set()
    queue: deque[str] = deque(direct)
    visited: set[str] = set(seeds) | direct
    while queue:
        current = queue.popleft()
        for dependent in reverse.get(current, set()):
            if dependent not in visited:
                visited.add(dependent)
                transitive.add(dependent)
                queue.append(dependent)
    transitive -= seeds
    transitive -= direct
    return direct, transitive


def _python_files(repo_root: Path, package: Package) -> list[str]:
    """List repo-relative POSIX paths of every ``*.py`` file under a package directory."""
    root = repo_root / PurePosixPath(package.directory)
    if not root.is_dir():
        return []
    files = [_normalize(str(p.relative_to(repo_root))) for p in root.rglob("*.py")]
    return sorted(files)


def build_impact(
    changed_files: list[str],
    repo_root: Path = _REPO_ROOT,
    search_dirs: tuple[str, ...] = _DEFAULT_SEARCH_DIRS,
) -> dict[str, dict[str, list[str]]]:
    """Classify changed files into changed / dependent / transitive Python-file buckets.

    Args:
        changed_files: Repo-relative paths changed by the pull request.
        repo_root: Absolute path to the repository root.
        search_dirs: Repo-relative directories scanned for extension packages.

    Returns:
        The impact document described in the module docstring. The three ``python`` lists
        are disjoint and each sorted.
    """
    packages = discover_packages(repo_root, search_dirs)
    reverse = build_reverse_graph(packages)

    changed_python = sorted({_normalize(f) for f in changed_files if f.strip().endswith(".py")})

    changed_packages = {owning_package(f, packages) for f in changed_python}
    changed_packages.discard(None)
    seeds: set[str] = {name for name in changed_packages if name is not None}

    direct, transitive = _reverse_levels(reverse, seeds)

    def files_for(names: set[str]) -> list[str]:
        collected: list[str] = []
        for name in sorted(names):
            collected.extend(_python_files(repo_root, packages[name]))
        return sorted(collected)

    return {
        "changed": {"python": changed_python},
        "dependents": {"python": files_for(direct)},
        "transitive": {"python": files_for(transitive)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--search-dir",
        action="append",
        dest="search_dirs",
        help="Repo-relative directory to scan for packages (repeatable; default: source).",
    )
    parser.add_argument(
        "--output",
        help="Write the JSON document to this path instead of stdout.",
    )
    args = parser.parse_args(argv)

    search_dirs = tuple(args.search_dirs) if args.search_dirs else _DEFAULT_SEARCH_DIRS
    changed_files = [line.strip() for line in sys.stdin if line.strip()]
    impact = build_impact(changed_files, search_dirs=search_dirs)

    document = json.dumps(impact, indent=2)
    if args.output:
        Path(args.output).write_text(document + "\n", encoding="utf-8")
    else:
        print(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
