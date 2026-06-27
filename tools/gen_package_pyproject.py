# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate each sub-package's dependency metadata from the root pyproject.

The root ``pyproject.toml`` ``[tool.isaaclab.packages]`` table is the single
source of truth for what every distributable ``isaaclab*`` sub-package requires.
Each entry (keyed by distribution name) lists that package's relaxed third-party
``dependencies`` and an ``optional-dependencies`` sub-table for its extras.

This script copies those declarations into the managed region of each
``source/<pkg>/pyproject.toml`` so the sub-packages keep complete, accurate,
self-contained metadata while remaining edited in exactly one place. Strict pins
(for the known-good development/CI environment and Isaac Sim interop) stay in the
root ``[tool.uv]`` configuration and ``uv.lock`` -- not in the package metadata.

Run without arguments to rewrite the managed regions, or with ``--check`` to
verify (in CI/pre-commit) that the files are up-to-date and consistent with the
root, without modifying them.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import tomllib

BEGIN_MARKER = (
    "# >>> BEGIN auto-generated dependencies -- edit [tool.isaaclab.packages] in the root"
    " pyproject.toml and run tools/gen_package_pyproject.py >>>"
)
END_MARKER = "# <<< END auto-generated dependencies <<<"

# Match a previously generated managed region (for idempotent re-runs).
_MANAGED_RE = re.compile(re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER) + r"\n", re.DOTALL)
# Match the initial ``dependencies = []`` placeholder plus any leading comment lines.
_PLACEHOLDER_RE = re.compile(r"(?:^[ \t]*#.*\n)*^dependencies = \[\][ \t]*\n", re.MULTILINE)


def _repo_root() -> Path:
    """Return the Isaac Lab repository root (the directory holding the root pyproject)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "source").is_dir():
            return parent
    raise RuntimeError("Could not find the Isaac Lab repository root.")


def _requirement_name(requirement: str) -> str:
    """Return the normalized distribution name of a PEP 508 requirement string."""
    name = re.split(r"[\s<>=!~;@\[]", requirement, maxsplit=1)[0].strip()
    return re.sub(r"[-_.]+", "-", name).lower()


def _format_array(name: str, requirements: list[str]) -> list[str]:
    """Format a TOML array assignment, one requirement per line (or ``[]`` if empty)."""
    if not requirements:
        return [f"{name} = []"]
    lines = [f"{name} = ["]
    lines += [f'    "{req}",' for req in requirements]
    lines.append("]")
    return lines


def _render_region(dependencies: list[str], optional: dict[str, list[str]]) -> str:
    """Render the managed dependency region (between the begin/end markers)."""
    lines = [BEGIN_MARKER]
    lines += _format_array("dependencies", dependencies)
    if optional:
        lines.append("")
        lines.append("[project.optional-dependencies]")
        for extra, requirements in optional.items():
            lines += _format_array(extra, requirements)
    lines.append(END_MARKER)
    return "\n".join(lines) + "\n"


def _apply_region(text: str, region: str, pkg_path: Path) -> str:
    """Splice the managed region into a sub-package pyproject's text."""
    if _MANAGED_RE.search(text):
        return _MANAGED_RE.sub(lambda _: region, text, count=1)
    if _PLACEHOLDER_RE.search(text):
        return _PLACEHOLDER_RE.sub(lambda _: region, text, count=1)
    raise RuntimeError(
        f"{pkg_path}: could not locate a managed region or a 'dependencies = []' placeholder to replace."
    )


def _packages(root: dict) -> dict[str, dict]:
    """Return the ``[tool.isaaclab.packages]`` table from the parsed root pyproject."""
    try:
        return root["tool"]["isaaclab"]["packages"]
    except KeyError as exc:
        raise RuntimeError("Root pyproject.toml is missing the [tool.isaaclab.packages] table.") from exc


def _root_specs(root: dict) -> dict[str, str]:
    """Map every third-party requirement name declared by the root to its spec string."""
    project = root["project"]
    specs: dict[str, str] = {}
    pools = [project.get("dependencies", [])] + list(project.get("optional-dependencies", {}).values())
    for pool in pools:
        for requirement in pool:
            name = _requirement_name(requirement)
            if name.startswith("isaaclab"):
                continue
            specs.setdefault(name, requirement)
    return specs


def generate(check: bool) -> int:
    """Generate (or, with ``check=True``, verify) all sub-package dependency regions."""
    repo_root = _repo_root()
    with (repo_root / "pyproject.toml").open("rb") as f:
        root = tomllib.load(f)

    packages = _packages(root)
    root_specs = _root_specs(root)

    stale: list[str] = []
    problems: list[str] = []

    for dist_name, entry in packages.items():
        dependencies = list(entry.get("dependencies", []))
        optional = {extra: list(reqs) for extra, reqs in entry.get("optional-dependencies", {}).items()}

        # Consistency checks against the curated root surface.
        for requirement in [*dependencies, *(r for reqs in optional.values() for r in reqs)]:
            name = _requirement_name(requirement)
            if name == "isaaclab-dev":
                problems.append(f"{dist_name}: must not depend on isaaclab-dev (development-only meta package).")
                continue
            if name.startswith("isaaclab"):
                continue  # sibling workspace member self-reference
            if name not in root_specs:
                problems.append(f"{dist_name}: '{requirement}' is not declared anywhere in the root pyproject.toml.")
            elif requirement != root_specs[name]:
                problems.append(
                    f"{dist_name}: '{requirement}' disagrees with the root spec '{root_specs[name]}' for '{name}'."
                )

        pkg_path = repo_root / "source" / dist_name.replace("-", "_") / "pyproject.toml"
        if not pkg_path.is_file():
            problems.append(f"{dist_name}: expected sub-package pyproject at {pkg_path} (not found).")
            continue

        original = pkg_path.read_text()
        updated = _apply_region(original, _render_region(dependencies, optional), pkg_path)
        if updated != original:
            if check:
                stale.append(dist_name)
            else:
                pkg_path.write_text(updated)
                print(f"Updated {pkg_path.relative_to(repo_root)}")

    if problems:
        print("Dependency consistency errors:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
    if check and stale:
        print(
            "These sub-package pyproject.toml files are out of date; run "
            "'python tools/gen_package_pyproject.py' and commit the result:",
            file=sys.stderr,
        )
        for dist_name in stale:
            print(f"  - {dist_name}", file=sys.stderr)

    return 1 if (problems or (check and stale)) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the generated files are up-to-date and consistent instead of writing them",
    )
    args = parser.parse_args()
    return generate(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
