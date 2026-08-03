# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate pyproject.toml for the isaaclab wheel from the root pyproject.toml.

The published wheel bundles every ``isaaclab*`` sub-package as a top-level
module, so the workspace self-references in the root ``[project.dependencies]``
and ``[project.optional-dependencies]`` are dropped here; only third-party
requirements end up in the wheel metadata.
"""

import re
import sys

import tomllib

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <root_pyproject> <output_path> <version>", file=sys.stderr)
    sys.exit(1)

root_pyproject_path = sys.argv[1]
output_path = sys.argv[2]
version = sys.argv[3]

with open(root_pyproject_path, "rb") as f:
    root = tomllib.load(f)
project = root["project"]


def _requirement_name(requirement: str) -> str:
    """Extract the normalized distribution name from a requirement string."""
    name = re.split(r"\s|<|>|=|!|~|\[|@|;", requirement, maxsplit=1)[0].strip()
    return re.sub(r"[-_.]+", "-", name).lower()


def _is_workspace_member(requirement: str) -> bool:
    """Return True for ``isaaclab*`` self-references (bundled into the wheel)."""
    return _requirement_name(requirement).startswith("isaaclab")


_project_name = _requirement_name(project["name"])
_optional = project.get("optional-dependencies", {})
_self_ref_pattern = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*\[([^\]]+)\]\s*$")


def _self_ref_extras(requirement: str) -> list[str] | None:
    """Return the referenced extra names for a bare self-reference.

    A self-reference looks like ``isaaclab-dev[sb3,skrl]`` (the root project's own
    name with extras and no version specifier or marker). Returns ``None`` for any
    other requirement, including third-party extras such as ``ray[default]>=2``.
    """
    match = _self_ref_pattern.match(requirement)
    if match is None or _requirement_name(match.group(1)) != _project_name:
        return None
    return [extra.strip() for extra in match.group(2).split(",")]


def _expand_self_refs(requirements: list[str], seen: set[str] | None = None) -> list[str]:
    """Inline self-referential extras into their concrete requirements."""
    seen = set() if seen is None else seen
    expanded = []
    for requirement in requirements:
        extras = _self_ref_extras(requirement)
        if extras is None:
            expanded.append(requirement)
            continue
        for extra in extras:
            if extra in seen:
                continue
            seen.add(extra)
            expanded.extend(_expand_self_refs(_optional.get(extra, []), seen))
    return expanded


def _dedup(requirements: list[str]) -> list[str]:
    """Drop duplicate requirements by distribution name, preserving order."""
    seen = set()
    result = []
    for requirement in requirements:
        key = _requirement_name(requirement)
        if key not in seen:
            seen.add(key)
            result.append(requirement)
    return result


# Required dependencies: third-party only (strip workspace members), deduped.
deps = _dedup([d for d in _expand_self_refs(project["dependencies"]) if not _is_workspace_member(d)])

# Optional dependencies: per extra, strip workspace members and dedup.
opt_deps = {}
for name, dep_list in project.get("optional-dependencies", {}).items():
    opt_deps[name] = _dedup([d for d in _expand_self_refs(dep_list) if not _is_workspace_member(d)])

# Write pyproject.toml
lines = []
lines.append("[build-system]")
lines.append('requires = ["setuptools >= 70.0, < 82.0.0"]')
lines.append('build-backend = "setuptools.build_meta"')
lines.append("")
lines.append("[tool.setuptools]")
lines.append("include-package-data = true")
lines.append('package-dir = {"" = "src"}')
lines.append("")
lines.append("[tool.setuptools.packages.find]")
lines.append('where = ["src"]')
lines.append("")
lines.append("# Include all non-.py files (kit apps, toml configs, usd, yaml, etc.)")
lines.append("[tool.setuptools.package-data]")
lines.append('"*" = ["**/*"]')
lines.append("")
lines.append("[project]")
lines.append('name = "isaaclab"')
lines.append(f'version = "{version}"')
lines.append('requires-python = ">=3.12"')
lines.append('description = "Isaac Lab"')
lines.append('license = {text = "BSD-3-Clause"}')
lines.append("dependencies = [")
for d in deps:
    lines.append(f'    "{d}",')
lines.append("]")
lines.append("")
lines.append("[project.scripts]")
lines.append('isaaclab = "isaaclab:main"')
lines.append("")
lines.append("[project.optional-dependencies]")
for name, dep_list in opt_deps.items():
    formatted = ", ".join(f'"{d}"' for d in dep_list)
    lines.append(f"{name} = [{formatted}]")
lines.append("")

with open(output_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Generated {output_path} with {len(deps)} dependencies and {len(opt_deps)} optional groups")
