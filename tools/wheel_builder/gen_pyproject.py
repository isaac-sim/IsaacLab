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
deps = _dedup([d for d in project["dependencies"] if not _is_workspace_member(d)])

# Optional dependencies: per extra, strip workspace members and dedup.
# Wheel-only extras (e.g. isaacsim) live under [tool.isaaclab.wheel-extras]; they
# are excluded from the uv workspace resolution but still shipped in the wheel.
wheel_extras = root.get("tool", {}).get("isaaclab", {}).get("wheel-extras", {})
opt_deps = {}
for name, dep_list in {**project.get("optional-dependencies", {}), **wheel_extras}.items():
    opt_deps[name] = _dedup([d for d in dep_list if not _is_workspace_member(d)])

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
