# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate pyproject.toml for the isaaclab wheel from python_packages.toml."""

import os
import sys

import tomllib

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <packages_toml> <output_path> <version>", file=sys.stderr)
    sys.exit(1)

packages_toml_path = sys.argv[1]
output_path = sys.argv[2]
version = sys.argv[3]

# Add `tools/` to sys.path so we can validate that the wheel TOML stays in
# sync with the canonical pins in tools/python_deps.py.
_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import python_deps  # noqa: E402

with open(packages_toml_path, "rb") as f:
    data = tomllib.load(f)
pkg = data["isaaclab"]


def _pkg_name(req: str) -> str:
    """Extract the lowercase package name from a PEP 508 requirement string."""
    return (
        req.split("[")[0]
        .split(";")[0]
        .split(">")[0]
        .split("<")[0]
        .split("=")[0]
        .split("!")[0]
        .split("~")[0]
        .split("@")[0]
        .strip()
        .lower()
    )


# Validate that every shared pin in tools/python_deps.py is the exact spec used in
# this TOML. Catches drift introduced when a version bumps in only one place.
_SHARED = {
    "NEWTON": "newton",
    "WARP_LANG": "warp-lang",
    "TORCH": "torch",
    "NUMPY": "numpy",
    "PRETTYTABLE": "prettytable",
    "PYOPENGL_ACCELERATE": "pyopengl-accelerate",
}
_seen: dict[str, set[str]] = {name: set() for name in _SHARED.values()}


def _scan(value):
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                name = _pkg_name(item)
                if name in _seen:
                    _seen[name].add(item)
            else:
                _scan(item)
    elif isinstance(value, dict):
        for v in value.values():
            _scan(v)


_scan(data)
for const_name, pkg_name in _SHARED.items():
    expected = getattr(python_deps, const_name)
    found = _seen[pkg_name]
    if not found or found != {expected}:
        msg = (
            f"Wheel TOML out of sync with tools/python_deps.py:{const_name}.\n"
            f"  Expected (from python_deps.py): {expected!r}\n"
            f"  Found in {packages_toml_path}: {sorted(found) or 'none'}\n"
            "  Update one to match the other; tools/python_deps.py is the source of truth."
        )
        raise SystemExit(msg)

# Collect dependencies (deduplicated, preserving order)
raw_deps = pkg["pyproject"]["dependencies"]["all"]
seen = set()
deps = []
for d in raw_deps:
    # Normalize for dedup: lowercase package name (before any version specifier)
    key = d.split(">")[0].split("<")[0].split("=")[0].split("!")[0].split("[")[0].strip().lower()
    if key not in seen:
        seen.add(key)
        deps.append(d)

# Collect optional dependencies
opt_deps = {}
for entry in pkg["pyproject"]["optional-dependencies"]["all"]:
    # Each entry is a dict like {"name": ["dep1", "dep2"]}
    for name, dep_list in entry.items():
        opt_deps[name] = dep_list

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
