# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print the exact ``ovrtx`` wheel version the kitless rendering containers install.

The kitless jobs install the ``ovrtx`` specifier from ``[tool.isaaclab.versions]``
in ``pyproject.toml`` (the same source the resolve-ov-pins action reads), which is
a range rather than a pin. The driver PSO blobs under ``nv_shadercache`` are
produced by whichever wheel that range resolves to, so the cache key has to name
that wheel: keyed by the range instead, a newly published wheel inside the range
would inherit - and then overwrite - the previous wheel's entry.

Resolution reads package metadata from the same public index the container
installs from, so it needs no credentials and does not download the wheel.

Usage:
    python3 resolve_ovrtx_version.py <python-platform>
"""

from __future__ import annotations

import re
import subprocess
import sys

import tomllib

PYPROJECT = "pyproject.toml"
INDEX_URL = "https://pypi.nvidia.com"


def read_pyproject() -> dict:
    """Return the parsed contents of :data:`PYPROJECT`."""
    with open(PYPROJECT, "rb") as handle:
        return tomllib.load(handle)


def ovrtx_specifier(pyproject: dict) -> str:
    """Return the ``ovrtx`` requirement specifier pinned in ``pyproject``."""
    value = pyproject.get("tool", {}).get("isaaclab", {}).get("versions", {}).get("ovrtx")
    if not value:
        raise SystemExit("[tool.isaaclab.versions] is missing a value for required key 'ovrtx'")

    # Mirror pyproject's convention, as resolve-ov-pins does: exact versions map
    # to ``ovrtx==x.y.z``, range specs (">=...", "<...") append directly.
    return f"ovrtx=={value}" if value[0].isdigit() else f"ovrtx{value}"


def python_version(pyproject: dict) -> str:
    """Return the ``major.minor`` interpreter version the containers run.

    Taken from ``requires-python``, which pins a single minor version, so its
    lower bound is the interpreter. Resolving for any other version could select
    a wheel the container would never install.
    """
    requires = pyproject.get("project", {}).get("requires-python", "")
    lower_bound = re.search(r">=\s*(\d+\.\d+)", requires)
    if lower_bound is None:
        raise SystemExit(f"[project] requires-python has no '>=' lower bound: {requires!r}")
    return lower_bound.group(1)


def resolve(specifier: str, python_platform: str, target_python: str) -> str:
    """Return the concrete version ``specifier`` resolves to for the given target."""
    result = subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            "-",
            "--no-deps",
            "--quiet",
            "--extra-index-url",
            INDEX_URL,
            "--python-platform",
            python_platform,
            "--python-version",
            target_python,
            "--prerelease=allow",
        ],
        input=specifier,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"resolving {specifier!r} failed: {result.stderr.strip()}")

    pin = re.search(r"^ovrtx==(\S+)$", result.stdout, flags=re.MULTILINE)
    if pin is None:
        raise RuntimeError(f"resolver produced no ovrtx pin for {specifier!r}: {result.stdout.strip()}")
    return pin.group(1)


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <python-platform>", file=sys.stderr)
        return 1

    pyproject = read_pyproject()
    specifier = ovrtx_specifier(pyproject)
    try:
        print(resolve(specifier, sys.argv[1], python_version(pyproject)))
    except (OSError, RuntimeError) as exc:
        # The caller degrades to its own collection rather than failing the job;
        # see the fallback in action.yml.
        print(f"::warning::Could not resolve the exact ovrtx version: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
