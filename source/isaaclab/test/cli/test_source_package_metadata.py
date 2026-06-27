# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for source package dependency metadata.

The sub-package dependency declarations are generated from the root
``[tool.isaaclab.packages]`` table by ``tools/gen_package_pyproject.py``. These
tests guard the properties that generation is meant to guarantee: the files stay
in sync with the root, every package ships complete self-contained metadata, and
no package leaks a dependency on the development-only ``isaaclab-dev`` meta
package.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import tomllib


def _repo_root() -> Path:
    """Find the Isaac Lab repository root from this test file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "source").is_dir():
            return parent
    raise RuntimeError("Could not find Isaac Lab repository root.")


def _subpackage_pyprojects() -> list[Path]:
    """Return every ``source/isaaclab*/pyproject.toml`` path."""
    return sorted((_repo_root() / "source").glob("isaaclab*/pyproject.toml"))


def _all_requirements(pyproject: dict) -> list[str]:
    """Return a package's base + optional dependencies as a flat list."""
    project = pyproject.get("project", {})
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    return requirements


def test_isaaclab_usd_core_pin_stays_on_isaacsim_compatible_usd25_abi():
    """The kit-less USD package must stay on the Isaac Sim compatible USD 25 ABI."""
    with (_repo_root() / "source/isaaclab/pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    usd_core_dependencies = [
        dependency for dependency in pyproject["project"]["dependencies"] if dependency.startswith("usd-core")
    ]

    assert usd_core_dependencies == ["usd-core>=25.11,<26.0 ; platform_machine in 'x86_64 AMD64'"]


def test_generated_subpackage_metadata_is_in_sync_with_root():
    """The sub-package pyprojects must match what the generator would produce from the root."""
    result = subprocess.run(
        [sys.executable, str(_repo_root() / "tools/gen_package_pyproject.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "source/isaaclab*/pyproject.toml are out of sync with the root"
        " [tool.isaaclab.packages] table. Run 'python tools/gen_package_pyproject.py'"
        f" and commit the result.\n{result.stdout}{result.stderr}"
    )


def test_subpackages_do_not_depend_on_isaaclab_dev():
    """No distributable sub-package may depend on the development-only ``isaaclab-dev`` meta package."""
    offenders = []
    for path in _subpackage_pyprojects():
        with path.open("rb") as f:
            requirements = _all_requirements(tomllib.load(f))
        if any(req.replace("_", "-").startswith("isaaclab-dev") for req in requirements):
            offenders.append(path.parent.name)
    assert not offenders, f"these sub-packages depend on isaaclab-dev: {offenders}"


def test_newton_package_declares_newton_dependency():
    """``isaaclab-newton`` must declare ``newton`` in its own metadata (self-contained)."""
    with (_repo_root() / "source/isaaclab_newton/pyproject.toml").open("rb") as f:
        dependencies = tomllib.load(f)["project"]["dependencies"]

    assert any(dep.startswith("newton[sim]") for dep in dependencies)
