# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers shared by tests that build package trees on disk.

One ``cli.py`` is cherry-picked to every branch the nightly targets, and
those branches disagree about where a package records its version:
``pyproject.toml`` under a ``[project]`` table on active branches,
``config/extension.toml`` with a bare top-level ``version`` on release
branches cut before that move.

Tests must not encode either answer. They ask :attr:`packages.Package.toml_path`
— the code under test — and write whatever shape that file implies. That is
what lets the same suite be cherry-picked alongside the code and keep
testing the truth rather than a layout it no longer runs on. (The code side
takes three coordinated edits to move layouts, not one; the tests take
none.)

This lives in ``conftest.py`` rather than being copied into each test module
on purpose: two copies of a layout rule is the same duplication-drift that
broke the nightly twice.
"""

from __future__ import annotations

from pathlib import Path

import packages


def version_file_for(root: Path) -> Path:
    """Path of the version metadata file for the package at ``root``."""
    return packages.Package(root).toml_path


def version_file_rel(name: str) -> str:
    """Repo-relative POSIX path of ``source/<name>``'s version metadata file."""
    return packages.Package(Path("source") / name).toml_path.as_posix()


def write_version_file(root: Path, name: str, version: str) -> Path:
    """Write ``version`` into the package's version metadata file.

    The file's *shape* follows from its name, because that is what the
    branch's parser expects: a ``[project]`` table for ``pyproject.toml``,
    a bare top-level ``version`` for ``config/extension.toml``.
    """
    path = version_file_for(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == "pyproject.toml":
        path.write_text(
            f'[build-system]\nrequires = ["setuptools"]\n\n[project]\nname = "{name}"\nversion = "{version}"\n',
            encoding="utf-8",
        )
    else:
        path.write_text(f'version = "{version}"\n', encoding="utf-8")
    return path
