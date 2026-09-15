# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Repository layout shared by changelog models and commands."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_ROOT = REPO_ROOT / "source"


class RepositoryPaths:
    """Construct and display paths using the changelog repository layout."""

    @staticmethod
    def package_prefix(name: str) -> str:
        """Repo-relative directory prefix for a source package."""
        return f"{PACKAGES_ROOT.name}/{name}/"

    @classmethod
    def fragment_dir_prefix(cls, name: str) -> str:
        """Repo-relative directory prefix for a package's fragments."""
        return f"{cls.package_prefix(name)}changelog.d/"

    @staticmethod
    def display(path: Path) -> str:
        """Display repository paths relatively and external paths unchanged."""
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)
