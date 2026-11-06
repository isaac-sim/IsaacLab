# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PEP 517 backend for building the aggregate Isaac Lab package from a source checkout."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from setuptools import build_meta
from stage import stage_package

_BUILDER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BUILDER_DIR.parents[1]


def _wheel_version() -> str:
    version = (_REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    build_number = os.getenv("WHEEL_BUILD_NUMBER")
    commit = os.getenv("WHEEL_SHA")
    if build_number and commit:
        return f"{version}+build{build_number}.{commit[:7]}"
    return version


@contextmanager
def _staged_project() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="isaaclab-wheel-") as temp_dir:
        stage_dir = Path(temp_dir)
        stage_package(_REPO_ROOT, stage_dir, _wheel_version())
        previous_directory = Path.cwd()
        os.chdir(stage_dir)
        try:
            yield stage_dir
        finally:
            os.chdir(previous_directory)


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build the aggregate Isaac Lab wheel."""
    with _staged_project():
        return build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    """Build an aggregate Isaac Lab source distribution."""
    with _staged_project():
        return build_meta.build_sdist(sdist_directory, config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    """Prepare metadata for the aggregate Isaac Lab wheel."""
    with _staged_project():
        return build_meta.prepare_metadata_for_build_wheel(metadata_directory, config_settings)
