# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Invariants asserted against a *built* container image.

The image under test is named by ``IMAGE_TAG``; the tests skip when it is unset so a plain
``pytest docker/test`` stays green on a machine with no image. Run one explicitly with::

    IMAGE_TAG=isaac-lab-base:latest pytest docker/test/test_image_invariants.py
"""

from __future__ import annotations

import os
import subprocess

import pytest

IMAGE_TAG = os.environ.get("IMAGE_TAG", "")


def _in_image(script: str) -> str:
    """Run ``script`` with bash inside the image under test and return its stdout."""
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "bash", IMAGE_TAG, "-lc", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.fixture(autouse=True)
def _require_image():
    if not IMAGE_TAG:
        pytest.skip("IMAGE_TAG is unset; no built image to assert against")


def test_no_prebundled_package_lost_its_entry_point():
    """A dangling ``__init__.py`` in a prebundle stops Isaac Sim extensions loading.

    Isaac Sim shares prebundled packages between extensions as per-file symlinks, so deleting
    or replacing one strands every symlink into it. #6329 added this invariant to the pip
    install path after nvbugs 6343978, where it cost 438 error lines and 14 failed extensions.
    The images now install with ``uv sync``, which never calls that code, so assert it here.

    Only ``__init__.py`` is fatal: the shipped image already carries dangling submodules and
    ``.pyi`` stubs that no extension imports - 41 of them, against develop's 48 - mostly
    generated protobuf stubs inside an Omniverse extension's own prebundle.
    """
    broken = _in_image('find / -path "*pip_prebundle*" -xtype l -name "__init__.py" 2>/dev/null || true').strip()
    assert not broken, "prebundled packages lost their entry point:\n" + broken


def test_uv_run_resolves_the_shipped_environment():
    """``uv run`` must use the venv the image ships, not build a second one.

    uv reads the project environment from ``UV_PROJECT_ENVIRONMENT`` and ignores ``VIRTUAL_ENV``,
    so an image setting only the latter has ``uv run`` create ``${ISAACLAB_PATH}/.venv`` and
    reinstall the whole lock. On a bind-mounted source tree that write fails outright
    (nvbugs 6732972, ``could not create '<package>.egg-info': Permission denied``).
    """
    virtual_env = _in_image('printf %s "$VIRTUAL_ENV"').strip()
    assert virtual_env, "image does not set VIRTUAL_ENV"

    prefix = _in_image('uv run python -c "import sys; print(sys.prefix)"').strip()
    assert prefix == virtual_env, f"uv run resolved {prefix}, not the image environment {virtual_env}"

    stray = _in_image('ls -d "${ISAACLAB_PATH}/.venv" 2>/dev/null || true').strip()
    assert not stray, f"uv run built a second environment at {stray}"
