# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for dependency installation in Docker images."""

from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_base_dockerfile_installs_compatible_ovstage_before_source_packages():
    """The base image must install the OVPhysX-compatible OVStage version first."""
    with (REPO_ROOT / "uv.lock").open("rb") as file:
        packages = tomllib.load(file)["package"]

    ovstage_versions = [package["version"] for package in packages if package["name"] == "ovstage"]
    assert ovstage_versions == ["0.1.0.346039"]

    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.base").read_text(encoding="utf-8")
    ovstage_install = f'${{ISAACLAB_PATH}}/isaaclab.sh -p -m pip install "ovstage=={ovstage_versions[0]}"'
    source_install = "${ISAACLAB_PATH}/isaaclab.sh --install"

    assert ovstage_install in dockerfile
    assert dockerfile.index(ovstage_install) < dockerfile.index(source_install)
