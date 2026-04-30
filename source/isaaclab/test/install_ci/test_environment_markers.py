# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for install_ci execution-environment marker handling."""

from __future__ import annotations

import pytest
from utils import (
    detect_execution_environment,
    get_execution_environment_skip_reason,
)


class TestDetectExecutionEnvironment:
    """Tests for detect_execution_environment()."""

    def test_uses_override(self, tmp_path):
        environment = detect_execution_environment(
            environ={"ISAACLAB_INSTALL_CI_ENV": "docker"},
            filesystem_root=tmp_path,
        )

        assert environment == "docker"

    def test_detects_marker_file(self, tmp_path):
        (tmp_path / ".dockerenv").touch()

        environment = detect_execution_environment(environ={}, filesystem_root=tmp_path)

        assert environment == "docker"

    def test_detects_cgroup_hint(self, tmp_path):
        cgroup_path = tmp_path / "proc" / "self"
        cgroup_path.mkdir(parents=True)
        (cgroup_path / "cgroup").write_text("0::/docker/container-id\n", encoding="utf-8")

        environment = detect_execution_environment(environ={}, filesystem_root=tmp_path)

        assert environment == "docker"

    def test_defaults_to_native(self, tmp_path):
        environment = detect_execution_environment(environ={}, filesystem_root=tmp_path)

        assert environment == "native"

    def test_rejects_invalid_override(self, tmp_path):
        with pytest.raises(ValueError, match="ISAACLAB_INSTALL_CI_ENV"):
            detect_execution_environment(
                environ={"ISAACLAB_INSTALL_CI_ENV": "virtual-machine"},
                filesystem_root=tmp_path,
            )


class TestGetExecutionEnvironmentSkipReason:
    """Tests for get_execution_environment_skip_reason()."""

    @pytest.mark.parametrize(
        ("marker_names", "execution_environment", "expected_reason"),
        [
            ({"docker"}, "native", "requires Docker execution environment, detected native"),
            ({"native"}, "docker", "requires native execution environment, detected docker"),
            ({"docker"}, "docker", None),
            ({"native"}, "native", None),
            (set(), "native", None),
        ],
    )
    def test_skip_reason(self, marker_names, execution_environment, expected_reason):
        skip_reason = get_execution_environment_skip_reason(marker_names, execution_environment)

        assert skip_reason == expected_reason

    def test_rejects_conflicting_markers(self):
        with pytest.raises(ValueError, match="docker"):
            get_execution_environment_skip_reason({"docker", "native"}, "native")
