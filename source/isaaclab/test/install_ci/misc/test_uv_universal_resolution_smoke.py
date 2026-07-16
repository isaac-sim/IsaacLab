# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - (none: the test resolves dependencies only and does not install anything)
Tests:
    - uv pip compile --universal --python-version 3.12 pyproject.toml
      -o <tmp>/requirements.txt -> verify the base dependency set resolves for every
      supported platform (no contradictory marker splits)
"""

from __future__ import annotations

import re
import shutil

import pytest
import tomllib
from utils import run_cmd


@pytest.mark.uv
@pytest.mark.smoke
class Test_Uv_Universal_Resolution_Smoke:
    """Universal (all-platform) resolution of the root project's base dependencies.

    ``uv run`` and ``uv lock`` resolve the dependency tree for every supported platform
    at once, so a requirement pair that is fine on each platform in isolation can still
    make the whole project unsatisfiable. uv cannot prove ``platform_machine in
    'x86_64 AMD64'`` and ``platform_machine in 'aarch64 arm64'`` mutually exclusive, so
    two such markers on the same package fork the resolution into a split where both
    pins apply (seen with the pytetwild pins; platform markers must be written as
    ``==`` disjunctions instead).

    ``uv pip compile --universal`` performs the same universal resolution in seconds
    without installing anything and without the optional extras, some of which
    (``ov``) pin versions that are not publicly resolvable yet.
    """

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.mark.timeout(900)
    def test_uv_pip_compile_universal_resolves_base_dependencies(self, isaaclab_root, tmp_path):
        """Verify the base dependencies resolve universally, as ``uv run`` would."""
        with (isaaclab_root / "pyproject.toml").open("rb") as f:
            requires_python = tomllib.load(f)["project"]["requires-python"]
        lower_bound = re.search(r">=\s*(\d+\.\d+)", requires_python)
        assert lower_bound, f"Could not read a lower bound from requires-python: {requires_python!r}"

        result = run_cmd(
            [
                "uv",
                "pip",
                "compile",
                "--universal",
                "--python-version",
                lower_bound.group(1),
                "pyproject.toml",
                "-o",
                str(tmp_path / "requirements.txt"),
            ],
            cwd=isaaclab_root,
            timeout=880,
        )
        assert result.returncode == 0, (
            "Universal resolution of the base dependencies failed, so `uv run` is broken on at"
            " least one supported platform. If the error mentions a split with overlapping"
            " platform markers, express the markers as `==` disjunctions (uv cannot prove"
            f" `platform_machine in '...'` markers disjoint):\n{result.stdout}\n{result.stderr}"
        )
