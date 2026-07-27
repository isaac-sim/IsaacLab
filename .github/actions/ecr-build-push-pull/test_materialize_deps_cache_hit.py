# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for materializing ECR dependency-cache hits as local images."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_ACTION_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ACTION_DIR.parents[2]
_SCRIPT_PATH = _ACTION_DIR / "materialize_deps_cache_hit.sh"
_ACTION_PATH = _ACTION_DIR / "action.yml"


def _write_fake_docker(bin_dir: Path) -> Path:
    """Create a Docker shim that records every argument list."""
    log_path = bin_dir / "docker.log"
    script = bin_dir / "docker"
    script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{log_path}"
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return log_path


def test_deps_cache_hit_creates_alias_then_pulls_and_tags_locally(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = _write_fake_docker(bin_dir)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(_SCRIPT_PATH),
            "registry.example/repo:deps-abc",
            "registry.example/repo:commit-123",
            "isaac-lab-ci:develop-123",
        ],
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert docker_log.read_text(encoding="utf-8").splitlines() == [
        "buildx imagetools create -t registry.example/repo:commit-123 registry.example/repo:deps-abc",
        "pull registry.example/repo:commit-123",
        "tag registry.example/repo:commit-123 isaac-lab-ci:develop-123",
    ]


def test_deps_cache_hit_action_materializes_local_image_before_reporting_hit():
    action = _ACTION_PATH.read_text(encoding="utf-8")
    helper_call = 'bash "${GITHUB_ACTION_PATH}/materialize_deps_cache_hit.sh"'

    assert action.index(helper_call) < action.index('echo "deps-cache-hit=true"')
