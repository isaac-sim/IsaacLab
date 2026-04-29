# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for CI Docker storage cleanup helpers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_ACTION_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ACTION_DIR.parents[2]
_ACTION_PATH = _ACTION_DIR / "action.yml"
_SCRIPT_PATH = _ACTION_DIR / "cleanup_docker_storage.sh"


def _write_fake_df(
    bin_dir: Path, first_usage: int, final_usage: int, failed_path: str | None = None, fallback_path: str | None = None
) -> None:
    count_file = bin_dir / "df-count"
    count_file.write_text("0", encoding="utf-8")
    script = bin_dir / "df"
    script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

path="${{2:-}}"
if [ -n "{failed_path or ""}" ] && [ "$path" = "{failed_path or ""}" ]; then
    exit 1
fi
if [ -n "{fallback_path or ""}" ] && [ "$path" = "{fallback_path or ""}" ]; then
    :
elif [ -n "{fallback_path or ""}" ]; then
    exit 1
fi

if [ "${{1:-}}" = "-P" ]; then
    count=$(cat "{count_file}")
    if [ "$count" -eq 0 ]; then
        usage="{first_usage}"
    else
        usage="{final_usage}"
    fi
    echo "$((count + 1))" > "{count_file}"
    echo "Filesystem 1024-blocks Used Available Capacity Mounted on"
    echo "/dev/test 100 90 10 ${{usage}}% /var/lib/docker"
else
    echo "Filesystem Size Used Avail Use% Mounted on"
    echo "/dev/test 100G 90G 10G {first_usage}% /var/lib/docker"
fi
""",
        encoding="utf-8",
    )
    script.chmod(0o755)


def _write_fake_docker(bin_dir: Path, docker_root_dir: str = "/var/lib/docker") -> Path:
    log_path = bin_dir / "docker.log"
    script = bin_dir / "docker"
    script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

printf '%s\\n' "$*" >> "{log_path}"
if [ "${{1:-}}" = "info" ] && [ "${{2:-}}" = "--format" ]; then
    if [ -z "{docker_root_dir}" ]; then
        exit 1
    fi
    echo "{docker_root_dir}"
    exit 0
fi
if [ "${{1:-}}" = "image" ] && [ "${{2:-}}" = "inspect" ]; then
    exit 0
fi
exit 0
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return log_path


def _run_cleanup(
    tmp_path: Path,
    first_usage: int,
    final_usage: int,
    failed_path: str | None = None,
    docker_root_dir: str = "/var/lib/docker",
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_df(bin_dir, first_usage, final_usage, failed_path=failed_path, fallback_path=docker_root_dir)
    _write_fake_docker(bin_dir, docker_root_dir=docker_root_dir)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["DOCKER_STORAGE_PATH"] = "/var/lib/docker"
    env["DOCKER_CLEANUP_THRESHOLD_PERCENT"] = "85"

    return subprocess.run(
        [
            "bash",
            str(_SCRIPT_PATH),
            "nvcr.io/nvidian/isaac-sim:latest-develop",
            "isaacsim-pin-test",
            "test-cleanup",
            "true",
        ],
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cleanup_uses_aggressive_pruning_when_docker_storage_exceeds_threshold(tmp_path: Path):
    result = _run_cleanup(tmp_path, first_usage=91, final_usage=70)

    assert result.returncode == 0, result.stderr
    docker_log = (tmp_path / "bin" / "docker.log").read_text(encoding="utf-8")
    assert "container prune -f --filter until=24h" in docker_log
    assert "builder prune -a -f --filter until=24h" in docker_log
    assert "image prune -a -f --filter until=24h" in docker_log


def test_cleanup_keeps_conservative_image_pruning_below_threshold(tmp_path: Path):
    result = _run_cleanup(tmp_path, first_usage=70, final_usage=70)

    assert result.returncode == 0, result.stderr
    docker_log = (tmp_path / "bin" / "docker.log").read_text(encoding="utf-8")
    assert "image prune -a -f --filter until=72h" in docker_log
    assert "builder prune" not in docker_log


def test_cleanup_fails_early_when_aggressive_pruning_cannot_recover_space(tmp_path: Path):
    result = _run_cleanup(tmp_path, first_usage=91, final_usage=89)

    assert result.returncode == 1
    assert "Docker storage remains at 89%" in result.stderr


def test_cleanup_uses_docker_root_dir_when_default_storage_path_is_unavailable(tmp_path: Path):
    result = _run_cleanup(
        tmp_path,
        first_usage=91,
        final_usage=89,
        failed_path="/var/lib/docker",
        docker_root_dir="/mnt/docker-root",
    )

    assert result.returncode == 1
    assert "Using Docker root from docker info: /mnt/docker-root" in result.stdout
    assert "Docker storage remains at 89%" in result.stderr


def test_cleanup_fails_early_when_docker_storage_usage_cannot_be_measured(tmp_path: Path):
    result = _run_cleanup(
        tmp_path,
        first_usage=91,
        final_usage=89,
        failed_path="/var/lib/docker",
        docker_root_dir="",
    )

    assert result.returncode == 1
    assert "Could not determine Docker storage usage after cleanup" in result.stderr


def test_run_package_tests_prepares_docker_space_before_image_pull():
    action_text = _ACTION_PATH.read_text(encoding="utf-8")

    cleanup_index = action_text.index("name: Prepare Docker disk space")
    pull_start_index = action_text.index("name: Record pull start time")
    pull_index = action_text.index("name: Pull image from ECR")

    assert cleanup_index < pull_start_index < pull_index
    assert "cleanup_docker_storage.sh" in action_text
