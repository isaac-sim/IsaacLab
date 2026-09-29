# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "docker"
SHIM_INSTALLER = DOCKER_DIR / "scripts" / "install_carb_env_shim.sh"
SHIMMED_DOCKERFILES = ["Dockerfile.base", "Dockerfile.curobo"]


@pytest.mark.parametrize("dockerfile_name", SHIMMED_DOCKERFILES)
def test_dockerfiles_run_shared_carb_env_shim_installer(dockerfile_name: str):
    """Verify each Isaac Sim Dockerfile runs the shared Carbonite environment shim installer."""
    dockerfile_text = (DOCKER_DIR / dockerfile_name).read_text(encoding="utf-8")

    assert "COPY docker/scripts/install_carb_env_shim.sh /tmp/install_carb_env_shim.sh" in dockerfile_text
    assert 'RUN /bin/bash /tmp/install_carb_env_shim.sh "${ISAACSIM_ROOT_PATH}"' in dockerfile_text


@pytest.mark.parametrize("preload_terminator", ["", "\n"], ids=["without_final_newline", "with_final_newline"])
def test_carb_env_shim_installer_copies_shim_and_preserves_preloads(tmp_path: Path, preload_terminator: str):
    """Verify the shared installer copies the shim and adds one preload entry."""
    isaacsim_root = tmp_path / "isaac-sim"
    source_shim = isaacsim_root / "kit" / "kernel" / "plugins" / "libcarb.env.shim.so"
    source_shim.parent.mkdir(parents=True)
    source_shim.write_bytes(b"carbonite environment shim")

    installed_shim = tmp_path / "usr" / "local" / "lib" / "libcarb.env.shim.so"
    preload_file = tmp_path / "etc" / "ld.so.preload"
    preload_file.parent.mkdir(parents=True)
    preload_file.write_text(f"/usr/local/lib/existing-shim.so{preload_terminator}", encoding="utf-8")

    command = ["bash", str(SHIM_INSTALLER), str(isaacsim_root), str(installed_shim), str(preload_file)]
    for _ in range(2):
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    assert installed_shim.read_bytes() == source_shim.read_bytes()
    assert stat.S_IMODE(installed_shim.stat().st_mode) == 0o755
    assert preload_file.read_text(encoding="utf-8").splitlines() == [
        "/usr/local/lib/existing-shim.so",
        str(installed_shim),
    ]


def test_dependency_cache_hash_tracks_carb_env_shim_installer():
    """Verify helper-only changes invalidate the Docker dependency image cache."""
    action = REPO_ROOT / ".github" / "actions" / "_lib" / "compute-deps-hash" / "action.yml"

    assert "docker/scripts/install_carb_env_shim.sh" in action.read_text(encoding="utf-8")
