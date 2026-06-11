# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "docker"


def _load_module(name: str, path: Path):
    """Import a module by file path (``docker`` is not an importable package here)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load module at {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Collect every Dockerfile.* from the entire repository tree.
DOCKERFILES = sorted(REPO_ROOT.glob("**/Dockerfile.*"))

ROOT_USERS = {"root", "0"}

# Keep every Dockerfile in this map so new containers must make an explicit
# runtime-user decision instead of silently escaping this regression test.
# Keys are Dockerfile *names* (unique across the repo); values are the
# expected final USER directive (None = not yet migrated, test skipped).
DOCKERFILE_RUNTIME_USERS = {
    "Dockerfile.base": "isaaclab",
    "Dockerfile.curobo": "isaaclab",
    "Dockerfile.installci": "isaaclab",
    "Dockerfile.ros2": "isaaclab",
}

# Dockerfiles that are expected to *create* the non-root runtime user
# (i.e. contain groupadd/useradd/USER isaaclab).
DOCKERFILES_CREATING_RUNTIME_USER = {"Dockerfile.base", "Dockerfile.curobo", "Dockerfile.installci"}

USER_DIRECTIVE_RE = re.compile(r"^USER\s+(\S+)\s*$")


def _user_directives(dockerfile_text: str) -> list[str]:
    users = []
    for raw_line in dockerfile_text.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            continue
        match = USER_DIRECTIVE_RE.match(line)
        if match:
            users.append(match.group(1))
    return users


def _final_user(dockerfile_path: Path) -> str | None:
    users = _user_directives(dockerfile_path.read_text(encoding="utf-8"))
    return users[-1] if users else None


def _find_dockerfile(name: str) -> Path:
    """Return the path of the unique Dockerfile with the given name."""
    matches = [p for p in DOCKERFILES if p.name == name]
    assert len(matches) == 1, f"Expected exactly one {name}, found: {matches}"
    return matches[0]


def test_all_dockerfiles_have_runtime_user_expectations():
    expected_dockerfiles = set(DOCKERFILE_RUNTIME_USERS)
    actual_dockerfiles = {dockerfile.name for dockerfile in DOCKERFILES}

    assert actual_dockerfiles == expected_dockerfiles


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_non_root_runtime_dockerfiles(dockerfile: Path):
    expected_user = DOCKERFILE_RUNTIME_USERS[dockerfile.name]

    if expected_user is None:
        pytest.skip(f"{dockerfile.name} has not been migrated to a non-root runtime user.")

    final_user = _final_user(dockerfile)
    assert final_user == expected_user
    assert final_user not in ROOT_USERS


@pytest.mark.parametrize("dockerfile_name", sorted(DOCKERFILES_CREATING_RUNTIME_USER))
def test_dockerfile_creates_non_root_runtime_user(dockerfile_name: str):
    dockerfile_text = _find_dockerfile(dockerfile_name).read_text(encoding="utf-8")

    assert re.search(r"\bgroupadd\b.*--gid\s+1000\b.*\bisaaclab\b", dockerfile_text, re.DOTALL)
    assert re.search(r"\buseradd\b.*--uid\s+1000\b.*--gid\s+1000\b.*\bisaaclab\b", dockerfile_text, re.DOTALL)
    assert "USER isaaclab" in dockerfile_text


def test_ros2_dockerfile_restores_non_root_runtime_user():
    dockerfile_text = (DOCKER_DIR / "Dockerfile.ros2").read_text(encoding="utf-8")

    assert _user_directives(dockerfile_text) == ["root", "isaaclab"]


# --------------------------------------------------------------------------- #
# Volume mount-point writability
#
# A fresh Docker named volume inherits ownership from the image directory at its
# mount path on first mount. If that directory is missing or root-owned, the
# volume comes up root-owned and the non-root ``isaaclab`` runtime user cannot
# write it (e.g. ``PermissionError`` creating ``logs/`` or ``omni.datastore``
# lock failures under ``kit/cache``). The image build therefore pre-creates and
# chowns every named-volume mount point, driven by a single source of truth:
# docker-compose.yaml, parsed by docker/utils/volume_mounts.py. These tests
# validate the parser and that each non-root Dockerfile wires it in.
# --------------------------------------------------------------------------- #

NONROOT_VOLUME_DOCKERFILES = ["Dockerfile.base", "Dockerfile.curobo"]


def _volume_mounts_module():
    """Load the parser the image build uses; skip the test if PyYAML is unavailable.

    The Docker image build exercises this parser for real, so a test environment
    without PyYAML simply skips the parser unit tests rather than failing.
    """
    pytest.importorskip("yaml")
    return _load_module("volume_mounts", DOCKER_DIR / "utils" / "volume_mounts.py")


def test_compose_volume_targets_parse():
    """The parser returns every ``type: volume`` mount point from docker-compose.yaml.

    Includes the directories that triggered the original regression so a compose
    edit that drops them is caught here.
    """
    targets = _volume_mounts_module().named_volume_targets(DOCKER_DIR / "docker-compose.yaml")

    assert targets, "no named-volume targets parsed from docker-compose.yaml"
    for required in (
        "${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache",
        "${DOCKER_ISAACLAB_PATH}/logs",
        "${DOCKER_ISAACLAB_PATH}/data_storage",
        "${DOCKER_ISAACLAB_PATH}/docs/_build",
    ):
        assert required in targets, f"{required} missing from parsed volume targets: {targets}"


def test_resolved_targets_are_absolute_paths(monkeypatch):
    """With the build's environment, every target resolves to an absolute path."""
    monkeypatch.setenv("DOCKER_ISAACSIM_ROOT_PATH", "/isaac-sim")
    monkeypatch.setenv("DOCKER_ISAACLAB_PATH", "/workspace/isaaclab")
    monkeypatch.setenv("DOCKER_USER_HOME", "/root")

    resolved = _volume_mounts_module().resolved_targets(DOCKER_DIR / "docker-compose.yaml")

    assert resolved, "no resolved targets"
    assert all(p.startswith("/") and "$" not in p for p in resolved), resolved
    assert "/isaac-sim/kit/cache" in resolved
    assert "/workspace/isaaclab/logs" in resolved


@pytest.mark.parametrize("dockerfile_name", NONROOT_VOLUME_DOCKERFILES)
def test_dockerfile_prepares_volume_mounts_from_compose(dockerfile_name: str):
    """Each non-root Dockerfile derives its mount points from the parser, with a guard.

    Guards the wiring: the build must call ``volume_mounts.py`` under
    ``set -o pipefail`` (so a parse failure aborts the build) rather than
    re-hardcoding the list or silently skipping preparation.
    """
    text = _find_dockerfile(dockerfile_name).read_text(encoding="utf-8")

    assert "set -o pipefail" in text
    assert "docker/utils/volume_mounts.py" in text
    assert "chown -R isaaclab:isaaclab ${dirs}" in text
