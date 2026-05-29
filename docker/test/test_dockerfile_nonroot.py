# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "docker"

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
