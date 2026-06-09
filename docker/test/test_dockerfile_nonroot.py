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


# --------------------------------------------------------------------------- #
# Volume mount-point writability
#
# A fresh Docker named volume inherits ownership from the image directory at its
# mount path on first mount. If that directory is missing or root-owned, the
# volume comes up root-owned and the non-root ``isaaclab`` runtime user cannot
# write it (e.g. ``PermissionError`` creating ``logs/`` or ``omni.datastore``
# lock failures under ``kit/cache``). The tests below statically guarantee every
# such mount point is both created in the image and owned by ``isaaclab``.
# --------------------------------------------------------------------------- #

COMPOSE_FILE = DOCKER_DIR / "docker-compose.yaml"
ENV_BASE_FILE = DOCKER_DIR / ".env.base"

_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _load_env_base() -> dict[str, str]:
    """Parse ``docker/.env.base`` into a ``{KEY: VALUE}`` map (quotes stripped)."""
    env: dict[str, str] = {}
    for raw in ENV_BASE_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _path_var_map() -> dict[str, str]:
    """Resolve the path vars shared by docker-compose and the Dockerfiles.

    The Dockerfile ENV names (``ISAACLAB_PATH`` ...) take their values from build
    args wired to the ``DOCKER_*`` vars in ``.env.base``, so both files resolve
    to the same concrete paths.
    """
    env = _load_env_base()
    sim_root = env["DOCKER_ISAACSIM_ROOT_PATH"]
    lab_path = env["DOCKER_ISAACLAB_PATH"]
    user_home = env["DOCKER_USER_HOME"]
    return {
        "DOCKER_ISAACSIM_ROOT_PATH": sim_root,
        "DOCKER_ISAACLAB_PATH": lab_path,
        "DOCKER_USER_HOME": user_home,
        "ISAACSIM_ROOT_PATH": sim_root,
        "ISAACLAB_PATH": lab_path,
        "HOME": user_home,
    }


def _expand(path: str, var_map: dict[str, str]) -> str:
    """Expand ``${VAR}`` references and strip any trailing slash."""
    return _VAR_RE.sub(lambda m: var_map.get(m.group(1), m.group(0)), path).rstrip("/")


def _is_within(child: str, parent: str) -> bool:
    """True if ``child`` equals ``parent`` or is nested under it."""
    return child == parent or child.startswith(parent + "/")


def _compose_volume_targets(var_map: dict[str, str]) -> list[str]:
    """Return every ``type: volume`` target path in docker-compose.yaml, expanded."""
    targets = []
    in_volume = False
    for raw in COMPOSE_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("- type:"):
            in_volume = line.endswith("volume")
        elif in_volume and line.startswith("target:"):
            targets.append(_expand(line.split("target:", 1)[1].strip(), var_map))
            in_volume = False
    return targets


def _command_path_args(dockerfile_text: str, command_re: str) -> list[str]:
    """Collect the absolute/``${VAR}`` path tokens of each matching RUN command.

    Line continuations are joined first; tokens are read up to the next shell
    separator (``&&``, ``||``, ``;`` or newline) so each command's args stay
    distinct.
    """
    joined = re.sub(r"\\\s*\n", " ", dockerfile_text)
    tokens: list[str] = []
    for match in re.finditer(command_re, joined):
        segment = re.split(r"&&|\|\||;|\n", joined[match.end() :], maxsplit=1)[0]
        tokens += [t for t in segment.split() if t.startswith(("$", "/"))]
    return tokens


def _mkdir_targets(text: str, var_map: dict[str, str]) -> list[str]:
    return [_expand(t, var_map) for t in _command_path_args(text, r"\bmkdir\b(?:\s+-\w+)*")]


def _chown_isaaclab_roots(text: str, var_map: dict[str, str]) -> list[str]:
    return [_expand(t, var_map) for t in _command_path_args(text, r"\bchown\s+-R\s+isaaclab:isaaclab\b")]


def test_base_compose_volume_mount_points_are_writable():
    """Every docker-compose named volume mounts onto an isaaclab-owned image dir.

    Guards the regression where ``kit/cache`` (root-owned) and ``logs`` (absent
    in the image) came up root-owned on a fresh volume, blocking training.
    """
    var_map = _path_var_map()
    text = _find_dockerfile("Dockerfile.base").read_text(encoding="utf-8")
    mkdirs = _mkdir_targets(text, var_map)
    chown_roots = _chown_isaaclab_roots(text, var_map)
    targets = _compose_volume_targets(var_map)

    assert targets, "no named-volume targets parsed from docker-compose.yaml"

    problems = []
    for target in targets:
        created = any(_is_within(m, target) for m in mkdirs)
        owned = any(_is_within(target, root) for root in chown_roots)
        if not (created and owned):
            problems.append({"target": target, "created_in_image": created, "owned_by_isaaclab": owned})
    assert not problems, "volume mount points not guaranteed writable by isaaclab:\n" + "\n".join(map(str, problems))


@pytest.mark.parametrize("dockerfile_name", ["Dockerfile.base", "Dockerfile.curobo"])
def test_created_dirs_outside_home_are_owned_by_runtime_user(dockerfile_name: str):
    """Dirs created under the Sim/Lab roots (outside ``$HOME``) must be chowned.

    ``$HOME`` is already handled by the recursive ``chown`` of the runtime home;
    anything created under the root-owned Isaac Sim tree (e.g. ``kit/cache``) is
    not, so it must be handed to ``isaaclab`` explicitly.
    """
    var_map = _path_var_map()
    home = var_map["DOCKER_USER_HOME"]
    managed_roots = (var_map["ISAACSIM_ROOT_PATH"], var_map["ISAACLAB_PATH"])
    text = _find_dockerfile(dockerfile_name).read_text(encoding="utf-8")
    mkdirs = _mkdir_targets(text, var_map)
    chown_roots = _chown_isaaclab_roots(text, var_map)

    problems = []
    for created_dir in mkdirs:
        if not any(_is_within(created_dir, root) for root in managed_roots):
            continue  # unrelated system dir (e.g. /var/run/...)
        if _is_within(created_dir, home):
            continue  # covered by the recursive chown of the runtime home
        if not any(_is_within(created_dir, root) for root in chown_roots):
            problems.append(created_dir)
    assert not problems, f"{dockerfile_name}: dirs created but not chowned to isaaclab: {problems}"
