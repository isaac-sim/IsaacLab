# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
from pathlib import Path

import pytest
import tomllib

from docker.utils import volume_mounts

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "docker"


# Source Dockerfiles only; generated wheel staging trees may contain duplicate copies.
DOCKERFILES = sorted(
    [*REPO_ROOT.glob("docker/Dockerfile.*"), REPO_ROOT / "source/isaaclab/test/install_ci/Dockerfile.installci"]
)

# Pinned by digest so a uv release cannot silently change how the lock resolves. Matches the uv
# that regenerated ``uv.lock``; a mismatch reintroduces the marker churn that refresh removed.
UV_PIN = "ghcr.io/astral-sh/uv:0.12.9@sha256:8b940d3a9d65bed080436972241af2e21c84b5e8c9193f7014ed71479ee795ff"


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_dockerfile_creates_and_uses_non_root_user(dockerfile: Path):
    """Every image creates uid/gid 1000 and finishes as the unprivileged user."""
    text = dockerfile.read_text(encoding="utf-8")
    users = re.findall(r"^\s*USER\s+(\S+)\s*$", text, re.MULTILINE)

    assert users and users[-1] == "isaaclab"
    assert re.search(r"\bgroupadd\b.*--gid\s+1000\b.*\bisaaclab\b", text, re.DOTALL)
    assert re.search(r"\buseradd\b.*--uid\s+1000\b.*--gid\s+1000\b.*\bisaaclab\b", text, re.DOTALL)


def test_images_share_one_pinned_uv():
    """Every image that installs with uv agrees on the pinned version."""
    pinned = {
        path.relative_to(REPO_ROOT).as_posix(): re.findall(
            r"FROM (ghcr\.io/astral-sh/uv:\S+) AS uv", path.read_text(encoding="utf-8")
        )
        for path in DOCKERFILES
    }
    pinned = {name: refs for name, refs in pinned.items() if refs}

    assert pinned, "no Dockerfile pins uv"
    offenders = {name: refs for name, refs in pinned.items() if refs != [UV_PIN]}
    assert not offenders, f"every image must pin {UV_PIN}; got {offenders}"


def test_curobo_compiler_matches_torch_without_replacing_runtime_libraries():
    """CUDA extensions use Torch's CUDA version while uv owns its runtime libraries."""
    text = (DOCKER_DIR / "Dockerfile.curobo").read_text(encoding="utf-8")
    with (REPO_ROOT / "uv.lock").open("rb") as file:
        torch_versions = {package["version"] for package in tomllib.load(file)["package"] if package["name"] == "torch"}

    toolkit = re.search(r"\bcuda-toolkit-(\d+)-(\d+)\b", text)
    assert toolkit is not None
    major, minor = toolkit.groups()
    assert {version.partition("+cu")[2] for version in torch_versions} == {f"{major}{minor}"}
    assert f"ENV CUDA_HOME=/usr/local/cuda-{major}.{minor}" in text
    assert "ENV PATH=${CUDA_HOME}/bin:${PATH}" in text
    # System CUDA libraries must not replace Kit's libraries or shadow the locked wheels.
    assert not re.search(r"^ENV LD_LIBRARY_PATH=.*\$\{CUDA_HOME\}", text, re.MULTILINE)
    assert not re.search(r"^\s+(?:libcudnn|libcusparselt|libnccl|libnvjitlink)\S*\s", text, re.MULTILINE)


def test_kitless_dockerfile_installs_newton_rl_ov_and_visualizers_without_isaac_sim():
    """The kit-less image installs its runtime features and importers without the full Isaac Sim runtime."""
    dockerfile_text = (DOCKER_DIR / "Dockerfile.kitless").read_text(encoding="utf-8")
    with (REPO_ROOT / "pyproject.toml").open("rb") as file:
        extras = tomllib.load(file)["project"]["optional-dependencies"]

    assert "uv sync --frozen --inexact --extra all --extra importers" in dockerfile_text
    assert "importers" in extras
    # ``all`` must not drag in the Isaac Sim runtime, or the kit-less image means nothing.
    assert "isaacsim" not in "".join(extras["all"])
    # The interpreter must sit outside ISAACLAB_PATH. CI bind-mounts the checkout over that path,
    # so a venv beneath it is masked and uv run isaaclab execs a missing interpreter (exit 127).
    assert "ARG VENV_PATH_ARG=/opt/isaaclab-venv" in dockerfile_text
    assert "ENV VIRTUAL_ENV=${VENV_PATH_ARG}" in dockerfile_text
    # ``uv sync`` honours the project's ``only-managed`` preference and would rebuild the venv
    # against a downloaded interpreter the runtime stage never receives, leaving bin/python
    # dangling. The image must pin uv to the system interpreter.
    assert "ENV UV_PYTHON=/usr/bin/python3.12" in dockerfile_text
    assert "ENV UV_PYTHON_PREFERENCE=only-system" in dockerfile_text
    assert "'isaacsim' not in names" in dockerfile_text
    assert "'isaacsim-asset-isolated' in names" in dockerfile_text
    assert "'ovphysx' in names" in dockerfile_text
    assert "'ovrtx' in names" in dockerfile_text
    assert "'viser' in names" in dockerfile_text
    assert "'rerun-sdk' in names" in dockerfile_text
    assert "libxrender1" in dockerfile_text
    assert 'test ! -e "${ISAACLAB_PATH}/_isaac_sim"' in dockerfile_text
    # volume_mounts.py parses docker-compose.yaml at runtime, so both must reach the image -
    # either named individually or via a whole-tree copy.
    for required in ("docker/docker-compose.yaml", "docker/utils/volume_mounts.py"):
        assert f"COPY {required} {required}" in dockerfile_text or "COPY . ." in dockerfile_text, (
            f"{required} must be copied into the kit-less image"
        )


def test_container_test_runner_only_links_an_actual_isaac_sim_runtime():
    """Cache mount points under /isaac-sim must not masquerade as an Isaac Sim installation."""
    runner_text = (REPO_ROOT / ".github/actions/run-tests/run_tests.sh").read_text(encoding="utf-8")
    guarded_link = re.compile(r"if \[ -x /isaac-sim/python\.sh \]; then\s+ln -s /isaac-sim _isaac_sim;?\s+fi")

    assert guarded_link.search(runner_text)
    assert runner_text.count("ln -s /isaac-sim _isaac_sim") == 1


NONROOT_VOLUME_DOCKERFILES = {
    "Dockerfile.base": "x-default-isaac-lab-volumes",
    "Dockerfile.curobo": "x-default-isaac-lab-volumes",
    "Dockerfile.kitless": "x-kitless-isaac-lab-volumes",
}


def test_compose_volume_targets_parse(monkeypatch):
    """The parser returns every ``type: volume`` mount point from docker-compose.yaml.

    Includes the directories that triggered the original regression so a compose
    edit that drops them is caught here.
    """
    targets = volume_mounts.named_volume_targets(DOCKER_DIR / "docker-compose.yaml")

    assert targets, "no named-volume targets parsed from docker-compose.yaml"
    for required in (
        "${DOCKER_ISAACSIM_ROOT_PATH:-/isaac-sim}/kit/cache",
        "${DOCKER_ISAACLAB_PATH}/logs",
        "${DOCKER_ISAACLAB_PATH}/data_storage",
        "${DOCKER_ISAACLAB_PATH}/docs/_build",
    ):
        assert required in targets, f"{required} missing from parsed volume targets: {targets}"

    monkeypatch.setenv("DOCKER_ISAACSIM_ROOT_PATH", "/isaac-sim")
    monkeypatch.setenv("DOCKER_ISAACLAB_PATH", "/workspace/isaaclab")
    monkeypatch.setenv("DOCKER_USER_HOME", "/root")

    resolved = volume_mounts.resolved_targets(DOCKER_DIR / "docker-compose.yaml")

    assert resolved, "no resolved targets"
    assert all(p.startswith("/") and "$" not in p for p in resolved), resolved
    assert "/isaac-sim/kit/cache" in resolved
    assert "/workspace/isaaclab/logs" in resolved


@pytest.mark.parametrize(("dockerfile_name", "volumes_key"), NONROOT_VOLUME_DOCKERFILES.items())
def test_dockerfile_prepares_volume_mounts_from_compose(dockerfile_name: str, volumes_key: str):
    """Each non-root Dockerfile derives its mount points from the parser, with a guard.

    Guards the wiring: the build must call ``volume_mounts.py`` under
    ``set -o pipefail`` (so a parse failure aborts the build) rather than
    re-hardcoding the list or silently skipping preparation.
    """
    text = (DOCKER_DIR / dockerfile_name).read_text(encoding="utf-8")

    assert "set -o pipefail" in text
    assert "docker/utils/volume_mounts.py" in text
    assert "chown -R isaaclab:isaaclab ${dirs}" in text
    if volumes_key != "x-default-isaac-lab-volumes":
        assert f"--volumes_key {volumes_key}" in text


@pytest.mark.parametrize("dockerfile_name", ["Dockerfile.base", "Dockerfile.curobo"])
def test_isaac_sim_dockerfiles_configure_writable_omnihub(dockerfile_name: str):
    """OmniHub can start and write its cache as the runtime user."""
    dockerfile_text = (DOCKER_DIR / dockerfile_name).read_text(encoding="utf-8")

    chown_block = re.search(r"chown -R isaaclab:isaaclab((?:\s*\\\s*\S+)+)", dockerfile_text)
    assert chown_block, f"{dockerfile_name} has no 'chown -R isaaclab:isaaclab' block"
    assert "/var/cache/hub" in chown_block.group(1)

    assert re.search(r"^ENV HUB__ARGS__DETECT_ONLY=false$", dockerfile_text, re.MULTILINE), (
        f"{dockerfile_name} must set 'ENV HUB__ARGS__DETECT_ONLY=false' exactly"
    )
