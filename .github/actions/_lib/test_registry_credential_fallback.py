# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the registry-scoped docker credential fallback.

Covers the helper that drops one registry's stored credential and the composite
action's guards around it, plus the base image digest resolution that feeds the
dependency-cache key. The action bodies are extracted from their action.yml and
run under bash against stubbed executables, so no registry access is needed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import yaml

_LIB_DIR = Path(__file__).resolve().parent
_SETUP_ACTION = _LIB_DIR / "setup-docker-config" / "action.yml"
_HASH_ACTION = _LIB_DIR / "compute-deps-hash" / "action.yml"
_STRIP_HELPER = _LIB_DIR / "setup-docker-config" / "strip_registry_auth.py"

_ECR = "968945269301.dkr.ecr.us-west-2.amazonaws.com"
_HUB = "https://index.docker.io/v1/"
_DIGEST = "sha256:" + "a" * 64


def _load_helper():
    spec = importlib.util.spec_from_file_location("strip_registry_auth", _STRIP_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _step_body(action_path: Path) -> str:
    action = yaml.safe_load(action_path.read_text(encoding="utf-8"))
    return action["runs"]["steps"][0]["run"]


def _write_config(path: Path, registries: list[str]) -> Path:
    config = {"credsStore": "", "auths": {name: {"auth": "redacted"} for name in registries}}
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _auths(path: Path) -> set[str]:
    return set((json.loads(path.read_text(encoding="utf-8")).get("auths") or {}).keys())


def _write_stub_docker(bin_dir: Path, body: str) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "docker"
    script.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    script.chmod(0o755)


def _run_setup(
    tmp_path: Path,
    base_image_ref: str,
    stub: str,
    *,
    owned: bool = True,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    _write_stub_docker(bin_dir, stub)
    config_dir = tmp_path / "cfg"
    config_dir.mkdir(exist_ok=True)
    _write_config(config_dir / "config.json", ["nvcr.io", _ECR, _HUB])
    body = tmp_path / "setup.sh"
    body.write_text(_step_body(_SETUP_ACTION), encoding="utf-8")
    env = {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(tmp_path / "home"),
        "DOCKER_CONFIG": str(config_dir),
        "GITHUB_ENV": str(tmp_path / "github-env"),
        "BASE_IMAGE_REF": base_image_ref,
        "STRIP_HELPER": str(_STRIP_HELPER),
    }
    if owned:
        env["SETUP_DOCKER_CONFIG_OWNED"] = str(config_dir)
    if extra_env:
        env.update(extra_env)
    Path(env["GITHUB_ENV"]).touch()
    return subprocess.run(["bash", str(body)], env=env, capture_output=True, text=True, check=False)


# A stub that refuses credentials but serves an anonymous config (empty auths).
_DENY_UNLESS_ANONYMOUS = """
config="${DOCKER_CONFIG:-}/config.json"
if [ -f "$config" ] && grep -q '"auths":{}' "$config"; then
  exit 0
fi
echo "denied: Access Denied" >&2
exit 1
"""


def test_public_nvcr_drops_only_that_registry(tmp_path: Path) -> None:
    """The refused nvcr.io credential goes; the push credentials stay."""
    result = _run_setup(tmp_path, "nvcr.io/nvidia/isaac-sim:6.1.0", _DENY_UNLESS_ANONYMOUS)
    assert result.returncode == 0, result.stderr
    assert _auths(tmp_path / "cfg" / "config.json") == {_ECR, _HUB}


def test_non_nvcr_reference_is_left_alone(tmp_path: Path) -> None:
    """A Docker Hub base image keeps its credential, which is also pull-rate budget."""
    result = _run_setup(tmp_path, "ubuntu:24.04", _DENY_UNLESS_ANONYMOUS)
    assert result.returncode == 0, result.stderr
    assert _auths(tmp_path / "cfg" / "config.json") == {"nvcr.io", _ECR, _HUB}


def test_readable_image_keeps_credentials(tmp_path: Path) -> None:
    """A private image the credentials can read must not trigger the fallback."""
    result = _run_setup(tmp_path, "nvcr.io/example/isaac-sim:latest", "exit 0")
    assert result.returncode == 0, result.stderr
    assert _auths(tmp_path / "cfg" / "config.json") == {"nvcr.io", _ECR, _HUB}


def test_transient_failure_keeps_credentials(tmp_path: Path) -> None:
    """A registry outage is not an authorization refusal, so nothing is dropped."""
    stub = 'echo "error: received unexpected HTTP status: 503" >&2\nexit 1'
    result = _run_setup(tmp_path, "nvcr.io/nvidia/isaac-sim:6.1.0", stub)
    assert result.returncode == 0, result.stderr
    assert _auths(tmp_path / "cfg" / "config.json") == {"nvcr.io", _ECR, _HUB}
    assert "::warning::" in result.stdout


def test_transient_anonymous_failure_is_retried(tmp_path: Path) -> None:
    """A flaky anonymous probe must not leave a credential that is already denied.

    Without a retry the action would keep the denied credential and the build
    would fail on the pull this fallback exists to rescue.
    """
    stub = """
config="${DOCKER_CONFIG:-}/config.json"
if [ -f "$config" ] && grep -q '"auths":{}' "$config"; then
  counter=/tmp/il_anon_attempts
  n=$(cat "$counter" 2>/dev/null || echo 0)
  n=$((n + 1))
  echo "$n" > "$counter"
  if [ "$n" -lt 2 ]; then
    echo "error: received unexpected HTTP status: 503" >&2
    exit 1
  fi
  exit 0
fi
echo "denied: Access Denied" >&2
exit 1
"""
    Path("/tmp/il_anon_attempts").unlink(missing_ok=True)
    try:
        result = _run_setup(tmp_path, "nvcr.io/nvidia/isaac-sim:6.1.0", stub)
        assert result.returncode == 0, result.stderr
        assert _auths(tmp_path / "cfg" / "config.json") == {_ECR, _HUB}
    finally:
        Path("/tmp/il_anon_attempts").unlink(missing_ok=True)


def test_unowned_config_is_never_modified(tmp_path: Path) -> None:
    """A config this action did not create belongs to the runner and stays intact."""
    result = _run_setup(tmp_path, "nvcr.io/nvidia/isaac-sim:6.1.0", _DENY_UNLESS_ANONYMOUS, owned=False)
    assert result.returncode == 0, result.stderr
    assert _auths(tmp_path / "cfg" / "config.json") == {"nvcr.io", _ECR, _HUB}


def test_already_checked_reference_is_not_probed_again(tmp_path: Path) -> None:
    """ecr-build-push-pull delegates to docker-build, so the probe must not repeat.

    The probes are network calls with backoff, and the first invocation already
    settled the outcome for this ref.
    """
    ref = "nvcr.io/nvidia/isaac-sim:6.1.0"
    stub = 'echo "the probe must not run again" >&2\nexit 1'
    result = _run_setup(
        tmp_path,
        ref,
        stub,
        extra_env={"SETUP_DOCKER_CONFIG_CHECKED_REF": ref},
    )
    assert result.returncode == 0, result.stderr
    assert "already checked" in result.stdout
    assert _auths(tmp_path / "cfg" / "config.json") == {"nvcr.io", _ECR, _HUB}


def test_registry_host_canonicalization() -> None:
    """Case, port, and Docker Hub spellings resolve to one registry identity."""
    helper = _load_helper()
    assert helper.registry_host("nvcr.io/nvidia/isaac-sim:6.1.0") == "nvcr.io"
    assert helper.registry_host("NVCR.IO/nvidia/isaac-sim:6.1.0") == "nvcr.io"
    assert helper.registry_host("nvcr.io:443/nvidia/isaac-sim:6.1.0") == "nvcr.io"
    assert helper.registry_host("ubuntu:24.04") == "index.docker.io"
    assert helper.registry_host("docker.io/library/ubuntu:24.04") == "index.docker.io"
    assert helper.registry_host("nvcr.io.example.com/team/image:1") == "nvcr.io.example.com"


def test_helper_reports_when_nothing_matched(tmp_path: Path) -> None:
    """No stored credential for the registry is reported rather than claimed as success."""
    config = _write_config(tmp_path / "config.json", [_ECR])
    result = subprocess.run(
        ["python3", str(_STRIP_HELPER), str(config), "nvcr.io/nvidia/isaac-sim:6.1.0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3
    assert _auths(config) == {_ECR}


def test_helper_also_clears_credential_helpers(tmp_path: Path) -> None:
    """A credHelpers entry would re-supply the credential, so it goes too."""
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"credHelpers": {"nvcr.io": "example"}, "auths": {"nvcr.io": {"auth": "redacted"}}}),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["python3", str(_STRIP_HELPER), str(config), "nvcr.io/nvidia/isaac-sim:6.1.0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    written = json.loads(config.read_text(encoding="utf-8"))
    assert not written.get("auths")
    assert not written.get("credHelpers")


def _run_hash(tmp_path: Path, image: str, version: str, stub: str) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    _write_stub_docker(bin_dir, stub)
    body = _step_body(_HASH_ACTION)
    marker = 'case "${ISAACSIM_VERSION}" in'
    lines = body.splitlines()
    start = next(index for index, line in enumerate(lines) if marker in line)
    end = next(index for index, line in enumerate(lines) if "base_image_uniq_id=" in line)
    script = tmp_path / "digest.sh"
    script.write_text(
        "\n".join(lines[start : end + 1]) + '\necho "RESULT=${base_image_uniq_id}"\n',
        encoding="utf-8",
    )
    env = {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "ISAACSIM_BASE_IMAGE": image,
        "ISAACSIM_VERSION": version,
    }
    return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=False)


def test_digest_pinned_tag_skips_the_network(tmp_path: Path) -> None:
    """A tag that already carries a digest is its own cache identity."""
    stub = 'echo "the manifest read must not run" >&2\nexit 1'
    result = _run_hash(tmp_path, "nvcr.io/example/isaac-sim", f"latest-develop@{_DIGEST}", stub)
    assert result.returncode == 0, result.stderr
    assert f"RESULT=nvcr.io/example/isaac-sim:latest-develop@{_DIGEST}:{_DIGEST}" in result.stdout


def test_malformed_digest_pin_fails(tmp_path: Path) -> None:
    """A malformed pin must not reach the cache key."""
    result = _run_hash(tmp_path, "nvcr.io/example/isaac-sim", "latest@sha256:short", "exit 0")
    assert result.returncode == 1
    assert "malformed digest pin" in result.stdout


def test_authorization_refusal_fails_without_retrying(tmp_path: Path) -> None:
    """An authorization refusal will not clear on its own, so it fails immediately."""
    stub = 'echo "denied: Access Denied" >&2\nexit 1'
    result = _run_hash(tmp_path, "nvcr.io/example/isaac-sim", "latest-release-6-1", stub)
    assert result.returncode == 1
    assert "cannot be read with the credentials available" in result.stdout
    assert "attempt 2/" not in result.stdout


def test_truncated_digest_is_rejected(tmp_path: Path) -> None:
    """A value that merely starts with sha256: is not a digest and must not be accepted.

    This distinguishes an exact-format check from a prefix check: a prefix check
    would accept the value below and fold it into the dependency-cache key.
    """
    stub = "echo '\"sha256:abc123\"'"
    result = _run_hash(tmp_path, "nvcr.io/example/isaac-sim", "latest-release-6-1", stub)
    assert result.returncode == 1
    assert "RESULT=" not in result.stdout


def test_trailing_output_after_a_digest_is_rejected(tmp_path: Path) -> None:
    """Extra stdout alongside a well-formed digest must not reach the cache key."""
    stub = f"echo '\"{_DIGEST} unexpected\"'"
    result = _run_hash(tmp_path, "nvcr.io/example/isaac-sim", "latest-release-6-1", stub)
    assert result.returncode == 1
    assert "RESULT=" not in result.stdout


def test_successful_read_ignores_stderr_noise(tmp_path: Path) -> None:
    """A warning emitted alongside a good digest is not concatenated into it."""
    stub = f'echo "buildx warning" >&2\necho \'"{_DIGEST}"\''
    result = _run_hash(tmp_path, "nvcr.io/example/isaac-sim", "latest-release-6-1", stub)
    assert result.returncode == 0, result.stderr
    assert f"RESULT=nvcr.io/example/isaac-sim:latest-release-6-1:{_DIGEST}" in result.stdout
