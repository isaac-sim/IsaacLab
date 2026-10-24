# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check registry credentials and image digests with local Docker stubs.

The actions resolve Bash from PATH; compute-deps-hash uses mapfile (Bash 4+).
On macOS, prepend the directory containing a newer Bash to PATH when running pytest.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
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
    if "nvcr.io" in registries:
        config["credHelpers"] = {"nvcr.io": "example"}
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _auths(path: Path) -> set[str]:
    return set((json.loads(path.read_text(encoding="utf-8")).get("auths") or {}).keys())


def _write_stub_docker(bin_dir: Path, body: str) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    trace = bin_dir.parent / "docker-calls.log"
    trace.write_text("")
    body = f"printf '%s\\n' \"$*\" >> {shlex.quote(str(trace))}\n{body}"
    for name, source in {"docker": body, "sleep": "exit 0"}.items():
        script = bin_dir / name
        script.write_text(f"#!/usr/bin/env bash\n{source}\n", encoding="utf-8")
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
    result = subprocess.run(["bash", str(body)], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return result


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
    result = _run_setup(tmp_path, "nvcr.io/nvidia/isaac-sim:6.2.0", _DENY_UNLESS_ANONYMOUS)
    assert _auths(tmp_path / "cfg" / "config.json") == {_ECR, _HUB}
    assert not json.loads((tmp_path / "cfg" / "config.json").read_text()).get("credHelpers")
    assert "::warning::" not in result.stdout


def test_non_nvcr_reference_is_left_alone(tmp_path: Path) -> None:
    """A Docker Hub base image keeps its credential, which is also pull-rate budget."""
    _run_setup(tmp_path, "ubuntu:24.04", _DENY_UNLESS_ANONYMOUS)
    assert _auths(tmp_path / "cfg" / "config.json") == {"nvcr.io", _ECR, _HUB}


def test_readable_image_keeps_credentials(tmp_path: Path) -> None:
    """A private image the credentials can read must not trigger the fallback."""
    _run_setup(tmp_path, "nvcr.io/example/isaac-sim:latest", "exit 0")
    assert _auths(tmp_path / "cfg" / "config.json") == {"nvcr.io", _ECR, _HUB}


def test_transient_failure_keeps_credentials(tmp_path: Path) -> None:
    """A registry outage is not an authorization refusal, so nothing is dropped."""
    stub = 'echo "error: received unexpected HTTP status: 503" >&2\nexit 1'
    result = _run_setup(tmp_path, "nvcr.io/nvidia/isaac-sim:6.2.0", stub)
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
  counter="$ANON_ATTEMPTS"
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
    _run_setup(
        tmp_path,
        "nvcr.io/nvidia/isaac-sim:6.2.0",
        stub,
        extra_env={"ANON_ATTEMPTS": str(tmp_path / "anonymous-attempts")},
    )
    assert _auths(tmp_path / "cfg" / "config.json") == {_ECR, _HUB}


def test_unowned_config_is_never_modified(tmp_path: Path) -> None:
    """A config this action did not create belongs to the runner and stays intact."""
    _run_setup(tmp_path, "nvcr.io/nvidia/isaac-sim:6.2.0", _DENY_UNLESS_ANONYMOUS, owned=False)
    assert _auths(tmp_path / "cfg" / "config.json") == {"nvcr.io", _ECR, _HUB}


def test_already_checked_reference_is_not_probed_again(tmp_path: Path) -> None:
    """ecr-build-push-pull delegates to docker-build, so the probe must not repeat.

    The probes are network calls with backoff, and the first invocation already
    settled the outcome for this ref.
    """
    ref = "nvcr.io/nvidia/isaac-sim:6.2.0"
    stub = 'echo "the probe must not run again" >&2\nexit 1'
    _run_setup(
        tmp_path,
        ref,
        stub,
        extra_env={"SETUP_DOCKER_CONFIG_CHECKED_REF": ref},
    )
    assert (tmp_path / "docker-calls.log").read_text() == ""
    assert _auths(tmp_path / "cfg" / "config.json") == {"nvcr.io", _ECR, _HUB}


def test_recreated_config_rechecks_previously_checked_image(tmp_path: Path) -> None:
    """Cleanup must not leave a cached probe valid for newly copied credentials."""
    runner_config = tmp_path / "home" / ".docker" / "config.json"
    runner_config.parent.mkdir(parents=True)
    _write_config(runner_config, ["nvcr.io", _ECR, _HUB])
    removed_config = tmp_path / "removed-config"
    ref = "nvcr.io/nvidia/isaac-sim:6.2.0"
    _run_setup(
        tmp_path,
        ref,
        _DENY_UNLESS_ANONYMOUS,
        extra_env={
            "DOCKER_CONFIG": str(removed_config),
            "SETUP_DOCKER_CONFIG_OWNED": str(removed_config),
            "SETUP_DOCKER_CONFIG_CHECKED_REF": ref,
            "TMPDIR": str(tmp_path),
        },
    )
    updates = dict(line.split("=", 1) for line in (tmp_path / "github-env").read_text().splitlines())
    recreated_config = Path(updates["DOCKER_CONFIG"]) / "config.json"
    assert _auths(recreated_config) == {_ECR, _HUB}
    assert _auths(runner_config) == {"nvcr.io", _ECR, _HUB}


def test_registry_host_canonicalization() -> None:
    """Case, port, and Docker Hub spellings resolve to one registry identity."""
    helper = _load_helper()
    assert helper.registry_host("NVCR.IO/nvidia/isaac-sim:6.2.0") == "nvcr.io"
    assert helper.registry_host("nvcr.io:443/nvidia/isaac-sim:6.2.0") == "nvcr.io"
    assert helper.registry_host("ubuntu:24.04") == "index.docker.io"
    assert helper.registry_host("docker.io/library/ubuntu:24.04") == "index.docker.io"
    assert helper.registry_host("nvcr.io.example.com/team/image:1") == "nvcr.io.example.com"


def test_helper_reports_when_nothing_matched(tmp_path: Path) -> None:
    """No stored credential for the registry is reported rather than claimed as success."""
    config = _write_config(tmp_path / "config.json", [_ECR])
    result = subprocess.run(
        ["python3", str(_STRIP_HELPER), str(config), "nvcr.io/nvidia/isaac-sim:6.2.0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3
    assert _auths(config) == {_ECR}


def _run_hash(
    tmp_path: Path, image: str, version: str, stub: str
) -> tuple[subprocess.CompletedProcess, dict[str, str]]:
    bin_dir = tmp_path / "bin"
    _write_stub_docker(bin_dir, stub)
    checkout = tmp_path / "checkout"
    if not checkout.exists():
        checkout.mkdir()
        subprocess.run(["git", "init", "--quiet", str(checkout)], check=True, capture_output=True)
        (checkout / "Dockerfile").write_text("FROM scratch\n")
        subprocess.run(["git", "add", "Dockerfile"], cwd=checkout, check=True, capture_output=True)
    script = tmp_path / "hash.sh"
    script.write_text(_step_body(_HASH_ACTION), encoding="utf-8")
    output = tmp_path / "github-output"
    output.write_text("")
    env = {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "DOCKERFILE_PATH": "Dockerfile",
        "ISAACSIM_BASE_IMAGE": image,
        "ISAACSIM_VERSION": version,
        "TARGET_PLATFORM": "linux/amd64",
        "GITHUB_OUTPUT": str(output),
    }
    result = subprocess.run(["bash", str(script)], cwd=checkout, env=env, capture_output=True, text=True, check=False)
    return result, dict(line.split("=", 1) for line in output.read_text().splitlines())


def test_digest_pinned_tag_skips_the_network(tmp_path: Path) -> None:
    hashes = []
    for digest in (_DIGEST, "sha256:" + "b" * 64):
        result, output = _run_hash(tmp_path, "nvcr.io/example/isaac-sim", f"latest-develop@{digest}", "exit 1")
        assert result.returncode == 0, result.stderr
        hashes.append(output["hash"])
        assert (tmp_path / "docker-calls.log").read_text() == ""
    assert hashes[0] != hashes[1]


def test_malformed_digest_pin_fails(tmp_path: Path) -> None:
    result, output = _run_hash(tmp_path, "nvcr.io/example/isaac-sim", "latest@sha256:short", "exit 0")
    assert result.returncode == 1
    assert "malformed digest pin" in result.stdout
    assert output == {}


def test_authorization_refusal_fails_without_retrying(tmp_path: Path) -> None:
    stub = 'echo "denied: Access Denied" >&2\nexit 1'
    result, output = _run_hash(tmp_path, "nvcr.io/example/isaac-sim", "latest-release-6-2", stub)
    assert result.returncode == 1
    assert "cannot be read with the credentials available" in result.stdout
    calls = (tmp_path / "docker-calls.log").read_text().splitlines()
    assert len(calls) == 1
    assert calls[0].startswith("buildx imagetools inspect ")
    assert output == {}


def test_truncated_digest_is_rejected(tmp_path: Path) -> None:
    stub = "echo '\"sha256:abc123\"'"
    result, output = _run_hash(tmp_path, "nvcr.io/example/isaac-sim", "latest-release-6-2", stub)
    assert result.returncode == 1
    assert output == {}


def test_trailing_output_after_a_digest_is_rejected(tmp_path: Path) -> None:
    stub = f"echo '\"{_DIGEST} unexpected\"'"
    result, output = _run_hash(tmp_path, "nvcr.io/example/isaac-sim", "latest-release-6-2", stub)
    assert result.returncode == 1
    assert output == {}


def test_successful_read_ignores_stderr_noise(tmp_path: Path) -> None:
    hashes = []
    for digest, warning in (
        (_DIGEST, ""),
        (_DIGEST, 'echo "buildx warning" >&2\n'),
        ("sha256:" + "b" * 64, 'echo "buildx warning" >&2\n'),
    ):
        stub = warning + f"echo '\"{digest}\"'"
        result, output = _run_hash(tmp_path, "nvcr.io/example/isaac-sim", "latest-release-6-2", stub)
        assert result.returncode == 0, result.stderr
        hashes.append(output["hash"])
    assert hashes[0] == hashes[1]
    assert hashes[1] != hashes[2]
