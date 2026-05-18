# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "tools" / "run_install_ci.py"
CONTAINER_RESULTS_XML = "/tmp/isaaclab-installci-results.xml"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_install_ci", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def _docker_args(**overrides):
    args = {
        "base_image": "ubuntu:24.04",
        "conda": False,
        "gpu": False,
        "no_cache": False,
        "no_pip_cache": True,
        "no_uv_cache": True,
        "pytest_args": ["--tb=short", "-sv", "-m", "uv"],
        "results_dir": None,
        "shell": False,
        "wheel": None,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def test_docker_results_dir_copies_junit_after_container_exit(tmp_path, monkeypatch):
    runner = _load_runner()
    docker_runs = []
    docker_side_effects = []

    def fake_build_image(*_args, **_kwargs):
        return 0

    def fake_call(cmd, timeout):
        docker_runs.append((cmd, timeout))
        return 0

    def fake_run_cmd(cmd, **kwargs):
        docker_side_effects.append((cmd, kwargs))
        if cmd[:2] == ["docker", "cp"]:
            Path(cmd[3]).write_text("<testsuite />\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(runner, "_build_image", fake_build_image)
    monkeypatch.setattr(runner, "_find_repo_root", lambda: REPO_ROOT)
    monkeypatch.setattr(runner, "run_cmd", fake_run_cmd)
    monkeypatch.setattr(runner.subprocess, "call", fake_call)

    rc = runner._cmd_docker(_docker_args(results_dir=str(tmp_path)))

    assert rc == 0
    assert len(docker_runs) == 1

    docker_run_cmd, timeout = docker_runs[0]
    assert timeout == 5400
    assert "--rm" not in docker_run_cmd
    assert "--name" in docker_run_cmd
    assert f"{tmp_path.resolve()}:/tmp/results" not in docker_run_cmd
    assert f"--junitxml={CONTAINER_RESULTS_XML}" in docker_run_cmd

    container_name = docker_run_cmd[docker_run_cmd.index("--name") + 1]
    host_results_xml = tmp_path / "results.xml"
    assert host_results_xml.read_text(encoding="utf-8") == "<testsuite />\n"

    side_effect_cmds = [cmd for cmd, _kwargs in docker_side_effects]
    assert [
        "docker",
        "cp",
        f"{container_name}:{CONTAINER_RESULTS_XML}",
        str(host_results_xml.resolve()),
    ] in side_effect_cmds
    assert ["docker", "rm", "-f", container_name] in side_effect_cmds


def test_docker_without_results_dir_uses_rm_and_no_junit_copy(monkeypatch):
    runner = _load_runner()
    docker_runs = []
    docker_side_effects = []

    monkeypatch.setattr(runner, "_build_image", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(runner, "_find_repo_root", lambda: REPO_ROOT)
    monkeypatch.setattr(runner, "run_cmd", lambda cmd, **kwargs: docker_side_effects.append((cmd, kwargs)))
    monkeypatch.setattr(runner.subprocess, "call", lambda cmd, timeout: docker_runs.append((cmd, timeout)) or 0)

    rc = runner._cmd_docker(_docker_args())

    assert rc == 0
    assert len(docker_runs) == 1

    docker_run_cmd, _timeout = docker_runs[0]
    assert "--rm" in docker_run_cmd
    assert "--name" not in docker_run_cmd
    assert all(not arg.startswith("--junitxml=") for arg in docker_run_cmd)
    assert docker_side_effects == []
