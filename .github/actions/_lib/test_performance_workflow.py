# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise workflow activation and cancellation without starting GPU jobs."""

import json
import os
import signal
import subprocess
import sys
import textwrap
from contextlib import suppress
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows/build.yaml"


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def labeled(login):
    return {"event": "labeled", "actor": {"login": login}}


@pytest.mark.parametrize(
    "event,action,label,pages,error,expected,cancel",
    [
        ("pull_request", "labeled", "", [], True, True, False),
        ("push", "", "", [], True, True, False),
        ("pull_request", "synchronize", "", [[labeled("github-actions[bot]")]], False, False, True),
        ("pull_request", "synchronize", "", [[labeled("maintainer"), {"event": "unlabeled"}]], False, True, True),
        (
            "pull_request",
            "synchronize",
            "",
            [[{"event": "commented"}], [labeled("isaaclab-bot[bot]")]],
            False,
            True,
            True,
        ),
        ("pull_request", "synchronize", "", [], True, False, True),
        ("pull_request", "labeled", "ci:run-docker", [], True, True, True),
        ("pull_request", "labeled", "triage", [], True, True, False),
    ],
)
def test_workflow_activation_routing_and_cancellation(workflow, event, action, label, pages, error, expected, cancel):
    jobs = workflow["jobs"]
    activation = next(step for step in jobs["changes"]["steps"] if step.get("id") == "activation")
    data = {
        "script": activation["with"]["script"],
        "continue_on_error": activation.get("continue-on-error", False),
        "context": {
            "eventName": event,
            "payload": {"action": action, "pull_request": {"number": 42}, "label": {"name": label}},
            "repo": {"owner": "owner", "repo": "lab"},
        },
        "pages": pages,
        "error": error,
        "runner": jobs["build"]["runs-on"].removeprefix("${{").removesuffix("}}"),
        "curobo": jobs["build-curobo"]["if"],
        "stop": next(step for step in jobs["build"]["steps"] if step.get("name") == "Explain missing PR activation"),
        "cancel": workflow["concurrency"]["cancel-in-progress"].removeprefix("${{").removesuffix("}}"),
    }
    runner = """
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const outputs = {};
const core = {setOutput: (k,v) => outputs[k] = v};
const github = {rest: {issues: {listEvents: 'events'}}, paginate: {iterator: async function*() {
  if (input.error) throw new Error('network unavailable');
  for (const data of input.pages) yield {data};
}}};
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
(async () => {
  try {
    await new AsyncFunction('context', 'github', 'core', input.script)(input.context, github, core);
  } catch (error) {
    if (!input.continue_on_error) throw error;
  }
  const needs = {changes: {outputs: {activated: outputs.activated, should_run: 'true'}}};
  const runner = new Function('needs', 'fromJSON', `return (${input.runner})`)(needs, JSON.parse);
  const event = {event_name: input.context.eventName, event: input.context.payload};
  const curobo = new Function('needs', 'github', `return (${input.curobo})`)(needs, event);
  const stop = new Function('needs', `return (${input.stop.if})`)(needs);
  const cancel = new Function('github', `return (${input.cancel})`)(event);
  process.stdout.write(JSON.stringify({runner, curobo, stop, cancel}));
})();
"""
    result = subprocess.run(["node", "-e", runner], input=json.dumps(data), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["runner"] == (["self-hosted", "gpu"] if expected else ["ubuntu-latest"])
    assert observed["curobo"] is (expected and event != "push")
    assert observed["stop"] is (not expected)
    assert observed["cancel"] is cancel
    if observed["stop"]:
        stopped = subprocess.run(["bash", "-e", "-c", data["stop"]["run"]], capture_output=True, text=True)
        assert stopped.returncode != 0


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_cancelled_workflow_removes_container_and_stops_benchmark(workflow, tmp_path, stop_signal):
    command = next(
        step["run"]
        for step in workflow["jobs"]["performance-smoke"]["steps"]
        if step.get("name") == "Run runtime benchmarks"
    )
    step_script = tmp_path / "workflow-step.sh"
    step_script.write_text(command)
    (tmp_path / "benchmark-legs.tsv").write_text("fixture|Fixture-Task|1|600|\n")
    executables = tmp_path / "bin"
    executables.mkdir()
    docker = executables / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """
            import json
            import os
            import sys
            import time
            from pathlib import Path

            workspace = Path(os.environ["GITHUB_WORKSPACE"])
            marker = workspace / "running-container"
            if sys.argv[1] == "run":
                name = sys.argv[sys.argv.index("--name") + 1]
                marker.write_text(name)
                print(json.dumps({"pid": os.getpid()}), flush=True)
                while marker.exists():
                    time.sleep(0.01)
            elif sys.argv[1] == "rm" and marker.exists():
                assert sys.argv[-1] == marker.read_text()
                marker.unlink()
            """
        )
    )
    docker.chmod(0o755)
    # Exercise Bash cancellation and Docker cleanup; the timeout wrapper is replaced by exec.
    timeout = executables / "timeout"
    timeout.write_text('#!/bin/bash\nshift\nexec "$@"\n')
    timeout.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{executables}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_WORKSPACE": str(tmp_path),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "CI_IMAGE_TAG": "fixture-image",
    }
    process = subprocess.Popen(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", str(step_script)],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        started = None
        for line in process.stdout:
            if line.startswith("{"):
                started = json.loads(line)
                break
        assert started is not None, process.stderr.read()
        process.send_signal(stop_signal)
        # GitHub escalates cancellation after 7.5 s + 2.5 s, then kills the process tree.
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 128 + stop_signal, stdout + stderr
        assert not (tmp_path / "running-container").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(started["pid"], 0)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
