# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise workflow activation and cancellation without starting GPU jobs."""

import json
import os
import runpy
import signal
import subprocess
import sys
import textwrap
from contextlib import suppress
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows/build.yaml"


def labeled(login):
    return {"event": "labeled", "actor": {"login": login}}


def test_performance_workflow(tmp_path, subtests):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    assert "synchronize" in workflow.get("on", workflow.get(True, {}))["pull_request"]["types"]
    jobs = workflow["jobs"]
    activation = next(step for step in jobs["changes"]["steps"] if step.get("id") == "activation")
    removed_label = [labeled("maintainer"), {"event": "unlabeled"}]
    paginated = [[{"event": "commented"}], [labeled("isaaclab-bot[bot]")]]
    cases = [
        ("push", "push", "", "", [], True, True, False),
        ("actions bot", "pull_request", "synchronize", "", [[labeled("github-actions[bot]")]], False, False, True),
        ("removed human label", "pull_request", "synchronize", "", [removed_label], False, True, True),
        ("paginated bot label", "pull_request", "synchronize", "", paginated, False, True, True),
        ("API error", "pull_request", "synchronize", "", [], True, False, True),
        ("explicit rerun", "pull_request", "labeled", "ci:run-docker", [], True, True, True),
        ("other label", "pull_request", "labeled", "triage", [], True, True, False),
    ]
    data = {
        "script": activation["with"]["script"],
        "continue_on_error": activation.get("continue-on-error", False),
        "cases": cases,
        "runner": jobs["build"]["runs-on"].removeprefix("${{").removesuffix("}}"),
        "curobo": jobs["build-curobo"]["if"],
        "stop": next(step for step in jobs["build"]["steps"] if step.get("name") == "Explain missing PR activation"),
        "cancel": workflow["concurrency"]["cancel-in-progress"].removeprefix("${{").removesuffix("}}"),
    }
    runner = """
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const assert = require('node:assert/strict');
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
(async () => {
  for (const [name, eventName, action, label, pages, error, activated, expectedCancel] of input.cases) {
    const outputs = {};
    const context = {eventName, payload: {action, pull_request: {number: 42}, label: {name: label}},
                     repo: {owner: 'owner', repo: 'lab'}};
    const core = {setOutput: (k,v) => outputs[k] = v};
    const github = {rest: {issues: {listEvents: 'events'}}, paginate: {iterator: async function*() {
      if (error) throw new Error('network unavailable');
      for (const data of pages) yield {data};
    }}};
    try {
      await new AsyncFunction('context', 'github', 'core', input.script)(context, github, core);
    } catch (error) {
      if (!input.continue_on_error) throw new Error(name + ': ' + error.message);
    }
    const needs = {changes: {outputs: {activated: outputs.activated, should_run: 'true'}}};
    const event = {event_name: eventName, event: context.payload};
    const evaluate = code => new Function('needs', 'github', 'fromJSON', `return (${code})`)(needs, event, JSON.parse);
    assert.deepEqual(
      [evaluate(input.runner), evaluate(input.curobo), evaluate(input.stop.if), evaluate(input.cancel)],
      [activated ? ['self-hosted', 'gpu'] : ['ubuntu-latest'],
       activated && eventName !== 'push', !activated, expectedCancel],
      name);
  }
})();
"""
    result = subprocess.run(["node", "-e", runner], input=json.dumps(data), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    stopped = subprocess.run(["bash", "-e", "-c", data["stop"]["run"]], capture_output=True, text=True)
    assert stopped.returncode != 0, "Unactivated PR explanation step exited successfully"

    command = next(
        step["run"]
        for step in workflow["jobs"]["performance-smoke"]["steps"]
        if step.get("name") == "Run runtime benchmarks"
    )
    step_script = tmp_path / "workflow-step.sh"
    step_script.write_text(command)
    scripts = tmp_path / "tools/perf_smoke"
    scripts.mkdir(parents=True)
    (scripts / "run_benchmarks.sh").write_bytes(
        (WORKFLOW.parents[2] / "tools/perf_smoke/run_benchmarks.sh").read_bytes()
    )
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
        "PERF_CONTROLLER": str(tmp_path),
        "PERF_MEASUREMENT_ROOT": str(tmp_path / "measurement"),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "CI_IMAGE_TAG": "fixture-image",
    }
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
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
            started = next((json.loads(line) for line in process.stdout if line.startswith("{")), None)
            assert started is not None, (stop_signal.name, process.stderr.read())
            process.send_signal(stop_signal)
            # GitHub escalates cancellation after 7.5 s + 2.5 s, then kills the process tree.
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 128 + stop_signal, (stop_signal.name, stdout + stderr)
            assert not (tmp_path / "running-container").exists(), stop_signal.name
            try:
                os.kill(started["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError(f"{stop_signal.name}: benchmark process is still running")
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()

    registry = runpy.run_path(str(Path(__file__).with_name("test_registry_credential_fallback.py")))
    registry["check_registry"](tmp_path / "registry", subtests)
