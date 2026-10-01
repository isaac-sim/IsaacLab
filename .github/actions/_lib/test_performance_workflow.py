# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute PR activation and scheduling expressions without starting GPU jobs."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows/build.yaml"


@pytest.mark.parametrize(
    "event,action,pages,error,expected",
    [
        ("pull_request", "labeled", [], False, True),
        ("workflow_dispatch", "", [], False, True),
        ("push", "", [], False, True),
        ("pull_request", "synchronize", [[{"event": "opened"}]], False, False),
        (
            "pull_request",
            "synchronize",
            [[{"event": "labeled", "actor": {"login": "github-actions[bot]"}}]],
            False,
            False,
        ),
        (
            "pull_request",
            "synchronize",
            [[{"event": "labeled", "actor": {"login": "maintainer"}}, {"event": "unlabeled"}]],
            False,
            True,
        ),
        (
            "pull_request",
            "synchronize",
            [[{"event": "commented"}], [{"event": "labeled", "actor": {"login": "isaaclab-bot[bot]"}}]],
            False,
            True,
        ),
        ("pull_request", "synchronize", [], True, False),
    ],
)
def test_update_activation_routes_gpu_only_after_initial_label(tmp_path, event, action, pages, error, expected):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    jobs = workflow["jobs"]
    activation = next(step for step in jobs["changes"]["steps"] if step.get("id") == "activation")
    data = {
        "script": activation["with"]["script"],
        "context": {
            "eventName": event,
            "payload": {"action": action, "pull_request": {"number": 42}},
            "repo": {"owner": "owner", "repo": "lab"},
        },
        "pages": pages,
        "error": error,
        "runner": jobs["build"]["runs-on"].removeprefix("${{").removesuffix("}}"),
        "curobo": jobs["build-curobo"]["if"],
        "stop": next(
            step["run"] for step in jobs["build"]["steps"] if step.get("name") == "Explain missing PR activation"
        ),
    }
    runner = """
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const outputs = {}, calls = [];
const core = {setOutput: (k,v) => outputs[k] = v, notice: () => {}, warning: () => {}};
const github = {rest: {issues: {listEvents: 'events'}}, paginate: {iterator: async function*(route, args) {
  calls.push(args);
  if (input.error) throw new Error('network unavailable');
  for (const data of input.pages) yield {data};
}}};
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
(async () => {
  await new AsyncFunction('context', 'github', 'core', input.script)(input.context, github, core);
  const needs = {changes: {outputs: {activated: outputs.activated, should_run: 'true'}}};
  const runner = new Function('needs', 'fromJSON', `return (${input.runner})`)(needs, JSON.parse);
  const curobo = new Function('needs', 'github', `return (${input.curobo})`)(
    needs, {event_name: input.context.eventName});
  process.stdout.write(JSON.stringify({outputs, runner, curobo, calls}));
})();
"""
    result = subprocess.run(["node", "-e", runner], input=json.dumps(data), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["outputs"]["activated"] == str(expected).lower()
    assert observed["runner"] == (["self-hosted", "gpu"] if expected else ["ubuntu-latest"])
    assert observed["curobo"] is (expected and event != "push")
    if event == "pull_request" and action == "synchronize":
        assert observed["calls"] == [
            {"owner": "owner", "repo": "lab", "issue_number": 42, "per_page": 100, "request": {"timeout": 60000}}
        ]
    else:
        assert observed["calls"] == []
    if not expected:
        stopped = subprocess.run(["bash", "-e", "-c", data["stop"]], capture_output=True, text=True)
        assert stopped.returncode != 0
        assert "No GPU build or benchmark was started" in stopped.stdout


@pytest.mark.parametrize(
    "action,label,expected",
    [("synchronize", "", True), ("labeled", "ci:run-docker", True), ("labeled", "triage", False), ("", "", False)],
)
def test_only_updates_and_explicit_reruns_cancel_running_ci(action, label, expected):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    expression = workflow["concurrency"]["cancel-in-progress"].removeprefix("${{").removesuffix("}}")
    data = {"expression": expression, "github": {"event": {"action": action, "label": {"name": label}}}}
    script = (
        "const x=JSON.parse(require('node:fs').readFileSync(0,'utf8'));"
        "process.stdout.write(JSON.stringify(new Function('github', `return (${x.expression})`)(x.github)));"
    )
    result = subprocess.run(["node", "-e", script], input=json.dumps(data), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) is expected
