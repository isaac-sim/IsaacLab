# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise workflow activation, source mounts, and cancellation without starting GPU jobs."""

import json
import os
import signal
import subprocess
import sys
import textwrap
from contextlib import suppress
from pathlib import Path

import yaml

_WORKFLOW = Path(__file__).resolve().parents[2] / "workflows/build.yaml"
_RUN_CI_WORKFLOW = Path(__file__).resolve().parents[2] / "workflows/run-docker-ci.yml"


def _labeled(login: str) -> dict[str, str | dict[str, str]]:
    return {"event": "labeled", "actor": {"login": login}}


def test_ci_activation_and_runner_selection() -> None:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert "synchronize" in workflow.get("on", workflow.get(True, {}))["pull_request"]["types"]
    jobs = workflow["jobs"]
    activation = next(step for step in jobs["changes"]["steps"] if step.get("id") == "activation")
    removed_label = [_labeled("maintainer"), {"event": "unlabeled"}]
    paginated = [[{"event": "commented"}], [{"event": "labeled", "actor": None}, _labeled("isaaclab-bot[bot]")]]
    ignored = [[_labeled("github-actions[bot]"), {"event": "commented", "actor": {"login": "human"}}]]
    cases = [
        ("push", "push", "", "", [], True, True, False, True),
        ("actions bot", "pull_request", "synchronize", "", ignored, False, False, True, False),
        ("removed human label", "pull_request", "synchronize", "", [removed_label], False, True, True, False),
        ("paginated bot label", "pull_request", "synchronize", "", paginated, False, True, True, False),
        ("API error", "pull_request", "synchronize", "", [], True, False, True, False),
        ("explicit rerun", "pull_request", "labeled", "ci:run-docker", [], True, True, True, False),
        ("ovstage rerun", "pull_request", "labeled", "ci:run-ovstage", [], True, True, True, True),
        ("other label", "pull_request", "labeled", "triage", [], True, True, False, False),
    ]
    data = {
        "script": activation["with"]["script"],
        "continue_on_error": activation.get("continue-on-error", False),
        "cases": cases,
        "runner": jobs["build"]["runs-on"].removeprefix("${{").removesuffix("}}"),
        "curobo": jobs["build-curobo"]["if"],
        "ovstage": jobs["test-rendering-correctness-kitless-ovstage"]["if"],
        "stop": next(step for step in jobs["build"]["steps"] if step.get("name") == "Explain missing PR activation"),
        "cancel": workflow["concurrency"]["cancel-in-progress"].removeprefix("${{").removesuffix("}}"),
    }
    runner = """
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const assert = require('node:assert/strict');
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
(async () => {
  for (const row of input.cases) {
    const [name, eventName, action, label, pages, error, activated, expectedCancel, expectedOvstage] = row;
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
    const needs = {build: {result: 'success'}, changes: {outputs: {activated: outputs.activated, should_run: 'true'}}};
    const event = {event_name: eventName, event: context.payload};
    const evaluate = code => new Function('needs', 'github', 'fromJSON', `return (${code})`)(needs, event, JSON.parse);
    assert.deepEqual(
      [
        evaluate(input.runner),
        evaluate(input.curobo),
        evaluate(input.stop.if),
        evaluate(input.cancel),
        evaluate(input.ovstage),
      ],
      [activated ? ['self-hosted', 'gpu'] : ['ubuntu-latest'],
       activated, !activated, expectedCancel, expectedOvstage],
      name);
  }
})();
"""
    result = subprocess.run(["node", "-e", runner], input=json.dumps(data), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    stopped = subprocess.run(["bash", "-e", "-c", data["stop"]["run"]], capture_output=True, text=True)
    assert stopped.returncode != 0, "Unactivated PR explanation step exited successfully"


def test_parallel_baseline_route() -> None:
    route = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["performance-regression-route"]["steps"][0]
    script = """
const source = require('node:fs').readFileSync(0, 'utf8');
const assert = require('node:assert/strict');
const run = new (Object.getPrototypeOf(async function() {}).constructor)('github', 'context', source);
const context = {repo: {owner: 'owner', repo: 'lab'}, payload: {pull_request: {number: 42, head: {ref: 'feature'}}}};
process.env.BASE_COMMIT = 'base';
(async () => {
  for (const [label, runCount, artifacts, artifactCount, expected] of [
    ['fresh', 1, [], 0, true], ['reusable on retry', 1, [{expired: false}], 1, false],
    ['expired', 1, [{expired: true}], 1, true], ['partial run list', 2, [], 0, false],
    ['partial artifact list', 1, [], 1, false],
  ]) {
    const actions = {
      listWorkflowRuns: async ({branch}) => {
        assert.equal(branch, 'feature');
        return {data: {workflow_runs: [{run_attempt: 2}], total_count: runCount}};
      },
      listArtifactsForRepo: async ({name}) => {
        assert.match(name, /^performance-pr-baseline-42-base-[12]$/);
        return {data: name.endsWith('-2') ? {artifacts, total_count: artifactCount} : {artifacts: [], total_count: 0}};
      },
    };
    assert.equal(await run({rest: {actions}}, context), expected, label);
  }
})();
"""
    subprocess.run(["node", "-e", script], input=route["with"]["script"], text=True, check=True)


def test_run_docker_ci_command_options(tmp_path: Path) -> None:
    workflow = yaml.safe_load(_RUN_CI_WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["request-docker-ci"]
    authorize_script = next(step["run"] for step in job["steps"] if step.get("id") == "authorize")
    trigger_script = next(step["run"] for step in job["steps"] if step.get("name") == "Trigger Docker CI")

    def gate_allows(is_pr: bool, body: str) -> bool:
        check = subprocess.run(
            [
                "node",
                "-e",
                r"""
const [rawExpr, isPr, body] = process.argv.slice(1);
// GitHub Actions single-quoted literals have no backslash escapes; only '' escapes '.
const jsExpr = rawExpr.replace(/'((?:[^']|'')*)'/g, (_, s) => JSON.stringify(s.replace(/''/g, "'")));
const contains = (search, item) => Array.isArray(search)
  ? search.some(v => String(v).toLowerCase() === String(item ?? '').toLowerCase())
  : String(search ?? '').toLowerCase().includes(String(item ?? '').toLowerCase());
const ok = new Function('github', 'fromJSON', 'contains', `return Boolean(${jsExpr})`)(
  {event: {issue: isPr === '1' ? {pull_request: {}} : {}, comment: {body}}}, JSON.parse, contains
);
process.stdout.write(ok ? '1' : '0');
""",
                job["if"],
                "1" if is_pr else "0",
                body,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return check.stdout == "1"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    api_log = tmp_path / "gh-calls.jsonl"
    gh = bin_dir / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            f"""
            import json
            import sys
            from pathlib import Path

            log_path = Path({str(api_log)!r})
            args = sys.argv[1:]
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(args) + "\\n")
            if "repos/owner/lab/pulls/42" in args:
                jq_expr = args[args.index("--jq") + 1]
                if "@tsv" in jq_expr:
                    print("author\\topen\\tabcdef1234567890")
                else:
                    print("abcdef1234567890")
            """
        ),
        encoding="utf-8",
    )
    gh.chmod(0o755)

    def run_authorize(
        comment_body: str, *, run_trigger: bool = False
    ) -> tuple[int, str, dict[str, str], list[list[str]]]:
        api_log.unlink(missing_ok=True)
        gh_output = tmp_path / "github_output"
        gh_summary = tmp_path / "github_summary"
        gh_output.write_text("", encoding="utf-8")
        gh_summary.write_text("", encoding="utf-8")
        env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "COMMENT_AUTHOR": "author",
            "COMMENT_BODY": comment_body,
            "GH_TOKEN": "token",
            "GITHUB_OUTPUT": str(gh_output),
            "GITHUB_STEP_SUMMARY": str(gh_summary),
            "PR_NUMBER": "42",
            "REPOSITORY": "owner/lab",
        }
        auth = subprocess.run(["bash", "-c", authorize_script], env=env, capture_output=True, text=True)
        outputs = dict(line.split("=", 1) for line in gh_output.read_text(encoding="utf-8").splitlines() if "=" in line)
        if auth.returncode == 0 and run_trigger:
            trig_env = {
                **env,
                "COMMAND": outputs["command"],
                "EXPECTED_HEAD_SHA": outputs["head_sha"],
                "LABEL": outputs["label"],
            }
            trig = subprocess.run(["bash", "-c", trigger_script], env=trig_env, capture_output=True, text=True)
            assert trig.returncode == 0, trig.stdout + trig.stderr
        calls = (
            [json.loads(line) for line in api_log.read_text(encoding="utf-8").splitlines()] if api_log.exists() else []
        )
        return auth.returncode, auth.stdout + auth.stderr, outputs, calls

    assert not gate_allows(False, "run-ci")
    for skipped_comment in (
        "LGTM",
        "`run-ci`",
        "  run-ci",
        "run-ci is already green",
        "run-ci\nplease",
        "run-ci-ovstage",
    ):
        assert not gate_allows(True, skipped_comment), skipped_comment
    code, _, _, calls = run_authorize("run-ci is already green")
    assert code != 0 and calls == []

    for body, expected_label, expected_cmd, run_trigger in [
        ("run-ci", "ci:run-docker", "run-ci", True),
        ("run-ci\r\n", "ci:run-docker", "run-ci", False),
        ("run-ci ovstage\r\n", "ci:run-ovstage", "run-ci ovstage", True),
        ("run-ci OVSTAGE\n", "ci:run-ovstage", "run-ci ovstage", False),
    ]:
        assert gate_allows(True, body), body
        code, output, outputs, calls = run_authorize(body, run_trigger=run_trigger)
        assert code == 0, (body, output)
        assert outputs == {"head_sha": "abcdef1234567890", "label": expected_label, "command": expected_cmd}
        if run_trigger:
            label_calls = [c for c in calls if any("labels" in arg for arg in c)]
            assert [c[c.index("--method") + 1] for c in label_calls] == ["DELETE", "POST", "DELETE"]
            assert f"repos/owner/lab/issues/42/labels/{expected_label}" in label_calls[0]
            assert f"labels[]={expected_label}" in label_calls[1]
            assert f"repos/owner/lab/issues/42/labels/{expected_label}" in label_calls[2]


def test_benchmark_cancellation(tmp_path: Path) -> None:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    command = next(
        step["run"]
        for step in workflow["jobs"]["performance-regression"]["steps"]
        if step.get("name") == "Run runtime benchmarks"
    )
    step_script = tmp_path / "workflow-step.sh"
    step_script.write_text(command, encoding="utf-8")
    controller = tmp_path / "controller"
    scripts = controller / "tools/performance_regression"
    scripts.mkdir(parents=True)
    (scripts / "run_benchmarks.sh").write_bytes(
        (_WORKFLOW.parents[2] / "tools/performance_regression/run_benchmarks.sh").read_bytes()
    )
    (scripts / "source_revision.py").write_text("", encoding="utf-8")
    (controller / "benchmark-legs.tsv").write_text("fixture|Fixture-Task|1|600|\n", encoding="utf-8")
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
            if sys.argv[1] == "run" and "--name" in sys.argv:
                name = sys.argv[sys.argv.index("--name") + 1]
                marker.write_text(name, encoding="utf-8")
                print(json.dumps({"pid": os.getpid(), "argv": sys.argv[1:]}), flush=True)
                while marker.exists():
                    time.sleep(0.01)
            elif sys.argv[1] == "rm" and marker.exists():
                assert sys.argv[-1] == marker.read_text(encoding="utf-8")
                marker.unlink()
            """
        ),
        encoding="utf-8",
    )
    docker.chmod(0o755)
    # Exercise Bash cancellation and Docker cleanup; the timeout wrapper is replaced by exec.
    timeout = executables / "timeout"
    timeout.write_text('#!/bin/bash\nshift\nexec "$@"\n', encoding="utf-8")
    timeout.chmod(0o755)
    df = executables / "df"
    df.write_text('#!/bin/bash\nprintf "Filesystem 1B-blocks\\nshm 1073741824\\n"\n', encoding="utf-8")
    df.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{executables}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_WORKSPACE": str(tmp_path),
        "PERF_CONTROLLER": str(controller),
        "PERF_MEASUREMENT_ROOT": str(controller / "measurement"),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "CI_IMAGE_TAG": "fixture-image",
    }
    expected_mounts = {
        f"{tmp_path}/{path}:/workspace/isaaclab/{path}:{mode}"
        for path, mode in (("source", "rw"), ("scripts", "ro"), ("apps", "ro"))
    }
    for path in ("source/isaaclab/isaaclab/benchmark", "source/isaaclab/isaaclab/utils/timer.py"):
        expected_mounts.add(f"{controller}/measurement/{path}:/workspace/isaaclab/{path}:ro")
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
            docker_args = started["argv"][: started["argv"].index(environment["CI_IMAGE_TAG"])]
            assert "--ipc=private" in docker_args and "--ipc=host" not in docker_args
            assert docker_args[docker_args.index("--shm-size") + 1] == "1073741824"
            mounts = {docker_args[index + 1] for index, arg in enumerate(docker_args) if arg == "-v"}
            assert expected_mounts <= mounts, (
                f"Missing or incorrect Docker source mounts: {sorted(expected_mounts - mounts)};"
                f" observed: {sorted(mounts)}"
            )
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
