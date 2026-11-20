# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise workflow activation, source mounts, and cancellation without starting GPU jobs."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import textwrap
import types
from contextlib import suppress
from pathlib import Path

import pytest
import tomllib
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW = _REPO_ROOT / ".github/workflows/build.yaml"
_NIGHTLY_IMAGE_WORKFLOW = _REPO_ROOT / ".github/workflows/nightly-isaacsim-image.yml"
_NIGHTLY_OVSTAGE_WORKFLOW = _REPO_ROOT / ".github/workflows/nightly-ovstage.yml"
_POSTMERGE_SUBSETS = _REPO_ROOT / ".github/test-subsets/postmerge-rendering.toml"
_RUN_CI_WORKFLOW = _REPO_ROOT / ".github/workflows/run-docker-ci.yml"
_RUN_TESTS_SCRIPT = _REPO_ROOT / ".github/actions/run-tests/run_tests.sh"


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
            if "repos/owner/lab/pulls" in args and "GET" in args:
                print("42")
            elif "repos/owner/lab/pulls/42" in args and "PATCH" in args:
                print("https://github.com/owner/lab/pull/42")
            elif "repos/owner/lab/pulls/42" in args:
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

    nightly_image = yaml.safe_load(_NIGHTLY_IMAGE_WORKFLOW.read_text(encoding="utf-8"))
    open_pr_script = next(
        step["run"]
        for step in nightly_image["jobs"]["update-image-pin"]["steps"]
        if step.get("name") == "Open or refresh the draft PR"
    )
    api_log.unlink(missing_ok=True)
    nightly_run = subprocess.run(
        ["bash", "-c", open_pr_script],
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "GH_TOKEN": "token",
            "REPOSITORY": "owner/lab",
            "TARGET_BRANCH": "main",
            "UPDATE_BRANCH": "ci/nightly-isaacsim-image-update",
            "IMAGE": "nvcr.io/org/isaac-sim",
            "SOURCE_TAG": "latest-release-6-3",
            "CURRENT_PIN": "latest-release-6-3@sha256:0",
            "CANDIDATE_PIN": "latest-release-6-3@sha256:1",
            "DIGEST": "sha256:111111111111",
            "BRANCH_CHANGED": "true",
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "github_summary"),
        },
        capture_output=True,
        text=True,
    )
    assert nightly_run.returncode == 0, nightly_run.stdout + nightly_run.stderr
    comment_call = next(
        c
        for c in (json.loads(line) for line in api_log.read_text(encoding="utf-8").splitlines())
        if "repos/owner/lab/issues/42/comments" in c
    )
    nightly_comment = next(arg.removeprefix("body=") for arg in comment_call if arg.startswith("body="))
    assert gate_allows(True, nightly_comment), nightly_comment
    code, output, outputs, _ = run_authorize(nightly_comment)
    assert code == 0 and outputs["label"] == "ci:run-ovstage", (nightly_comment, output, outputs)


def test_ovstage_ci_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract_spec = importlib.util.spec_from_file_location(
        "isaaclab.renderers.output_contract",
        _REPO_ROOT / "source/isaaclab/isaaclab/renderers/output_contract.py",
    )
    assert contract_spec and contract_spec.loader
    contract_mod = importlib.util.module_from_spec(contract_spec)
    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.renderers", types.ModuleType("isaaclab.renderers"))
    monkeypatch.setitem(sys.modules, "isaaclab.renderers.output_contract", contract_mod)
    monkeypatch.delitem(sys.modules, "rendering_cases", raising=False)
    contract_spec.loader.exec_module(contract_mod)
    monkeypatch.syspath_prepend(str(_REPO_ROOT / "source/_isaaclab_testing/test/rendering"))
    from rendering_cases import KITLESS_RENDERING_CASES

    monkeypatch.delitem(sys.modules, "rendering_cases", raising=False)

    expected_ovstage_files = {
        f"test_rendering_kitless_{case.physics}_{case.renderer}.py"
        for stage, case in KITLESS_RENDERING_CASES
        if stage == "ovstage"
    }
    assert expected_ovstage_files == {
        "test_rendering_kitless_newton_ovrtx.py",
        "test_rendering_kitless_ovphysx_ovrtx.py",
    }

    subsets = tomllib.loads(_POSTMERGE_SUBSETS.read_text(encoding="utf-8"))
    assert "rendering-correctness-kitless-ovstage" not in subsets
    assert {"rendering-correctness", "rendering-correctness-kitless-legacy", "ovrtx-cache-warm"} <= subsets.keys()

    build_workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    nightly_workflow = yaml.safe_load(_NIGHTLY_OVSTAGE_WORKFLOW.read_text(encoding="utf-8"))

    # Nightly runs daily and manually without change-detection, image-bump gating, or full-pipeline fan-out.
    nightly_on = nightly_workflow.get("on", nightly_workflow.get(True, {}))
    assert nightly_on["schedule"] and "workflow_dispatch" in nightly_on
    assert set(nightly_workflow["jobs"]) == {"config", "build", "test-rendering-correctness-kitless-ovstage"}
    assert "if" not in nightly_workflow["jobs"]["config"]
    assert "if" not in nightly_workflow["jobs"]["build"]
    assert nightly_workflow["jobs"]["test-rendering-correctness-kitless-ovstage"]["if"] == (
        "needs.build.result == 'success'"
    )
    assert all("continue-on-error" not in job for job in nightly_workflow["jobs"].values())

    build_ovstage = build_workflow["jobs"]["test-rendering-correctness-kitless-ovstage"]
    coe_expr = build_ovstage["continue-on-error"].removeprefix("${{").removesuffix("}}")
    node_eval = (
        "process.stdout.write(new Function('github', `return Boolean(${process.argv[1]})`)"
        "({event_name: process.argv[2]}) ? '1' : '0')"
    )
    for event_name in ("push", "pull_request"):
        check = subprocess.run(
            ["node", "-e", node_eval, coe_expr, event_name],
            capture_output=True,
            text=True,
            check=True,
        )
        assert check.stdout == "0", event_name

    # Both config jobs produce the same ECR image tag on main so nightly reuses the post-merge image.
    tags = []
    for wf, event_name in ((build_workflow, "push"), (nightly_workflow, "schedule")):
        step = next(s for s in wf["jobs"]["config"]["steps"] if s.get("id") == "image_tag")
        out_file = tmp_path / f"tag-{event_name}"
        out_file.write_text("", encoding="utf-8")
        subprocess.run(
            ["bash", "-c", step["run"]],
            env={
                **os.environ,
                "EVENT_NAME": event_name,
                "PR_NUMBER": "",
                "REF_NAME": "main",
                "SHA": "abcdef123456",
                "CI_IMAGE_REPOSITORY": "isaac-lab-ci",
                "CONFIG_REF": "",
                "CONFIG_HASH": "",
                "GITHUB_OUTPUT": str(out_file),
            },
            check=True,
            capture_output=True,
            text=True,
        )
        outputs = dict(line.split("=", 1) for line in out_file.read_text(encoding="utf-8").splitlines() if "=" in line)
        tags.append(outputs["ci_image_tag"])
    assert tags == ["isaac-lab-ci:main-abcdef123456", "isaac-lab-ci:main-abcdef123456"]

    # Both post-merge and nightly invoke run-package-tests with the full OVStage suite (no subset node IDs).
    build_with = next(
        s["with"] for s in build_ovstage["steps"] if s.get("uses") == "./.github/actions/run-package-tests"
    )
    nightly_with = next(
        s["with"]
        for s in nightly_workflow["jobs"]["test-rendering-correctness-kitless-ovstage"]["steps"]
        if s.get("uses") == "./.github/actions/run-package-tests"
    )
    assert "test-node-ids-file" not in build_with and "test-node-ids-key" not in build_with
    assert {k: v for k, v in nightly_with.items() if k != "container-name"} == {
        k: v for k, v in build_with.items() if k != "container-name"
    }

    bin_dir = tmp_path / "ovstage-bin"
    bin_dir.mkdir()
    docker_log = tmp_path / "docker-run.json"
    docker = bin_dir / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            f"""
            import json
            import os
            import sys
            from pathlib import Path

            cmd = sys.argv[1]
            if cmd == "run":
                Path({str(docker_log)!r}).write_text(
                    json.dumps({{
                        "argv": sys.argv[1:],
                        "env": {{
                            k: os.environ[k]
                            for k in ("TEST_K_EXPR", "TEST_NODE_IDS_FILE", "TEST_NODE_IDS_KEY", "TEST_NODE_IDS")
                            if k in os.environ
                        }},
                    }}),
                    encoding="utf-8",
                )
                print("cid")
            elif cmd == "wait":
                print("0")
            elif cmd == "cp" and sys.argv[-1].endswith(".xml"):
                Path(sys.argv[-1]).write_text('<testsuite tests="1" failures="0" errors="0"/>', encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    docker.chmod(0o755)
    for helper, body in (
        ("bc", "#!/bin/bash\necho 1\n"),
        ("free", "#!/bin/bash\nprintf 'Mem: 16000\\n'\n"),
        ("nproc", "#!/bin/bash\necho 4\n"),
    ):
        tool = bin_dir / helper
        tool.write_text(body, encoding="utf-8")
        tool.chmod(0o755)

    for label, pkg_with in (("post-merge", build_with), ("nightly", nightly_with)):
        docker_log.unlink(missing_ok=True)
        reports_dir = tmp_path / f"reports-{label}"
        include_files = " ".join(str(pkg_with["include-files"]).split())
        proc = subprocess.run(
            [
                "bash",
                str(_RUN_TESTS_SCRIPT),
                "tools",
                "report.xml",
                f"ovstage-{label}",
                "isaac-lab-ci:main-abcdef123456",
                str(reports_dir),
                "",
                str(pkg_with["filter-pattern"]),
                "",
                "false",
                include_files,
                "false",
                "",
                "",
                "",
                "",
                str(pkg_with.get("test-node-ids-file", "")),
                str(pkg_with.get("test-node-ids-key", "")),
                "",
                "",
                str(pkg_with["test-k-expr"]),
                "",
                "",
                "default",
                "",
                "",
                "",
                "",
                "",
            ],
            cwd=_REPO_ROOT,
            env={**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (label, proc.stdout + proc.stderr)
        recorded = json.loads(docker_log.read_text(encoding="utf-8"))
        argv = recorded["argv"]
        env_flags = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "-e"]
        kv_flags = dict(item.split("=", 1) for item in env_flags if "=" in item)
        assert kv_flags["TEST_FILTER_PATTERN"] == "isaaclab", label
        assert set(kv_flags["TEST_INCLUDE_FILES"].split(",")) == expected_ovstage_files, label
        assert "TEST_K_EXPR" in env_flags and recorded["env"] == {"TEST_K_EXPR": "ovstage"}, label
        assert not ({"TEST_NODE_IDS_FILE", "TEST_NODE_IDS_KEY", "TEST_NODE_IDS"} & set(env_flags)), label


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
