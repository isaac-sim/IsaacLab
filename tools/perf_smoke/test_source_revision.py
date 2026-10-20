# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute source-provenance checks against distinct checkout and cached code."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import suppress
from pathlib import Path

import yaml

LAUNCHER = Path(__file__).with_name("source_revision.py")
TASK = Path("source/isaaclab_tasks/isaaclab_tasks/fixture_task.py")
IMPORT_MAP = TASK.with_name("__init__.pyi")
RUNTIME_MODULE = "isaaclab.benchmark.entrypoints.runtime"


class SourceRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.checkout = self.directory / "checkout"
        self.checkout.mkdir()
        self._git("init", "--quiet")
        self._write(".gitignore", "__pycache__/\n")
        for name in (
            "source/isaaclab/isaaclab/__init__.py",
            "source/isaaclab/isaaclab/benchmark/__init__.py",
            "source/isaaclab/isaaclab/benchmark/entrypoints/__init__.py",
            "source/isaaclab_tasks/isaaclab_tasks/__init__.py",
        ):
            self._write(name, "")
        self._write(
            "source/isaaclab/isaaclab/cli/__init__.py",
            """
            import os
            import subprocess
            import sys

            def cli():
                initialized = dict(os.environ, FIXTURE_INITIALIZED="1")
                if sys.argv[1] == "-p":
                    raise SystemExit(subprocess.call([sys.executable, *sys.argv[2:]], env=initialized))
                if os.environ.get("FIXTURE_INITIALIZED") != "1":
                    raise SystemExit(subprocess.call(
                        [sys.executable, "-m", "isaaclab.cli", *sys.argv[1:]], env=initialized
                    ))
                if os.environ.get("FIXTURE_EARLY_EXIT"):
                    raise SystemExit(17)
                if os.environ.get("FIXTURE_INTERRUPT"):
                    raise KeyboardInterrupt
                from isaaclab.benchmark.entrypoints import runtime
                runtime.run(sys.argv[1:])
            """,
        )
        self._write(
            "source/isaaclab/isaaclab/benchmark/entrypoints/runtime.py",
            """
            import json
            import os
            from pathlib import Path

            def run(argv):
                from isaaclab_tasks.fixture_task import observed
                marker = observed()
                if os.environ.get("FIXTURE_NO_OUTPUT"):
                    return
                output = Path(argv[argv.index("--output_path") + 1])
                output.mkdir(parents=True, exist_ok=True)
                result = {
                    "executed_revision": marker,
                    "worker_pid": os.getpid(),
                    "argv": argv,
                }
                (output / "benchmark_runtime_fixture.json").write_text(json.dumps(result))
            """,
        )
        self._write("source/isaaclab/isaaclab/cli/__main__.py", "from . import cli\ncli()\n")
        self._write(IMPORT_MAP, "from .fixture_task import observed\n")
        self._task("A")
        self.commit_a = self._commit()
        self.cached = self.directory / "cached-image"
        shutil.copytree(self.checkout / "source", self.cached / "source")
        (self.cached / TASK).write_text("def observed():\n    return 'C'\n")

    def _write(self, relative, contents):
        destination = self.checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(textwrap.dedent(contents).lstrip("\n"))

    def _task(self, marker):
        self._write(TASK, f"def observed():\n    return {marker!r}\n")

    def _git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.checkout), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def _commit(self):
        self._git("add", ".")
        self._git(
            "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "Fixture"
        )
        return self._git("rev-parse", "HEAD")

    def _environment(self, roots=None, **extra):
        environment = dict(os.environ)
        environment.pop("PYTHONPYCACHEPREFIX", None)
        environment["PYTHONPATH"] = os.pathsep.join(
            str(root / "source" / package)
            for root in (roots or [self.checkout])
            for package in ("isaaclab", "isaaclab_tasks")
        )
        environment["FIXTURE_INITIALIZED"] = "1"
        environment.update(extra)
        return environment

    def _invoke(self, operation, *arguments, command=None, roots=None, **extra):
        launcher = [sys.executable, str(LAUNCHER), operation, "--checkout-root", str(self.checkout)]
        return subprocess.run(
            command or launcher + list(map(str, arguments)),
            cwd=self.directory,
            env=self._environment(roots, **extra),
            capture_output=True,
            text=True,
        )

    def _prepare(self, name="manifest", *, success=True):
        path = self.directory / f"{name}.json"
        process = self._invoke("prepare", "--output", path, "--commit", self._git("rev-parse", "HEAD"))
        if success:
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(json.loads(path.read_text())["commit"], self._git("rev-parse", "HEAD"))
        return path, process

    def _run(self, manifest, name, *, roots=None, command=None, proof_expected=True, expected_exit=0, **extra):
        output = self.directory / name
        verifier_args = ["--manifest", manifest, "--output-dir", output]
        benchmark_args = ["benchmark", "runtime", "--output_path", output]
        process = self._invoke("run", *verifier_args, "--", *benchmark_args, command=command, roots=roots, **extra)
        if expected_exit is None:
            self.assertNotEqual(process.returncode, 0)
        else:
            self.assertEqual(process.returncode, expected_exit, process.stdout + process.stderr)
        sidecar_path = output / "source-revision.json"
        self.assertEqual(sidecar_path.is_file(), proof_expected, process.stdout + process.stderr)
        sidecar = json.loads(sidecar_path.read_text()) if proof_expected else None
        if sidecar is not None:
            self.assertEqual(sidecar["status"], "verified" if expected_exit == 0 else "failed")
        return process, output, sidecar

    def _assert_verified(self, output, sidecar, marker, commit):
        self.assertEqual(sidecar["commit"], commit)
        self.assertEqual(sidecar["mismatches"], [])
        result_path = output / "benchmark_runtime_fixture.json"
        data = result_path.read_bytes()
        result = json.loads(data)
        self.assertEqual(result["executed_revision"], marker)
        self.assertEqual(result["worker_pid"], sidecar["pid"])
        self.assertEqual(
            sidecar["outputs"],
            [{"path": result_path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}],
        )
        modules = {item["name"]: item for item in sidecar["modules"]}
        self.assertIn(RUNTIME_MODULE, modules)
        task = modules["isaaclab_tasks.fixture_task"]
        self.assertEqual(task["relative_path"], TASK.as_posix())
        return result

    def test_source_verification_workflow(self):
        manifest_a, _ = self._prepare("manifest-a")
        with self.subTest(case="cached package origin is rejected"):
            process, output, sidecar = self._run(
                manifest_a, "stale", expected_exit=None, roots=[self.cached, self.checkout]
            )
            self.assertTrue(sidecar["mismatches"])
            result = json.loads((output / "benchmark_runtime_fixture.json").read_text())
            self.assertEqual(result["executed_revision"], "C")
            self.assertEqual(result["worker_pid"], sidecar["pid"])
        with self.subTest(case="initialized CLI worker and benchmark arguments"):
            process, output, proof = self._run(manifest_a, "direct-wrapper", expected_exit=None, FIXTURE_INITIALIZED="")
            self.assertTrue(
                any("Runtime entrypoint was not imported" in item["reason"] for item in proof["mismatches"])
            )
            result = json.loads((output / "benchmark_runtime_fixture.json").read_text())
            self.assertNotEqual(result["worker_pid"], proof["pid"])

            runner = LAUNCHER.with_name("run_benchmarks.sh").read_text()
            command_text = runner[runner.index("uv run --no-sync ") :].split('$args"', 1)[0] + "$args"
            output = self.directory / "ci-wrapper"
            replacements = {
                "/tmp/source_revision.py": str(LAUNCHER),
                "/tmp/source-manifest.json": str(manifest_a),
                "/workspace/isaaclab": str(self.checkout),
                "/tmp/benchmark-output": str(output),
                "$task": "Fixture-Task",
                "$num_envs": "512",
            }
            command = []
            for argument in shlex.split(command_text.replace("\\\n", " "))[3:]:
                command.extend(
                    ["physics=newton_mjwarp", "renderer=newton_renderer"]
                    if argument == "$args"
                    else [replacements.get(argument, argument)]
                )
            executables = {"python": [sys.executable], "isaaclab": [sys.executable, "-m", "isaaclab.cli"]}
            command = executables[command[0]] + command[1:]

            _, output, proof = self._run(manifest_a, "ci-wrapper", command=command, FIXTURE_INITIALIZED="")
            result = self._assert_verified(output, proof, "A", self.commit_a)
            expected = shlex.split(
                "benchmark runtime --task Fixture-Task --num_envs 512 --num_steps 200 "
                "--warmup_steps 100 --seed 42 --benchmark_formatter schema --output_path"
            )
            expected += [str(output), "--visualizer", "none", "physics=newton_mjwarp", "renderer=newton_renderer"]
            self.assertEqual(result["argv"], expected)
        with self.subTest(case="CLI failures and interruption"):
            stale = self.directory / "no-output" / "benchmark_runtime_stale.json"
            stale.parent.mkdir()
            stale.write_bytes(b'{"stale": true}')
            process, _, _ = self._run(
                manifest_a, "early-failure", proof_expected=False, expected_exit=17, FIXTURE_EARLY_EXIT="1"
            )
            process, output, sidecar = self._run(manifest_a, "no-output", expected_exit=None, FIXTURE_NO_OUTPUT="1")
            self.assertEqual(sidecar["outputs"], [])
            self.assertEqual(list(output.glob("benchmark_runtime_*.json")), [stale])
            self.assertEqual(stale.read_bytes(), b'{"stale": true}')
            process, output, _ = self._run(
                manifest_a, "interrupted", proof_expected=False, expected_exit=None, FIXTURE_INTERRUPT="1"
            )
            self.assertIn("KeyboardInterrupt", process.stderr)
            self.assertFalse(list(output.glob("benchmark_runtime_*.json")))
        with self.subTest(case="host Git check rejects changed source and import maps"):
            self._git("config", "diff.autoRefreshIndex", "false")
            os.utime(self.checkout / TASK, (1700000000, 1700000000))
            self.assertEqual(self._git("diff", "--name-only", "HEAD", "--", TASK.as_posix()), TASK.as_posix())
            self._prepare("unchanged-source")
            _, output, _ = self._run(manifest_a, "permission-only")
            self._git("config", "core.filemode", "true")
            (self.checkout / TASK).chmod(0o755)
            self.assertIn("mode change", self._git("diff", "--summary", "HEAD", "--", TASK.as_posix()))
            checked = self._invoke("check", "--manifest", manifest_a, "--output-dir", output)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertEqual(json.loads((output / "source-revision.json").read_text())["status"], "verified")
            self._prepare("permission-only")
            for path, contents in (
                (TASK, "def observed():\n    return 'B'\n"),
                (IMPORT_MAP, "# changed exports\n"),
                (IMPORT_MAP, None),
            ):
                with self.subTest(path=path, contents=contents):
                    self._git("checkout", "--force", "--detach", self.commit_a)
                    if contents is None:
                        (self.checkout / path).unlink()
                    else:
                        (self.checkout / path).write_text(contents)
                    # The worker checks paths; the host checks for source edits after the attempt.
                    process, output, sidecar = self._run(manifest_a, "wrong-bytes")
                    if path == TASK:
                        (self.checkout / path).chmod(0o755)
                        self.assertEqual(
                            json.loads((output / "benchmark_runtime_fixture.json").read_text())["executed_revision"],
                            "B",
                        )
                    checked = self._invoke("check", "--manifest", manifest_a, "--output-dir", output)
                    self.assertNotEqual(checked.returncode, 0, checked.stdout + checked.stderr)
                    proof = json.loads((output / "source-revision.json").read_text())
                    self.assertEqual(proof["status"], "failed")
                    self.assertTrue(any(path.as_posix() in item["reason"] for item in proof["mismatches"]))
                    if path == TASK:
                        (self.checkout / path).chmod(0o644)
                        self._git("add", "--", path.as_posix())
                    _, process = self._prepare(success=False)
                    self.assertNotEqual(process.returncode, 0)
                    self.assertIn("Traceback (most recent call last):", process.stderr)
                    self.assertIn(path.as_posix(), process.stderr)
        with self.subTest(case="tracked test package cannot impersonate the installed package"):
            self._git("checkout", "--force", "--detach", self.commit_a)
            shadow = Path("source/isaaclab_tasks/test/isaaclab_tasks/fixture_task.py")
            self._write(shadow.parent / "__init__.py", "")
            self._write(shadow, "def observed():\n    return 'T'\n")
            self._commit()
            manifest, _ = self._prepare("tracked-shadow")
            pythonpath = os.pathsep.join((str(self.checkout / shadow.parent.parent), self._environment()["PYTHONPATH"]))
            process, output, sidecar = self._run(manifest, "tracked-shadow", expected_exit=None, PYTHONPATH=pythonpath)
            result = json.loads((output / "benchmark_runtime_fixture.json").read_text())
            self.assertEqual(result["executed_revision"], "T")
            task = next(item for item in sidecar["modules"] if item["name"] == "isaaclab_tasks.fixture_task")
            self.assertEqual(task["relative_path"], shadow.as_posix())
        with self.subTest(case="revision B overrides cached packages and stale bytecode"):
            self._git("checkout", "--force", "--detach", self.commit_a)
            task = self.checkout / TASK
            os.utime(task, (1700000000, 1700000000))
            subprocess.run([sys.executable, "-m", "py_compile", str(task)], check=True, env=self._environment())
            original_size = task.stat().st_size
            self._task("B")
            self.assertEqual(task.stat().st_size, original_size)
            os.utime(task, (1700000000, 1700000000))
            commit_b = self._commit()
            # Confirm that an ordinary import loads stale bytecode before testing the fix.
            uncorrected = subprocess.run(
                [sys.executable, "-c", "from isaaclab_tasks.fixture_task import observed; print(observed())"],
                cwd=self.directory,
                env=self._environment(),
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(uncorrected.stdout.strip(), "A")
            rejected = self._invoke("prepare", "--output", manifest_a, "--commit", self.commit_a)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("differs from expected commit", rejected.stderr)
            manifest, _ = self._prepare()
            _, output, proof = self._run(manifest, "fresh-bytecode", roots=[self.checkout, self.cached])
            self._assert_verified(output, proof, "B", commit_b)
        self._check_workflow_cancellation()

    def _check_workflow_cancellation(self):
        root = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load((root / ".github/workflows/build.yaml").read_text())
        command = next(
            step["run"]
            for step in workflow["jobs"]["performance-smoke"]["steps"]
            if step.get("name") == "Run runtime benchmarks"
        )
        for stop_signal in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(case="workflow cancellation", signal=stop_signal):
                workspace = self.directory / f"cancel-{stop_signal}"
                shutil.copytree(self.checkout, workspace)
                step_script = workspace / "workflow-step.sh"
                step_script.write_text(command)
                scripts = workspace / "tools/perf_smoke"
                scripts.mkdir(parents=True)
                runner = root / "tools/perf_smoke/run_benchmarks.sh"
                shutil.copy2(runner, scripts)
                shutil.copy2(LAUNCHER, scripts)
                manifest, _ = self._prepare()
                (workspace / "benchmark-output").mkdir()
                shutil.copy2(manifest, workspace / "benchmark-output/source-manifest.json")
                (workspace / "benchmark-legs.tsv").write_text("fixture|Fixture-Task|1|600|\n")
                executables = workspace / "bin"
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
                # The child stays blocked until cancellation; timeout's expiry is not under test.
                timeout = executables / "timeout"
                timeout.write_text('#!/bin/bash\nshift\nexec "$@"\n')
                timeout.chmod(0o755)
                environment = {
                    **os.environ,
                    "PATH": f"{executables}{os.pathsep}{os.environ['PATH']}",
                    "GITHUB_WORKSPACE": str(workspace),
                    "GITHUB_RUN_ID": "123",
                    "GITHUB_RUN_ATTEMPT": "2",
                    "CI_IMAGE_TAG": "fixture-image",
                }
                process = subprocess.Popen(
                    ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", str(step_script)],
                    cwd=workspace,
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
                    if started is None:
                        self.fail(process.stderr.read())
                    process.send_signal(stop_signal)
                    # GitHub escalates cancellation after 7.5 s + 2.5 s, then kills the process tree.
                    stdout, stderr = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 128 + stop_signal, stdout + stderr)
                    self.assertFalse((workspace / "running-container").exists())
                    with self.assertRaises(ProcessLookupError):
                        os.kill(started["pid"], 0)
                finally:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()


if __name__ == "__main__":
    unittest.main()
