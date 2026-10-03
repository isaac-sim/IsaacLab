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
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).with_name("source_revision.py")
TASK = Path("source/isaaclab_tasks/isaaclab_tasks/fixture_task.py")
RUNTIME_MODULE = "isaaclab.benchmark.entrypoints.runtime"
RUNTIME_PATH = Path("source/isaaclab/isaaclab/benchmark/entrypoints/runtime.py")


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
            import sys

            def cli():
                if os.environ.get("FIXTURE_EARLY_EXIT"):
                    raise SystemExit(17)
                from isaaclab.benchmark.entrypoints import runtime
                if os.environ.get("FIXTURE_REPLACE_RUNTIME"):
                    from pathlib import Path
                    altered = Path(runtime.__file__).read_text().replace("marker = observed()", "marker = 'X'")
                    namespace = {}
                    exec(compile(altered, runtime.__file__, "exec"), namespace)
                    runtime.run.__code__ = namespace["run"].__code__
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
                    "image_identity": os.environ["FIXTURE_IMAGE_IDENTITY"],
                    "run": {"status": "completed"},
                    "runtime": {"total_fps": {"mean": 100.0}},
                }
                (output / "benchmark_runtime_fixture.json").write_text(json.dumps(result))
            """,
        )
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
        environment["FIXTURE_IMAGE_IDENTITY"] = "same-cached-image"
        environment.update(extra)
        return environment

    def _prepare(self, name="manifest", *, success=True):
        path = self.directory / f"{name}.json"
        process = subprocess.run(
            [sys.executable, str(LAUNCHER), "prepare", "--checkout-root", str(self.checkout), "--output", str(path)],
            cwd=self.directory,
            env=self._environment(),
            capture_output=True,
            text=True,
        )
        if success:
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(json.loads(path.read_text())["commit"], self._git("rev-parse", "HEAD"))
        return path, process

    def _run(self, manifest, name, *, roots=None, command=None, **extra):
        output = self.directory / name
        process = subprocess.run(
            command
            or [
                sys.executable,
                str(LAUNCHER),
                "run",
                "--manifest",
                str(manifest),
                "--checkout-root",
                str(self.checkout),
                "--output-dir",
                str(output),
                "--",
                "benchmark",
                "runtime",
                "--output_path",
                str(output),
            ],
            cwd=self.directory,
            env=self._environment(roots, **extra),
            capture_output=True,
            text=True,
        )
        sidecar = json.loads((output / "source-revision.json").read_text())
        return process, output, sidecar

    def _assert_verified(self, process, output, sidecar, marker, commit):
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(sidecar["status"], "verified")
        self.assertEqual(sidecar["commit"], commit)
        self.assertEqual(sidecar["mismatches"], [])
        result_path = output / "benchmark_runtime_fixture.json"
        result = json.loads(result_path.read_text())
        self.assertEqual(result["executed_revision"], marker)
        self.assertEqual(result["worker_pid"], sidecar["pid"])
        self.assertNotEqual(result["worker_pid"], os.getpid())
        self.assertEqual(result["image_identity"], "same-cached-image")
        self.assertIn(
            hashlib.sha256(result_path.read_bytes()).hexdigest(), [item["sha256"] for item in sidecar["outputs"]]
        )
        modules = {item["name"]: item for item in sidecar["modules"]}
        self.assertIn(RUNTIME_MODULE, modules)
        self.assertEqual(modules["isaaclab_tasks.fixture_task"]["relative_path"], TASK.as_posix())
        self.assertEqual(
            modules["isaaclab_tasks.fixture_task"]["sha256"],
            hashlib.sha256((self.checkout / TASK).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            modules["isaaclab_tasks.fixture_task"]["sha256"], modules["isaaclab_tasks.fixture_task"]["expected_sha256"]
        )
        return result

    def test_same_cached_image_executes_each_selected_commit(self):
        manifest_a, _ = self._prepare("manifest-a")
        run_a = self._run(manifest_a, "result-a", roots=[self.checkout, self.cached])
        result_a = self._assert_verified(*run_a, "A", self.commit_a)
        self._task("B")
        commit_b = self._commit()
        manifest_b, _ = self._prepare("manifest-b")
        run_b = self._run(manifest_b, "result-b", roots=[self.checkout, self.cached])
        result_b = self._assert_verified(*run_b, "B", commit_b)
        self.assertNotEqual(self.commit_a, commit_b)
        self.assertNotEqual(result_a["executed_revision"], result_b["executed_revision"])
        self.assertEqual(result_a["image_identity"], result_b["image_identity"])

    def test_ci_launcher_verifies_the_initialized_benchmark_worker(self):
        self._write(
            "source/isaaclab/isaaclab/cli/__init__.py",
            """
            import argparse
            import os
            import subprocess
            import sys

            def cli():
                initialized = dict(os.environ, FIXTURE_INITIALIZED="1")
                if sys.argv[1] == "-p":
                    parser = argparse.ArgumentParser()
                    parser.add_argument("-p", "--python", nargs=argparse.REMAINDER)
                    args = parser.parse_args()
                    raise SystemExit(subprocess.call([sys.executable, *args.python], env=initialized))
                if os.environ.get("FIXTURE_INITIALIZED") != "1":
                    raise SystemExit(subprocess.call(
                        [sys.executable, "-m", "isaaclab.cli", *sys.argv[1:]], env=initialized
                    ))
                from isaaclab.benchmark.entrypoints import runtime
                runtime.run(sys.argv[1:])
            """,
        )
        self._write("source/isaaclab/isaaclab/cli/__main__.py", "from . import cli\ncli()\n")
        runtime = self.checkout / RUNTIME_PATH
        runtime.write_text(
            runtime.read_text().replace('"worker_pid": os.getpid(),', '"worker_pid": os.getpid(), "argv": argv,')
        )
        commit = self._commit()
        manifest, _ = self._prepare()
        process, output, proof = self._run(manifest, "direct-wrapper", FIXTURE_INITIALIZED="")
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(proof["benchmark_exit_code"], 0)
        self.assertEqual(proof["status"], "failed")
        self.assertIsNone(proof["runtime_entrypoint"])
        self.assertTrue(any("Runtime entrypoint was not imported" in item["reason"] for item in proof["mismatches"]))
        result = json.loads((output / "benchmark_runtime_fixture.json").read_text())
        self.assertNotEqual(result["worker_pid"], proof["pid"])

        # Use the CI command so the test catches regressions in the production launch choice.
        runner = LAUNCHER.with_name("run_benchmarks.sh").read_text()
        command_text = runner[runner.index("uv run --no-sync ") :].split('$args"', 1)[0] + "$args"
        output = self.directory / "ci-wrapper"
        replacements = {
            "/tmp/source_revision.py": str(LAUNCHER),
            "/tmp/source-manifest.json": str(manifest),
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
        executables = {
            "python": [sys.executable],
            "isaaclab": [sys.executable, "-m", "isaaclab.cli"],
        }
        command = executables[command[0]] + command[1:]
        rejected = command.copy()
        rejected.insert(rejected.index("benchmark"), "--")
        process = subprocess.run(
            rejected, cwd=self.directory, env=self._environment(FIXTURE_INITIALIZED=""), capture_output=True, text=True
        )
        self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
        self.assertIn("unrecognized arguments: -- benchmark runtime", process.stderr)
        self.assertFalse((output / "source-revision.json").exists())

        verified = self._run(manifest, "ci-wrapper", command=command, FIXTURE_INITIALIZED="")
        result = self._assert_verified(*verified, "A", commit)
        expected = shlex.split(
            "benchmark runtime --task Fixture-Task --num_envs 512 --num_steps 200 "
            "--warmup_steps 100 --seed 42 --benchmark_formatter schema --output_path"
        )
        expected += [str(output), "--visualizer", "none", "physics=newton_mjwarp", "renderer=newton_renderer"]
        self.assertEqual(result["argv"], expected)

    def test_docstring_only_revision_changes_live_function_proof_in_same_image(self):
        runtime = self.checkout / RUNTIME_PATH
        runtime.write_text(
            runtime.read_text().replace("def run(argv):\n", "def run(argv):\n    'Runtime revision A.'\n")
        )
        commit_a = self._commit()
        manifest_a, _ = self._prepare("docstring-a")
        run_a = self._run(manifest_a, "docstring-output-a", roots=[self.checkout, self.cached])
        self._assert_verified(*run_a, "A", commit_a)
        runtime.write_text(runtime.read_text().replace("Runtime revision A.", "Runtime revision B."))
        commit_b = self._commit()
        manifest_b, _ = self._prepare("docstring-b")
        run_b = self._run(manifest_b, "docstring-output-b", roots=[self.checkout, self.cached])
        self._assert_verified(*run_b, "A", commit_b)
        proof_a = run_a[2]["runtime_entrypoint"]
        proof_b = run_b[2]["runtime_entrypoint"]
        self.assertTrue(proof_a["source_code_matches"])
        self.assertTrue(proof_b["source_code_matches"])
        self.assertEqual(proof_a["doc_sha256"], hashlib.sha256(b"Runtime revision A.").hexdigest())
        self.assertEqual(proof_b["doc_sha256"], hashlib.sha256(b"Runtime revision B.").hexdigest())
        self.assertNotEqual(proof_a["code_sha256"], proof_b["code_sha256"])

    def test_stale_installed_package_does_not_verify_as_checkout(self):
        manifest, _ = self._prepare()
        process, output, sidecar = self._run(manifest, "stale", roots=[self.cached, self.checkout])
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(sidecar["status"], "failed")
        self.assertTrue(sidecar["mismatches"])
        result = json.loads((output / "benchmark_runtime_fixture.json").read_text())
        self.assertEqual(result["executed_revision"], "C")
        self.assertEqual(result["worker_pid"], sidecar["pid"])

    def test_correct_path_with_changed_bytes_does_not_verify(self):
        manifest, _ = self._prepare()
        self._task("B")
        process, output, sidecar = self._run(manifest, "wrong-bytes")
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(sidecar["status"], "failed")
        self.assertTrue(sidecar["mismatches"])
        task = next(item for item in sidecar["modules"] if item["name"] == "isaaclab_tasks.fixture_task")
        self.assertEqual(task["relative_path"], TASK.as_posix())
        self.assertNotEqual(task["sha256"], task["expected_sha256"])
        self.assertEqual(json.loads((output / "benchmark_runtime_fixture.json").read_text())["executed_revision"], "B")

    def test_prepare_detects_checkout_modified_from_commit(self):
        self._task("B")
        _, process = self._prepare(success=False)
        self.assertNotEqual(process.returncode, 0)

    def test_replaced_runtime_code_with_matching_source_bytes_is_detected(self):
        manifest, _ = self._prepare()
        process, output, sidecar = self._run(manifest, "replaced-code", FIXTURE_REPLACE_RUNTIME="1")
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(sidecar["status"], "failed")
        self.assertTrue(sidecar["mismatches"])
        runtime = next(item for item in sidecar["modules"] if item["name"] == RUNTIME_MODULE)
        self.assertEqual(runtime["sha256"], runtime["expected_sha256"])
        self.assertFalse(sidecar["runtime_entrypoint"]["source_code_matches"])
        self.assertEqual(json.loads((output / "benchmark_runtime_fixture.json").read_text())["executed_revision"], "X")

    def test_early_cli_failure_retains_exit_without_completed_proof(self):
        manifest, _ = self._prepare()
        process, output, sidecar = self._run(manifest, "early-failure", FIXTURE_EARLY_EXIT="1")
        self.assertEqual(process.returncode, 17)
        self.assertEqual(sidecar["benchmark_exit_code"], 17)
        self.assertEqual(sidecar["status"], "failed")
        self.assertEqual(sidecar["outputs"], [])
        self.assertFalse(list(output.glob("benchmark_runtime_*.json")))

    def test_successful_cli_without_runtime_output_is_not_verified(self):
        manifest, _ = self._prepare()
        process, _, sidecar = self._run(manifest, "no-output", FIXTURE_NO_OUTPUT="1")
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(sidecar["benchmark_exit_code"], 0)
        self.assertEqual(sidecar["status"], "failed")
        self.assertEqual(sidecar["outputs"], [])

    def test_stale_timestamp_bytecode_is_not_the_executed_revision(self):
        task = self.checkout / TASK
        os.utime(task, (1700000000, 1700000000))
        subprocess.run([sys.executable, "-m", "py_compile", str(task)], check=True, env=self._environment())
        original_size = task.stat().st_size
        self._task("B")
        self.assertEqual(task.stat().st_size, original_size)
        os.utime(task, (1700000000, 1700000000))
        commit_b = self._commit()
        # Establish that the fixture really exposes stale bytecode to an ordinary import.
        uncorrected = subprocess.run(
            [sys.executable, "-c", "from isaaclab_tasks.fixture_task import observed; print(observed())"],
            cwd=self.directory,
            env=self._environment(),
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(uncorrected.stdout.strip(), "A")
        manifest, _ = self._prepare()
        self._assert_verified(*self._run(manifest, "fresh-bytecode"), "B", commit_b)


if __name__ == "__main__":
    unittest.main()
