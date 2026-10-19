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
IMPORT_MAP = TASK.with_name("__init__.pyi")
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
                    "run": {"status": "completed"},
                    "runtime": {"total_fps": {"mean": 100.0}},
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
        process = self._invoke("prepare", "--output", path)
        if success:
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(json.loads(path.read_text())["commit"], self._git("rev-parse", "HEAD"))
        return path, process

    def _run(self, manifest, name, *, roots=None, command=None, proof_expected=True, **extra):
        output = self.directory / name
        verifier_args = ["--manifest", manifest, "--output-dir", output]
        benchmark_args = ["benchmark", "runtime", "--output_path", output]
        process = self._invoke("run", *verifier_args, "--", *benchmark_args, command=command, roots=roots, **extra)
        sidecar_path = output / "source-revision.json"
        self.assertEqual(sidecar_path.is_file(), proof_expected, process.stdout + process.stderr)
        sidecar = json.loads(sidecar_path.read_text()) if proof_expected else None
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
        self.assertIn(
            hashlib.sha256(result_path.read_bytes()).hexdigest(), [item["sha256"] for item in sidecar["outputs"]]
        )
        modules = {item["name"]: item for item in sidecar["modules"]}
        self.assertIn(RUNTIME_MODULE, modules)
        task = modules["isaaclab_tasks.fixture_task"]
        expected_hash = hashlib.sha256((self.checkout / TASK).read_bytes()).hexdigest()
        self.assertEqual(task["relative_path"], TASK.as_posix())
        self.assertEqual(task["sha256"], expected_hash)
        self.assertEqual(task["expected_sha256"], expected_hash)
        return result

    def test_selected_checkout_wins_over_cached_packages(self):
        manifest_a, _ = self._prepare("manifest-a")
        run_a = self._run(manifest_a, "result-a", roots=[self.checkout, self.cached])
        self._assert_verified(*run_a, "A", self.commit_a)
        self._task("B")
        commit_b = self._commit()
        manifest_b, _ = self._prepare("manifest-b")
        run_b = self._run(manifest_b, "result-b", roots=[self.checkout, self.cached])
        self._assert_verified(*run_b, "B", commit_b)

    def test_ci_launcher_verifies_the_initialized_benchmark_worker(self):
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
        executables = {"python": [sys.executable], "isaaclab": [sys.executable, "-m", "isaaclab.cli"]}
        command = executables[command[0]] + command[1:]

        verified = self._run(manifest, "ci-wrapper", command=command, FIXTURE_INITIALIZED="")
        result = self._assert_verified(*verified, "A", commit)
        expected = shlex.split(
            "benchmark runtime --task Fixture-Task --num_envs 512 --num_steps 200 "
            "--warmup_steps 100 --seed 42 --benchmark_formatter schema --output_path"
        )
        expected += [str(output), "--visualizer", "none", "physics=newton_mjwarp", "renderer=newton_renderer"]
        self.assertEqual(result["argv"], expected)

    def test_wrong_package_path_does_not_verify_as_checkout(self):
        manifest, _ = self._prepare()
        process, output, sidecar = self._run(manifest, "stale", roots=[self.cached, self.checkout])
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(sidecar["status"], "failed")
        self.assertTrue(sidecar["mismatches"])
        result = json.loads((output / "benchmark_runtime_fixture.json").read_text())
        self.assertEqual(result["executed_revision"], "C")
        self.assertEqual(result["worker_pid"], sidecar["pid"])

        shadow = Path("source/isaaclab_tasks/test/isaaclab_tasks/fixture_task.py")
        self._write(shadow.parent / "__init__.py", "")
        self._write(shadow, "def observed():\n    return 'T'\n")
        self._commit()
        manifest, _ = self._prepare("tracked-shadow")
        pythonpath = os.pathsep.join((str(self.checkout / shadow.parent.parent), self._environment()["PYTHONPATH"]))
        process, output, sidecar = self._run(manifest, "tracked-shadow", PYTHONPATH=pythonpath)
        result = json.loads((output / "benchmark_runtime_fixture.json").read_text())
        self.assertEqual(result["executed_revision"], "T")
        task = next(item for item in sidecar["modules"] if item["name"] == "isaaclab_tasks.fixture_task")
        self.assertEqual(task["relative_path"], shadow.as_posix())
        self.assertEqual(task["expected_sha256"], json.loads(manifest.read_text())["files"][shadow.as_posix()])
        self.assertEqual(task["sha256"], task["expected_sha256"])
        self.assertEqual(task["status"], "failed")
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(sidecar["status"], "failed")

    def test_correct_path_with_changed_bytes_does_not_verify(self):
        for path, contents in (
            (TASK, "def observed():\n    return 'B'\n"),
            (IMPORT_MAP, "# changed exports\n"),
            (IMPORT_MAP, None),
        ):
            with self.subTest(path=path, contents=contents):
                self._git("checkout", "--force", "--detach", self.commit_a)
                manifest, _ = self._prepare()
                if contents is None:
                    (self.checkout / path).unlink()
                else:
                    (self.checkout / path).write_text(contents)
                process, output, sidecar = self._run(manifest, "wrong-bytes")
                self.assertNotEqual(process.returncode, 0)
                self.assertEqual(sidecar["status"], "failed")
                self.assertTrue(sidecar["mismatches"])
                if path == TASK:
                    task = next(item for item in sidecar["modules"] if item["name"] == "isaaclab_tasks.fixture_task")
                    self.assertEqual(task["relative_path"], TASK.as_posix())
                    self.assertNotEqual(task["sha256"], task["expected_sha256"])
                    self.assertEqual(
                        json.loads((output / "benchmark_runtime_fixture.json").read_text())["executed_revision"], "B"
                    )
                else:
                    self.assertTrue(any(item.get("path") == path.as_posix() for item in sidecar["mismatches"]))
                _, process = self._prepare(success=False)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn("Traceback (most recent call last):", process.stderr)
                self.assertIn(path.as_posix(), process.stderr)

    def test_cli_failures_preserve_exit_status_and_interruptions(self):
        manifest, _ = self._prepare()
        process, output, sidecar = self._run(manifest, "early-failure", FIXTURE_EARLY_EXIT="1")
        self.assertEqual(process.returncode, 17)
        self.assertEqual(sidecar["benchmark_exit_code"], 17)
        self.assertEqual(sidecar["status"], "failed")
        self.assertEqual(sidecar["outputs"], [])
        self.assertFalse(list(output.glob("benchmark_runtime_*.json")))
        process, output, _ = self._run(manifest, "interrupted", proof_expected=False, FIXTURE_INTERRUPT="1")
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("KeyboardInterrupt", process.stderr)
        self.assertFalse(list(output.glob("benchmark_runtime_*.json")))

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
        manifest, _ = self._prepare()
        self._assert_verified(*self._run(manifest, "fresh-bytecode"), "B", commit_b)


if __name__ == "__main__":
    unittest.main()
