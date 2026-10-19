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
TIMER_PATH = Path("source/isaaclab/isaaclab/utils/timer.py")


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
            "source/isaaclab/isaaclab/utils/__init__.py",
            "source/isaaclab_tasks/isaaclab_tasks/__init__.py",
        ):
            self._write(name, "")
        self._write("source/isaaclab/isaaclab/benchmark/__init__.pyi", "from .entrypoints.runtime import run\n")
        self._write(TIMER_PATH, "FPS = 100.0\n")
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
                from isaaclab.utils.timer import FPS
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
                    "run": {"status": "completed"},
                    "runtime": {"total_fps": {"mean": FPS}},
                }
                (output / "benchmark_runtime_fixture.json").write_text(json.dumps(result))
            """,
        )
        self._write(IMPORT_MAP, "from .fixture_task import observed\n")
        self._write("source/isaaclab/isaaclab/cli/__main__.py", "from . import cli\ncli()\n")
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

    def _prepare(self, name="manifest", *, measurement=None, success=True):
        path = self.directory / f"{name}.json"
        measurement_args = ["--measurement-manifest", measurement] if measurement else []
        process = self._invoke("prepare", "--output", path, *measurement_args)
        if success:
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(json.loads(path.read_text())["commit"], self._git("rev-parse", "HEAD"))
        return path, process

    def _snapshot(self, name="measurement"):
        output = self.directory / name
        process = self._invoke("snapshot", "--output-dir", output)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        return output / "measurement-manifest.json"

    def _overlay(self, measurement):
        shutil.rmtree(self.checkout / "source/isaaclab/isaaclab/benchmark")
        for relative in json.loads(measurement.read_text())["files"]:
            destination = self.checkout / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(measurement.parent / relative, destination)

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

    def _assert_verified(self, process, output, sidecar, marker, commit):
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
            self.assertEqual(proof["benchmark_exit_code"], 0)
            self.assertIsNone(proof["runtime_entrypoint"])
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

            verified = self._run(manifest_a, "ci-wrapper", command=command, FIXTURE_INITIALIZED="")
            result = self._assert_verified(*verified, "A", self.commit_a)
            expected = shlex.split(
                "benchmark runtime --task Fixture-Task --num_envs 512 --num_steps 200 "
                "--warmup_steps 100 --seed 42 --benchmark_formatter schema --output_path"
            )
            expected += [str(output), "--visualizer", "none", "physics=newton_mjwarp", "renderer=newton_renderer"]
            self.assertEqual(result["argv"], expected)
        with self.subTest(case="CLI failures and interruption"):
            for name, environment, exit_code, benchmark_code in (
                ("early-failure", {"FIXTURE_EARLY_EXIT": "1"}, 17, 17),
                ("no-output", {"FIXTURE_NO_OUTPUT": "1"}, None, 0),
            ):
                with self.subTest(failure=name):
                    process, output, sidecar = self._run(manifest_a, name, expected_exit=exit_code, **environment)
                    self.assertEqual(sidecar["benchmark_exit_code"], benchmark_code)
                    self.assertEqual(sidecar["outputs"], [])
                    self.assertFalse(list(output.glob("benchmark_runtime_*.json")))
            process, output, _ = self._run(
                manifest_a, "interrupted", proof_expected=False, expected_exit=None, FIXTURE_INTERRUPT="1"
            )
            self.assertIn("KeyboardInterrupt", process.stderr)
            self.assertFalse(list(output.glob("benchmark_runtime_*.json")))
        with self.subTest(case="changed source and runtime import maps"):
            measurement = self._snapshot()
            manifest, _ = self._prepare("measurement-source", measurement=measurement)
            stub = Path("source/isaaclab/isaaclab/benchmark/__init__.pyi")
            for name, path in (
                ("shared workload", TASK),
                ("workload import map", IMPORT_MAP),
                ("missing import map", IMPORT_MAP),
                ("measurement resource", stub),
            ):
                with self.subTest(change=name):
                    self._git("checkout", "--force", "--detach", self.commit_a)
                    if name == "missing import map":
                        (self.checkout / path).unlink()
                    else:
                        (self.checkout / path).write_text(
                            "def observed():\n    return 'B'\n" if path == TASK else "# changed exports\n"
                        )
                    process, output, sidecar = self._run(manifest, name, expected_exit=None)
                    self.assertTrue(sidecar["mismatches"])
                    self.assertEqual(sidecar["measurement"]["status"], "failed" if path == stub else "verified")
                    if path == TASK:
                        task = next(
                            item for item in sidecar["modules"] if item["name"] == "isaaclab_tasks.fixture_task"
                        )
                        self.assertEqual(task["relative_path"], TASK.as_posix())
                        self.assertNotEqual(task["sha256"], task["expected_sha256"])
                        self.assertTrue(any(item.get("module") == task["name"] for item in sidecar["mismatches"]))
                        result = json.loads((output / "benchmark_runtime_fixture.json").read_text())
                        self.assertEqual(result["executed_revision"], "B")
                    else:
                        self.assertTrue(any(item.get("path") == path.as_posix() for item in sidecar["mismatches"]))
                    _, process = self._prepare(name + "-dirty", measurement=measurement, success=False)
                    self.assertNotEqual(process.returncode, 0)
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
            self.assertEqual(task["expected_sha256"], json.loads(manifest.read_text())["files"][shadow.as_posix()])
            self.assertEqual(task["sha256"], task["expected_sha256"])
            self.assertEqual(task["status"], "failed")
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
            manifest, _ = self._prepare()
            self._assert_verified(
                *self._run(manifest, "fresh-bytecode", roots=[self.checkout, self.cached]), "B", commit_b
            )
        with self.subTest(case="shared measurement overlays both workload revisions"):
            (self.checkout / "source/isaaclab/isaaclab/benchmark/__init__.pyi").unlink()
            self._write(TIMER_PATH, "FPS = 150.0\n")
            runtime = self.checkout / RUNTIME_PATH
            runtime.write_text(
                runtime.read_text()
                .replace("def run(argv):", "def measure(**kwargs):\n    return kwargs['fps'] * 2\n\ndef run(argv):")
                .replace('"mean": FPS', '"mean": measure(**{"fps": FPS})')
            )
            commit_b = self._commit()
            measurement = self._snapshot()
            expected = json.loads(measurement.read_text())
            for marker, commit in (("A", self.commit_a), ("B", commit_b)):
                with self.subTest(workload=marker):
                    self._git("checkout", "--force", "--detach", commit)
                    manifest, _ = self._prepare(f"manifest-{marker}", measurement=measurement)
                    self._overlay(measurement)
                    run = self._run(manifest, f"result-{marker}", roots=[self.checkout, self.cached])
                    result = self._assert_verified(*run, marker, commit)
                    self.assertEqual(result["runtime"]["total_fps"]["mean"], 300.0)
                    proof = run[2]
                    self.assertEqual(proof["measurement"], dict(expected, status="verified"))
                    origins = {item["name"]: item["origin"] for item in proof["modules"]}
                    self.assertEqual(origins[RUNTIME_MODULE], "measurement")
                    self.assertEqual(origins["isaaclab.utils.timer"], "measurement")
                    self.assertEqual(origins["isaaclab_tasks.fixture_task"], "workload")
                    self.assertEqual(proof["runtime_entrypoint"]["origin"], "measurement")
            self._task("C")
            self._commit()
            later = json.loads(self._snapshot("later-measurement").read_text())
            self.assertNotEqual(later["commit"], expected["commit"])
            self.assertEqual(later["digest"], expected["digest"])


if __name__ == "__main__":
    unittest.main()
