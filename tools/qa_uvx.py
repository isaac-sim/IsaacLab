# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test released or candidate demos through uvx without a checkout or Isaac Lab installation.

Run with ``uv run --no-project python tools/qa_uvx.py --help``. Only Python's standard library
and uvx are required. Additional arguments after ``--`` are forwarded to simulation runs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

_PROBE = """
import importlib.metadata
import json
import sys
from pathlib import Path
import isaaclab
from isaaclab.programs import DEMOS, EXAMPLES

package_root = Path(isaaclab.__file__).resolve().parent
catalogs = {}
for command, catalog in (("demo", DEMOS), ("example", EXAMPLES)):
    if not catalog:
        raise RuntimeError(f"Empty {command} catalog")
    catalogs[command] = []
    for program in catalog:
        path = program.path.resolve()
        if not path.is_relative_to(package_root):
            raise RuntimeError(f"Program outside installed package: {path}")
        if not path.is_file():
            raise RuntimeError(f"Missing packaged program: {path}")
        catalogs[command].append({
            "name": program.name,
            "extras": program.extras,
            "missing_modules": program.missing_modules(),
        })
Path(sys.argv[1]).write_text(json.dumps({
    "version": importlib.metadata.version("isaaclab"),
    "package_root": str(package_root),
    "python": sys.version,
    "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "newton", "warp-lang")},
    "catalogs": catalogs,
}), encoding="utf-8")
"""


def _run_check(name: str, command: list[str], directory: Path, env: dict[str, str], log: Path, timeout: float) -> dict:
    """Run a check, saving output and killing its process tree on timeout or interruption."""
    started = time.monotonic()
    result = {"name": name, "command": command, "log": str(log), "status": "failed"}
    with log.open("w", encoding="utf-8") as output:
        try:
            process = subprocess.Popen(
                command,
                cwd=directory,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
        except OSError as error:
            result["reason"] = str(error)
        else:
            try:
                result["returncode"] = process.wait(timeout=timeout)
                if process.returncode == 0:
                    result["status"] = "passed"
                else:
                    result["reason"] = f"Exit code {process.returncode}"
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                else:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                if isinstance(error, KeyboardInterrupt):
                    raise
                result["reason"] = f"Timed out after {timeout:g} seconds"
    result["seconds"] = round(time.monotonic() - started, 2)
    print(f"{result['status'].upper():7} {name}" + (f": {result['reason']}" if "reason" in result else ""))
    return result


def main(argv: list[str] | None = None) -> int:
    """Check uvx packaging, catalogs and help; optionally launch bounded demo simulations."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", default="isaaclab", help="Requirement, wheel path/URL, or Git URL for uvx --from.")
    parser.add_argument("--python", default="3.12", help="Python requested from uvx (default: 3.12).")
    parser.add_argument("--index", action="append", default=[], help="Additional package index; repeat as needed.")
    parser.add_argument("--uvx-arg", action="append", default=[], help="Extra uvx option, e.g. --uvx-arg=--offline.")
    parser.add_argument("--demo", action="append", help="Demo to check; repeat to select several (default: catalog).")
    parser.add_argument("--examples", action="store_true", help="Also check example help (does not simulate examples).")
    parser.add_argument("--run", action="store_true", help="Run selected demos with --max_steps; default is CLI only.")
    parser.add_argument("--steps", type=int, default=60, help="Simulation steps per demo (default: 60).")
    parser.add_argument(
        "--timeout", type=float, default=1800, help="Deadline per command, including install (seconds)."
    )
    parser.add_argument("--output", type=Path, help="New directory for logs and report.json.")
    parser.add_argument("demo_args", nargs=argparse.REMAINDER, help="Arguments after -- forwarded to demo runs.")
    args = parser.parse_args(argv)
    if args.steps < 1 or not 0 < args.timeout < float("inf"):
        parser.error("--steps and --timeout must be positive and finite")
    if args.demo_args and not args.run:
        parser.error("arguments after -- require --run")
    uvx = shutil.which("uvx")
    if uvx is None:
        parser.error("uvx is not on PATH; install uv first")
    # Resolve local wheels before changing to a temporary working directory.
    package = str(Path(args.package).resolve()) if Path(args.package).is_file() else args.package
    output_dir = (args.output or Path(f"uvx-qa-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}")).resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(f"cannot create output directory: {error}")
    env = os.environ.copy()
    for variable in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "ISAAC_PATH", "LIVESTREAM"):
        env.pop(variable, None)
    env["PYTHONUNBUFFERED"] = "1"
    base = [uvx, "--no-config", "--isolated", "--python", args.python]
    if package != "isaaclab":
        base.extend(["--from", package])
    for index in args.index:
        base.extend(["--index", index])
    base.extend(args.uvx_arg)
    base.append("isaaclab")
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "package": package,
        "requested_python": args.python,
        "simulation": args.run,
        "checks": [],
    }
    checks = report["checks"]
    try:
        with tempfile.TemporaryDirectory(prefix="isaaclab-uvx-qa-") as temporary:
            directory = Path(temporary)
            checks.append(
                _run_check(
                    "uvx-version", [uvx, "--version"], directory, env, output_dir / "uvx-version.log", args.timeout
                )
            )
            gpu_tool = shutil.which("nvidia-smi")
            if gpu_tool:
                try:
                    gpu = subprocess.run(
                        [gpu_tool, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    report["gpu"] = {"returncode": gpu.returncode, "output": gpu.stdout, "error": gpu.stderr}
                except (OSError, subprocess.TimeoutExpired) as error:
                    report["gpu"] = {"error": str(error)}

            def check(name: str, arguments: list[str]) -> dict:
                result = _run_check(name, [*base, *arguments], directory, env, output_dir / f"{name}.log", args.timeout)
                checks.append(result)
                return result

            # Exercise the user-facing catalog before probing the installed wheel.
            if check("demo-list", ["demo", "list"])["status"] != "passed":
                return 1
            probe = directory / "probe.py"
            probe.write_text(_PROBE, encoding="utf-8")
            metadata = directory / "package.json"
            if check("installed-package", ["-p", str(probe), str(metadata)])["status"] != "passed":
                return 1
            report["installed"] = json.loads(metadata.read_text(encoding="utf-8"))
            catalogs = report["installed"]["catalogs"]
            if args.demo:
                unknown = set(args.demo) - {program["name"] for program in catalogs["demo"]}
                if unknown:
                    checks.append(
                        {"name": "demo-selection", "status": "failed", "reason": f"Unknown demos: {sorted(unknown)}"}
                    )
                    return 1
            if args.examples:
                check("example-list", ["example", "list"])
            for command in ("demo", "example") if args.examples else ("demo",):
                for program in catalogs[command]:
                    name = program["name"]
                    if command == "demo" and args.demo and name not in args.demo:
                        continue
                    if program["missing_modules"] and not (command == "demo" and args.demo):
                        reason = f"Missing {program['missing_modules']}; requires extras {program['extras']}"
                        checks.append({"name": f"{command}-{name}", "status": "skipped", "reason": reason})
                        print(f"SKIPPED {command}-{name}: {reason}")
                        continue
                    result = check(f"{command}-{name}-help", [command, name, "--help"])
                    if result["status"] == "passed" and command == "demo" and args.run:
                        forwarded = args.demo_args[1:] if args.demo_args[:1] == ["--"] else args.demo_args
                        check(f"demo-{name}-run", [command, name, *forwarded, "--max_steps", str(args.steps), "--info"])
    except (OSError, ValueError, KeyError) as error:
        checks.append({"name": "qa-runner", "status": "failed", "reason": str(error)})
        print(f"FAILED  qa-runner: {error}")
    except KeyboardInterrupt:
        checks.append({"name": "qa-runner", "status": "failed", "reason": "Interrupted"})
    finally:
        report["passed"] = bool(checks) and all(result["status"] != "failed" for result in checks)
        report["counts"] = {
            status: sum(result["status"] == status for result in checks) for status in ("passed", "failed", "skipped")
        }
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Results: {report['counts']}")
        print(f"Report: {output_dir / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
