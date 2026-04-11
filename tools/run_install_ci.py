#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified runner for Isaac Lab installation CI tests.

Modes

``docker``
    Build a clean-room Docker container and execute pytest inside it.
``native``
    Run pytest directly on the host (no Docker).

Examples

.. code-block:: bash

    # Run all tests inside Docker (Ubuntu 24.04)
    tools/run_install_ci.py docker

    # Run only pip tests with a custom base image
    tools/run_install_ci.py docker --base-image ubuntu:22.04 -- -vs -k "testname"

    # Run with GPU support (passes --gpus all to Docker)
    tools/run_install_ci.py docker --gpu

    # Filter by marker (e.g. only uv tests, only slow tests)
    tools/run_install_ci.py docker -- -m uv
    tools/run_install_ci.py docker --gpu -- -m "slow and gpu"

    # Filter by bug ID (dashes become underscores)
    tools/run_install_ci.py docker --gpu -- -m nvbugs_5968136

    # Drop into a shell for debugging
    tools/run_install_ci.py docker --shell

    # Run natively (no Docker)
    tools/run_install_ci.py native -- -vs

    # Pass a pre-built wheel
    tools/run_install_ci.py docker --wheel /tmp/isaaclab-3.0.0-py3-none-any.whl
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_DIM = "\033[2m"
_MAGENTA = "\033[95m"
_RESET = "\033[0m"

# Controls whether run_cmd() streams output by default.
stream_output: bool = False


def run_cmd(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 600,
    check: bool = True,
    stream: bool | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, merging *env* into the current environment.

    Args:
        args: Command and arguments to run.
        cwd: Working directory for the subprocess.
        env: Extra environment variables merged into the current environment.
        timeout: Timeout in seconds.
        check: Raise CalledProcessError on non-zero exit.
        stream: When True, stream stdout/stderr to the console in
            real time instead of capturing them.

    Returns:
        The CompletedProcess; raises CalledProcessError when *check* is
        True and return code != 0.
    """
    if stream is None:
        stream = stream_output
    merged_env = {**os.environ, **(env or {})}
    cmd_str = " ".join(str(a) for a in args)
    if stream:
        sys.stdout.write(f"{_MAGENTA}[COMMAND] {cmd_str}{_RESET}\n")
        sys.stdout.flush()
        t0 = time.monotonic()
        proc = subprocess.Popen(
            [str(a) for a in args],
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        lines: list[str] = []
        try:
            for line in proc.stdout:
                lines.append(line)
                sys.stdout.write(f"{_DIM}{line}{_RESET}")
                sys.stdout.flush()
        except Exception:
            proc.kill()
            raise
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        elapsed = time.monotonic() - t0
        sys.stdout.write(f"{_MAGENTA}[{elapsed:.1f}s]{_RESET}\n")
        sys.stdout.flush()
        result = subprocess.CompletedProcess(
            args=proc.args,
            returncode=proc.returncode,
            stdout="".join(lines),
            stderr="",
        )
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
        return result
    return subprocess.run(
        [str(a) for a in args],
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def _find_repo_root() -> Path:
    """Walk up from CWD or this file to find the repo root."""
    for anchor in [Path.cwd(), Path(__file__).resolve().parent]:
        for parent in [anchor] + list(anchor.parents):
            if (parent / "isaaclab.sh").exists():
                return parent
    raise FileNotFoundError("Could not locate IsaacLab repository root")


# Docker mode


def _cmd_docker(args: argparse.Namespace) -> int:
    """Build the Docker image and run tests inside the container based on *args*."""

    repo_root = _find_repo_root()
    dockerfile = repo_root / "docker" / "Dockerfile.installci"
    image_tag = f"isaaclab-installci:{args.base_image.replace(':', '-').replace('/', '-')}"

    # Build the Docker image
    build_cmd = [
        "docker",
        "build",
        "--build-arg",
        f"BASE_IMAGE={args.base_image}",
        "-f",
        str(dockerfile),
        "-t",
        image_tag,
        "--progress=quiet",
    ]

    if args.no_cache:
        build_cmd.append("--no-cache")

    build_cmd.append(str(repo_root))

    result = run_cmd(build_cmd, check=False, stream=True)
    if result.returncode != 0:
        print(f"Docker build failed (exit {result.returncode})")
        return result.returncode

    print(f"Docker image built successfully: {image_tag}")

    # Run
    docker_run_cmd: list[str] = [
        "docker",
        "run",
        "--rm",
        "--network=host",
    ]

    if args.gpu:
        docker_run_cmd.extend(["--gpus", "all"])

    # Persist pip and uv caches across runs via named Docker volumes
    if not args.no_pip_cache:
        docker_run_cmd.extend(["-v", "isaaclab-installci-pip-cache:/root/.cache/pip"])
    if not args.no_uv_cache:
        docker_run_cmd.extend(["-v", "isaaclab-installci-uv-cache:/root/.cache/uv"])

    # Pass environment variables
    docker_run_cmd.extend(["-e", "OMNI_KIT_ACCEPT_EULA=Y"])
    docker_run_cmd.extend(["-e", "ACCEPT_EULA=Y"])

    if args.results_dir:
        results_abs = Path(args.results_dir).resolve()
        results_abs.mkdir(parents=True, exist_ok=True)
        docker_run_cmd.extend(["-v", f"{results_abs}:/tmp/results"])

    if args.wheel:
        wheel_abs = Path(args.wheel).resolve()
        container_wheel = f"/tmp/wheels/{wheel_abs.name}"
        docker_run_cmd.extend(["-v", f"{wheel_abs}:{container_wheel}:ro"])
        docker_run_cmd.extend(["-e", f"ISAACLAB_WHEEL={container_wheel}"])

    if args.shell:
        # Interactive debugging mode
        docker_run_cmd.extend(["-it", "--entrypoint", "bash", image_tag])
    else:
        # Test execution mode
        pytest_args = args.pytest_args or ["--tb=short"]
        if args.results_dir:
            pytest_args = ["--junitxml=/tmp/results/results.xml"] + pytest_args
        docker_run_cmd.extend([image_tag] + pytest_args)

    print(f"{_MAGENTA}[COMMAND] {' '.join(docker_run_cmd)}{_RESET}")

    t0 = time.monotonic()
    try:
        ret = subprocess.call(docker_run_cmd, timeout=5400)
    except subprocess.TimeoutExpired:
        print("Docker run timed out after 90 minutes", file=sys.stderr)
        ret = 124
    elapsed = time.monotonic() - t0
    print(f"{_MAGENTA}[{elapsed:.1f}s]{_RESET}")
    return ret


# Native mode


def _cmd_native(args: argparse.Namespace) -> int:
    """Run tests directly on the host OS."""

    repo_root = _find_repo_root()
    test_dir = repo_root / "source" / "isaaclab" / "test" / "install_ci"

    env = os.environ.copy()
    if args.wheel:
        env["ISAACLAB_WHEEL"] = str(Path(args.wheel).resolve())

    pytest_args = args.pytest_args or ["--tb=short"]
    cmd = [sys.executable, "-m", "pytest"] + pytest_args

    print(f"{_MAGENTA}[COMMAND] {' '.join(cmd)}{_RESET}")

    t0 = time.monotonic()
    ret = subprocess.call(cmd, cwd=str(test_dir), env=env)
    elapsed = time.monotonic() - t0
    print(f"{_MAGENTA}[{elapsed:.1f}s]{_RESET}")
    return ret


# CLI


def main() -> int:
    """Parse CLI arguments and dispatch to the docker or native test runner.

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    parser = argparse.ArgumentParser(
        description="Isaac Lab Installation CI test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
docker options:
  --base-image IMAGE   Docker base image (default: ubuntu:24.04)
  --gpu                Pass --gpus all to docker run
  --shell              Drop into interactive bash instead of running tests
  --no-cache           Build Docker image without layer cache
  --no-pip-cache       Disable persistent pip cache volume
  --no-uv-cache        Disable persistent uv cache volume
  --results-dir DIR    Host directory for test results (auto-adds --junitxml)
  --wheel PATH         Path to pre-built isaaclab wheel file

native options:
  --wheel PATH         Path to pre-built isaaclab wheel file

pytest arguments:
  Pass pytest options after '--'. Without '--', defaults to '-sv --tb=short'.

  examples:
    %(prog)s docker                                          # run all tests in Docker
    %(prog)s docker --base-image ubuntu:22.04 -- -vs -k "testname"  # custom base image
    %(prog)s docker --gpu                                    # GPU support (--gpus all)
    %(prog)s docker -- -m uv                                 # filter by marker
    %(prog)s docker --gpu -- -m "slow and gpu"               # combine markers with GPU
    %(prog)s docker --gpu -- -m nvbugs_5968136               # filter by bug ID
    %(prog)s docker --shell                                  # drop into shell for debugging
    %(prog)s native -- -vs                                   # run natively (no Docker)
    %(prog)s docker --wheel /tmp/isaaclab.whl                # pass a pre-built wheel
    %(prog)s docker -- --collect-only                        # list tests without running
""",
    )
    sub = parser.add_subparsers(dest="mode")

    # docker subcommand
    docker_p = sub.add_parser("docker", help="Build container and run tests inside Docker")
    docker_p.add_argument(
        "--base-image",
        default="ubuntu:24.04",
        help="Docker base image (default: ubuntu:24.04)",
    )
    docker_p.add_argument("--gpu", action="store_true", help="Pass --gpus all to docker run")
    docker_p.add_argument(
        "--shell", action="store_true", help="Drop into an interactive bash shell instead of running tests"
    )
    docker_p.add_argument("--no-cache", action="store_true", help="Build Docker image without cache")
    docker_p.add_argument("--no-pip-cache", action="store_true", help="Disable persistent pip cache volume")
    docker_p.add_argument("--no-uv-cache", action="store_true", help="Disable persistent uv cache volume")
    docker_p.add_argument(
        "--results-dir", type=str, default=None, help="Host directory for test results (auto-adds --junitxml)"
    )
    docker_p.add_argument("--wheel", type=str, default=None, help="Path to pre-built isaaclab wheel file")
    docker_p.add_argument("pytest_args", nargs="*", help="Arguments forwarded to pytest (use -- to separate)")

    # native subcommand
    native_p = sub.add_parser("native", help="Run tests directly on the host OS")
    native_p.add_argument("--wheel", type=str, default=None, help="Path to pre-built isaaclab wheel file")
    native_p.add_argument("pytest_args", nargs="*", help="Arguments forwarded to pytest (use -- to separate)")

    # If '--' is in sys.argv, split there so pytest args are captured correctly
    argv = sys.argv[1:]
    if "--" in argv:
        split_idx = argv.index("--")
        our_args = argv[:split_idx]
        pytest_extra = argv[split_idx + 1 :]
    else:
        our_args = argv
        pytest_extra = []

    args = parser.parse_args(our_args)

    # Merge any args after '--' into pytest_args
    if pytest_extra:
        args.pytest_args = (args.pytest_args or []) + pytest_extra

    if args.mode == "docker":
        if not shutil.which("docker"):
            print("Error: docker is not available on PATH", file=sys.stderr)
            return 1
        return _cmd_docker(args)
    elif args.mode == "native":
        return _cmd_native(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
