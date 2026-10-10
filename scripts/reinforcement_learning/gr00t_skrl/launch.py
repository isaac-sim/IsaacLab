# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin two-environment launcher; training interaction stays in skrl SequentialTrainer."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from contextlib import ExitStack, suppress
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from scripts.reinforcement_learning.gr00t_skrl.protocol import RunConfig

REPO_ROOT = Path(__file__).absolute().parents[3]


def stop_process(process: subprocess.Popen) -> None:
    """Reap a complete child process group, including uv and runtime descendants."""
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def interrupt(signum: int, frame) -> None:
    """Route termination through the launcher cleanup path."""
    raise KeyboardInterrupt


def main() -> None:
    """Validate paths, launch isolated children and propagate either child's failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path", default=str(REPO_ROOT.parent / "embodied-template/models/stack-cube-n1.7-sft/checkpoint")
    )
    parser.add_argument(
        "--backbone_path", default=str(REPO_ROOT.parent / "embodied-template/models/nvidia/Cosmos-Reason2-2B")
    )
    parser.add_argument("--model_project", default=str(REPO_ROOT.parent / "Isaac-GR00T"))
    parser.add_argument("--run_dir")
    parser.add_argument("--mode", choices=("train", "inference"), default="train")
    parser.add_argument("--resume")
    parser.add_argument("--timesteps", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--learning_epochs", type=int, default=2)
    parser.add_argument("--generation_steps", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--episode_steps", type=int)
    parser.add_argument("--rpc_timeout", type=float, default=300)
    parser.add_argument(
        "--debug_subprocesses",
        action="store_true",
        help="Launch .venv Python interpreters directly so VS Code can automatically debug both children",
    )
    args = parser.parse_args()
    for path in (args.model_path, args.backbone_path, args.model_project):
        if not Path(path).is_dir():
            parser.error(f"Missing directory: {path}")
    if args.resume is not None and not Path(args.resume).is_file():
        parser.error(f"Missing checkpoint: {args.resume}")
    if args.debug_subprocesses:
        for project in (REPO_ROOT, Path(args.model_project).absolute()):
            if not (project / ".venv/bin/python").is_file():
                parser.error(f"Missing debug interpreter: {project / '.venv/bin/python'}")
    run_dir = Path(
        args.run_dir or REPO_ROOT / "logs/gr00t_skrl" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ).absolute()
    run_dir.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, interrupt)
    processes = []
    gpu_samples = []
    next_sample_time = 0.0
    with tempfile.TemporaryDirectory(prefix="gr00t-skrl-") as socket_dir, ExitStack() as stack:
        values = vars(args).copy()
        model_project = values.pop("model_project")
        debug_subprocesses = values.pop("debug_subprocesses")
        values["run_dir"] = str(run_dir)
        values["socket_path"] = str(Path(socket_dir) / "simulation.sock")
        # absolute(), rather than resolve(), preserves the model-family symlink name.
        values["model_path"] = str(Path(values["model_path"]).absolute())
        values["backbone_path"] = str(Path(values["backbone_path"]).absolute())
        if values["resume"]:
            values["resume"] = str(Path(values["resume"]).absolute())
        cfg = RunConfig(**values)
        cfg.validate()
        config_file = run_dir / "config.json"
        config_file.write_text(json.dumps(asdict(cfg), indent=2))
        environment = os.environ.copy()
        environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
        environment.pop("VIRTUAL_ENV", None)
        (run_dir / "simulation_versions.json").write_text(
            json.dumps(
                {package: version(package) for package in ("torch", "numpy", "transformers", "isaacsim")}, indent=2
            )
        )
        environment["TOKENIZERS_PARALLELISM"] = "false"
        environment["HF_HUB_OFFLINE"] = "1"
        environment["NO_ALBUMENTATIONS_UPDATE"] = "1"
        environment["OMP_NUM_THREADS"] = "4"
        environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        try:
            for child, project in (("simulation", str(REPO_ROOT)), ("train", model_project)):
                log = stack.enter_context((run_dir / f"{child}.log").open("w"))
                # The debugger can inject into Python, but cannot follow uv's Rust process into Python.
                command = (
                    [str(Path(project).absolute() / ".venv/bin/python")]
                    if debug_subprocesses
                    else ["uv", "run", "--project", project, "--no-sync", "python"]
                )
                process = subprocess.Popen(
                    [
                        *command,
                        "-u",
                        "-m",
                        f"scripts.reinforcement_learning.gr00t_skrl.{child}",
                        "--config",
                        str(config_file),
                    ],
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append(process)
            print(f"Run directory: {run_dir}", flush=True)
            while processes[1].poll() is None:
                if time.monotonic() >= next_sample_time:
                    query = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    gpu_samples.append(int(query.stdout.strip().splitlines()[0]))
                    next_sample_time = time.monotonic() + 1
                simulation_status = processes[0].poll()
                if simulation_status is not None:
                    # The simulator can exit just before a successful model process finishes.
                    try:
                        model_status = processes[1].wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError(
                            f"Simulation exited with {simulation_status}; inspect simulation.log"
                        ) from None
                    if simulation_status or model_status:
                        raise RuntimeError(f"Children exited with simulation={simulation_status}, model={model_status}")
                    break
                time.sleep(0.2)
            if processes[1].returncode:
                raise RuntimeError(f"Model process exited with {processes[1].returncode}; inspect train.log")
            try:
                simulation_status = processes[0].wait(timeout=30)
            except subprocess.TimeoutExpired:
                raise RuntimeError("Simulation did not exit after model shutdown") from None
            if simulation_status:
                raise RuntimeError(f"Simulation exited with {simulation_status}; inspect simulation.log")
            print("Both processes exited successfully", flush=True)
        finally:
            for process in reversed(processes):
                stop_process(process)
            (run_dir / "processes.json").write_text(
                json.dumps(
                    {
                        "child_pids": [process.pid for process in processes],
                        "exit_codes": [process.returncode for process in processes],
                        "peak_total_gpu_memory_mib": max(gpu_samples, default=0),
                        "socket_path": cfg.socket_path,
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
