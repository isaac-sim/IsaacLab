# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path

from .commands.format import command_format
from .commands.list_envs import command_list_envs
from .commands.misc import (
    command_build_docs,
    command_build_isaacsim,
    command_editor,
    command_new,
    command_run_docker,
    command_run_isaacsim,
    command_test,
)
from .utils import (
    DEFAULT_ISAAC_SIM_PATH,
    ISAACLAB_ROOT,
    is_isaac_sim_source_build,
    runs_isaac_sim_python,
    run_python_command,
)

_TASK_ENTRY_POINT_GROUP = "isaaclab.tasks"


def _exit_on_error(status: int) -> None:
    """Raise ``SystemExit`` when an in-process command reports a failure."""
    if status != 0:
        raise SystemExit(status)


def _load_external_tasks() -> None:
    """Import task packages registered by installed downstream projects."""
    for entry_point in importlib.metadata.entry_points(group=_TASK_ENTRY_POINT_GROUP):
        entry_point.load()


def train(args: list[str] | None = None) -> None:
    """Run unified reinforcement learning training."""
    from isaaclab_rl.entrypoints import run_train_cli

    _exit_on_error(run_train_cli(args))


def train_multigpu(args: list[str] | None = None) -> None:
    """Run unified multi-GPU reinforcement learning training."""
    from isaaclab_rl.entrypoints import run_train_multigpu_cli

    _exit_on_error(run_train_multigpu_cli(args))


def play(args: list[str] | None = None) -> None:
    """Run unified reinforcement learning playback."""
    from isaaclab_rl.entrypoints import run_play_cli

    _exit_on_error(run_play_cli(args))


def leapp(args: list[str] | None = None) -> None:
    """Export or deploy a policy with LEAPP."""
    parser = argparse.ArgumentParser(
        description="Export or deploy policies with LEAPP.",
        prog=f"{Path(sys.argv[0]).name} leapp",
    )
    parser.add_argument("command", choices=("export", "deploy"), help="LEAPP workflow to run.")
    if args is None:
        args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        parser.parse_args(args)
    parsed_args = parser.parse_args(args[:1])
    command_args = args[1:]

    if parsed_args.command == "export":
        from isaaclab_rl.entrypoints import run_export_cli

        _exit_on_error(run_export_cli(command_args))
    else:
        from .commands.deploy import command_deploy_leapp

        _exit_on_error(command_deploy_leapp(command_args))


def zero_agent(args: list[str] | None = None) -> None:
    """Run an environment with a zero-action agent."""
    from isaaclab_rl.entrypoints import run_zero_agent_cli

    _exit_on_error(run_zero_agent_cli(args))


def random_agent(args: list[str] | None = None) -> None:
    """Run an environment with a random-action agent."""
    from isaaclab_rl.entrypoints import run_random_agent_cli

    _exit_on_error(run_random_agent_cli(args))


def list_envs(args: list[str] | None = None) -> None:
    """List registered Isaac Lab environments."""
    command_list_envs(args)


def demo(args: list[str] | None = None) -> None:
    """List or run a packaged Isaac Lab demo.

    Args:
        args: Command-line arguments. Uses ``sys.argv`` when omitted.
    """
    from isaaclab.programs import DEMOS, run_program_cli

    run_program_cli("demo", DEMOS, args)


def example(args: list[str] | None = None) -> None:
    """List or run a packaged Isaac Lab example.

    Args:
        args: Command-line arguments. Uses ``sys.argv`` when omitted.
    """
    from isaaclab.programs import EXAMPLES, run_program_cli

    run_program_cli("example", EXAMPLES, args)


def teleop(args: list[str] | None = None) -> None:
    """Run a live teleoperation, demonstration recording, or demonstration replay workflow.

    Args:
        args: Command-line arguments. Uses ``sys.argv`` when omitted.
    """
    workflow_scripts = {
        "run": ISAACLAB_ROOT / "scripts" / "environments" / "teleoperation" / "teleop_se3_agent.py",
        "record": ISAACLAB_ROOT / "scripts" / "tools" / "record_demos.py",
        "replay": ISAACLAB_ROOT / "scripts" / "tools" / "replay_demos.py",
    }
    parser = argparse.ArgumentParser(description="Run an Isaac Lab teleoperation workflow.")
    parser.add_argument("command", choices=tuple(workflow_scripts), help="Teleoperation workflow to run.")
    if args is None:
        args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        parser.parse_args(args)
    parsed_args = parser.parse_args(args[:1])
    run_python_command(workflow_scripts[parsed_args.command], args[1:], check=True)


def benchmark(args: list[str] | None = None) -> None:
    """Run a runtime, startup, training, or play benchmark, optionally across several GPUs.

    Args:
        args: Command-line arguments. Uses sys.argv when omitted.
    """
    from ..benchmark import run_benchmark_cli

    _exit_on_error(run_benchmark_cli(args))


def microbenchmark(args: list[str] | None = None) -> None:
    """Run a component micro-benchmark with an exact physics variant."""
    from ..benchmark import run_microbenchmark_cli

    _exit_on_error(run_microbenchmark_cli(args))


def cli() -> None:
    """Parse CLI arguments and run the requested command."""
    subcommands = {
        "benchmark": benchmark,
        "leapp": leapp,
        "microbenchmark": microbenchmark,
        "train": train,
        "train_multigpu": train_multigpu,
        "play": play,
        "zero_agent": zero_agent,
        "random_agent": random_agent,
    }
    # uv selects the environment; local Kit builds still need their native runtime paths.
    # Delegate before importing tasks or simulator libraries, and avoid wrapping the child twice.
    runtime_commands = {*subcommands, "list_envs", "demo", "example", "teleop"}
    configured_path = os.environ.get("ISAAC_PATH")
    if (
        len(sys.argv) > 1
        and sys.argv[1] in runtime_commands
        and (DEFAULT_ISAAC_SIM_PATH / ("python.bat" if os.name == "nt" else "python.sh")).is_file()
        and (
            is_isaac_sim_source_build(DEFAULT_ISAAC_SIM_PATH)
            or runs_isaac_sim_python(DEFAULT_ISAAC_SIM_PATH, sys.executable, os.environ.get("VIRTUAL_ENV"))
        )
        and (configured_path is None or Path(configured_path).resolve() != DEFAULT_ISAAC_SIM_PATH.resolve())
    ):
        run_python_command("-m", ["isaaclab", *sys.argv[1:]], check=True)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "list_envs":
        list_envs(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "example":
        example(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] in subcommands:
        _load_external_tasks()
        subcommands[sys.argv[1]](sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "teleop":
        teleop(sys.argv[2:])
        return

    executable_name = Path(sys.argv[0]).name
    default_prog = "isaaclab"
    parser = argparse.ArgumentParser(
        description="Isaac Lab CLI",
        prog=executable_name if executable_name != "__main__.py" else default_prog,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "commands:\n"
            "  benchmark       Run a runtime, startup, training, or play benchmark\n"
            "                  (append _multigpu to a workflow to run it across GPUs)\n"
            "  microbenchmark  Run a component micro-benchmark\n"
            "  demo            List or run packaged demonstrations\n"
            "  example         List or run packaged standalone examples\n"
            "  leapp           Export or deploy a policy with LEAPP\n"
            "  list_envs       List registered environments and presets\n"
            "  train           Train an RL policy\n"
            "  train_multigpu  Train an RL policy across multiple GPUs\n"
            "  play            Play a trained RL policy\n"
            "  zero_agent      Run an environment with zero actions\n"
            "  random_agent    Run an environment with random actions\n"
            "  teleop          Run a live teleoperation, demo recording, or demo replay workflow"
        ),
    )

    parser.add_argument(
        "-f",
        "--format",
        action="store_true",
        help="Run pre-commit to format the code and check lints.",
    )
    parser.add_argument(
        "-p",
        "--python",
        nargs=argparse.REMAINDER,
        help="Run Python in the active environment, initializing the linked Isaac Sim runtime when needed.",
    )
    parser.add_argument(
        "-s",
        "--sim",
        nargs=argparse.REMAINDER,
        help="Run the simulator executable (isaac-sim.sh) provided by Isaac Sim.",
    )
    parser.add_argument(
        "-t",
        "--test",
        nargs=argparse.REMAINDER,
        help="Run the repository tooling tests under tools/ with pytest.",
    )
    parser.add_argument(
        "-o",
        "--docker",
        nargs=argparse.REMAINDER,
        help="Run the docker container helper script (docker/container.py).",
    )
    parser.add_argument(
        "--editor",
        nargs=argparse.REMAINDER,
        help="Generate editor settings and import paths for the current workspace.",
    )
    parser.add_argument(
        "-d",
        "--docs",
        action="store_true",
        help="Build the documentation from source using sphinx.",
    )
    parser.add_argument(
        "--docs_multi",
        action="store_true",
        help="Build the multi-version documentation from source using sphinx-multiversion.",
    )
    parser.add_argument(
        "-n",
        "--new",
        nargs=argparse.REMAINDER,
        help="Create a new external project or internal task from template.",
    )
    parser.add_argument(
        "--isaacsim_source",
        metavar="PATH",
        help=(
            "Incrementally build the Isaac Sim source checkout at PATH and link its live release\n"
            "tree as '_isaac_sim'. Python commands keep using the active uv environment."
        ),
    )

    args = parser.parse_args()

    if (
        args.format
        or args.docs
        or args.docs_multi
        or args.docker is not None
        or args.test is not None
        or args.isaacsim_source is not None
    ) and not (ISAACLAB_ROOT / "pyproject.toml").is_file():
        parser.error("This command requires an Isaac Lab source checkout. Run it with uv run isaaclab from that checkout.")

    if args.format:
        command_format()

    elif args.isaacsim_source:
        command_build_isaacsim(args.isaacsim_source)

    elif args.editor is not None:
        command_editor(args.editor)

    elif args.docs:
        command_build_docs()

    elif args.docs_multi:
        command_build_docs(multi_version=True)

    elif args.docker is not None:
        command_run_docker(args.docker)

    elif args.python is not None:
        if args.python:
            run_python_command(args.python[0], args.python[1:], check=True)
        else:
            run_python_command("-i", [], check=True)

    elif args.sim is not None:
        command_run_isaacsim(args.sim)

    elif args.new is not None:
        command_new(args.new)

    elif args.test is not None:
        command_test(args.test)

    else:
        parser.print_help()
