# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
from pathlib import Path

from .commands.envs import command_setup_conda, command_setup_uv
from .commands.format import command_format
from .commands.install import (
    CORE_ISAACLAB_SUBMODULES,
    OPTIONAL_ISAACLAB_SUBMODULES,
    VALID_EXTRA_FEATURES,
    command_install,
)
from .commands.misc import (
    command_build_docs,
    command_new,
    command_run_docker,
    command_run_isaacsim,
    command_test,
    command_vscode_settings,
)
from .utils import (
    ISAACLAB_ROOT,
    is_windows,
    run_python_command,
)


def train(args: list[str] | None = None) -> None:
    """Run the unified reinforcement learning training script."""
    if args is None:
        args = sys.argv[1:]
    run_python_command(ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "train.py", args, check=True)


def train_multigpu(args: list[str] | None = None) -> None:
    """Run the unified multi-GPU reinforcement learning training script."""
    if args is None:
        args = sys.argv[1:]
    run_python_command(
        ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "train_multigpu.py", args, check=True
    )


def play(args: list[str] | None = None) -> None:
    """Run the unified reinforcement learning play script."""
    if args is None:
        args = sys.argv[1:]
    run_python_command(ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "play.py", args, check=True)


def cli() -> None:
    """Parse CLI arguments and run the requested command."""
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "train_multigpu":
        train_multigpu(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "play":
        play(sys.argv[2:])
        return

    executable_name = Path(sys.argv[0]).name
    default_prog = "isaaclab.bat" if is_windows() else "isaaclab.sh"
    parser = argparse.ArgumentParser(
        description="Isaac Lab CLI",
        prog=executable_name if executable_name != "__main__.py" else default_prog,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "commands:\n"
            "  train           Run scripts/reinforcement_learning/train.py\n"
            "  train_multigpu  Run scripts/reinforcement_learning/train_multigpu.py\n"
            "  play            Run scripts/reinforcement_learning/play.py"
        ),
    )

    _optional_str = ", ".join(sorted(OPTIONAL_ISAACLAB_SUBMODULES))
    _extras_str = ", ".join(sorted(VALID_EXTRA_FEATURES))
    _core_str = ", ".join(CORE_ISAACLAB_SUBMODULES)
    parser.add_argument(
        "-i",
        "--install",
        nargs="?",
        const="all",
        help=(
            "Install Isaac Lab submodules and optional extra dependencies.\n"
            "\n"
            "All core submodules are always installed:\n"
            f"  {_core_str}\n"
            "\n"
            "Accepts a comma-separated list of optional submodule names and/or\n"
            "extra feature selectors, or one of the special values below.\n"
            "\n"
            f"* Optional submodules: {_optional_str}\n"
            "  Installed by 'all' or by explicit token.\n"
            "\n"
            f"* Extra feature sets: {_extras_str}\n"
            "  Install optional heavy dependencies for a feature on top of the core.\n"
            "  Supports an optional selector in brackets:\n"
            "    contrib[rlinf]\n"
            "    ov[ovrtx|ovphysx|all]\n"
            "    rl[rsl-rl|skrl|sb3|rl-games]  (default: all)\n"
            "    visualizer[kit|newton|rerun|viser]  (default: all)\n"
            "  On Linux/macOS, quote selectors containing brackets:\n"
            "    --install 'rl[rsl-rl]'\n"
            "\n"
            "* Special values:\n"
            "  all   - Core + optional submodules (mimic, teleop) + auto extra\n"
            "          features (newton, rl, visualizer). Does not install contrib/ov\n"
            "          dependency extras (default).\n"
            "  core  - Core submodules only; no optional submodules, no extra features.\n"
            "  <empty> (-i with no value) - Same as 'all'.\n"
            "\n"
            "Note: Contrib and OV source packages are core; runtime dependencies require selectors:\n"
            "  ./isaaclab.sh -i 'contrib[rlinf]'\n"
            "  ./isaaclab.sh -i 'ov[ovrtx]'\n"
            "\n"
            "Examples:\n"
            "  ./isaaclab.sh -i\n"
            "  ./isaaclab.sh -i core\n"
            "  ./isaaclab.sh -i newton,'rl[rsl-rl]'\n"
            "  ./isaaclab.sh -i mimic,teleop,'visualizer[rerun]'\n"
            "  ./isaaclab.sh -i 'ov[ovrtx]'\n"
            "\n"
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
        help="Run the python executable provided by Isaac Sim or virtual environment (if active).",
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
        help="Run all python pytest tests.",
    )
    parser.add_argument(
        "-o",
        "--docker",
        nargs=argparse.REMAINDER,
        help="Run the docker container helper script (docker/container.sh).",
    )
    parser.add_argument(
        "-v",
        "--vscode",
        action="store_true",
        help="Generate the VSCode settings file from template.",
    )
    parser.add_argument(
        "-d",
        "--docs",
        action="store_true",
        help="Build the documentation from source using sphinx.",
    )
    parser.add_argument(
        "-n",
        "--new",
        nargs=argparse.REMAINDER,
        help="Create a new external project or internal task from template.",
    )
    parser.add_argument(
        "-c",
        "--conda",
        nargs="?",
        const="env_isaaclab",
        help="Create a new conda environment for Isaac Lab. Default name is 'env_isaaclab'.",
    )
    parser.add_argument(
        "-u",
        "--uv",
        nargs="?",
        const="env_isaaclab",
        help="Create a new uv environment for Isaac Lab. Default name is 'env_isaaclab'.",
    )

    args = parser.parse_args()

    if args.install:
        command_install(args.install)

    elif args.format:
        command_format()

    elif args.conda:
        command_setup_conda(args.conda)

    elif args.uv:
        command_setup_uv(args.uv)

    elif args.vscode:
        command_vscode_settings()

    elif args.docs:
        command_build_docs()

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
