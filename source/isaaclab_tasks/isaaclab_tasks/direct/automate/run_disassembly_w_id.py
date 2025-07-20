# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import re
import subprocess
import sys


def update_task_param(task_cfg, assembly_id, disassembly_dir):
    # Read the file lines.
    with open(task_cfg) as f:
        lines = f.readlines()

    updated_lines = []

    # Regex patterns to capture the assignment lines
    assembly_pattern = re.compile(r"^(.*assembly_id\s*=\s*).*$")
    disassembly_dir_pattern = re.compile(r"^(.*disassembly_dir\s*=\s*).*$")

    for line in lines:
        if "assembly_id =" in line:
            line = assembly_pattern.sub(rf"\1'{assembly_id}'", line)
        elif "disassembly_dir = " in line:
            line = disassembly_dir_pattern.sub(rf"\1'{disassembly_dir}'", line)

        updated_lines.append(line)

    # Write the modified lines back to the file.
    with open(task_cfg, "w") as f:
        f.writelines(updated_lines)


def main():
    parser = argparse.ArgumentParser(description="Update assembly_id and run training script.")
    parser.add_argument(
        "--disassembly_dir",
        type=str,
        help="Path to the directory containing output disassembly trajectories.",
        default="disassembly_dir",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Path to the file containing assembly_id.",
        default="source/isaaclab_tasks/isaaclab_tasks/direct/automate/disassembly_tasks_cfg.py",
    )
    parser.add_argument("--assembly_id", type=str, default="00731", help="New assembly ID to set.")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environment.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed.")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode.")
    args = parser.parse_args()

    os.makedirs(args.disassembly_dir, exist_ok=True)

    update_task_param(
        args.cfg_path,
        args.assembly_id,
        args.disassembly_dir,
    )

    if sys.platform.startswith("win"):
        bash_command = "isaaclab.bat -p"
    elif sys.platform.startswith("linux"):
        bash_command = "./isaaclab.sh -p"

    bash_command += " scripts/reinforcement_learning/rl_games/train.py --task=Isaac-AutoMate-Disassembly-Direct-v0"

    bash_command += f" --num_envs={str(args.num_envs)}"
    bash_command += f" --seed={str(args.seed)}"

    if args.headless:
        bash_command += " --headless"

    # Run the bash command
    subprocess.run(bash_command, shell=True, check=True)


if __name__ == "__main__":
    main()
