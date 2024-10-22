# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import subprocess
import sys

import isaac_ray_util


def exec(cmd):
    print("---")
    print(f"{cmd = }")
    cmd = " ".join(cmd)
    if "*" in cmd:
        formatted_cmd = cmd.split("*")
        print(f"{formatted_cmd = }")
        for idx, cmd in enumerate(formatted_cmd):
            print(f"Submitting job {idx} of {len(formatted_cmd)} | {cmd = }")
            res = subprocess.run(cmd, shell=True, text=True)
            print(res)
    else:
        formatted_cmd = cmd.split("+")
        print(f"{formatted_cmd = }")
        for idx, cmd in enumerate(formatted_cmd):
            print(f"Submitting job {idx} of {len(formatted_cmd)} | {cmd = }")
            res = subprocess.run(cmd, shell=True, text=True)
            print(res)


if __name__ == "__main__":
    print("Running Arg Parser Test..")
    print(f"{sys.argv = }")
    parser = argparse.ArgumentParser(description="test")
    isaac_ray_util.add_cluster_args(parser)
    parser.add_argument(
        "--jobs",
        type=str,
        nargs=argparse.REMAINDER,
        help="!!Should be last argument supplied!!! Separate clusters by asterisk (*), and jobs for each by +",
    )
    args = parser.parse_args()

    exec(args.jobs)
