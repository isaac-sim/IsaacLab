# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility wrapper for ``isaaclab list_envs``."""

import sys

from isaaclab.cli.commands.list_envs import command_list_envs

if __name__ == "__main__":
    command_list_envs(sys.argv[1:])
