# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility executable for ``isaaclab demo rizon-sharpa-coffee``."""

import sys

from isaaclab.cli import demo

if __name__ == "__main__":
    demo(["rizon-sharpa-coffee", *sys.argv[1:]])
