# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""System-identification task family.

Environments here are not RL tasks: they replay recorded joint trajectories
open-loop through the actuators while an external optimizer (see
``scripts/sysid/fit.py``) assigns a different physical-parameter candidate to
each parallel env and minimizes the simulated-vs-measured joint position error.
"""
