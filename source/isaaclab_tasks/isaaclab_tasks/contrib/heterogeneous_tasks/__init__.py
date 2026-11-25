# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heterogeneous tasks for vision-language-action policies: a Franka with a Robotiq gripper manipulating objects, on
Newton.

One flat catalog supplies bounded object combinations. Reset places objects on the reachable part of
a variable NVIDIA-hosted SimReady table, and category/size rules expose the tasks that can be evaluated in each world.
"""
