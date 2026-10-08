# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A Franka picks deformable raspberries: MPM physics on Newton, rendered as 3D Gaussians.

Each step of the environment (:mod:`.pick_berries_env`) runs two coupled Newton solvers: MuJoCo-Warp moves the arm
and an MPM solver deforms, bruises and tears the berries' tissue, which the finger pads grip by friction
(:mod:`.physics`). Each frame then moves the berries' hundreds of thousands of Gaussians with their few thousand tissue
particles and streams them to the RTX renderer (:mod:`.gaussians`).

Core: :mod:`.pick_berries_env_cfg` (the scene), :mod:`.pick_berries_env`, :mod:`.physics` and :mod:`.gaussians`.
Helpers: :mod:`.mdp` (actions), :mod:`.control` (teleoperation and the scripted sorting), :mod:`.rendering` (viewer,
camera director, video), :mod:`.scene` (table, tableware, room) and :mod:`.assets`. ``scripts/pick_berries.py`` runs
the demo and ``setup/setup.sh`` installs its runtime.
"""
