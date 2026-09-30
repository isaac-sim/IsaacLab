# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Continuous-grasp Franka teleoperation with deformable Gaussian berries.

The environment lives in :mod:`pick_berries_env` / :mod:`pick_berries_env_cfg`, the action terms in
:mod:`mdp`, and the gym registration in :mod:`config.franka`. Runtime support is grouped into
:mod:`physics`, :mod:`rendering`, :mod:`scene`, :mod:`assets` and :mod:`control`. Interactive and
benchmark CLIs are in ``scripts/``, offline asset tools in ``offline/`` and the runtime installer in
``setup/``.
"""
