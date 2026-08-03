# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-hand object reorientation task family (allegro hand and shadow hand).

This package consolidates the direct-workflow and manager-based-workflow in-hand
manipulation tasks, where a dexterous hand reorients an object to match a goal
orientation. The shared direct base environment lives in
:mod:`~isaaclab_tasks.core.reorient.reorient_direct_env` and the shared manager-based
base configuration in :mod:`~isaaclab_tasks.core.reorient.reorient_manager_env_cfg`.
Robot-specific tasks are organized under the ``config`` subpackage
(``config/allegro_hand`` and ``config/shadow_hand``).

These environments are based on the `dexterous cube manipulation`_ environments
provided in IsaacGymEnvs repository from NVIDIA. However, they contain certain
modifications and additional features.

.. _dexterous cube manipulation: https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/allegro_hand.py

"""
