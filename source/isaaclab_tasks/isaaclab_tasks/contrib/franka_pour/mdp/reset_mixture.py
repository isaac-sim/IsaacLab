# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated compatibility names for the Franka Pour reset-dataset curriculum."""

from .reset_dataset import PourResetDatasetCurriculum

RESET_MIXTURE_REGION_NAMES = ("reaching", "near_object", "grasped", "near_goal")
"""Deprecated region names retained for import compatibility."""

RESET_MIXTURE_STAGE_NAMES = ("randomized", "grasp", "carry", "tilt")
"""Deprecated stage names retained for import compatibility."""

PourResetMixture = PourResetDatasetCurriculum
"""Deprecated compatibility name for :class:`PourResetDatasetCurriculum`."""
