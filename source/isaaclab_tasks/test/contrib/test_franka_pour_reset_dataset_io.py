# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Franka Pour reset-artifact provenance."""

import pytest

from isaaclab_tasks.contrib.franka_pour.reset_dataset_io import (
    FRANKA_POUR_RESET_DATASET_STATIC_VALIDATION_CHECKS,
    FRANKA_POUR_RESET_DATASET_STATIC_VALIDATION_POLICY,
    _validate_production_marker,
)


def test_one_file_generator_marker_is_accepted():
    """Artifacts truthfully marked as Newton-screened are loadable without fake calibration."""
    _validate_production_marker(
        {
            "static_validation": {
                "policy": FRANKA_POUR_RESET_DATASET_STATIC_VALIDATION_POLICY,
                "all_rows_statically_validated": True,
                "per_row_mpm_rollout": False,
                "checks": FRANKA_POUR_RESET_DATASET_STATIC_VALIDATION_CHECKS,
            }
        }
    )


def test_one_file_generator_marker_requires_every_check():
    """A shortened check list cannot claim the one-file generator provenance."""
    with pytest.raises(ValueError, match="checks are incomplete"):
        _validate_production_marker(
            {
                "static_validation": {
                    "policy": FRANKA_POUR_RESET_DATASET_STATIC_VALIDATION_POLICY,
                    "all_rows_statically_validated": True,
                    "per_row_mpm_rollout": False,
                    "checks": FRANKA_POUR_RESET_DATASET_STATIC_VALIDATION_CHECKS[:-1],
                }
            }
        )
