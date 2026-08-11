# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the one-file Franka Pour reset generator."""

import pytest

from scripts.tools.generate_franka_pour_reset_dataset import GeneratorCfg, phase_counts


def test_default_phase_counts_match_reference_artifact():
    """The default 20,000 rows preserve the calibrated reference mixture exactly."""
    assert phase_counts(GeneratorCfg()) == (
        1667,
        1667,
        1666,
        1666,
        1667,
        1667,
        1400,
        1400,
        1400,
        1400,
        1400,
        2600,
        300,
        100,
    )


@pytest.mark.parametrize("grasping_count, non_grasping_count", ((99, 10_000), (10_000, 5)))
def test_generator_requires_every_phase(grasping_count: int, non_grasping_count: int):
    """Diagnostic row overrides cannot silently remove a connected phase."""
    with pytest.raises(ValueError, match="cover all 14 phases"):
        GeneratorCfg(grasping_count=grasping_count, non_grasping_count=non_grasping_count)
