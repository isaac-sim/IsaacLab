# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark CLI value validation."""

import pytest

from isaaclab.test.benchmark._cli import validate_warmup_steps


@pytest.mark.parametrize(("warmup_steps", "available_steps"), [(0, 1), (15, 16)])
def test_validate_warmup_steps_accepts_a_remaining_sample(warmup_steps: int, available_steps: int):
    validate_warmup_steps(warmup_steps, available_steps)


@pytest.mark.parametrize(("warmup_steps", "available_steps"), [(1, 1), (17, 16)])
def test_validate_warmup_steps_rejects_exhausted_workload(warmup_steps: int, available_steps: int):
    with pytest.raises(ValueError, match="must be less than resolved training environment steps"):
        validate_warmup_steps(warmup_steps, available_steps)
