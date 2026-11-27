# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for configured first-subtask start offsets."""

from types import SimpleNamespace

import numpy as np
import pytest

from isaaclab_mimic.datagen.data_generator import DataGenerator

pytestmark = pytest.mark.kitless


@pytest.mark.parametrize("offset_range", [(0, 0), (2, 9)])
def test_first_subtask_start_offsets(offset_range: tuple[int, int]) -> None:
    """Honor both configured bounds while leaving source boundaries and end indices unchanged."""
    source = np.tile([[0, 80], [80, 160]], (256, 1, 1))
    configs = [
        SimpleNamespace(first_subtask_start_offset_range=offset_range, subtask_term_offset_range=(0, 0)),
        SimpleNamespace(subtask_term_offset_range=(0, 0)),
    ]
    generator = DataGenerator.__new__(DataGenerator)
    generator.env_cfg = SimpleNamespace(
        subtask_configs={"arm": configs}, datagen_config=SimpleNamespace(use_skillgen=False)
    )
    generator.src_demo_datagen_info_pool = SimpleNamespace(subtask_boundaries={"arm": source.tolist()})
    rng_state = np.random.get_state()
    try:
        np.random.seed(42)
        result = generator.randomize_subtask_boundaries()["arm"]
    finally:
        np.random.set_state(rng_state)

    starts = result[:, 0, 0]
    # The fixed seed and batch exercise both inclusive endpoints without a flaky retry loop.
    assert starts.min() == offset_range[0]
    assert starts.max() == offset_range[1]
    np.testing.assert_array_equal(result[:, :, 1], source[:, :, 1])
    np.testing.assert_array_equal(result[:, 1, 0], source[:, 1, 0])
    np.testing.assert_array_equal(generator.src_demo_datagen_info_pool.subtask_boundaries["arm"], source)
