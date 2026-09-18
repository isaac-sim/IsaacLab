# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from isaaclab.envs.utils.io_descriptors import generic_io_descriptor

pytestmark = pytest.mark.unit


def test_generic_io_descriptor_accepts_function_without_docstring():
    def observation(env):
        return env

    assert observation.__doc__ is None

    decorated = generic_io_descriptor()(observation)

    assert decorated._descriptor.description is None
    marker = object()
    assert decorated(marker) is marker
