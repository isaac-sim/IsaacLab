# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A dictionary of tests and their timeouts in seconds.

Any tests not listed here will use the default timeout.
"""
PER_TEST_TIMEOUTS = {
    "test_articulation.py": 200,
    "test_deformable_object.py": 200,
    "test_environments.py": 1650,  # This test runs through all the environments for 100 steps each
    "test_environment_determinism.py": 200,  # This test runs through many the environments for 100 steps each
    "test_factory_environments.py": 300,  # This test runs through Factory environments for 100 steps each
    "test_env_rendering_logic.py": 300,
    "test_camera.py": 500,
    "test_tiled_camera.py": 300,
    "test_generate_dataset.py": 300,  # This test runs annotation for 10 demos and generation until one succeeds
    "test_rsl_rl_wrapper.py": 200,
    "test_sb3_wrapper.py": 200,
    "test_skrl_wrapper.py": 200,
    "test_operational_space.py": 300,
    "test_terrain_importer.py": 200,
}
