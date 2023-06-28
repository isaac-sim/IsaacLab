# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import os
import shutil

from omni.isaac.orbit.terrains.config.rough import ASSORTED_TERRAINS_CFG
from omni.isaac.orbit.terrains.terrain_generator import TerrainGenerator

if __name__ == "__main__":
    # Create directory to dump results
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "generator")
    # remove directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # create directory
    os.makedirs(output_dir, exist_ok=True)
    # modify the config to cache
    ASSORTED_TERRAINS_CFG.use_cache = True
    ASSORTED_TERRAINS_CFG.cache_dir = output_dir
    # generate terrains
    terrain_generator = TerrainGenerator(cfg=ASSORTED_TERRAINS_CFG, curriculum=False)
