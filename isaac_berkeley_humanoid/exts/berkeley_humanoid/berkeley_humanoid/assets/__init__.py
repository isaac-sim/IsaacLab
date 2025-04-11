# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing assets implementations for various robotic environments."""

import os
import toml

# Conveniences to other module directories via relative paths
ISAAC_ASSET_DIR = os.path.abspath(os.path.dirname(__file__))
