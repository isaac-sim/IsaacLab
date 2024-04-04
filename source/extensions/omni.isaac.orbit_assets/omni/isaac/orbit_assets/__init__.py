# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations."""

import os
import toml

# Conveniences to other module directories via relative paths
ORBIT_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

ORBIT_ASSETS_DATA_DIR = os.path.join(ORBIT_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

ORBIT_ASSETS_METADATA = toml.load(os.path.join(ORBIT_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ORBIT_ASSETS_METADATA["package"]["version"]


##
# Configuration for different assets.
##

from .allegro import *
from .anymal import *
from .cartpole import *
from .franka import *
from .kinova import *
from .ridgeback_franka import *
from .sawyer import *
from .shadow_hand import *
from .unitree import *
from .universal_robots import *
