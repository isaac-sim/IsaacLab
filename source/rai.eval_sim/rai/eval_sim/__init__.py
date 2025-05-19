# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

import os

import toml

# Conveniences to other module directories via relative paths
EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
"""Path to the extension source directory."""

EXT_METADATA = toml.load(os.path.join(EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = EXT_METADATA["package"]["version"]


# Import relevant modules within EvalSim

from .eval_sim import *
from .ros_manager import *
from .tasks import *
from .utils import *
