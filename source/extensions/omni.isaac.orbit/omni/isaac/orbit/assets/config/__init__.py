# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration instances for different assets.

This sub-module contains the configuration instances for different assets. The configuration
instances are used to spawn and configure the assets in the simulation. They are passed to
their corresponding asset classes during construction.
"""

from .anymal import *
from .franka import *
from .ridgeback_franka import *
from .unitree import *
from .universal_robots import *
