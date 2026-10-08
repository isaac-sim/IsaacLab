# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental conveyor support for contributed tasks.

Backend adapters are imported explicitly from :mod:`.newton` or :mod:`.physx` so
importing the surface descriptions does not load either optional backend.
"""
