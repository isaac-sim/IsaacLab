# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf integration for IsaacLab.

This package provides the extension mechanism for integrating IsaacLab tasks
with RLinf's distributed RL training framework for VLA models like GR00T.

Extension Module
----------------

The extension module (``extension.py``) is loaded by RLinf via the
``RLINF_EXT_MODULE`` environment variable and handles:

1. Registering IsaacLab tasks into RLinf's ``REGISTER_ISAACLAB_ENVS``
2. Registering GR00T obs/action converters
3. Patching GR00T ``get_model`` for custom embodiments

Usage:
    .. code-block:: bash

        export RLINF_EXT_MODULE="isaaclab_contrib.rl.rlinf.extension"
        export RLINF_CONFIG_FILE="/path/to/config.yaml"
"""
