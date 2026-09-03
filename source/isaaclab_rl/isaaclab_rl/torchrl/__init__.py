# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrappers and utilities to configure an environment for the TorchRL library.

The following example shows how to wrap an environment for TorchRL:

.. code-block:: python

    from isaaclab_rl.torchrl import IsaacLabTorchRLWrapper

    env = IsaacLabTorchRLWrapper(env)

"""

from isaaclab.utils.module import lazy_export

lazy_export()
