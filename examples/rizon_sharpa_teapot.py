# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physically grasp the Utah teapot and pour MPM water during a 25 cm rise.

This experimental example uses a fixed index grasp and motion.

.. code-block:: bash

    # Newton GL:
    uv run isaaclab example rizon-sharpa-teapot

The pinned robot USD and textures are downloaded and verified on first use.
Set ``ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT`` to an existing bundle directory
to use it without downloading.
"""

from argparse import Namespace

if __name__ == "__main__":
    from demos.teapot_fill import SIMULATION_HZ, main

    from isaaclab_tasks.contrib.rizon_sharpa_teapot.motion import SEQUENCE_DURATION

    main(
        Namespace(
            robot="rizon_sharpa", fluid_render_mode="particles", max_steps=round(SIMULATION_HZ * SEQUENCE_DURATION)
        )
    )
