# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Historic replicator texture randomization reproducer (retired).

This script previously depended on Isaac Sim core extensions that Isaac Sim now carries under
``source/deprecated`` in the Isaac Sim repository. It has been retired from Isaac Lab to avoid
depending on those extension IDs and import paths.

Use :class:`~isaaclab.sim.SimulationContext`, :class:`isaacsim.core.cloner.GridCloner`, and
``isaacsim.core.experimental.*`` when experimenting with Omniverse Replicator workflows.
"""

raise RuntimeError(
    "This standalone reproducer was retired; it depended on deprecated Isaac Sim core extensions. "
    "Use Isaac Lab :mod:`isaaclab.sim` and ``isaacsim.core.experimental.*`` instead."
)
