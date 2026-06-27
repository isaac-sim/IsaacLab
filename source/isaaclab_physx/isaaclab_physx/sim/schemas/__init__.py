# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing PhysX schema configuration exports."""

from isaaclab.sim.schemas._backend_hooks import register_joint_drive_skip_predicate
from isaaclab.utils.module import lazy_export

lazy_export()


def _is_physx_tendon_child(prim) -> bool:
    """Return whether ``prim`` is a non-root PhysX fixed-tendon member.

    A tendon-child joint carries the multi-instance ``PhysxTendonAxisAPI`` without the
    ``PhysxTendonAxisRootAPI`` root instance. Applied-schema *type* names are compared exactly (the
    part before the ``:`` instance suffix), so this is robust to schema-name changes. A drive must
    not be authored on such joints -- the tendon controls them.
    """
    types = {schema.split(":", 1)[0] for schema in prim.GetAppliedSchemas()}
    return "PhysxTendonAxisAPI" in types and "PhysxTendonAxisRootAPI" not in types


# Keep PhysX tendon knowledge out of core: register the detector with the core joint-drive writers so
# they skip tendon-controlled joints. Registered on import of this package (which a caller does to
# construct PhysX schema fragments), inverting the dependency so core carries no PhysX schema name.
register_joint_drive_skip_predicate(_is_physx_tendon_child)
