# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing PhysX schema configuration exports."""

from isaaclab.sim.schemas._backend_hooks import register_fixed_root_joint_creator
from isaaclab.utils.module import lazy_export

from isaaclab_physx.physics import PhysxCfg

lazy_export()


def _create_fixed_root_joint(articulation_prim, stage) -> None:
    """Fix an articulation base by authoring a world<->root fixed joint (PhysX parser semantics).

    Backend creator registered with the core articulation-root writer. Creating the joint relies on
    the PhysX parser, which does not treat a fixed joint on a rigid body as a fixed-base articulation;
    relocating the ``UsdPhysics.ArticulationRootAPI`` (and its ``physxArticulation:*`` attributes) to
    the parent prim works around that. This PhysX-specific logic lives in the PhysX package so core
    carries none of it.

    Args:
        articulation_prim: The resolved articulation-root prim to fix to the world frame.
        stage: The stage the prim lives on.

    Raises:
        NotImplementedError: When the root prim is not a rigid body (the first rigid-body link cannot
            be determined to attach the fixed joint).
    """
    from pxr import UsdPhysics

    from omni.physx.scripts import utils as physx_utils

    # note: we assume the root prim is a rigid body; there is no obvious way to get the first rigid
    #   body link identified by the PhysX parser when it is not.
    if not articulation_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise NotImplementedError(
            f"The articulation prim '{articulation_prim.GetPath().pathString}' does not have the"
            " RigidBodyAPI applied. To create a fixed joint, we need to determine the first rigid body"
            " link in the articulation tree. However, this is not implemented yet."
        )

    # create a fixed joint between the root link and the world frame
    physx_utils.createJoint(stage=stage, joint_type="Fixed", from_prim=None, to_prim=articulation_prim)

    # Having a fixed joint on a rigid body is not treated as "fixed base articulation"; it is treated as
    # part of the maximal coordinate tree. Moving the articulation root to the parent solves this (a
    # limitation of the PhysX parser).
    parent_prim = articulation_prim.GetParent()
    UsdPhysics.ArticulationRootAPI.Apply(parent_prim)
    if "PhysxArticulationAPI" not in parent_prim.GetAppliedSchemas():
        parent_prim.AddAppliedSchema("PhysxArticulationAPI")

    # copy the attributes to the parent
    # -- usd attributes
    usd_articulation_api = UsdPhysics.ArticulationRootAPI(articulation_prim)
    for attr_name in usd_articulation_api.GetSchemaAttributeNames():
        attr = articulation_prim.GetAttribute(attr_name)
        parent_attr = parent_prim.GetAttribute(attr_name)
        if not parent_attr:
            parent_attr = parent_prim.CreateAttribute(attr_name, attr.GetTypeName())
        parent_attr.Set(attr.Get())
    # -- physx attributes (copy by name prefix)
    for attr in articulation_prim.GetAttributes():
        aname = attr.GetName()
        if aname.startswith("physxArticulation:"):
            parent_attr = parent_prim.GetAttribute(aname)
            if not parent_attr:
                parent_attr = parent_prim.CreateAttribute(aname, attr.GetTypeName())
            parent_attr.Set(attr.Get())

    # remove the api from the (former) root
    articulation_prim.RemoveAppliedSchema("PhysxArticulationAPI")
    articulation_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)


# Keep PhysX articulation-root logic out of core: register the creator with the core articulation-root
# writer keyed by ``PhysxCfg``, so the writer selects it only when PhysX is the active simulation
# backend (``cfg.physics`` is a ``PhysxCfg``). Registered on import of this package (which a caller does
# to construct PhysX schema fragments), inverting the dependency so core carries no PhysX logic.
register_fixed_root_joint_creator(PhysxCfg, _create_fixed_root_joint)
