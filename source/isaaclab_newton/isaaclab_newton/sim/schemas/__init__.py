# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing Newton schema configuration exports."""

from isaaclab.sim.schemas._backend_hooks import register_fixed_root_joint_creator
from isaaclab.utils.module import lazy_export

lazy_export()


def _create_fixed_root_joint_newton(articulation_prim, stage) -> None:
    """Fix an articulation base by authoring a world<->root ``UsdPhysics.FixedJoint``.

    Backend creator registered with the core articulation-root writer for the Newton backend. Newton's
    importer reads a ``UsdPhysics.FixedJoint`` directly as a fixed root joint (a jointless root would
    otherwise default to floating), so -- unlike PhysX -- no articulation-root relocation is needed. The
    joint is authored with the Kit ``omni.physx`` utility (a USD-authoring helper present in any Isaac
    Sim app regardless of the active solver), so nothing is hand-rolled.

    Args:
        articulation_prim: The resolved articulation-root prim to fix to the world frame.
        stage: The stage the prim lives on.

    Raises:
        NotImplementedError: When the root prim is not a rigid body (the first rigid-body link cannot be
            determined to anchor the fixed joint).
    """
    from pxr import UsdPhysics

    from omni.physx.scripts import utils as physx_utils

    if not articulation_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise NotImplementedError(
            f"The articulation prim '{articulation_prim.GetPath().pathString}' does not have the"
            " RigidBodyAPI applied. To create a fixed joint, we need to determine the first rigid body"
            " link in the articulation tree. However, this is not implemented yet."
        )
    physx_utils.createJoint(stage=stage, joint_type="Fixed", from_prim=None, to_prim=articulation_prim)


def _newton_is_active() -> bool:
    """Return whether the running simulation uses the Newton backend."""
    from isaaclab.sim import SimulationContext

    from isaaclab_newton.physics import NewtonCfg

    sim = SimulationContext.instance()
    return sim is not None and isinstance(sim.cfg.physics, NewtonCfg)


# Register the Newton fixed-root-joint creator (matched to the active Newton backend) so fixing an
# articulation base works in a Newton-only run, without core carrying any backend logic. Registered on
# import of this package (which a caller does to construct Newton schema fragments).
register_fixed_root_joint_creator(_create_fixed_root_joint_newton, _newton_is_active)
