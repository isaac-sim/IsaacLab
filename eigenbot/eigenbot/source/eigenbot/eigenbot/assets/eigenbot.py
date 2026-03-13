"""Configuration for the Eigenbot hexapod modular robot."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg

# Path to the USDZ file
EIGENBOT_USDZ_PATH = os.path.join(os.path.dirname(__file__), "eigenbot_new.usdz")


def _spawn_eigenbot_usdz(prim_path: str, cfg: sim_utils.UsdFileCfg, *args, **kwargs):
    """Custom spawner that adds physics APIs to a geometry-only USDZ.

    The eigenbot_new.usdz is a CAD export with no physics data. This spawner
    creates the USD reference and then applies RigidBodyAPI so it can be used
    as a RigidObject in the simulation.
    """
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()

    # Create prim and add USD reference
    prim = stage.DefinePrim(prim_path)
    prim.GetReferences().AddReference(cfg.usd_path)

    # Apply RigidBodyAPI so the object participates in physics
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)

    # Apply collision API to all mesh children so the object doesn't fall through the ground
    from pxr import UsdGeom

    for child_prim in Usd.PrimRange(prim):
        if child_prim.IsA(UsdGeom.Mesh):
            if not child_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(child_prim)

    # Now modify rigid body properties if specified
    if cfg.rigid_props is not None:
        from isaaclab.sim import schemas
        schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props)

    return prim


EIGENBOT_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        func=_spawn_eigenbot_usdz,
        usd_path=EIGENBOT_USDZ_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
    ),
)
