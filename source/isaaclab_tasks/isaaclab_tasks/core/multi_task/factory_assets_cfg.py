# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg

# This is where we will get the Robot that we want to use
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, LOCAL_ASSET_PATH_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from .assembly_keypoints import NIST_BOARD_CFG

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
)

ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
)


@configclass
class _SocketCollisionPropsCfg(PresetCfg):
    """Backend-aware collision props for assembly sockets (bolts, holes, bases)."""

    default = sim_utils.CollisionPropertiesCfg(contact_offset=0.0025, rest_offset=0.0)
    newton_mjwarp = sim_utils.NewtonSDFCollisionPropertiesCfg(
        rest_offset=0.0,
        contact_gap=0.005,
        sdf_max_resolution=256,
        sdf_narrow_band_inner=-0.005,
        sdf_narrow_band_outer=0.005,
    )
    physx = default


@configclass
class _PlugCollisionPropsCfg(PresetCfg):
    """Backend-aware collision props for assembly plugs (nuts, pegs, gears)."""

    default = sim_utils.CollisionPropertiesCfg(contact_offset=0.0025, rest_offset=0.0)
    newton_mjwarp = sim_utils.NewtonSDFCollisionPropertiesCfg(
        contact_offset=0.0025,
        rest_offset=0.0,
        contact_gap=0.005,
        sdf_max_resolution=256,
        sdf_narrow_band_inner=-0.005,
        sdf_narrow_band_outer=0.005,
    )
    physx = default


ASSEMBLY_SOCKET_COLLISION_PROPS_CFG = _SocketCollisionPropsCfg()

ASSEMBLY_PLUG_COLLISION_PROPS_CFG = _PlugCollisionPropsCfg()

# Newton-only stiff contact material for the assembly pair. Raising stiffness globally
# (default_shape_cfg) makes the grasped-nut-vs-gripper contact explode on banked reset
# states (the reset acceptance gate cannot see the closed-finger pose); scoping ke/kd to
# the nut/bolt shapes keeps threading stiff while robot/board/table contacts stay soft.
# newton:contactStiffness/Damping map to shape ke/kd -> geom_solref = (2/kd, (kd/2)/sqrt(ke)).
# The friction fragment is REQUIRED: binding a material that authors no physics:*Friction
# resolves the shape's friction to 0, and mu=0 contacts with condim=3 zero the pyramidal
# row invweight -> efc_D is floored to 1/MJ_MINVAL (~1e15) -> the float32 Newton-solver
# Hessian degenerates -> NaN on contact-rich states (seated insertion; mjwarp warns
# "friction[0] < MJ_MINMU with condim=3 may cause NaN" at model build).
ASSEMBLY_CONTACT_MATERIAL_CFG = preset(
    default=None,
    newton_mjwarp=[
        UsdPhysicsRigidBodyMaterialCfg(static_friction=0.75, dynamic_friction=0.75),
        NewtonMaterialCfg(contact_stiffness=2.56e6, contact_damping=3200.0),
    ],
)

# Newton-only stiff contact material for the ROBOT colliders. Same friction-fragment
# requirement as above (stiffness-only binding would zero mu). Friction 1.0 matches the
# robot USD's authored value; the robot_material startup event overrides it at init.
# (1e6, 2000) -> solref (1e-3, 1.0), damping-matched.
ROBOT_CONTACT_MATERIAL_CFG = preset(
    default=None,
    newton_mjwarp=[
        UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        NewtonMaterialCfg(contact_stiffness=1.0e6, contact_damping=2000.0),
    ],
)

GROUND_CFG = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.GroundPlaneCfg(),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.868)),
)

DOMELIGHT_CFG = AssetBaseCfg(
    prim_path="/World/DomeLight",
    spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
)


FRANKA_ACTUATORS_CFG = {
    "panda_shoulder": ImplicitActuatorCfg(  # type:ignore
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit=87.0,
        effort_limit_sim=870.0,
        velocity_limit=2.175,
        velocity_limit_sim=21.75,
        stiffness=80.0,
        damping=4.0,
        armature=0.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(  # type:ignore
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit=12.0,
        effort_limit_sim=120.0,
        velocity_limit=2.61,
        velocity_limit_sim=26.1,
        stiffness=80.0,
        damping=4.0,
        armature=0.0,
    ),
    "panda_hand": ImplicitActuatorCfg(  # type:ignore
        joint_names_expr=["panda_finger_joint.*"],
        effort_limit=40.0,
        effort_limit_sim=400.0,
        velocity_limit=0.2,
        velocity_limit_sim=2.0,
        stiffness=7500.0,
        damping=173.0,
        friction=0.1,
        armature=0.0,
    ),
}


# FRANKA_ACTUATORS_CFG = {
#     "panda_shoulder": ImplicitActuatorCfg(  # type:ignore
#         joint_names_expr=["panda_joint[1-4]"],
#         effort_limit_sim=87.0,
#         velocity_limit_sim=2.175,
#         stiffness=80.0,
#         damping=4.0,
#         armature=0.0,
#     ),
#     "panda_forearm": ImplicitActuatorCfg(  # type:ignore
#         joint_names_expr=["panda_joint[5-7]"],
#         effort_limit_sim=12.0,
#         velocity_limit_sim=2.61,
#         stiffness=80.0,
#         damping=4.0,
#         armature=0.0,
#     ),
#     "panda_hand": ImplicitActuatorCfg(  # type:ignore
#         joint_names_expr=["panda_finger_joint.*"],
#         effort_limit_sim=40.0,
#         velocity_limit_sim=0.2,
#         stiffness=7500.0,
#         damping=173.0,
#         friction=0.1,
#         armature=0.0,
#     ),
# }

FRANKA_DEFAULT_STATE_CFG = ArticulationCfg.InitialStateCfg(
    joint_pos={
        "panda_joint1": 0.00871,
        "panda_joint2": -0.10368,
        "panda_joint3": -0.00794,
        "panda_joint4": -1.49139,
        "panda_joint5": -0.00083,
        "panda_joint6": 1.38774,
        "panda_joint7": 0.0,
        "panda_finger_joint2": 0.04,
    },
    pos=(0.0, 0.0, 0.0),
    rot=(0.0, 0.0, 0.0, 1.0),
)

FRANKA_PANDA_PHYSX_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/franka_mimic.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=FRANKA_DEFAULT_STATE_CFG,
    actuators=FRANKA_ACTUATORS_CFG,
)


FRANKA_PANDA_NEWTON_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/franka_mimic.usd",
        activate_contact_sensors=True,
        # Newton/MuJoCo: leave gravity ON and compensate it per-body via mjc:gravcomp,
        # instead of PhysX's disable_gravity hack. gravcomp=1.0 = full compensation.
        rigid_props=sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        # Convex-hull the robot colliders (esp. the ~8.2k-tri fingers) so they use the
        # GJK convex path instead of the BVH triangle path against the SDF nut/bolt.
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.0,
            mesh_collision_property=sim_utils.NewtonMeshCollisionPropertiesCfg(mesh_approximation_name="convexHull"),
        ),
        physics_material=ROBOT_CONTACT_MATERIAL_CFG,
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(ensure_drives_exist=True),
    ),
    init_state=FRANKA_DEFAULT_STATE_CFG,
    actuators=FRANKA_ACTUATORS_CFG,
)

# Table
TABLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/Mounts/UWPatVention/pat_vention.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.868), rot=(0.0, 0.0, -0.70711, 0.70711)),
)

x, y, z = NIST_BOARD_CFG.nist_board_center.pos
# NIST Board
NISTBOARD_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/NistBoard",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/Taskboard/nistboard.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        scale=(1.0, 1.0, 0.5),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65 - x, 0.0 - y, 0.0206 - z), rot=(0.0, 1.0, 0.0, 0.0)),
)

##
# Assembly Tools
##


BOLT_M16_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BOLT_M16",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/thread_m16.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


NUT_M16_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/NUT_M16",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/nut_m16.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


HOLE_16MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/HOLE_16MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_hole_16mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


ROD_16MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/ROD_16MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_peg_16mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


HOLE_12MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/HOLE_12MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_hole_12mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


ROD_12MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/ROD_12MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_peg_12mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


HOLE_8MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/HOLE_8MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_hole_8mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


ROD_8MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/ROD_8MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_peg_8mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


HOLE_4MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/HOLE_4MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_hole_4mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


ROD_4MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/ROD_4MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_peg_4mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


RECTANGULAR_HOLE_16MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_HOLE_16MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_hole_16mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


RECTANGULAR_PEG_16MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_PEG_16MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_peg_16mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


RECTANGULAR_HOLE_12MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_HOLE_12MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_hole_12mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


RECTANGULAR_PEG_12MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_PEG_12MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_peg_12mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


RECTANGULAR_HOLE_8MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_HOLE_8MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_hole_8mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


RECTANGULAR_PEG_8MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_PEG_8MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_peg_8mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


RECTANGULAR_HOLE_4MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_HOLE_4MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_hole_4mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


RECTANGULAR_PEG_4MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_PEG_4MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_peg_4mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


LARGE_GEAR_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/LARGE_GEAR",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/gear_large.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


MEDIUM_GEAR_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/MEDIUM_GEAR",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/gear_medium.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)


SMALL_GEAR_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/SMALL_GEAR",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/gear_small.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)

GEAR_BASE_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/GEAR_BASE",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/gear_base.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
        physics_material=ASSEMBLY_CONTACT_MATERIAL_CFG,
    ),
)
