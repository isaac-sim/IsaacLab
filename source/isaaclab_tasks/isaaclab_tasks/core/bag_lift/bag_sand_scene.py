# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene assets for the bag-lift task: work tray, arm, bag, contact sensor, light.

Isaac Lab 3.x on the PhysX backend. Sand and gravel particles are off (plain rigid tray) so
training is cheap. Flip USE_SAND_PARTICLES to put them back.

Everything you are likely to tune by hand is in the "TUNE ME" blocks. This file is imported by
bag_lift_env_cfg.py.

HOW TO EDIT (quick index)
    USE_SAND_PARTICLES     on/off sand+gravel particles (rebuild sandbox_particles.usdc first)
    USE_DEFORMABLE_HANDLE  True = bendable PhysX surface-deformable strap, False = the original rigid strap
    HANDLE_*               stiffness / thickness / mass of the deformable strap (see "BAG HANDLE (deformable)")
    EE_FLOOR_CLEARANCE     how far the arm can go through the tray floor before a collider stops it
    ARM_BASE_POS / ROT     where the arm sits relative to the tray
    BAG_SPAWN_XY / ROT     bag xy and lying-flat rotation (handle must sit 0.18-0.30 m in front of the arm)
    ARM_HOME_JOINT_POS     start pose
    GRIPPER_*_POS          open vs closed jaw angles (must match revolute_7 limits)
    EE_TIP_OFFSET          fixed-jaw tip used by reach/insert/grasp rewards (gripper-link frame)
    RIGID_JAW_LENGTH / INSERT_THROUGH_FRAC
                           how far through the loop the tip must go (reach target)
    JAW_INSERT_AXIS        direction the rigid jaw points (gripper-link -Y); must follow the hole
    ARM_CONTACT_CFG        contact forces between every arm link and the bag (collision penalty)
    stiffness / effort     how strong the motors are (too low = cannot lift; too high = violent)
"""

from pathlib import Path

from isaaclab_physx.sim.schemas import (
    PhysxArticulationCfg,
    PhysxCollisionCfg,
    PhysxConvexDecompositionCfg,
    PhysxDeformableBodyCfg,
    PhysxSurfaceDeformableBodyCfg,
)
from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import OmniPhysicsDeformableBodyCfg

# ---------------------------------------------------------------------------
# Paths -- your original files are never loaded directly. The wrappers in assets/ reference them and fix
# what Isaac Lab cannot load as-is (read the doc string at the top of each wrapper).
# ---------------------------------------------------------------------------
DOCS = Path("/home/joshua/Documents")
SAND_USD = str(DOCS / "sand.usd")  # your world        -> assets/sandbox_particles.usdc
BAG_USD = str(DOCS / "Rigid Bag.usd")  # your bag          -> assets/bag_single_body.usda
ARM_USD = str(DOCS / "SO_101_sim.usd")  # your arm          -> assets/so101_physics.usda

_ASSETS = Path(__file__).parent / "assets"
# Sand/gravel switch. False = fast: plain rigid tray, no particles and no sand/gravel visuals.
# True = PhysX PBD particles cropped from your sand.usd (slow; rebuild sandbox_particles.usdc first).
# Edit: True also needs cloning_contexts = UsdReplicateContext on SANDBOX_CFG (already wired below).
USE_SAND_PARTICLES = False
SANDBOX_SPAWN_USD = str(_ASSETS / ("sandbox_particles.usdc" if USE_SAND_PARTICLES else "sandbox_tray.usda"))
# Handle switch. True = the strap is a bendable PhysX surface deformable clamped to a body-only rigid bag (stiff enough
# to hold its arch under gravity, but yields when the jaws press it). False = the original fully rigid bag (body + strap
# as one rigid body, bag_single_body.usda). Needs the files made by assets/build_handle_assets.py.
USE_DEFORMABLE_HANDLE = True
BAG_SPAWN_USD = str(_ASSETS / ("bag_body_only.usda" if USE_DEFORMABLE_HANDLE else "bag_single_body.usda"))
HANDLE_SPAWN_USD = str(_ASSETS / "bag_handle_ribbon.usda")
HANDLE_INFO_JSON = str(_ASSETS / "bag_handle_info.json")  # clamped vertices etc. (build_handle_assets.py)
JAW_POINTS_JSON = str(_ASSETS / "jaw_points.json")  # proxy points on both jaws, for the geometric grasp test
ARM_SPAWN_USD = str(_ASSETS / "so101_physics.usda")

# ===========================================================================
# SANDBOX  (TUNE ME)
# The particle tray is a crop of sand.usd around x = 8.25. Those two numbers shift that crop onto the
# env origin and the floor onto z = 0. The plain tray (USE_SAND_PARTICLES = False) is already at the
# origin with its floor at z = 0, so the spawn pose is (0, 0, 0).
# ===========================================================================
SANDBOX_WORK_CENTER_X = 8.25
SANDBOX_FLOOR_Z = 0.4108
# Surface the bag rests on. 0 with the plain tray; ~3 cm when the particle sand bed is on.
SAND_SURFACE_Z = 0.030 if USE_SAND_PARTICLES else 0.0
# The bag still sits on TrayFloor (z = 0). The arm ignores that collider and hits an invisible
# cube TrayEeLimit whose top is this far below the visual floor, so the jaw can thread the loop.
EE_FLOOR_CLEARANCE = 0.04  # [m]

# ===========================================================================
# BAG  (from Rigid Bag.usd; bag frame = origin at the body centre)
#   body   : 0.165 x 0.089 x 0.152 m, z in [-0.076, 0.076]
#   handle : plate 0.065 x 0.013 x 0.038 m on top, z in [0.076, 0.114]
# The bag cannot stand on end (rounded bottom, it tips over in ~0.4 s) so it is spawned lying on a big
# face: 90 deg roll about x, then 180 deg yaw. Body centre is 0.0445 m above the tray. The extra yaw
# points the handle +y (into the open tray). Roll-only put the loop at y=-0.095, 10.5 cm in front of
# the arm and against the near wall — inside the SO-101's inner dead zone (usable reach is ~0.18-0.30 m
# in front of the base; the cube-stack task puts objects at 0.18-0.30 m).
# Handle after this rot is +0.095 m in y from the body centre. Hole axis is still world +Z.
# ===========================================================================
BAG_MASS = 0.08  # [kg] body + handle. Was 0.2 (the strap popped through the jaw under that load). Edit: your real bag.
BAG_HALF_THICKNESS = 0.0445
BAG_REST_Z = SAND_SURFACE_Z + BAG_HALF_THICKNESS + 0.005
BAG_REST_ROT = (0.0, 0.70710678, 0.70710678, 0.0)  # yaw 180 * roll 90, (x, y, z, w)
# Body at this xy keeps the loop ~0.265 m in front of ARM_BASE_POS y=-0.20, with clearance from the
# south wall (inner face y≈-0.12). Edit y toward 0 to push the handle farther (max ~0.30 m from the arm).
BAG_SPAWN_XY = (0.0, -0.03)

# ===========================================================================
# ARM  (TUNE ME)
# SO_101_sim.usd is a 7-joint chain. Joint names are the prim names the URDF converter gave them:
#   revolute_1 base pan      revolute_2 shoulder lift   revolute_3 elbow       revolute_4 wrist pitch
#   revolute_5 wrist yaw     revolute_6 wrist roll      revolute_7 gripper jaw
# The CAD arm faces -y, so a 180 deg turn about +z makes it face +y, toward the bag.
# Edit ARM_BASE_POS y to slide the arm toward/away from the tray; z to raise the base off the floor.
# Edit ARM_BASE_ROT if the arm faces the wrong way (quaternion is x,y,z,w; (0,0,1,0) = 180 deg yaw).
# ===========================================================================
TRAY_WALL_HEIGHT = 0.06  # [m] tray walls span z = 0 .. 0.06 (see assets/sandbox_tray.usda)
# The base sits on top of the wall height so the arm clears the south wall (y = -0.125) in front of it.
ARM_BASE_POS = (0.0, -0.20, TRAY_WALL_HEIGHT)
ARM_BASE_ROT = (0.0, 0.0, 1.0, 0.0)  # (x, y, z, w)
# Grip strength. The gripper motor used to be the same 3 N*m as every other joint: when the jaws stall on the strap
# the position error saturates it, so it squeezed with ~50 N (3 N*m / ~6 cm jaw lever) although ~2 N is enough to hold
# a light bag. Over-squeezing a 1 mm shell drives it deep into the jaw colliders, and the pull of the bag then snaps it
# through. Edit: raise if the bag slips out of the jaws; lower if the strap still pops through the jaw mesh.
GRIPPER_EFFORT_LIMIT = 0.8  # [N*m] ~13 N total pinch
ARM_JOINT_NAMES = [f"revolute_{i}" for i in range(1, 8)]
ARM_GRIPPER_JOINT = "revolute_7"
# Home pose [rad]: every episode starts here (gripper revolute_7 starts OPEN at 0.6).
# Edit: change these to move the start/return pose. If the arm cannot reach the bag, raise the
# shoulder (revolute_2) or unfold the elbow (revolute_3). If it punches the floor, fold it up.
ARM_HOME_JOINT_POS = {
    "revolute_1": 0.11,
    "revolute_2": 0.94,
    "revolute_3": -2.17,
    "revolute_4": 0.71,
    "revolute_5": -0.09,
    "revolute_6": -0.37,
    "revolute_7": 0.6,
}
GRIPPER_OPEN_POS = 0.6  # [rad] jaw gap ~8 cm (what the episode starts with)
GRIPPER_CLOSED_POS = -0.38  # [rad] jaw gap ~1 cm = joint lower limit; holding the 1.3 cm handle plate stalls near -0.33
# The "hand" is the FIXED jaw on the URDF link wrist_roll_08c. That link's own visual is
# Wrist_Roll_08c.stl (see assembly_1.urdf). moving_jaw_08d is a *child rigid body* (revolute_7);
# walking every mesh under wrist_roll_08c used to pick the moving-jaw min-Y vertex
# (0.0216, -0.107, 0.0145), which sits 4.5 cm off the rigid-jaw tip and is why the policy
# stopped ~4 cm to the side of the loop.
# Tip = centroid of the distal 5 mm of Wrist_Roll_08c only, in the wrist_roll_08c body frame.
# JAW_INSERT_AXIS is that paddle's long axis (link -Y). The rotating jaw is penalised for
# hitting the bag; the rigid jaw is allowed to occupy the hole.
EE_BODY_NAME = "wrist_roll_08c"
EE_TIP_OFFSET = (-0.0065, -0.0934, 0.0464)
JAW_INSERT_AXIS = (0.0, -1.0, 0.0)
MOVING_JAW_BODY_NAME = "moving_jaw_08d"
# Rigid-jaw paddle length along gripper -Y, from Wrist_Roll_08c only: housing (y ≈ -0.040) to
# distal tip (y ≈ -0.095). Reach/insert target is this fraction PAST the loop centre.
RIGID_JAW_LENGTH = 0.055  # [m]
INSERT_THROUGH_FRAC = 0.90
INSERT_THROUGH = -INSERT_THROUGH_FRAC * RIGID_JAW_LENGTH  # -0.0495 m along bag +Y (through the hole)

# ---------------------------------------------------------------------------
# Config objects
# ---------------------------------------------------------------------------
SANDBOX_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Sandbox",
    spawn=sim_utils.UsdFileCfg(usd_path=SANDBOX_SPAWN_USD),
    # PhysX cannot replicate particle systems / PBD materials, so the particle tray is copied as plain USD.
    cloning_contexts=("isaaclab.cloner:UsdReplicateContext",) if USE_SAND_PARTICLES else None,
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(-SANDBOX_WORK_CENTER_X, 0.0, -SANDBOX_FLOOR_Z) if USE_SAND_PARTICLES else (0.0, 0.0, 0.0)
    ),
)

ARM_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Arm",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ARM_SPAWN_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
        # Tight contact generation so "EE vs bag" means real scrape, not the default ~2 cm PhysX halo.
        make_uninstanceable=True,
        collision_props={"/.*": [PhysxCollisionCfg(contact_offset=0.002, rest_offset=0.0)]},
        # fix_root_link pins the base to the world (Isaac Lab adds the anchor joint; see assets/so101_physics.usda)
        fix_root_link=True,
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=ARM_BASE_POS, rot=ARM_BASE_ROT, joint_pos=ARM_HOME_JOINT_POS),
    actuators={
        # stiffness 20 N*m/rad matches the 0.35 N*m/deg in your USD; max torque 3 N*m is the USD maxForce.
        # Edit: raise stiffness for snappier tracking (can overshoot / shake); raise effort_limit if
        # the arm cannot lift the bag. Split into two ImplicitActuatorCfg entries (arm vs
        # gripper) if you want different gains on revolute_7.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["revolute_[1-6]"],
            stiffness=20.0,
            damping=1.0,
            joint_effort_limit=3.0,
            # Solver clamp [rad/s]. Stock Isaac Lab SO-101 is 10. Also fills the actuator velocity
            # limit used by the PD model. Edit: raise toward 5-10 for snappier motion.
            joint_velocity_limit=0.5,
        ),
        # Same gains, but a weaker motor so the pinch cannot crush the strap (see GRIPPER_EFFORT_LIMIT).
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[ARM_GRIPPER_JOINT],
            stiffness=20.0,
            damping=1.0,
            joint_effort_limit=GRIPPER_EFFORT_LIMIT,
            joint_velocity_limit=0.5,
        ),
    },
)

# Rigid-handle mode only: the strap is a thin ARCHED band (13 mm wide) and the USD ships it as a single convexHull
# collider, and the hull of an arch is a SOLID block: the opening does not exist for physics. Cook the handle as a
# convex decomposition instead (several convex pieces that follow the arch). The bag body keeps its convex hull.
# To inspect: Kit > Window > Debug > Physics > Collision > show collision shapes.
_RIGID_HANDLE_COLLISION = {
    "/Handle/.*": [
        PhysxConvexDecompositionCfg(
            max_convex_hulls=32,
            hull_vertex_limit=32,
            voxel_resolution=1_000_000,
            error_percentage=0.5,
            min_thickness=0.001,
            shrink_wrap=True,
        )
    ]
}

BAG_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Bag",
    spawn=sim_utils.UsdFileCfg(
        usd_path=BAG_SPAWN_USD,
        activate_contact_sensors=True,
        collision_props={"/.*": [PhysxCollisionCfg(contact_offset=0.002, rest_offset=0.0)]},
        mass_props={"": [sim_utils.MassCfg(mass=BAG_MASS)]},  # overrides physics:mass in the wrapper USD
        mesh_collision_props=None if USE_DEFORMABLE_HANDLE else _RIGID_HANDLE_COLLISION,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(*BAG_SPAWN_XY, BAG_REST_Z), rot=BAG_REST_ROT),
)

# ===========================================================================
# BAG HANDLE (deformable)  (TUNE ME; only used when USE_DEFORMABLE_HANDLE = True)
# A 13 x ~120 mm, 1 mm thick strap simulated as a PhysX surface deformable (cloth-like FEM shell). The two feet are
# clamped to the bag body (see BagLiftEnv._setup_scene). It spawns at the bag's pose, so its mesh is in the bag frame.
#
# Feel of the strap (all stiffnesses in Pa, "force per area"):
#   HANDLE_STIFFNESS       one number that sets Young's modulus AND the stretch / shear / bend stiffness overrides
#                          (the schema's derived defaults turned out too soft: the arch collapsed at sim start).
#                          3e9 = rigid-ish plastic strap. Too floppy / sags: raise it (1e10, 3e10). Jaws cannot
#                          dent it: lower it (1e8..1e9). Very high values can make the solver jitter: raise
#                          HANDLE_SOLVER_ITERS then.
#   HANDLE_BEND_FACTOR     bend stiffness = HANDLE_STIFFNESS * this. Bending is what holds the arch up, so it is
#                          set higher than stretch.
#   HANDLE_THICKNESS       [m] shell thickness.
#   HANDLE_DAMPING         damping of stretch / bending; higher = settles faster, less ringing.
#   HANDLE_SOLVER_ITERS    position solver iterations. Stiff thin shells need many (cost grows with it).
# ===========================================================================
HANDLE_STIFFNESS = 3.0e9  # [Pa]
HANDLE_BEND_FACTOR = 10.0
HANDLE_THICKNESS = 0.001  # [m]
HANDLE_DENSITY = 1000.0  # [kg/m^3] -> the strap weighs ~1.5 g
HANDLE_DAMPING = 0.2
HANDLE_SOLVER_ITERS = 64

HANDLE_CFG = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Handle",
    # PhysX deformables are copied per env as plain USD (physics replication cannot clone them).
    cloning_contexts=("isaaclab.cloner:UsdReplicateContext",),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(*BAG_SPAWN_XY, BAG_REST_Z), rot=BAG_REST_ROT),
    spawn=sim_utils.UsdFileCfg(
        usd_path=HANDLE_SPAWN_USD,
        surface_deformable_props=[
            OmniPhysicsDeformableBodyCfg(kinematic_enabled=False),
            PhysxDeformableBodyCfg(solver_position_iteration_count=HANDLE_SOLVER_ITERS, enable_speculative_c_c_d=True),
            # contact against the jaws is where the strap tunnelled: refresh contact pairs more often and apply the
            # collision constraints several times per solver iteration (limit: HANDLE_SOLVER_ITERS / 2)
            PhysxSurfaceDeformableBodyCfg(collision_pair_update_frequency=2, collision_iteration_multiplier=4.0),
        ],
        # collision offsets ride on the simulation mesh the spawner creates under the Handle prim. A larger rest
        # offset keeps the 1 mm shell this far off the jaw surface, so it is harder to push through.
        collision_props={"/sim_mesh": [PhysxCollisionCfg(rest_offset=0.001, contact_offset=0.005)]},
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.1)),
        physics_material=PhysxSurfaceDeformableBodyMaterialCfg(
            density=HANDLE_DENSITY,
            surface_thickness=HANDLE_THICKNESS,
            youngs_modulus=HANDLE_STIFFNESS,
            surface_stretch_stiffness=HANDLE_STIFFNESS,
            surface_shear_stiffness=HANDLE_STIFFNESS,
            surface_bend_stiffness=HANDLE_STIFFNESS * HANDLE_BEND_FACTOR,
            poissons_ratio=0.3,
            elasticity_damping=HANDLE_DAMPING,
            bend_damping=HANDLE_DAMPING,
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    ),
)

# Contact reporter on EVERY arm link, filtered to the bag. The env uses per-link forces and contact
# points to penalise collisions on the lower part of the bag body (not to detect the grasp).
ARM_CONTACT_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Arm/.*",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Bag"],
    history_length=1,
    update_period=0.0,
    # Where each link touches the bag (average contact point). Collision uses bag-frame z of that point
    # so the top rim and handle are free; the bag is one rigid body so location is needed, not force alone.
    track_contact_points=True,
    max_contact_data_count_per_prim=16,
)

# One dim, constant dome light so nothing is pitch black. (Per-env randomised lamps and the D415 cameras
# were removed to keep the task small; the old code is in the backup noted in README.md.)
AMBIENT_LIGHT_CFG = AssetBaseCfg(
    prim_path="/World/Ambient",
    spawn=sim_utils.DomeLightCfg(intensity=30.0, color=(1.0, 1.0, 1.0)),
)
