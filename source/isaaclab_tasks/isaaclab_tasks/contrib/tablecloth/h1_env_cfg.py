# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based configuration for the H1 tablecloth task."""

from __future__ import annotations

from isaaclab_newton.controllers.ik import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg, NewtonIKSolverCfg
from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.physics import (
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg, NewtonSDFCollisionCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.core.lift import mdp as lift_mdp

from . import mdp
from .assets import (
    BOWL_LOCAL_HEIGHT,
    BOWL_LOCAL_RADIUS,
    BOWL_LOCAL_Z_MIN,
    BOWL_SCALE,
    BOWL_USD,
    FORK_CENTER_OF_MASS,
    FORK_DIAGONAL_INERTIA,
    FORK_LOCAL_Z_MIN,
    FORK_ROTATION,
    FORK_SCALE,
    FORK_USD,
    KITCHEN_ISLAND_SIZE,
    KITCHEN_ISLAND_USD,
    RIGID_GAP,
    WINE_GLASS_CENTER_OF_MASS,
    WINE_GLASS_DIAGONAL_INERTIA,
    WINE_GLASS_LOCAL_Z_MIN,
    WINE_GLASS_USD,
    VisualTableUsdFileCfg,
    collision_properties,
    rigid_material,
    rigid_object_cfg,
    spawn_h1_from_usd,
    tabletop_collider_cfg,
)

FPS = 60
SUBSTEPS = 16
# The straight-leg H1 sole is 1.0442 m below its pelvis; keep the tabletop 1 cm below the pelvis.
H1_STANDING_HEIGHT = 1.0442 + RIGID_GAP
TABLE_TOP_Z = H1_STANDING_HEIGHT - 0.01
TABLEWARE_CLEARANCE = 0.008
CLOTH_Z = TABLE_TOP_Z + 0.002
TABLEWARE_NAMES = ("bowl", "glass", "fork")

KITCHEN_ISLAND_SCALE = (
    0.40 / KITCHEN_ISLAND_SIZE[0],
    0.72 / KITCHEN_ISLAND_SIZE[1],
    TABLE_TOP_Z / KITCHEN_ISLAND_SIZE[2],
)

HAND_OFFSETS = ((0.146273, -0.068447, 0.028077), (0.148808, 0.068652, 0.026675))
FINGER_JOINT_NAMES = [
    f"{side}_{joint}"
    for side in ("L", "R")
    for joint in (
        "thumb_proximal_yaw_joint",
        "thumb_proximal_pitch_joint",
        "thumb_intermediate_joint",
        "thumb_distal_joint",
        "index_proximal_joint",
        "index_intermediate_joint",
        "middle_proximal_joint",
        "middle_intermediate_joint",
        "ring_proximal_joint",
        "ring_intermediate_joint",
        "pinky_proximal_joint",
        "pinky_intermediate_joint",
    )
]


@configclass
class H1ArticulationCfg(ArticulationCfg):
    """H1 articulation with SDF hand collisions and a high-friction grasp material."""

    def _post_spawn(self, stage) -> None:
        super()._post_spawn(stage)
        from pxr import UsdGeom, UsdPhysics  # noqa: PLC0415

        from isaaclab.sim.schemas import apply_namespaced  # noqa: PLC0415
        from isaaclab.sim.spawners.materials.physics_materials import (  # noqa: PLC0415
            spawn_physics_material_from_fragments,
        )

        root_path = self.spawn.spawn_path if self.spawn is not None and self.spawn.spawn_path else self.prim_path
        meshes = sim_utils.get_all_matching_child_prims(
            root_path,
            predicate=lambda prim: prim.IsA(UsdGeom.Mesh),
            stage=stage,
        )
        for mesh in meshes:
            mesh_path = mesh.GetPath().pathString
            if "/left_hand_link/" in mesh_path or "/right_hand_link/" in mesh_path:
                usd_mesh = UsdGeom.Mesh(mesh)
                usd_mesh.CreateDisplayColorAttr([(0.0100228, 0.0100228, 0.0100228)])
                usd_mesh.CreateDisplayOpacityAttr([1.0])

        # Full-surface rigid-soft contact requires an SDF for every participating mesh.
        for collider in (mesh for mesh in meshes if mesh.HasAPI(UsdPhysics.CollisionAPI)):
            collider.RemoveAppliedSchema("NewtonMeshCollisionAPI")
            UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Set("none")
            apply_namespaced(
                NewtonSDFCollisionCfg(sdf_max_resolution=64, sdf_padding=0.012),
                collider.GetPath().pathString,
                stage,
            )

        # The global cloth friction is low so tableware can stay put; pinching needs a strong local contact.
        grasp_material_path = f"{root_path}/GraspMaterial"
        spawn_physics_material_from_fragments(
            grasp_material_path,
            rigid_material(
                density=None,
                friction=2000.0,
                contact_stiffness=8.0e3,
                contact_damping=2.0e1,
            ),
            stage=stage,
        )
        grasp_roots = {"L_thumb_proximal_base", "L_index_proximal", "R_thumb_proximal_base", "R_index_proximal"}
        finger_prims = sim_utils.get_all_matching_child_prims(
            root_path,
            predicate=lambda prim: prim.GetName() in grasp_roots,
            stage=stage,
        )
        if len(finger_prims) != len(grasp_roots):
            raise RuntimeError(f"Expected {len(grasp_roots)} H1 grasp roots, found {len(finger_prims)}")
        for prim in finger_prims:
            sim_utils.bind_physics_material(prim.GetPath(), grasp_material_path, stage=stage)


H1_CFG = H1ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    articulation_root_prim_path="/fix_base_joint",
    spawn=sim_utils.UsdFileCfg(
        func=spawn_h1_from_usd,
        usd_path="",
        make_uninstanceable=True,
        fix_root_link=True,
        # A compliant VBD world joint alone lets the base bounce under manipulation loads.
        rigid_props={"/Geometry/pelvis": [sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)]},
        collision_props=collision_properties(),
        physics_material=rigid_material(
            density=None,
            friction=0.50,
            contact_stiffness=1.0e3,
            contact_damping=1.0e-2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.75, 0.0, H1_STANDING_HEIGHT),
        joint_pos={
            ".*(hip|knee|ankle).*": 0.0,
            "torso_joint": 0.0,
            "left_shoulder_pitch_joint": -0.333745,
            "left_shoulder_roll_joint": 1.362674,
            "left_shoulder_yaw_joint": -1.246960,
            "left_elbow_joint": -0.403194,
            "left_hand_joint": -1.755706,
            "right_shoulder_pitch_joint": -0.347129,
            "right_shoulder_roll_joint": -1.375690,
            "right_shoulder_yaw_joint": 1.260851,
            "right_elbow_joint": -0.409848,
            "right_hand_joint": 1.769167,
        },
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[r"(?!torso_joint$)(?![LR]_(?:thumb|index|middle|ring|pinky)_).+"],
            stiffness=5.0e4,
            damping=5.0e2,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint"],
            stiffness=2.0e5,
            damping=2.0e3,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[r"[LR]_(?:thumb|index|middle|ring|pinky)_.+"],
            stiffness=1.0e3,
            damping=2.0,
        ),
    },
)


@configclass
class H1TableclothSceneCfg(InteractiveSceneCfg):
    """Scene containing H1, a VBD cloth, and dynamic tableware."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )
    robot: ArticulationCfg = H1_CFG
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=VisualTableUsdFileCfg(
            usd_path=KITCHEN_ISLAND_USD,
            scale=KITCHEN_ISLAND_SCALE,
            variants={"Physics": "none"},
            make_uninstanceable=True,
            rigid_props={"(/.*)?": [sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=False)]},
            collision_props={"(/.*)?": [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)]},
        ),
    )
    tabletop_collider = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tabletop",
        spawn=tabletop_collider_cfg(
            (0.40, 0.72, 0.08),
            # VBD mixes this with soft_contact_mu=0.01, yielding about 0.2 cloth-table friction.
            friction=4.0,
            contact_stiffness=1.0e4,
            contact_damping=1.0e1,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_TOP_Z - 0.04)),
    )
    cloth = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cloth",
        spawn=sim_utils.MeshRectangleCfg(
            # Corner overhang lets the index fingers lift the cloth without pressing against the tabletop.
            size=(0.48, 0.80),
            edge_refinement=24,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.76, 0.05, 0.05)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=0.24,
                particle_radius=0.001,
                tri_ke=5.0e4,
                tri_ka=5.0e4,
                tri_kd=5.0e1,
                edge_ke=0.10,
                edge_kd=1.0e-3,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, CLOTH_Z)),
    )
    bowl = rigid_object_cfg(
        usd_path=BOWL_USD,
        position=(0.08, -0.07, CLOTH_Z + TABLEWARE_CLEARANCE - BOWL_SCALE * BOWL_LOCAL_Z_MIN),
        mass=0.45,
        collision_proxy_cfg=sim_utils.CylinderCfg(
            radius=BOWL_LOCAL_RADIUS,
            height=BOWL_LOCAL_HEIGHT,
            visible=False,
            collision_props=collision_properties(),
            physics_material=rigid_material(
                density=None,
                friction=0.05,
                contact_stiffness=1.0e4,
                contact_damping=1.0e1,
            ),
        ),
        collision_proxy_position=(0.0, 0.0, BOWL_LOCAL_Z_MIN + 0.5 * BOWL_LOCAL_HEIGHT),
        scale=(BOWL_SCALE,) * 3,
    ).replace(prim_path="{ENV_REGEX_NS}/Bowl")
    glass = rigid_object_cfg(
        usd_path=WINE_GLASS_USD,
        position=(0.10, 0.13, CLOTH_Z + TABLEWARE_CLEARANCE - WINE_GLASS_LOCAL_Z_MIN),
        mass=0.35,
        collision_proxy_cfg=sim_utils.CylinderCfg(
            radius=0.040,
            height=0.006,
            visible=False,
            collision_props=collision_properties(),
            physics_material=rigid_material(
                density=None,
                friction=0.01,
                contact_stiffness=1.0e4,
                contact_damping=1.0e1,
            ),
        ),
        collision_proxy_position=(0.0, 0.0, 0.003),
        center_of_mass=WINE_GLASS_CENTER_OF_MASS,
        diagonal_inertia=WINE_GLASS_DIAGONAL_INERTIA,
        visual_color=(0.55, 0.75, 0.90),
        visual_opacity=0.90,
        visual_roughness=0.12,
    ).replace(prim_path="{ENV_REGEX_NS}/Glass")
    fork = rigid_object_cfg(
        usd_path=FORK_USD,
        position=(0.12, -0.22, CLOTH_Z + TABLEWARE_CLEARANCE - FORK_SCALE * FORK_LOCAL_Z_MIN),
        mass=0.07,
        collision_proxy_cfg=sim_utils.CuboidCfg(
            size=(0.014, 0.160, 0.003),
            visible=False,
            collision_props=collision_properties(),
            physics_material=rigid_material(
                density=None,
                friction=0.02,
                contact_stiffness=1.0e4,
                contact_damping=1.0e1,
            ),
        ),
        collision_proxy_position=(0.0, 0.0092, 0.0016),
        orientation=FORK_ROTATION,
        scale=(FORK_SCALE,) * 3,
        center_of_mass=FORK_CENTER_OF_MASS,
        diagonal_inertia=FORK_DIAGONAL_INERTIA,
    ).replace(prim_path="{ENV_REGEX_NS}/Fork")
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ActionsCfg:
    """Absolute arm/torso poses and finger targets; leg actuators hold the standing pose."""

    arm_action = NewtonInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["torso_joint", "(left|right)_(shoulder_.*|elbow|hand)_joint"],
        controller=NewtonIKSolverCfg(iterations=24, lambda_initial=0.1),
        objectives=[
            NewtonIKPoseObjectiveCfg(
                body_name="left_hand_link",
                name="left_hand",
                body_offset_pos=HAND_OFFSETS[0],
                use_relative_mode=False,
                position_weight=5.0,
                rotation_weight=0.2,
            ),
            NewtonIKPoseObjectiveCfg(
                body_name="right_hand_link",
                name="right_hand",
                body_offset_pos=HAND_OFFSETS[1],
                use_relative_mode=False,
                position_weight=5.0,
                rotation_weight=0.2,
            ),
            NewtonIKPoseObjectiveCfg(
                body_name="torso_link",
                name="torso",
                use_relative_mode=False,
                position_weight=50.0,
                rotation_weight=50.0,
            ),
            NewtonIKJointLimitObjectiveCfg(weight=1.0),
        ],
    )
    finger_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=FINGER_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Proprioception and task-object state exposed to an agent."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        hand_poses = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_link", "right_hand_link"])},
        )
        cloth_points = ObsTerm(
            func=lift_mdp.DeformableSampledPointsInRobotRootFrame,
            params={"asset_cfg": SceneEntityCfg("cloth"), "num_points": 20},
        )
        bowl_pose = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("bowl")})
        bowl_orientation = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("bowl")})
        glass_pose = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("glass")})
        glass_orientation = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("glass")})
        fork_pose = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("fork")})
        fork_orientation = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("fork")})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset the complete scene to the validated tablecloth start state."""

    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class RewardsCfg:
    """Task rewards that favor a clean pull with stationary tableware."""

    cloth_progress = RewTerm(func=mdp.cloth_pull_progress, weight=2.0, params={"target_distance": 0.30})
    tableware_position = RewTerm(
        func=mdp.tableware_displacement,
        weight=2.0,
        params={"asset_names": TABLEWARE_NAMES, "distance_scale": 0.05},
    )
    tableware_orientation = RewTerm(
        func=mdp.tableware_upright,
        weight=1.0,
        params={"asset_names": TABLEWARE_NAMES},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    success = RewTerm(
        func=mdp.is_terminated_term,
        weight=10.0,
        params={"term_keys": ["success"]},
    )


@configclass
class TerminationsCfg:
    """Success, failure, and time-limit conditions for the tablecloth task."""

    success = DoneTerm(
        func=mdp.tablecloth_success,
        params={
            "asset_names": TABLEWARE_NAMES,
            "table_height": TABLE_TOP_Z,
            "pull_distance": 0.30,
            "maximum_tilt": 0.35,
            "maximum_displacement": 0.05,
        },
    )
    tableware_fallen = DoneTerm(
        func=mdp.tableware_fallen,
        params={"asset_names": TABLEWARE_NAMES, "minimum_height": TABLE_TOP_Z - 0.15},
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class H1TableclothEnvCfg(ManagerBasedRLEnvCfg):
    """H1 tablecloth task using Newton VBD and Newton IK."""

    class_type: type | str = "{DIR}.h1_env:H1TableclothEnv"
    scene: H1TableclothSceneCfg = H1TableclothSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    curriculum = None

    def __post_init__(self) -> None:
        self.seed = 0
        self.decimation = 1
        self.episode_length_s = 6.0
        self.sim = SimulationCfg(
            dt=1.0 / FPS,
            render_interval=1,
            physics=NewtonCfg(
                num_substeps=SUBSTEPS,
                collision_decimation=1,
                default_shape_cfg=NewtonShapeCfg(gap=RIGID_GAP, ke=1.0e4, kd=1.0e1, mu=0.35),
                soft_contact_cfg=NewtonSoftContactCfg(
                    soft_contact_ke=1.0e3,
                    soft_contact_kd=1.0e-2,
                    soft_contact_mu=0.01,
                ),
                collision_cfg=NewtonCollisionPipelineCfg(
                    broad_phase="sap",
                    soft_contact_margin=0.008,
                    enable_rigid_soft_full_surface_contact=True,
                ),
                solver_cfg=VBDSolverCfg(
                    iterations=30,
                    rigid_compliant_alm=True,
                    # Structural attachment stiffness is independent of the actuator's position gains.
                    rigid_joint_linear_ke=1.0e7,
                    rigid_joint_angular_ke=1.0e7,
                    rigid_body_particle_contact_buffer_size=8192,
                ),
            ),
            default_visualizer_cfg=VisualizerCfg(
                eye=(-2.435, -2.70, 1.725),
                lookat=(-0.08, 0.0, 0.93),
            ),
        )
