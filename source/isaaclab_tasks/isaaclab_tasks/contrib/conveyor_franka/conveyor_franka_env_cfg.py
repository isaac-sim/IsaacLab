# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the force-driven conveyor and Franka demonstration scene."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonMaterialPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

from . import mdp
from .conveyor_geometry import (
    BELT_CENTER_Y,
    BELT_COLOR,
    BELT_TURN_RADIUS,
    GUARD_COLOR,
    PARCEL_COLOR,
    MeshSpec,
    belt_mesh_spec,
    guard_mesh_specs,
)
from .franka_robot_cfg import FRANKA_PANDA_CONVEYOR_CFG

_DYNAMIC_PROPERTIES = sim_utils.RigidBodyBaseCfg()


def _srgb_to_linear_channel(value: float) -> float:
    """Convert an sRGB channel to the linear value expected by USD displayColor."""
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


@configclass
class ActionsCfg:
    """Relative arm and binary gripper actions."""

    arm_action = mdp.ConveyorRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint[1-7]"],
        scale=0.12,
        max_delta=0.12,
        gravity_compensation=True,
    )
    gripper_action = mdp.ResetBufferedGripperActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint[1-2]"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
        force_close_steps=5,
    )


@configclass
class ObservationsCfg:
    """Policy observations with stable cube identity and transfer commands."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Fully observed transfer policy input."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint[1-7]"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint[1-7]"])},
        )
        gripper_pos = ObsTerm(
            func=mdp.gripper_joint_positions,
            params={"robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint[1-2]"])},
        )
        objects = ObsTerm(func=mdp.transfer_object_observation)
        active_transfer = ObsTerm(func=mdp.active_transfer_features)
        target_cube = ObsTerm(func=mdp.target_cube_one_hot)
        cube_conveyors = ObsTerm(func=mdp.cube_conveyor_state)
        target_side = ObsTerm(func=mdp.target_side_one_hot)
        eef_velocity = ObsTerm(func=mdp.end_effector_velocity)
        eef_axes = ObsTerm(func=mdp.end_effector_axes)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset the scene, then restore one validated transfer state."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_from_state_table = EventTerm(
        func=mdp.ConveyorResetStateTable,
        mode="reset",
        params={
            "fixed_recipe": None,
            "fixed_variant_id": None,
            "fixed_target_cube_id": None,
            "fixed_source_side_id": None,
            "belt_start_x_range": (0.30, 0.82),
            "cube_position_noise": 0.015,
            "arm_joint_noise": 0.015,
        },
    )


@configclass
class RewardsCfg:
    """Transfer progress, completion, and regularization rewards."""

    progress = RewTerm(func=mdp.ConveyorTransferProgressReward, weight=60.0)
    success = RewTerm(func=mdp.transfer_success_reward, weight=600.0)
    failure = RewTerm(func=mdp.terminal_failure, weight=-60.0)
    arm_action_l2 = RewTerm(
        func=mdp.action_term_l2,
        params={"action_name": "arm_action"},
        weight=-1.0e-3,
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-3)
    joint_velocity_l2 = RewTerm(
        func=mdp.finite_joint_velocity_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint[1-7]"])},
        weight=-1.0e-4,
    )


@configclass
class TerminationsCfg:
    """Successful placement, physical failure, and horizon terms."""

    learning_progress_context = DoneTerm(
        func=mdp.ConveyorResetLearningProgress,
        params={
            "minimum_episode_steps": 3,
            "minimum_progress": 0.35,
            "maximum_target_potential": 5.0,
        },
    )
    success = DoneTerm(
        func=mdp.StableConveyorTransfer,
        params={
            "minimum_episode_steps": 2,
            "hold_steps": 3,
            "lateral_tolerance": 0.055,
            "maximum_cube_speed": 0.65,
            "minimum_finger_position": 0.027,
            "minimum_tool_clearance": 0.055,
        },
    )
    cube_out_of_workspace = DoneTerm(func=mdp.cube_out_of_workspace)
    nonfinite_scene_state = DoneTerm(func=mdp.nonfinite_scene_state)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Adaptive phase-balanced reset-state sampling."""

    reset_sampling = CurrTerm(
        func=mdp.ConveyorResetCurriculum,
        params={
            "progress_context_name": "learning_progress_context",
            "final_success_termination_name": "success",
            # Match Franka Stack: sampling follows each row's recent policy
            # competence instead of retaining stale early failures forever.
            "monitored_history_len": 50,
            # Keep a deployment-facing stream while the remaining starts
            # adapt around the rolling pickup-to-placement frontier. Adaptive
            # rows remain balanced across recipe, cube identity, and side.
            "deployment_probability": 0.35,
        },
    )


@configclass
class ConveyorForceCfg:
    """Configuration for force-based conveyor traction."""

    speed: float = 0.35
    """Tangential conveyor surface speed [m/s]."""

    friction: float = 0.5
    """Coulomb friction coefficient used to limit traction."""

    normal_threshold: float = 0.95
    """Minimum upward contact-normal alignment in the range [0, 1]."""

    def __post_init__(self) -> None:
        """Validate conveyor force parameters."""
        if self.speed < 0.0:
            raise ValueError(f"Conveyor speed must be non-negative, got {self.speed}.")
        if self.friction < 0.0:
            raise ValueError(f"Conveyor friction must be non-negative, got {self.friction}.")
        if not 0.0 <= self.normal_threshold <= 1.0:
            raise ValueError(f"Conveyor normal threshold must be in [0, 1], got {self.normal_threshold}.")


@sim_utils.clone
def _spawn_shape_with_display_color(
    prim_path: str,
    cfg: sim_utils.ShapeCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a primitive and author a renderer-independent USD display color."""
    if isinstance(cfg, sim_utils.CuboidCfg):
        prim = sim_utils.spawn_cuboid(prim_path, cfg, translation, orientation, **kwargs)
    elif isinstance(cfg, sim_utils.CylinderCfg):
        prim = sim_utils.spawn_cylinder(prim_path, cfg, translation, orientation, **kwargs)
    elif isinstance(cfg, sim_utils.CapsuleCfg):
        prim = sim_utils.spawn_capsule(prim_path, cfg, translation, orientation, **kwargs)
    elif isinstance(cfg, sim_utils.SphereCfg):
        prim = sim_utils.spawn_sphere(prim_path, cfg, translation, orientation, **kwargs)
    else:
        raise TypeError(f"Unsupported colored primitive configuration: {type(cfg).__name__}")

    if cfg.visual_material is not None:
        from pxr import Usd, UsdGeom

        display_color = tuple(_srgb_to_linear_channel(value) for value in cfg.visual_material.diffuse_color)
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Gprim):
                UsdGeom.Gprim(child).CreateDisplayColorAttr([display_color])
    return prim


def _static_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    friction: float = 0.7,
    roughness: float = 0.75,
    metallic: float = 0.0,
) -> AssetBaseCfg:
    """Build a static colliding cuboid configuration."""
    spawn = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionBaseCfg(),
        physics_material=RigidBodyMaterialBaseCfg(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=roughness,
            metallic=metallic,
        ),
    )
    spawn.func = _spawn_shape_with_display_color
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=spawn,
    )


def _static_mesh(
    prim_path: str,
    spec: MeshSpec,
    color: tuple[float, float, float],
    friction: float,
    roughness: float,
    metallic: float,
    mujoco_priority: int | None = None,
) -> AssetBaseCfg:
    """Build a static colliding triangle-mesh configuration."""
    collision_props = sim_utils.CollisionBaseCfg()
    if mujoco_priority is not None:
        collision_props = [
            UsdPhysicsCollisionCfg(collision_enabled=True),
            MujocoCollisionCfg(priority=mujoco_priority),
        ]
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.MeshCustomCfg(
            vertices=spec.vertices,
            faces=spec.faces,
            collision_props=collision_props,
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=roughness,
                metallic=metallic,
            ),
        ),
    )


def _cube(
    name: str,
    color: tuple[float, float, float],
    pos: tuple[float, float, float],
) -> RigidObjectCfg:
    """Build one numbered dynamic transfer cube."""
    spawn = sim_utils.CuboidCfg(
        size=(0.04, 0.04, 0.04),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.75),
    )
    spawn.rigid_props = _DYNAMIC_PROPERTIES
    spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.05)
    spawn.collision_props = sim_utils.CollisionBaseCfg(contact_offset=0.0, rest_offset=0.0)
    spawn.physics_material = NewtonMaterialPropertiesCfg(
        # The belt's higher MuJoCo contact priority overrides this friction
        # only for belt/cube pairs, leaving physical finger/cube friction.
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.0,
        torsional_friction=0.002,
        rolling_friction=0.0001,
        contact_stiffness=1.0e4,
        contact_damping=200.0,
    )
    spawn.func = _spawn_shape_with_display_color
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        spawn=spawn,
    )


@configclass
class ConveyorFrankaSceneCfg(InteractiveSceneCfg):
    """Scene with two counter-rotating racetrack conveyors around a table-mounted Franka."""

    # Use the MuJoCo Menagerie-derived model with explicit manipulation gains.
    robot = FRANKA_PANDA_CONVEYOR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.35,
                "panda_joint3": 0.0,
                "panda_joint4": -2.35,
                "panda_joint5": 0.0,
                "panda_joint6": 2.0,
                "panda_joint7": 0.78,
                "panda_finger_joint.*": 0.04,
            }
        ),
    )

    tabletop = _static_cuboid(
        prim_path="{ENV_REGEX_NS}/Tabletop",
        size=(2.0, 1.9, 0.08),
        pos=(0.50, 0.0, -0.04),
        color=(0.32, 0.34, 0.37),
    )
    table_pedestal = _static_cuboid(
        prim_path="{ENV_REGEX_NS}/TablePedestal",
        size=(0.75, 0.55, 0.76),
        pos=(0.25, 0.0, -0.46),
        color=(0.18, 0.20, 0.23),
    )

    cube_0 = _cube("Cube0", (0.15, 0.35, 0.90), (0.30, BELT_CENTER_Y - BELT_TURN_RADIUS, 0.06))
    cube_1 = _cube("Cube1", (0.90, 0.20, 0.15), (0.78, BELT_CENTER_Y - BELT_TURN_RADIUS, 0.06))
    cube_2 = _cube("Cube2", (0.15, 0.75, 0.25), (0.30, -BELT_CENTER_Y + BELT_TURN_RADIUS, 0.06))
    cube_3 = _cube("Cube3", PARCEL_COLOR, (0.78, -BELT_CENTER_Y + BELT_TURN_RADIUS, 0.06))

    cube_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cube.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.85)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )

    def __post_init__(self) -> None:
        """Generate both racetrack belts and their inner/outer guardrails."""
        for side in ("Left", "Right"):
            belt_spec = belt_mesh_spec(side)
            setattr(
                self,
                f"conveyor_{side.lower()}_belt",
                _static_mesh(
                    prim_path=f"{{ENV_REGEX_NS}}/{belt_spec.name}",
                    spec=belt_spec,
                    color=BELT_COLOR,
                    # MuJoCo requires a tiny positive value even though the force driver,
                    # rather than solver friction, supplies the belt motion.
                    friction=1.1e-5,
                    roughness=0.9,
                    metallic=0.0,
                    # MuJoCo otherwise resolves equal-priority pair friction
                    # with max(belt, cube), pinning parcels to the static mesh.
                    mujoco_priority=1,
                ),
            )

            for spec in guard_mesh_specs(side):
                boundary = "inner" if spec.name.endswith("Inner") else "outer"
                setattr(
                    self,
                    f"guard_{side.lower()}_{boundary}",
                    _static_mesh(
                        prim_path=f"{{ENV_REGEX_NS}}/{spec.name}",
                        spec=spec,
                        color=GUARD_COLOR,
                        friction=0.2,
                        roughness=0.3,
                        metallic=0.8,
                    ),
                )


@configclass
class ConveyorFrankaEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL task for commanded conveyor-to-conveyor cube transfer."""

    scene: ConveyorFrankaSceneCfg = ConveyorFrankaSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
    conveyor_force: ConveyorForceCfg = ConveyorForceCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    decimation: int = 2
    episode_length_s: float = 10.0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=300,
                nconmax=256,
                impratio=10.0,
                cone="elliptic",
                update_data_interval=2,
                iterations=100,
                ls_iterations=15,
                ls_parallel=False,
                use_mujoco_contacts=False,
                ccd_iterations=35,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(),
            # Refresh contacts between the two 240 Hz solver substeps and
            # preserve the authored surfaces without speculative separation.
            collision_decimation=1,
            default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=0.0),
            num_substeps=2,
            use_cuda_graph=False,
            load_visual_shapes=True,
        ),
    )

    def __post_init__(self) -> None:
        self.seed = 42
        # Frame the complete robot and both conveyor lanes in the Newton viewer.
        try:
            import isaaclab_visualizers  # noqa: F401
        except ModuleNotFoundError as exc:
            if exc.name != "isaaclab_visualizers":
                raise
            return
        from isaaclab_visualizers.newton import NewtonVisualizerCfg

        self.sim.default_visualizer_cfg = NewtonVisualizerCfg(
            eye=(2.3, -2.7, 1.8),
            lookat=(0.45, 0.0, 0.35),
        )

    def play_mode(self) -> None:
        """Evaluate complete transfers from randomized moving-belt starts."""
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 8)
        self.events.reset_from_state_table.params["fixed_recipe"] = int(mdp.ConveyorResetRecipe.BELT)
        self.events.reset_from_state_table.params["fixed_variant_id"] = mdp.BELT_DEPLOYMENT_VARIANT
        self.curriculum = None
