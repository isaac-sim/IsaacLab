# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat (hand-written) single-robot multi-task env config.

Demonstrates how to write a heterogeneous multi-task environment that uses
**one robot type** (Franka) across all environments, sharing a single
``ArticulationView``, while each group of environments runs a different task.
No ``MultiTaskRegistryConfig`` is used -- everything is specified explicitly.

The Franka robot is declared once with ``prim_path="{ENV_REGEX_NS}/Robot"``,
which creates a single PhysX ``ArticulationView`` covering every environment.
Because the robot and its action space are identical for all groups, a single
set of action terms (``arm_action`` + ``gripper_action``) applies to all envs.

Per-task objects (lift cube, stacking cubes) use ``{ENV_REGEX_NS}`` in their
``prim_path`` just like shared assets, but declare ``task_group`` to indicate
which task group they belong to.  The scene's ``task_groups`` dict defines the
groups; env-id partitioning and cloning masks are resolved automatically.

The stack task uses a different Franka default joint pose, which is applied
per-env via ``_default_joint_pos_per_env`` and the corresponding reset event.

Layout (3 groups, evenly split):
    Group 0 (LIFT):   Franka -- Lift Cube        (1 cube)
    Group 1 (STACK):  Franka -- Stack Cubes      (3 cubes)
    Group 2 (REACH):  Franka -- Reach            (no objects)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg, partition_env_ids
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.tasks.manipulation.multitask import mdp

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

# ---------------------------------------------------------------------------
# Task / group constants
# ---------------------------------------------------------------------------


@configclass
class MultitaskPhysicsCfg(PresetCfg):
    """Physics backend presets for the single-robot multitask environment."""

    default: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=16 * 1024,
        friction_correlation_distance=0.00625,
    )
    physx: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=16 * 1024,
        friction_correlation_distance=0.00625,
    )
    newton: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=60,
            nconmax=80,
            ls_iterations=20,
            cone="pyramidal",
            ls_parallel=True,
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=1,
        debug_mode=False,
    )


NUM_ENVS = 25

# Stack task uses a different default arm pose than lift/reach.
# (From the official Franka stack env config.)
_STACK_DEFAULT_JOINT_POS = [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400]

# Default Franka joint pose from FRANKA_PANDA_CFG (lift / reach tasks).
_DEFAULT_JOINT_POS = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04]


# ---------------------------------------------------------------------------
# Task / group names
# ---------------------------------------------------------------------------

TASK_LIFT_NAME = "lift"
TASK_STACK_NAME = "stack"
TASK_REACH_NAME = "reach"


# ---------------------------------------------------------------------------
# Per-group event functions
# ---------------------------------------------------------------------------


def reset_default_joint_pos_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Write per-env default joint positions directly into the simulation.

    Instead of mutating ``default_joint_pos`` (which is immutable after
    physics priming), this writes the desired joint positions and zero
    velocities straight into the sim for the requested environments.
    """
    per_env_list = getattr(env.cfg, "_default_joint_pos_per_env", None)
    if per_env_list is None:
        return
    asset = env.scene[asset_cfg.name]
    per_env = torch.tensor(per_env_list, dtype=torch.float32, device=asset.device)
    joint_pos = per_env[env_ids]
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Shared spawn helpers
# ---------------------------------------------------------------------------

_LIFT_CUBE_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

_LIFT_CUBE_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    scale=(0.8, 0.8, 0.8),
    rigid_props=_LIFT_CUBE_RIGID_PROPS,
)

_STACK_CUBE_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

_STACK_CUBE_1_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
    scale=(1.0, 1.0, 1.0),
    rigid_props=_STACK_CUBE_RIGID_PROPS,
)

_STACK_CUBE_2_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
    scale=(1.0, 1.0, 1.0),
    rigid_props=_STACK_CUBE_RIGID_PROPS,
)

_STACK_CUBE_3_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
    scale=(1.0, 1.0, 1.0),
    rigid_props=_STACK_CUBE_RIGID_PROPS,
)

_TABLE_SPAWN = UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class FlatSingleRobotSceneCfg(InteractiveSceneCfg):
    """Scene for a single-robot multi-task environment.

    **One Franka** is placed in every environment (``{ENV_REGEX_NS}/Robot``).
    This results in a single ``ArticulationView`` spanning all ``num_envs``.

    Per-task objects declare ``task_group`` to indicate which group they
    belong to.  Group 2 (reach) has no objects at all.
    """

    # Values are relative weights: 1:1:1 means equal split across the three groups.
    task_groups = {TASK_LIFT_NAME: 2, TASK_STACK_NAME: 1, TASK_REACH_NAME: 1}

    # -- shared (all envs) ---------------------------------------------------
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # -- Franka robot (ALL envs, one ArticulationView) -----------------------
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # -- Table (ALL envs) ----------------------------------------------------
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )

    # -- Group 0: Lift – single cube ----------------------------------------
    lift_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_LIFT_CUBE_SPAWN,
        task_group=TASK_LIFT_NAME,
    )

    # -- Group 1: Stack – three cubes ----------------------------------------
    stack_cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_1_SPAWN,
        task_group=TASK_STACK_NAME,
    )
    stack_cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.05, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_2_SPAWN,
        task_group=TASK_STACK_NAME,
    )
    stack_cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, -0.1, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_3_SPAWN,
        task_group=TASK_STACK_NAME,
    )

    # -- Group 2: Reach – no objects -----------------------------------------
    # (nothing to declare)


# ---------------------------------------------------------------------------
# Actions  (single set – same robot, same action space for every env)
# ---------------------------------------------------------------------------


@configclass
class FlatSingleRobotActionsCfg:
    """Joint-position actions applied to the shared ``robot`` asset.

    Because every environment uses the same Franka with the same action space,
    only one ``arm_action`` and one ``gripper_action`` are needed.  The action
    terms reference ``asset_name="robot"`` whose ``prim_path`` is
    ``{ENV_REGEX_NS}/Robot`` – the robot has no layout key (present in all
    envs), so the actions are dispatched to every environment.
    """

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


# ---------------------------------------------------------------------------
# Observations / Rewards / Terminations / Events
# ---------------------------------------------------------------------------


@configclass
class FlatCommandsCfg:
    """Command terms – reach target for the shared Franka end-effector.

    The command is generated for every env in Group 2 (REACH) task.
    """

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        task_group=TASK_REACH_NAME,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class FlatObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        sim_time_s = ObsTerm(func=mdp.current_time_s)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FlatRewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class FlatTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class FlatEventsCfg:
    """Events including per-env default joint pose reset for heterogeneous tasks."""

    reset_scene_to_default = EventTerm(func=mdp.reset_multitask_scene_to_default, mode="reset")
    reset_default_joint_pos = EventTerm(
        func=reset_default_joint_pos_per_env,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ---------------------------------------------------------------------------
# Top-level environment config
# ---------------------------------------------------------------------------


@configclass
class FlatSingleRobotMultiTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Flat single-robot multi-task env: one Franka, three tasks.

    No ``MultiTaskRegistryConfig`` -- everything is explicit.

    The Franka robot has ``prim_path="{ENV_REGEX_NS}/Robot"``, so it is
    instantiated in every environment and backed by **one** PhysX
    ``ArticulationView``.  Actions (``arm_action``, ``gripper_action``)
    reference ``asset_name="robot"`` and therefore apply to all envs.

    Per-task objects (cubes) use ``{GROUP<N>}`` placeholder tokens that are
    resolved in ``__post_init__`` into concrete env-id regexes.

    ``__post_init__`` stores a ``_default_joint_pos_per_env`` nested list
    (OmegaConf-safe) so the reset event can write per-env default joint
    positions (stack envs get a different pose than lift/reach envs).

    Group 0 (LIFT):  Franka -- lift 1 cube       (joint-position actions)
    Group 1 (STACK): Franka -- stack 3 cubes     (joint-position actions)
    Group 2 (REACH): Franka -- reach (no objects) (joint-position actions)
    """

    scene: FlatSingleRobotSceneCfg = FlatSingleRobotSceneCfg(
        num_envs=NUM_ENVS, env_spacing=2.0, replicate_physics=False
    )
    actions: FlatSingleRobotActionsCfg = FlatSingleRobotActionsCfg()
    commands: FlatCommandsCfg = FlatCommandsCfg()
    observations: FlatObservationsCfg = FlatObservationsCfg()
    rewards: FlatRewardsCfg = FlatRewardsCfg()
    terminations: FlatTerminationsCfg = FlatTerminationsCfg()
    events: FlatEventsCfg = FlatEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 3
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physics = MultitaskPhysicsCfg()

        # --- per-env default joint positions (stored as nested list for OmegaConf) ---
        # The partition is computed eagerly here because we need env_ids before
        # the scene is constructed (for the reset event's joint-pose table).
        groups = partition_env_ids(self.scene.num_envs, self.scene.task_groups)
        default_jpos = [list(_DEFAULT_JOINT_POS) for _ in range(self.scene.num_envs)]
        for eid in groups[TASK_STACK_NAME]:
            default_jpos[eid] = list(_STACK_DEFAULT_JOINT_POS)
        self._default_joint_pos_per_env: list[list[float]] = default_jpos
