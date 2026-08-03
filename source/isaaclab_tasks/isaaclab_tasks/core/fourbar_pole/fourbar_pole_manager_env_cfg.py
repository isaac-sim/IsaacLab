# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import math
from dataclasses import MISSING

from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.core.fourbar_pole.mdp as mdp
from isaaclab_tasks.utils import PresetCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.fourbar_pole import FOURBAR_POLE_CFG  # isort:skip


##
# Physics backend presets
##


@configclass
class FourbarPolePhysicsCfg(PresetCfg):
    """Physics backends for the fourbar-pole task.

    Only the kamino (Newton constraint) solver is wired up for now; the closed
    four-bar kinematic loop is resolved as an equality constraint by kamino. The
    ``default`` field mirrors ``newton_kamino`` so the task uses it without any
    ``physics=`` selector. Additional backends will be added later.
    """

    default: NewtonCfg = MISSING
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoSolverCfg(
            integrator="euler",
            use_fk_solver=True,
            sparse_jacobian=True,
            constraints_alpha=0.1,
            padmm_max_iterations=100,
            padmm_rho_0=0.1,
            padmm_warmstart_mode="containers",
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )

    def __post_init__(self):
        if not isinstance(self.default, NewtonCfg):
            self.default = copy.deepcopy(self.newton_kamino)


##
# Scene definition
##


@configclass
class FourbarPoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a fourbar-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # fourbar-pole
    robot: ArticulationCfg = FOURBAR_POLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["ground_to_crank"], scale=50.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # pole encoded as (cos, sin, vel) to avoid the +-pi wrap discontinuity during swing-up
        pole_cos = ObsTerm(
            func=mdp.joint_pos_cos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["coupler_to_pole"])},
        )
        pole_sin = ObsTerm(
            func=mdp.joint_pos_sin,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["coupler_to_pole"])},
        )
        pole_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["coupler_to_pole"])},
        )

        # crank encoded with (pos, vel). No wrapping possible due to joint limits.
        crank_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["ground_to_crank"])},
        )
        crank_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["ground_to_crank"])},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset crank around its default (0) so the cart starts at varied positions
    reset_crank_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["ground_to_crank"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # reset pole around its default (0) with a small perturbation
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["coupler_to_pole"]),
            "position_range": (-math.pi, math.pi),
            "velocity_range": (-0.5, 0.5),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Primary task + success tracking: swing the pole up and keep it upright (cos is maximal upright)
    pole_upright = RewTerm(
        func=mdp.pole_upright,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["coupler_to_pole"]),
            "success_threshold": 0.9,
            "hold_time_s": 0.5,
        },
    )
    # (2) Shaping: damp the pole angular velocity to settle at the top
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["coupler_to_pole"])},
    )
    # (3) Shaping: discourage excessive crank motion
    crank_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["ground_to_crank"])},
    )
    # (4) Shaping: penalize control effort
    effort = RewTerm(
        func=mdp.action_l2,
        weight=-0.01,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Time out only -- the pole must be free to hang and the parallelogram is geometrically bounded.
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class FourbarPoleSwingupEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the fourbar-pole swing-up environment."""

    # Scene settings
    scene: FourbarPoleSceneCfg = FourbarPoleSceneCfg(num_envs=4096, env_spacing=4.0, clone_in_fabric=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (12.0, 0.0, 4.0)
        # Match Newton GL / --video camera to the task viewport when --viz newton creates the visualizer.
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=self.viewer.eye, lookat=self.viewer.lookat)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = FourbarPolePhysicsCfg()
