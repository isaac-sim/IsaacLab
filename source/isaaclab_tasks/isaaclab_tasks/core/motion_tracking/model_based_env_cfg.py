# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native physical MDP for differentiable model-based motion tracking."""

from dataclasses import MISSING

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import PresetCfg

from .mdp.model_based import observations
from .mdp.model_based.commands import randomize_context
from .mdp.model_based.commands_cfg import ReferenceCommandCfg


@configclass
class ModelBasedSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=0.3, dynamic_friction=0.3)
        ),
    )
    robot: ArticulationCfg = MISSING


@configclass
class ModelBasedCommandsCfg:
    motion = ReferenceCommandCfg()


@configclass
class ModelBasedActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        class_type="isaaclab_tasks.core.motion_tracking.mdp.model_based.commands:BiasedJointPositionAction",
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
    )


@configclass
class ModelBasedPhysicsCfg(PresetCfg):
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=500, nconmax=128, cone="pyramidal", integrator="implicitfast", use_mujoco_contacts=True
        ),
        num_substeps=1,
        use_cuda_graph=True,
    )
    default = newton_mjwarp


@configclass
class ReferenceObservationsCfg:
    @configclass
    class StateCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=observations.full_state)

    @configclass
    class ActorCfg(ObservationGroupCfg):
        sensor = ObservationTermCfg(func=observations.actor_sensor)
        enable_corruption = False

    full_state = StateCfg()
    actor = ActorCfg()


@configclass
class ReferenceRewardsCfg:
    tracking = RewardTermCfg(func=observations.reward, weight=1.0)


@configclass
class ReferenceTerminationsCfg:
    anchor_pos = TerminationTermCfg(func=observations.anchor_height_failure, params={"threshold": 0.25})
    anchor_ori = TerminationTermCfg(func=observations.anchor_orientation_failure, params={"threshold": 0.8})
    motion_end = TerminationTermCfg(func=observations.motion_end, time_out=True)
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class ReferenceEventsCfg:
    context = EventTermCfg(func=randomize_context, mode="startup")


@configclass
class ModelBasedMotionTrackingEnvCfg(ManagerBasedRLEnvCfg):
    scene = ModelBasedSceneCfg(num_envs=4096, env_spacing=0.0)
    commands = ModelBasedCommandsCfg()
    actions = ModelBasedActionsCfg()
    observations = ReferenceObservationsCfg()
    rewards = ReferenceRewardsCfg()
    terminations = ReferenceTerminationsCfg()
    events = ReferenceEventsCfg()
    # Four 5 ms integration steps per policy action, matching the reference integrator.
    sim = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics=ModelBasedPhysicsCfg(),
        visualizer_cfgs=VisualizerCfg(eye=(3.0, -3.0, 2.0), lookat=(0.0, 0.0, 0.75), focal_length=24.0),
    )
    decimation = 4
    episode_length_s = 10.0
