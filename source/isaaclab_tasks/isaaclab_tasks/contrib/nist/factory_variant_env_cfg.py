# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory composition with reset-selectable assembly variants."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from . import mdp
from .assembly_variants import ASSEMBLY_VARIANT_NAMES
from .factory_env_cfg import (
    FIXED_ASSET_MATERIAL_EVENT,
    HELD_ASSET_MATERIAL_EVENT,
    ROBOT_MATERIAL_EVENT,
    FactoryEnvCfg,
    FactoryPhysicsCfg,
    FactoryRewardsCfg,
    FactoryTerminationsCfg,
)
from .factory_presets import EndEffectorBodyCfg, GripperGraspOffsetCfg
from .factory_variant_scene_cfg import FactoryVariantSceneCfg
from .utils import SamplerCfg, UniformSamplingStrategyCfg
from .variant_reset_env_cfg import VARIANT_ACCUMULATOR_RESET


@configclass
class FactoryVariantObservationsCfg:
    """Policy state and scene geometry observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        held_asset_in_fixed_asset_frame = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            history_length=5,
            params={
                "target_asset_cfg": SceneEntityCfg("held_asset"),
                "root_asset_cfg": SceneEntityCfg("fixed_asset"),
                "target_asset_offset": "held_align",
                "root_asset_offset": "fixed_tip",
            },
        )
        end_effector_vel_lin_ang_b = ObsTerm(
            func=mdp.asset_link_velocity_in_root_asset_frame,
            history_length=5,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),  # type: ignore
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset": GripperGraspOffsetCfg(),
            },
        )
        joint_pos = ObsTerm(func=mdp.joint_pos, history_length=5)
        prev_action = ObsTerm(func=mdp.last_action, history_length=5)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PerceptionCfg(ObsGroup):
        scene_point_cloud = ObsTerm(
            func=mdp.scene_point_cloud_b,
            clip=(-2.0, 2.0),
            params={
                "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                "held_asset_cfg": SceneEntityCfg("held_asset"),
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "fixed_num_points": 256,
                "held_num_points": 256,
                "robot_num_points": 256,
                "flatten": True,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    perception: PerceptionCfg = PerceptionCfg()


@configclass
class FactoryVariantEventCfg:
    """Add assembly selection to the shared Factory startup events."""

    assembly_variants = EventTerm(
        func=mdp.AssemblyVariantContext,
        mode="startup",
        params={
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "variant_names": ASSEMBLY_VARIANT_NAMES,
        },
    )
    held_asset_material = HELD_ASSET_MATERIAL_EVENT
    fixed_asset_material = FIXED_ASSET_MATERIAL_EVENT
    robot_material = ROBOT_MATERIAL_EVENT
    reset_strategies = VARIANT_ACCUMULATOR_RESET


@configclass
class FactoryVariantTerminationsCfg(FactoryTerminationsCfg):
    """Use variant geometry and Newton failure detection."""

    progress_context = DoneTerm(
        func=mdp.variant_progress_context,
        params={
            "success_threshold": 0.001,
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
        },
    )


@configclass
class FactoryVariantEnvCfg(FactoryEnvCfg):
    """Factory environment with reset-selectable assembly pairs."""

    scene: FactoryVariantSceneCfg = FactoryVariantSceneCfg()
    observations: FactoryVariantObservationsCfg = FactoryVariantObservationsCfg()
    events: FactoryVariantEventCfg = FactoryVariantEventCfg()
    terminations: FactoryVariantTerminationsCfg = FactoryVariantTerminationsCfg()
    rewards: FactoryRewardsCfg = FactoryRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.render_interval = self.decimation
        self.sim.physics = FactoryPhysicsCfg().newton_mjwarp

    def play_mode(self) -> None:
        """Evaluate from uniformly sampled random-start states."""
        self.scene.num_envs = 128
        uniform = SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0)
        reset = self.events.reset_strategies.params
        reset["state_table_size"] = 500
        reset["sampling"] = uniform
        scene_reset = reset.get("reset_term", self.events.reset_strategies)
        choice = scene_reset.params["terms"]["reset_strategies"].params
        choice["terms"] = {"start_random": choice["terms"]["start_random"]}
