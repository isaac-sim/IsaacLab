# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import MultiAssetSpawnerCfg, SimulationCfg
from isaaclab.utils import config_field, replace_config
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import PresetCfg, preset

from . import mdp
from .keyboards import TYPING_KEYBOARD_POOL

##
# Pre-defined configs
##
from isaaclab_assets.robots.so101 import SO101_CFG  # isort: skip
from typing import Any


@dataclass
class KeyboardAssetCfg(PresetCfg):
    """Backend-dependent keyboard articulation, selected by the ``physics=`` preset.

    PhysX flattens all per-env articulation instances onto one axis, so the ``fixed_dof`` partition
    (18 roots/env) exposes every key. IsaacLab's Newton :class:`ArticulationData` reads only the first
    articulation per env (``[:, 0]``), so Newton uses a single 108-DOF articulation instead.
    """

    # PhysX: 18 articulation roots/env under ``parts/part_*`` (its ArticulationView flattens them).
    # (Octi) partitioned keyboard investigate if topologies can be harnessed to improve performance
    default: Any = config_field(
        ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Keyboard",
            articulation_root_prim_path="/parts/part_.*",
            spawn=MultiAssetSpawnerCfg(assets_cfg=list(TYPING_KEYBOARD_POOL.spawners_partitioned), random_choice=False),
            init_state=ArticulationCfg.InitialStateCfg(
                # Pose used by the training checkpoint: centered in the SO-101 workspace and rotated -90 degrees.
                pos=(0.285, 0.0, 0.01),
                rot=(0.0, 0.0, -0.7071068, 0.7071068),  # -90 deg about Z (xyzw)
                joint_pos={"key_.*_joint": 0.0},
                joint_vel={"key_.*_joint": 0.0},
            ),
            actuators={},
        )
    )
    # Newton: one 108-DOF articulation/env, root auto-resolved at ``prim_path`` (Newton's
    # ArticulationData reads only the first articulation per env, so separate parts would be invisible).
    newton_mjwarp: Any = config_field(
        replace_config(
            default,
            articulation_root_prim_path=None,
            spawn=MultiAssetSpawnerCfg(assets_cfg=list(TYPING_KEYBOARD_POOL.spawners_single), random_choice=False),
        )
    )
    isaacsim_physx: Any = config_field(default)
    physx: Any = config_field(default)
    default: Any = config_field(newton_mjwarp)


@dataclass
class SO101SceneCfg(InteractiveSceneCfg):
    """SO-101 keyboard-typing scene."""

    robot: ArticulationCfg = config_field(
        replace_config(
            SO101_CFG,
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=replace_config(
                SO101_CFG.spawn,
                variants={
                    "Robot": "robot",
                    "Sensor": "sensors",
                    "Physics": preset(
                        default="physics",
                        isaacsim_physx="physx",
                        physx="physx",
                        newton_mjwarp="physics",
                    ),
                },
            ),
        )
    )

    # keyboard
    keyboard: ArticulationCfg = config_field(KeyboardAssetCfg())

    # contact sensor
    robot_contact: Any = config_field(
        ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            update_period=0.0,
            history_length=1,
            track_pose=False,
        )
    )

    # plane
    plane: Any = config_field(
        AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(),
            spawn=sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0)),
            collision_group=-1,
        )
    )

    # lights
    sky_light: Any = config_field(
        AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
    )


@dataclass
class CommandsCfg:
    """Command terms for the MDP."""

    typing: Any = config_field(
        mdp.LetterTypingCommandCfg(
            asset_name="robot",
            object_name="keyboard",
            resampling_time_range=(10.0, 10.0),
            debug_vis=False,
            letter_length=(1, 5),
            max_len=5,
            command_mode="letter_full",
            typeable_slots=tuple(
                slot for slot in TYPING_KEYBOARD_POOL.active_slots if slot != TYPING_KEYBOARD_POOL.backspace_slot
            ),
            backspace_slot=TYPING_KEYBOARD_POOL.backspace_slot,
            slot_labels=TYPING_KEYBOARD_POOL.slot_labels,
            reset=mdp.LetterTypingCommandCfg.ResetCfg(
                enabled=True,
                ik=mdp.DifferentialInverseKinematicsActionCfg(
                    asset_name="robot",
                    joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                    body_name="gripper(_link)?",
                    controller=DifferentialIKControllerCfg(
                        command_type="pose", use_relative_mode=False, ik_method="dls"
                    ),
                    body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
                        pos=(-0.0079, -0.000218121, -0.0981274)
                    ),
                ),
                ik_rpy_deg=(0.0, 45.0, 0.0),  # (roll, pitch, yaw) [deg]
                ik_hover_height=0.02,
                ik_iters=(1, 4),
                ik_seed_joint_noise=0.25,
                buffer_size=8192,
                normal_weight=0.1,
                pre_solve_reset=EventTerm(
                    func=mdp.reset_root_state_uniform,
                    mode="reset",
                    params={
                        "pose_range": {
                            "x": [-0.0, 0.0],
                            "y": [-0.0, 0.0],
                            "z": [0.015, 0.05],
                            "yaw": [-0.1, 0.1],
                            "roll": [0.0, 0.75],
                        },
                        "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
                        "asset_cfg": SceneEntityCfg("keyboard"),
                    },
                ),
            ),
        )
    )


@dataclass
class SO101RelJointPosActionCfg:
    action: Any = config_field(mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.02))


@dataclass
class SO101ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        target_keys_onehot: Any = config_field(ObsTerm(func=mdp.target_keys_onehot, params={"command_name": "typing"}))
        typed_keys_onehot: Any = config_field(ObsTerm(func=mdp.typed_keys_onehot, params={"command_name": "typing"}))

    @dataclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""

        actions: Any = config_field(ObsTerm(func=mdp.last_action))
        joint_pos: Any = config_field(ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0)))
        joint_vel: Any = config_field(ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0)))

    @dataclass
    class PerceptionObsCfg(ObsGroup):
        """Observations for perception group."""

        key_positions: Any = config_field(
            ObsTerm(
                func=mdp.key_positions_b,
                clip=(-2.0, 2.0),
                params={
                    "command_name": "typing",
                    "base_asset_cfg": SceneEntityCfg("robot"),
                    "active_slots": TYPING_KEYBOARD_POOL.active_slots,
                },
            )
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = config_field(PolicyCfg())
    proprio: ProprioObsCfg = config_field(ProprioObsCfg())
    perception: PerceptionObsCfg = config_field(PerceptionObsCfg())


@dataclass
class EventCfg:
    """Reset-mode events (shared by all physics backends)."""

    reset_keyboard: Any = config_field(
        EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": [-0.0, 0.0],
                    "y": [-0.0, 0.0],
                    "z": [0.015, 0.05],
                    "yaw": [-0.1, 0.1],
                    "roll": [0.0, 0.75],
                },
                "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
                "asset_cfg": SceneEntityCfg("keyboard"),
            },
        )
    )


@dataclass
class SO101ReorientRewardCfg:
    typing_progress: Any = config_field(
        RewTerm(func=mdp.letter_typing_progress, weight=2.0, params={"command_name": "typing"})
    )

    success: Any = config_field(RewTerm(func=mdp.typing_success, weight=50.0, params={"command_name": "typing"}))

    mechanical_power: Any = config_field(RewTerm(func=mdp.mechanical_power, weight=-0.0005))

    early_termination: Any = config_field(
        RewTerm(func=mdp.is_terminated_term, weight=-10, params={"term_keys": ["abnormal_robot"]})
    )


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: Any = config_field(DoneTerm(func=mdp.time_out, time_out=True))

    abnormal_robot: Any = config_field(DoneTerm(func=mdp.joint_vel_out_of_limit))

    excessive_contact: Any = config_field(
        DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("robot_contact"), "threshold": 20.0},
        )
    )

    success: Any = config_field(DoneTerm(func=mdp.typing_complete, params={"command_name": "typing"}))


@dataclass
class PhysicsCfg(PresetCfg):
    # Octi: note
    # physx is usable but extremely slow for this task and is only for evaluation purposes
    # for training please only use newton_mjwarp.
    isaacsim_physx: Any = config_field(
        PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_max_rigid_patch_count=16 * 5 * 2**15,
            gpu_found_lost_pairs_capacity=2**27,
            gpu_total_aggregate_pairs_capacity=2**27,
        )
    )
    newton_mjwarp: Any = config_field(
        NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=600,
                nconmax=600,
                impratio=1.0,
                cone="pyramidal",
                update_data_interval=2,
                iterations=100,
                ls_iterations=15,
                use_mujoco_contacts=False,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(),
            default_shape_cfg=NewtonShapeCfg(),
            num_substeps=2,
            debug_mode=False,
        )
    )
    physx: Any = config_field(PhysxAutoCfg(isaacsim_physx=isaacsim_physx))
    default: Any = config_field(newton_mjwarp)


@dataclass
class SO101KeyboardEnvCfg(ManagerBasedRLEnvCfg):
    scene: SO101SceneCfg = config_field(SO101SceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True))
    observations: SO101ObservationsCfg = config_field(SO101ObservationsCfg())
    actions: SO101RelJointPosActionCfg = config_field(SO101RelJointPosActionCfg())
    commands: CommandsCfg = config_field(CommandsCfg())
    rewards: SO101ReorientRewardCfg = config_field(SO101ReorientRewardCfg())
    terminations: TerminationsCfg = config_field(TerminationsCfg())
    events: EventCfg = config_field(EventCfg())
    sim: SimulationCfg = config_field(SimulationCfg(physics=PhysicsCfg(), dt=0.01))

    def __post_init__(self):
        self.decimation = 4  # 100 Hz sim -> 25 Hz control
        self.episode_length_s = 6.0
        self.sim.render_interval = self.decimation
        self.sim.default_visualizer_cfg = VisualizerCfg(
            eye=(0.85, -0.75, 1.0), lookat=(0.25, 0.0, 0.1), focal_length=28.0
        )

    def play_mode(self):
        """Enable typing markers for every playback environment."""
        super().play_mode()
        self.num_envs = 36
        self.commands.typing.debug_vis = True
