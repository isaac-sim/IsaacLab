# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small scenes combining the supported manager export operations in one rollout."""

import torch
from isaaclab_newton.physics import FeatherstoneSolverCfg, NewtonCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import ActionTermCfg, EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg, ImuCfg, PvaCfg, RayCasterCfg, SensorBaseCfg, patterns
from isaaclab.utils import configclass, replace
from isaaclab.utils.modifiers import DigitalFilterCfg, IntegratorCfg, ModifierCfg, scale
from isaaclab.utils.noise import GaussianNoiseCfg

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg

from isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg import CartpolePPORunnerCfg
from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg, CartpoleSceneCfg
from isaaclab_tasks.core.lift.mdp.observations import body_state_b, object_point_cloud_b, object_quat_b
from isaaclab_tasks.utils import load_cfg_from_registry

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


def contact_history(env) -> torch.Tensor:
    """Read normal-force history [N] through the sensor mapping."""
    return env.scene.sensors["contact"].data.net_normal_forces_w_history.torch.flatten(start_dim=1)


def data_observation(env, entity: str, properties: tuple[str, ...]) -> torch.Tensor:
    """Read sensor/asset properties through their annotated public data interface."""
    data = env.scene[entity].data
    return torch.cat([getattr(data, name).torch.flatten(start_dim=1) for name in properties], dim=-1)


def motion_vectors(env) -> torch.Tensor:
    """Read floating-point optical flow through the generic camera output mapping."""
    return env.scene["camera"].data.output["motion_vectors"].torch.flatten(start_dim=1)


def image_encoder() -> torch.nn.Module:
    """Small frozen feature encoder with no downloads or optional vision dependencies."""
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 4), torch.nn.Tanh()).requires_grad_(False)


def encode_image(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run the feature encoder on normalized pixels."""
    return model.to(images.device)(images.float() / 255.0)


def set_gripper_limits(env, env_ids):
    """Give the prismatic joint finite asymmetric limits [m] for action rescaling."""
    asset = env.scene["gripper"]
    asset.write_joint_position_limit_to_sim_index(
        limits=torch.tensor([[[-0.3, 0.7]]], device=env.device), joint_ids=asset.find_joints("slider_to_cart")[0]
    )


@configclass
class SyntheticSceneCfg(CartpoleSceneCfg):
    """Reuse three small articulations for action and sensor coverage."""

    relative = replace(CARTPOLE_CFG, prim_path="{ENV_REGEX_NS}/Relative")
    gripper = replace(CARTPOLE_CFG, prim_path="{ENV_REGEX_NS}/Gripper")
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=0.1),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 1.0, 0.06)),
    )
    collection = RigidObjectCollectionCfg(
        rigid_objects={
            "first": replace(object, prim_path="{ENV_REGEX_NS}/CollectionA"),
            "second": replace(
                object,
                prim_path="{ENV_REGEX_NS}/CollectionB",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 2.0, 0.06)),
            ),
        }
    )
    camera = SensorBaseCfg(class_type="leapp_synthetic_env:InjectedCamera", prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pole",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.5, size=(0.5, 0.5)),
        mesh_prim_paths=["/World/ground"],
    )
    contact = SensorBaseCfg(class_type="leapp_synthetic_env:InjectedContact", prim_path="{ENV_REGEX_NS}/Object")
    deformable = SensorBaseCfg(class_type="leapp_synthetic_env:InjectedDeformable", prim_path="{ENV_REGEX_NS}/Object")
    wrench = SensorBaseCfg(class_type="leapp_synthetic_env:InjectedWrench", prim_path="{ENV_REGEX_NS}/Robot")
    cable = SensorBaseCfg(class_type="leapp_synthetic_env:InjectedCable", prim_path="{ENV_REGEX_NS}/Robot")
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/pole", offset=ImuCfg.OffsetCfg(pos=(0.1, 0.0, 0.0)))
    pva = PvaCfg(prim_path="{ENV_REGEX_NS}/Robot/pole")
    frames = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/cart",
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/pole", name="tip")],
    )


@configclass
class SyntheticActionsCfg:
    """Capture target writes, subsets, state-dependent targets, and EMA feedback."""

    position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["cart_to_pole", "slider_to_cart"],
        preserve_order=True,
        scale={"cart_to_pole": 0.2, "slider_to_cart": 0.1},
        clip={".*": (-0.3, 0.5)},
    )
    velocity = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["cart_to_pole"], scale=0.2)
    relative = mdp.RelativeJointPositionActionCfg(asset_name="relative", joint_names=["slider_to_cart"], scale=0.05)
    effort = mdp.JointEffortActionCfg(asset_name="relative", joint_names=["cart_to_pole"], scale=0.1)
    binary = mdp.BinaryJointPositionActionCfg(
        asset_name="gripper",
        joint_names=["cart_to_pole"],
        open_command_expr={".*": 0.1},
        close_command_expr={".*": -0.1},
    )
    ema = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="gripper", joint_names=["slider_to_cart"], alpha=0.3, rescale_to_limits=True, scale=0.1
    )
    command = ActionTermCfg(class_type="leapp_synthetic_env:CommandAction", asset_name="robot")


@configclass
class SyntheticCommandsCfg:
    """Export both pose and velocity command metadata."""

    velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.02, 0.02),
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.2, 0.2)
        ),
    )
    pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="pole",
        resampling_time_range=(0.02, 0.02),
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.1, 0.2), pos_y=(0.1, 0.2), pos_z=(0.1, 0.2), roll=(-0.1, 0.1), pitch=(-0.1, 0.1), yaw=(-0.1, 0.1)
        ),
    )


@configclass
class SyntheticPolicyCfg(ObservationGroupCfg):
    """Exercise tensor preprocessing, live camera inputs, and stateful observations."""

    joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, history_length=3, clip=(-2.0, 2.0), scale=0.5)
    joint_vel = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        modifiers=[ModifierCfg(func=scale, params={"multiplier": 0.7})],
        noise=GaussianNoiseCfg(mean=0.0, std=0.01),
    )
    relative_pos = ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("relative")})
    gravity = ObservationTermCfg(func=mdp.projected_gravity)
    linear_velocity = ObservationTermCfg(func=mdp.base_lin_vel)
    angular_velocity = ObservationTermCfg(func=mdp.base_ang_vel)
    object_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
    object_quat = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
    object_orientation = ObservationTermCfg(func=object_quat_b)
    body_state = ObservationTermCfg(
        func=body_state_b,
        params={
            "body_asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "base_asset_cfg": SceneEntityCfg("robot"),
        },
    )
    point_cloud = ObservationTermCfg(
        func=object_point_cloud_b, params={"num_points": 4, "flatten": True, "visualize": False}
    )
    height_scan = ObservationTermCfg(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")})
    contact = ObservationTermCfg(func=contact_history)
    contact_details = ObservationTermCfg(
        func=data_observation,
        params={
            "entity": "contact",
            "properties": ("force_matrix_w", "friction_force_matrix_w", "contact_pos_w", "current_air_time"),
        },
    )
    nodes = ObservationTermCfg(
        func=data_observation,
        params={
            "entity": "deformable",
            "properties": ("nodal_pos_w", "nodal_vel_w", "nodal_state_w", "root_pos_w", "root_vel_w"),
        },
    )
    imu = ObservationTermCfg(func=data_observation, params={"entity": "imu", "properties": ("ang_vel_b", "lin_acc_b")})
    pva = ObservationTermCfg(
        func=data_observation,
        params={
            "entity": "pva",
            "properties": ("pose_w", "projected_gravity_b", "lin_vel_b", "ang_vel_b", "lin_acc_b", "ang_acc_b"),
        },
    )
    frames = ObservationTermCfg(
        func=data_observation,
        params={"entity": "frames", "properties": ("target_pose_source", "target_pose_w", "source_pose_w")},
    )
    camera_geometry = ObservationTermCfg(
        func=data_observation,
        params={
            "entity": "camera",
            "properties": ("pos_w", "quat_w_world", "intrinsic_matrices", "quat_w_ros", "quat_w_opengl"),
        },
    )
    wrench = ObservationTermCfg(func=data_observation, params={"entity": "wrench", "properties": ("force", "torque")})
    cable = ObservationTermCfg(
        func=data_observation, params={"entity": "cable", "properties": ("segment_pose_w", "segment_velocity_w")}
    )
    dynamics = ObservationTermCfg(
        func=data_observation,
        params={
            "entity": "robot",
            "properties": (
                "body_joint_wrench",
                "body_link_jacobian_w",
                "body_com_jacobian_w",
                "mass_matrix",
                "gravity_compensation_forces",
            ),
        },
    )
    collection = ObservationTermCfg(
        func=data_observation, params={"entity": "collection", "properties": ("object_state_w",)}
    )
    delayed = ObservationTermCfg(func=mdp.joint_pos, delay_min_lag=2, delay_max_lag=2)
    filtered = ObservationTermCfg(func=mdp.joint_vel, modifiers=[DigitalFilterCfg(A=[-0.3], B=[0.4, 0.3])])
    integrated = ObservationTermCfg(func=mdp.joint_vel, modifiers=[IntegratorCfg(dt=0.02)])
    velocity_command = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "velocity"})
    pose_command = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "pose"})
    previous = ObservationTermCfg(func=mdp.last_action)
    previous_position = ObservationTermCfg(func=mdp.last_action, params={"action_name": "position"})
    previous_relative = ObservationTermCfg(func=mdp.last_action, params={"action_name": "relative"})
    rgb = ObservationTermCfg(
        func=mdp.image_rgb,
        params={"sensor_cfg": SceneEntityCfg("camera"), "frame_stack": 2, "channel_first": True},
        history_length=1,
    )
    depth = ObservationTermCfg(
        func=mdp.image_depth,
        params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "depth", "max_depth": 2.0},
        history_length=1,
    )
    perspective_depth = ObservationTermCfg(
        func=mdp.image_depth,
        params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "data_type": "distance_to_camera",
            "convert_perspective_to_orthogonal": True,
            "tanh_scale": 2.0,
        },
        history_length=1,
    )
    normals = ObservationTermCfg(
        func=mdp.image_normals, params={"sensor_cfg": SceneEntityCfg("camera")}, history_length=1
    )
    labels = ObservationTermCfg(
        func=mdp.image_segmentation, params={"sensor_cfg": SceneEntityCfg("camera")}, history_length=1
    )
    flow = ObservationTermCfg(func=motion_vectors)
    features = ObservationTermCfg(
        func=mdp.image_features,
        params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "model_name": "tiny",
            "model_zoo_cfg": {"tiny": {"model": image_encoder, "inference": encode_image}},
        },
    )


@configclass
class SyntheticObservationsCfg:
    """Separate actor groups from a critic-only input that must not reach the graph."""

    @configclass
    class ProprioCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("gripper")})

    @configclass
    class CriticCfg(ObservationGroupCfg):
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("relative")})

    @configclass
    class VisionCfg(ObservationGroupCfg):
        rgb = ObservationTermCfg(
            func=mdp.image_rgb, params={"sensor_cfg": SceneEntityCfg("camera"), "channel_first": True}
        )

    policy = SyntheticPolicyCfg(enable_corruption=True)
    proprio = ProprioCfg()
    critic = CriticCfg()
    vision = VisionCfg()


def manager_rsl_rl_cfg() -> CartpolePPORunnerCfg:
    """Exercise multi-group actor selection and observation normalization."""
    cfg = CartpolePPORunnerCfg()
    cfg.obs_groups = {"actor": ["policy", "proprio", "vision"], "critic": ["critic"]}
    cfg.actor = RslRlCNNModelCfg(
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=cfg.actor.distribution_cfg,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(output_channels=[4], kernel_size=3, activation="elu"),
    )
    return cfg


def manager_rl_games_cfg() -> dict:
    """Reuse the RL-Games case to cover LSTM state without another scene build."""
    cfg = load_cfg_from_registry("Isaac-Cartpole", "rl_games_cfg_entry_point")
    cfg["params"]["network"]["rnn"] = {"name": "lstm", "units": 16, "layers": 1, "before_mlp": True, "layer_norm": True}
    return cfg


def manager_sb3_cfg() -> dict:
    """Exercise dictionary observations and saved per-key normalization in the SB3 case."""
    cfg = load_cfg_from_registry("Isaac-Cartpole", "sb3_cfg_entry_point")
    cfg.update(policy="MultiInputPolicy", normalize_input=True)
    return cfg


def manager_env_cfg() -> CartpoleEnvCfg:
    """Build a single small scene with real observation and action managers."""
    cfg = CartpoleEnvCfg()
    cfg.scene = SyntheticSceneCfg(num_envs=1, env_spacing=8.0)
    cfg.scene.relative.init_state.pos = (2.0, 0.0, 2.0)
    cfg.scene.gripper.init_state.pos = (-2.0, 0.0, 2.0)
    cfg.scene.robot.enable_joint_wrench = True
    cfg.scene.robot.actuators = {
        "position": ImplicitActuatorCfg(joint_names_expr=["slider_to_cart"], stiffness=20.0, damping=2.0),
        "velocity": IdealPDActuatorCfg(joint_names_expr=["cart_to_pole"], stiffness=0.0, damping=3.0),
    }
    cfg.actions = SyntheticActionsCfg()
    cfg.commands = SyntheticCommandsCfg()
    cfg.observations = SyntheticObservationsCfg()
    cfg.events.gripper_limits = EventTermCfg(func=set_gripper_limits, mode="startup")
    cfg.sim.physics = NewtonCfg(solver_cfg=FeatherstoneSolverCfg(), use_cuda_graph=False)
    return cfg
