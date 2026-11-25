# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a model-based actor checkpoint in the saved physical task."""

import argparse
import copy
from pathlib import Path

import torch
from rsl_rl.models import SuperMimicActor

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import update_from_dict
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_mul

from isaaclab_tasks.core.motion_tracking.mdp.commands import MotionCommand
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from ..common import (
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    create_isaaclab_env,
    enable_cameras_for_video,
    pre_launch_video_config,
)
from .runner_rsl_rl import wrap_rsl_rl_env

REFERENCE_REVISION = "5efbf114e11f725c31d83e217be987c13c0e4ce2"


def run(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_play_args(parser, agent_default="rsl_rl_cfg_entry_point", agent_help="Saved model-based configuration.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion_dir")
    parser.add_argument("--asset_path")
    parser.add_argument("--steps", type=int, default=1000)
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, argv)
    for value in overrides:
        if value.startswith("visualizer="):
            args.visualizer = value.partition("=")[2]
    if (
        any(value != "physics=newton_mjwarp" and not value.startswith("visualizer=") for value in overrides)
        or args.train_env_cfg
        or args.frontend != "torch"
    ):
        parser.error("Playback restores the saved configuration; use explicit relocation/device/seed options.")
    cfg, agent, checkpoint, checkpoint_path = load_legacy_task(args)
    apply_env_overrides(args, cfg)
    output = str(Path(checkpoint_path).parent)
    enable_cameras_for_video(args)
    pre_launch_video_config(cfg, args)
    with launch_simulation(cfg, args):
        apply_video_recording(cfg, output, args, subdir="play", checkpoint_path=checkpoint_path)
        env = create_isaaclab_env("Isaac-Motion-Tracking-G1", cfg, args, convert_marl_to_single_agent=False)
        try:
            env = wrap_rsl_rl_env(env, agent)
            obs, _ = env.reset()
            actor = load_legacy_actor(checkpoint, obs, agent, env.unwrapped.device)
            from .play_rsl_rl import play_policy

            play_policy(env, actor, args, cfg, max_steps=args.steps)
        finally:
            env.close()


def load_legacy_task(args):
    """Restore the published checkpoint's task and actor settings for playback or export."""
    if not args.checkpoint or args.checkpoint == "pretrained":
        raise ValueError("Pass a checkpoint file path or URL.")
    checkpoint_path = retrieve_file_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["reference_revision"] != REFERENCE_REVISION:
        raise ValueError("The checkpoint uses a different reference algorithm revision.")
    task = checkpoint["task"]
    if args.task is not None and args.task != task:
        raise ValueError("The task must match the saved checkpoint.")
    args.task = task
    cfg, _ = resolve_task_config("Isaac-Motion-Tracking-G1", "rsl_rl_cfg_entry_point", overrides=[])
    agent = legacy_agent_cfg(checkpoint)
    _restore_environment(cfg, checkpoint["env_cfg"])

    cfg.sim.device = args.device
    cfg.episode_length_s = 1.0e9
    if args.motion_dir:
        cfg.commands.motion.motion_file = args.motion_dir
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    motion_path = cfg.commands.motion.motion_file
    if not Path(motion_path).is_dir():
        cfg.commands.motion.motion_file = retrieve_file_path(motion_path)
    configure_legacy_observations(cfg, agent)
    return cfg, agent, checkpoint, checkpoint_path


def _restore_environment(cfg, saved: dict) -> None:
    """Restore physical settings, including checkpoints from the pre-core task layout."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.envs import mdp
    from isaaclab.sim import GroundPlaneCfg

    original_action_scale = saved["actions"]["joint_pos"]["scale"]
    saved = copy.deepcopy(saved)
    command = saved["commands"]["motion"]
    # Playback keeps the published controller and reference layout on the shared physical task.
    cfg.commands.motion.class_type = _PublishedMotionCommand
    cfg.commands.motion.joint_names = command["joint_names"]
    cfg.commands.motion.motion_file = command["motion_dir"]
    cfg.commands.motion.source_fps = command["source_fps"]
    update_from_dict(cfg.commands.motion.csv, command["csv"])
    cfg.commands.motion.anchor_body_name = "pelvis"
    cfg.commands.motion.body_names = command["tracked_bodies"] or cfg.commands.motion.body_names
    cfg.commands.motion.sampling_mode = "start"
    cfg.commands.motion.pose_range = {}
    cfg.commands.motion.velocity_range = {}
    cfg.commands.motion.joint_position_range = (0.0, 0.0)
    cfg.scene.terrain = None
    cfg.scene.self_collision = None
    cfg.scene.ground = AssetBaseCfg(prim_path="/World/ground", spawn=GroundPlaneCfg())
    for name in vars(cfg.events):
        if not name.startswith("_"):
            setattr(cfg.events, name, None)
    cfg.events.material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=command["friction_bodies"]),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    for name in vars(cfg.rewards):
        if not name.startswith("_"):
            setattr(cfg.rewards, name, None)
    cfg.terminations.ee_body_pos = None
    cfg.terminations.anchor_pos.params["threshold"] = saved["terminations"]["anchor_pos"]["params"]["threshold"]
    cfg.terminations.anchor_ori.params["threshold"] = saved["terminations"]["anchor_ori"]["params"]["threshold"]
    for name in ("observations", "commands", "events", "rewards", "terminations", "actions"):
        saved.pop(name, None)
    actuators = saved["scene"]["robot"]["actuators"]
    for values in actuators.values():
        for old, new in (
            ("effort_limit_sim", "joint_effort_limit"),
            ("velocity_limit_sim", "joint_velocity_limit"),
            ("effort_limit", "actuator_effort_limit"),
            ("velocity_limit", "actuator_velocity_limit"),
        ):
            value = values.pop(old, None)
            if value is not None:
                values[new] = value
    if set(actuators) == {"joints"} and "joints" not in cfg.scene.robot.actuators:
        cfg.scene.robot.actuators = {"joints": ImplicitActuatorCfg(joint_names_expr=[".*"])}
        cfg.scene.robot.actuators["joints"].joint_effort_limit = dict(actuators["joints"]["joint_effort_limit"])
        cfg.scene.robot.actuators["joints"].joint_velocity_limit = dict(actuators["joints"]["joint_velocity_limit"])
        for key in ("stiffness", "damping", "armature"):
            setattr(cfg.scene.robot.actuators["joints"], key, dict(actuators["joints"][key]))
    cfg.actions.joint_pos.scale = original_action_scale
    if saved.get("recorders", {}).get("transition", False) is None:
        saved["recorders"].pop("transition")
    spawn = saved["scene"]["robot"]["spawn"]
    if "/PretrainedCheckpoints/model_based/assets/g1_29dof.usdc" in spawn["usd_path"]:
        spawn["usd_path"] = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_29dof.usd"
        spawn["variants"] = {"Physics": "SimplifiedPhysX"}
    spawn["variants"].update(left_hand="None", right_hand="None")
    cfg.scene.robot.spawn.variants = spawn["variants"]
    update_from_dict(cfg, saved)


NUM_GOAL_FRAMES = 10
GOAL_FRAME_STRIDE = 5


def goal_window_indices(motion, frames):
    """Published actor's future window, clamped to the current clip."""
    offsets = torch.arange(NUM_GOAL_FRAMES, device=frames.device) * GOAL_FRAME_STRIDE
    return torch.minimum(frames[:, None] + offsets, (motion.clip_end(frames) - 1)[:, None])


def _actor_goal(env):
    command = env.command_manager.get_term("motion")
    motion = command.motion
    idx = goal_window_indices(motion, command.time_steps)
    root = command.robot.data.root_link_quat_w.torch[:, None].expand(-1, NUM_GOAL_FRAMES, -1)
    relative = quat_mul(quat_inv(root), motion.root_quat_w[idx])
    return torch.cat(
        (
            torch.cat((motion.joint_pos[idx], motion.joint_vel[idx]), -1).flatten(1),
            matrix_from_quat(relative)[..., :2].flatten(1),
        ),
        -1,
    )


def configure_legacy_observations(cfg, agent):
    """Keep published sensor/history and future-goal layouts out of the training task."""
    for name in vars(cfg.observations):
        if not name.startswith("_"):
            setattr(cfg.observations, name, None)
    cfg.observations.policy = ObservationGroupCfg(
        enable_corruption=False,
        concatenate_terms=True,
        history_length=agent.algorithm.history_length,
        flatten_history_dim=False,
    )
    from isaaclab.envs import mdp

    cfg.observations.policy.gravity = ObservationTermCfg(func=mdp.projected_gravity)
    cfg.observations.policy.angular_velocity = ObservationTermCfg(func=mdp.base_ang_vel)
    cfg.observations.policy.joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
    cfg.observations.policy.joint_vel = ObservationTermCfg(func=mdp.joint_vel)
    cfg.observations.policy.actions = ObservationTermCfg(func=mdp.last_action)
    cfg.observations.actor_goal = ObservationGroupCfg(enable_corruption=False, concatenate_terms=True)
    cfg.observations.actor_goal.goal = ObservationTermCfg(func=_actor_goal)


class _PublishedMotionCommand(MotionCommand):
    """Restore the published controller's deterministic reference resets."""

    def __init__(self, cfg, env):
        if cfg.joint_names is not None and env.scene[cfg.asset_name].joint_names != cfg.joint_names:
            raise ValueError("The published reference joint order must match the articulation.")
        super().__init__(cfg, env)

    def _update_command(self):
        last = self.motion.clip_end(self.time_steps) - 1
        self.time_steps.copy_(torch.minimum(self.time_steps + (self._env.episode_length_buf > 0).long(), last))

    def _resample_command(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        frames = self.motion.motion_starts[ids % len(self.motion.names)]
        motion = self.motion
        joints = motion.joint_pos[frames].clone()
        limits = self.robot.data.soft_joint_pos_limits.torch[ids]
        joints = joints.clamp(limits[..., 0], limits[..., 1])
        self.robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat(
                (motion.root_pos_w[frames] + self._env.scene.env_origins[ids], motion.root_quat_w[frames]), -1
            ),
            env_ids=ids,
        )
        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=torch.cat((motion.root_lin_vel_w[frames], motion.root_ang_vel_w[frames]), -1), env_ids=ids
        )
        self.robot.write_joint_state_to_sim_index(position=joints, velocity=motion.joint_vel[frames], env_ids=ids)
        self.robot.actuators.target_command.set_position_index(value=joints, env_ids=ids)
        self.time_steps[ids] = frames


def legacy_agent_cfg(checkpoint):
    """Map the published standalone actor settings to the RSL-RL policy interface."""
    from isaaclab_rl.rsl_rl import RslRlSuperMimicActorCfg, RslRlSuperMimicAlgorithmCfg, RslRlSuperMimicRunnerCfg

    saved = checkpoint["agent_cfg"]
    if [name for name, weight in saved["slot_mix"].items() if weight > 0] != ["motion"]:
        raise ValueError("Published checkpoint playback supports the motion encoder only.")
    if saved["actor_obs"] != "proprio":
        raise ValueError("Published playback requires a proprioceptive actor.")
    cfg = RslRlSuperMimicRunnerCfg()
    cfg.algorithm = RslRlSuperMimicAlgorithmCfg()
    cfg.obs_groups = {"actor": ["policy", "history", "actor_goal"]}
    cfg.actor = RslRlSuperMimicActorCfg(activation="gelu", norm="layer", output_gain=1.0)
    cfg.algorithm.history_length = saved["history_length"]
    cfg.actor.latent_dim = saved["latent_dim"]
    cfg.actor.encoder_hidden_dims = tuple(saved["actor_hidden_dims"])
    cfg.actor.decoder_hidden_dims = tuple(saved["actor_hidden_dims"])
    if saved["history_length"] == 1:
        cfg.obs_groups["actor"].remove("history")
    return cfg


def load_legacy_actor(checkpoint, observations, cfg, device):
    """Load motion weights without rewriting the published checkpoint."""
    params = cfg.actor.to_dict()
    params.pop("class_name")
    actor = SuperMimicActor(observations, cfg.obs_groups, "actor", 29, **params).to(device)
    prefixes = {
        "encoders.motion.": "policy.encoder.",
        "decoder.": "policy.decoder.",
        "obs_normalizer.": "policy.obs_norm.",
        "goal_normalizers.motion.": "policy.goal_norm.",
    }
    mapped = {}
    for name, value in checkpoint["actor"].items():
        for old, new in prefixes.items():
            if name.startswith(old):
                mapped[new + name[len(old) :]] = value
                break
        else:
            if not name.startswith(("encoders.velocity.", "goal_normalizers.velocity.")):
                raise ValueError(f"Unexpected published actor parameter: {name}")
    actor.load_state_dict(mapped, strict=True)
    return actor.eval()
