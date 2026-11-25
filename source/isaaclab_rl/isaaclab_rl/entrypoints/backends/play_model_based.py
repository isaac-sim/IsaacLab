# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a model-based actor checkpoint in the saved physical task."""

import argparse
import time
from pathlib import Path

import torch
from rsl_rl.models import SuperMimicActor

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import update_from_dict
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from ..common import (
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    create_isaaclab_env,
    enable_cameras_for_video,
    pre_launch_video_config,
    resolve_published_checkpoint,
    video_playback_steps,
)
from .runner_rsl_rl import configure_rsl_rl_env_cfg, wrap_rsl_rl_env

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
    if args.checkpoint == "pretrained":
        if args.task is None:
            parser.error("--task is required with --checkpoint pretrained.")
        cfg, _ = resolve_task_config(args.task, args.agent, overrides=[])
        checkpoint_path = resolve_published_checkpoint("model_based", args.task, cfg)
        if checkpoint_path is None:
            return
    else:
        checkpoint_path = retrieve_file_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["reference_revision"] != REFERENCE_REVISION:
        parser.error("The checkpoint uses a different reference algorithm revision.")
    task = checkpoint["task"]
    if args.task is not None and args.task != task:
        parser.error("The task must match the saved checkpoint.")
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    agent = legacy_agent_cfg(checkpoint)
    _restore_environment(cfg, checkpoint["env_cfg"])

    apply_env_overrides(args, cfg)
    cfg.commands.motion.evaluation = True
    cfg.commands.motion.reset_noise = False
    cfg.episode_length_s = 1.0e9
    if args.motion_dir:
        cfg.commands.motion.motion_dir = args.motion_dir
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    motion_path = cfg.commands.motion.motion_dir
    if not Path(motion_path).is_dir():
        cfg.commands.motion.motion_dir = retrieve_file_path(motion_path)
    configure_rsl_rl_env_cfg(cfg, agent)
    output = str(Path(checkpoint_path).parent)
    enable_cameras_for_video(args)
    pre_launch_video_config(cfg, args)
    with launch_simulation(cfg, args):
        apply_video_recording(cfg, output, args, subdir="play", checkpoint_path=checkpoint_path)
        env = create_isaaclab_env(task, cfg, args, convert_marl_to_single_agent=False)
        try:
            env = wrap_rsl_rl_env(env, agent)
            obs, _ = env.reset()
            actor = load_legacy_actor(checkpoint, obs, agent, env.unwrapped.device)
            for _ in range(video_playback_steps(args, cfg) or args.steps):
                start = time.monotonic()
                with torch.inference_mode():
                    obs, _, _, _ = env.step(actor(obs))
                if args.real_time:
                    time.sleep(max(0.0, env.unwrapped.step_dt - (time.monotonic() - start)))
        finally:
            env.close()


def _restore_environment(cfg, saved: dict) -> None:
    """Restore physical settings, including checkpoints from the pre-core task layout."""
    from isaaclab.actuators import ImplicitActuatorCfg

    old_prefix = "isaaclab_tasks.contrib.motion_tracking."
    new_prefix = "isaaclab_tasks.core.motion_tracking.mdp.model_based."

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and value.startswith(old_prefix):
            value = new_prefix + value[len(old_prefix) :]
        if isinstance(value, str):
            value = value.replace("mdp.model_based.sim_context:", "mdp.model_based.commands:")
        return value

    saved = relocate(saved)
    # Observation layout belongs to the active learner; restore the physical task only.
    saved.pop("observations", None)
    # The original controller used one actuator group. Restore its serialized gains
    # rather than merging it into the shared G1 configuration's five groups.
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
    cfg.actions.joint_pos.scale = saved["actions"]["joint_pos"]["scale"]
    command = saved["commands"]["motion"]
    command.pop("kinematics_path", None)  # Older checkpoints stored a separate MJCF input.
    for key in ("joint_names", "tracked_bodies"):
        if command[key] is None:
            command[key] = getattr(cfg.commands.motion, key)
    if saved.get("recorders", {}).get("transition", False) is None:
        saved["recorders"].pop("transition")
    spawn = saved["scene"]["robot"]["spawn"]
    if "/PretrainedCheckpoints/model_based/assets/g1_29dof.usdc" in spawn["usd_path"]:
        spawn["usd_path"] = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_29dof.usd"
        spawn["variants"] = {"Physics": "SimplifiedPhysX", "left_hand": "RubberHand", "right_hand": "RubberHand"}
    cfg.scene.robot.spawn.variants = spawn["variants"]
    update_from_dict(cfg, saved)


def legacy_agent_cfg(checkpoint):
    """Map the published standalone actor settings to the RSL-RL policy interface."""
    from isaaclab_tasks.core.motion_tracking.config.g1.agents.rsl_rl_supermimic_cfg import PaperTrackingModelBasedCfg

    saved = checkpoint["agent_cfg"]
    if [name for name, weight in saved["slot_mix"].items() if weight > 0] != ["motion"]:
        raise ValueError("Published checkpoint playback supports the motion encoder only.")
    cfg = PaperTrackingModelBasedCfg()
    cfg.algorithm.task_cfg.actor_obs = saved["actor_obs"]
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
