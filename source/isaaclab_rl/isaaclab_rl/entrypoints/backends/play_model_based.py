# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a model-based actor checkpoint in the saved physical task."""

import argparse
import time
from pathlib import Path

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import update_from_dict
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.model_based.rollout import PhysicalRollout

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from ...model_based.runner import REFERENCE_REVISION, build_actor
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


def run(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_play_args(
        parser, agent_default="model_based_cfg_entry_point", agent_help="Saved model-based configuration."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion_dir")
    parser.add_argument("--asset_path")
    parser.add_argument("--steps", type=int, default=1000)
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, argv)
    if any(value != "physics=newton_mjwarp" for value in overrides) or args.train_env_cfg or args.frontend != "torch":
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
    cfg, agent = resolve_task_config(task, checkpoint["agent"], overrides=[])
    _restore_environment(cfg, checkpoint["env_cfg"])
    update_from_dict(agent, checkpoint["agent_cfg"])
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
    output = str(Path(checkpoint_path).parent)
    enable_cameras_for_video(args)
    pre_launch_video_config(cfg, args)
    with launch_simulation(cfg, args):
        apply_video_recording(cfg, output, args, subdir="play", checkpoint_path=checkpoint_path)
        env = create_isaaclab_env(task, cfg, args, convert_marl_to_single_agent=False)
        try:
            adapter = PhysicalRollout(env, agent)
            actor = build_actor(adapter, agent).eval()
            actor.load_state_dict(checkpoint["actor"])
            adapter.reset()
            for _ in range(video_playback_steps(args, cfg) or args.steps):
                start = time.monotonic()
                with torch.inference_mode():
                    adapter.advance(actor(adapter.actor_input()))
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
    update_from_dict(cfg, saved)
