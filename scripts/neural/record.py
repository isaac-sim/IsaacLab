# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a task's declared exploration environment into a native NeRD dataset.

The task's model definition names the contact input and the labels that the data must hold; ``model.<field>``
overrides configure it. The dataset describes the recorded robot, and ``train.py`` builds the model on it.
"""

# ruff: noqa: E402
# Configure Warp before importing simulation modules.

import argparse
from functools import partial
from pathlib import Path
from types import MethodType

import warp as wp

wp.config.enable_backward = False

import torch
from isaaclab_neural.record.recording import NeRDDynamicsRecorder

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.array import index_fill_
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli


def step_recording(env, action: torch.Tensor, *, capture: NeRDDynamicsRecorder):
    """Capture states and consumed controls before resetting completed exploration segments."""
    env.extras.pop("final_obs", None)
    env.extras.pop("final_obs_every_step", None)
    env.action_manager.process_action(action.to(env.device))
    capture.record_pre_step()
    if env._physics_handles_actions is None:
        env._handle_physics_actions()

    # check if we need to do rendering within the physics loop
    # note: evaluated once per step rather than once per physics substep
    is_rendering = env.sim.is_rendering

    # physics-owned decimation covers all substeps in one call
    steps_per_call = env.cfg.decimation if env._physics_handles_decimation else 1
    for _ in range(env.cfg.decimation // steps_per_call):
        env._sim_step_counter += steps_per_call
        if not env._physics_handles_actions:
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
        env.sim.step(render=False)
        # render_enabled=False skips Kit (camera/GUI); standalone visualizers still update
        if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
            env.sim.render(skip_app_pumping=not env.render_enabled)
        env.scene.update(dt=env.physics_dt * steps_per_call)

    env.episode_length_buf += 1  # step in current episode (per env)
    env.common_step_counter += 1  # total step (common for all envs)
    # -- check terminations
    env.reset_buf = env.termination_manager.compute()
    env.reset_terminated = env.termination_manager.terminated
    env.reset_time_outs = env.termination_manager.time_outs
    env.reward_buf = env.reward_manager.compute(dt=env.step_dt)

    # -- reset envs that terminated/timed-out and log the episode information
    capture.record_post_step()
    reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
    if len(reset_env_ids) > 0:
        # Capture the terminal observation before the Same-Step autoreset.
        if env.cfg.compute_final_obs:
            env.extras["final_obs"] = env.observation_manager.compute()

        env._reset_idx(reset_env_ids)
        capture.reset(reset_env_ids)

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if env.render_enabled and is_rendering and env.has_rtx_sensors and env.cfg.num_rerenders_on_reset > 0:
            for _ in range(env.cfg.num_rerenders_on_reset):
                env.sim.render()

    # -- handle episode reset requested from visualizer UI controls
    manual_reset_ids = reset_env_ids[:0]
    if env.sim.consume_reset_request():
        # Only reset envs not already reset this step to avoid redundant resets.
        not_yet_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        index_fill_(not_yet_reset, reset_env_ids, False)
        manual_reset_ids = not_yet_reset.nonzero(as_tuple=False).squeeze(-1).int()
        if len(manual_reset_ids) > 0:
            # mark as terminated so RL wrappers observe the episode boundary
            index_fill_(env.reset_terminated, manual_reset_ids, True)
            env._reset_idx(manual_reset_ids)
            capture.reset(manual_reset_ids)

    if len(reset_env_ids) > 0 or len(manual_reset_ids) > 0:
        # Reset writes invalidate derived body poses used by observations and rewards.
        env.sim.forward()

    env.command_manager.compute(dt=env.step_dt)
    if "interval" in env.event_manager.active_terms:
        env.event_manager.apply(mode="interval", dt=env.step_dt)
    # -- advance video recorders (after render and resets, before final obs)
    for recorder in env.video_recorders:
        recorder.step()
    # -- compute observations
    # note: done after reset to get the correct observations for reset envs
    env.obs_buf = env.observation_manager.compute(update_history=True)

    return env.obs_buf, env.reward_buf, env.reset_terminated, env.reset_time_outs, env.extras


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--num_frames", type=int, default=2048)
    parser.add_argument("--shard_frames", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, argv)
    # The task resolver composes the model definition in its agent slot.
    overrides = [
        f"agent.{token.removeprefix('model.')}" if token.startswith("model.") else token for token in overrides
    ]
    env_cfg, model_cfg = resolve_task_config(
        args.task,
        "nerd_cfg_entry_point",
        overrides=overrides,
        env_cfg_entry_point="nerd_recording_cfg_entry_point",
    )
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    with launch_simulation(env_cfg, args):
        env = ManagerBasedRLEnv(env_cfg)
        capture = NeRDDynamicsRecorder(
            env,
            args.dataset,
            contact_bodies=model_cfg.contact_bodies,
            prediction=model_cfg.prediction,
            shard_frames=args.shard_frames,
        )
        env.step = MethodType(partial(step_recording, capture=capture), env)
        try:
            action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            env.reset()
            capture.reset(None)
            for _ in range(args.num_frames):
                env.step(action)
            capture.close()
            dump_yaml(str(args.dataset / "params" / "env.yaml"), env_cfg)
        except BaseException:
            capture.close(complete=False)
            raise
        finally:
            env.close()


if __name__ == "__main__":
    main()
