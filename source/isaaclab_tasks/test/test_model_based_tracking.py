# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical/imagined state conventions and reference timing at the native task boundary."""

import os
import shutil
from pathlib import Path

import gymnasium as gym
import pytest
import torch

from isaaclab.app import launch_simulation
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_rl.model_based.quaternion import matrix_from_quat
from isaaclab_rl.model_based.runner import build_actor

from isaaclab_tasks.core.motion_tracking.mdp.model_based.adapter import MotionTrackingAdapter
from isaaclab_tasks.core.motion_tracking.mdp.model_based.observations import body_state
from isaaclab_tasks.core.motion_tracking.mdp.model_based.sim_context import randomize_context
from isaaclab_tasks.utils import resolve_task_config


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_physical_reward_reference_timing_and_reset_history(tmp_path, device):
    """Physical body buffers independently validate FK/reward; real resets validate replay timing."""
    data = os.environ.get("ISAACLAB_G1_TRACKING_DATA")
    if data is None:
        pytest.skip("Set ISAACLAB_G1_TRACKING_DATA to the prepared G1 tracking directory.")
    source = Path(data)
    clip = next(iter(sorted((source / "lafan1_g1").rglob("*.csv"))))
    shutil.copyfile(clip, tmp_path / clip.name)
    cfg, agent = resolve_task_config(
        "IsaacContrib-Motion-Tracking-GenCoord-G1",
        "model_based_cfg_entry_point",
        overrides=["env.scene.num_envs=4", "env.seed=123"],
    )
    cfg.sim.device = device
    cfg.commands.motion.motion_dir = str(tmp_path)
    cfg.commands.motion.kinematics_path = str(source / "g1_description/g1_29dof.xml")
    cfg.scene.robot.spawn.usd_path = str(source / "g1_usd/g1_29dof/g1_29dof.usda")
    cfg.commands.motion.sampling_mode = agent.sampling_mode = "start"
    cfg.commands.motion.reset_noise = False
    cfg.events.context.mode = "reset"
    agent.actor_obs, agent.history_length = "proprio", 3
    agent.actor_hidden_dims = (16, 16)
    with launch_simulation(cfg):
        env = gym.make("IsaacContrib-Motion-Tracking-GenCoord-G1", cfg=cfg).unwrapped
        try:
            adapter = MotionTrackingAdapter(env, agent)
            actor = build_actor(adapter, agent)
            adapter.reset()
            command = adapter.command
            context = adapter.sim_context
            decoded_com = context.denormalize(context.read(env), context.find("base_com"))
            actual_com = command.robot.data.body_com_pos_b.torch[
                :, command.robot.body_names.index(command.cfg.context_body)
            ]
            torch.testing.assert_close(decoded_com, actual_com, atol=1e-6, rtol=1e-5)
            before_frame = command.time_steps.clone()
            before_state = adapter.obs["full_state"].clone()
            with torch.inference_mode():
                transition, _, _ = adapter.collect(actor, 0.0)
            torch.testing.assert_close(transition["state"], before_state, rtol=0, atol=0)
            torch.testing.assert_close(transition["motion_time"], before_frame)
            alive = ~(transition["terminated"] | transition["truncated"])
            assert alive.any()
            assert (command.time_steps[alive] == before_frame[alive] + 1).all()
            full = adapter.obs["full_state"]
            rep, aux = adapter.representation.represent(full), adapter.representation.extract_aux_state(full)
            predicted = adapter.representation.world_frame_bodies(aux, rep)
            actual = body_state(env)
            for expected, value in zip((actual[0], matrix_from_quat(actual[1]), *actual[2:]), predicted, strict=True):
                torch.testing.assert_close(value[alive], expected[alive], atol=2e-4, rtol=2e-4)
            imagined_reward = adapter.representation.compute_wm_reward_from_reference(
                rep,
                aux,
                adapter.representation.joint_state(rep),
                transition["action"],
                transition["prev_action"],
                *adapter.motion.body_state(before_frame),
            )
            torch.testing.assert_close(imagined_reward[alive], env.reward_buf[alive], atol=2e-5, rtol=2e-4)
            command.time_steps[0] = adapter.motion.motion_ends[0] - 1
            last_frame = command.time_steps[0].clone()
            with torch.inference_mode():
                transition, _, _ = adapter.collect(actor, 0.0)
            assert transition["motion_time"][0] == last_frame
            assert transition["truncated"][0]
            assert command.time_steps[0] == adapter.motion.motion_starts[0]
            torch.testing.assert_close(adapter.history[0, 0], adapter.history[0, -1], rtol=0, atol=0)
            # Reset randomization must preserve other environments and draw offsets from nominal CoM.
            preserved = context.read(env)[1:].clone()
            ids = torch.tensor([0], device=device)
            for _ in range(5):
                randomize_context(env, ids)
                bounds = command.com.new_tensor((0.025, 0.05, 0.05))
                assert ((command.com[0] - command.nominal_com[0]).abs() <= bounds + 1e-6).all()
            torch.testing.assert_close(context.read(env)[1:], preserved, rtol=0, atol=0)
            actual_com = command.robot.data.body_com_pos_b.torch[
                :, command.robot.body_names.index(command.cfg.context_body)
            ]
            torch.testing.assert_close(command.com, actual_com, atol=1e-6, rtol=1e-5)
            command.cfg.evaluation = True
            randomize_context(env, torch.arange(env.num_envs, device=device))
            adapter = MotionTrackingAdapter(env, agent)
            adapter.reset()
            torch.testing.assert_close(adapter.history, adapter.obs["actor"].unsqueeze(1).expand_as(adapter.history))
            torch.testing.assert_close(command.friction, torch.ones_like(command.friction))
            torch.testing.assert_close(command.com, command.nominal_com)
        finally:
            env.close()
