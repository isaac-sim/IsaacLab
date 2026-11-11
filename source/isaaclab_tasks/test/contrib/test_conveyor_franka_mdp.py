# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint, inventory identity, and selected-reset contracts for conveyor tasks."""

import subprocess
import sys
from contextlib import closing

import gymnasium as gym
import pytest
import torch
from torch.distributions import Bernoulli, Normal

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.conveyor_franka import mdp
from isaaclab_tasks.contrib.conveyor_franka.agents.rsl_rl_ppo_cfg import ConveyorGaussianBernoulliDistribution
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

_BASE = "IsaacContrib-Conveyor-Racetrack-Transfer-v0"
_SORTER = "IsaacContrib-Conveyor-Warehouse-Sorting-v0"

pytestmark = pytest.mark.kitless


def _environment(task):
    cfg = parse_env_cfg(task, device="cpu", num_envs=2)
    cfg.events.reset_from_state_table.params.update(
        fixed_recipe=int(mdp.ConveyorResetRecipe.BELT), fixed_variant_id=mdp.BELT_DEPLOYMENT_VARIANT
    )
    cfg.sim.physics.use_cuda_graph = False
    cfg.sim.visualizer_cfgs = []
    cfg.sim.use_fabric = False
    cfg.sim.save_logs_to_file = False
    cfg.seed = 41
    return gym.make(task, cfg=cfg).unwrapped


def test_warehouse_config_import_does_not_preload_usd():
    """Contrib configuration remains importable before a Kit application starts."""
    code = (
        "import sys; "
        "import isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_asset_env_cfg; "
        "raise SystemExit('pxr' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_checkpoint_interface_and_invalid_action_recovery():
    """The trained 123/8 interface survives invalid actions and selected action resets."""
    with closing(_environment(_BASE)) as env:
        obs, _ = env.reset()
        assert obs["policy"].shape == (2, 123)
        assert env.action_manager.total_action_dim == 8
        result = env.step(torch.zeros(2, 8))
        assert torch.isfinite(result[0]["policy"]).all() and torch.isfinite(result[1]).all()

        bad = torch.zeros(2, 8)
        bad[0, 0], bad[1, 7] = float("nan"), float("inf")
        env.action_manager.process_action(bad)
        for name in ("arm_action", "gripper_action"):
            assert torch.isfinite(env.action_manager.get_term(name).processed_actions).all()
        assert mdp.invalid_action(env).tolist() == [True, True]
        env.action_manager.reset(env_ids=[0])
        assert mdp.invalid_action(env).tolist() == [False, True]
        terminated = env.step(bad)
        assert terminated[2].tolist() == [True, True]
        assert torch.isfinite(terminated[0]["policy"]).all() and torch.isfinite(terminated[1]).all()


def test_warehouse_slot_mapping_and_selected_reset_counters():
    """Four policy slots track physical inventory; resetting one world preserves the other."""
    with closing(_environment(_SORTER)) as env:
        env.reset()
        pool = env.conveyor_cube_pool
        assert len(pool.assets) == 24
        positions = torch.tensor([[[2.5, 0.8, 0.5]] * 24] * 2)
        positions[:, :4] = torch.tensor([[0.4, 0.27, 0.06], [0.6, -0.27, 0.06], [0.8, 0.0, 0.2], [0.9, -0.27, 0.06]])
        positions[0, 4] = torch.tensor([0.52, 0.27, 0.06])
        positions[1, 5] = torch.tensor([0.72, -0.27, 0.06])
        pool.slot_ids[:] = torch.tensor([[4, 1, 2, 3], [0, 5, 2, 3]])
        for index, parcel in enumerate(pool.assets):
            pose = parcel.data.default_root_pose.torch.clone()
            pose[:, :3] = positions[:, index] + env.scene.env_origins
            parcel.write_root_pose_to_sim_index(root_pose=pose, skip_forward=True)
            parcel.write_root_velocity_to_sim_index(root_velocity=torch.zeros(2, 6), skip_forward=True)
        env.sim.forward()
        env.scene.update(env.step_dt)
        policy = env.observation_manager.compute()["policy"]
        expected = positions[torch.arange(2)[:, None], pool.slot_ids]
        torch.testing.assert_close(policy[:, 16:28], expected.flatten(1))
        pool.record_transfers(torch.tensor([0, 1]), torch.tensor([0, 1]))
        assert pool.transfer_counts[0, 4] == 1 and pool.transfer_counts[1, 5] == 1
        assert pool.transfer_counts.sum() == 2

        pool.assignment_counts.fill_(7)
        pool.transfer_counts.fill_(3)
        before_slots = pool.slot_ids[0].clone()
        before_poses = torch.stack([parcel.data.root_pose_w.torch[0].clone() for parcel in pool.assets])
        before_velocities = torch.stack([parcel.data.root_vel_w.torch[0].clone() for parcel in pool.assets])
        velocity = pool.assets[-1].data.root_vel_w.torch.clone()
        velocity[1].fill_(0.2)
        pool.assets[-1].write_root_velocity_to_sim_index(root_velocity=velocity)
        env._reset_idx(slice(1, 2))

        assert not pool.transfer_counts[1].any(), "Selected reset must clear the completed batch's transfer counters."
        torch.testing.assert_close(pool.assignment_counts[1], torch.tensor([1] * 4 + [0] * 20))
        assert (pool.assignment_counts[0] == 7).all() and (pool.transfer_counts[0] == 3).all()
        torch.testing.assert_close(pool.slot_ids[1], torch.arange(4))
        torch.testing.assert_close(pool.slot_ids[0], before_slots)
        assert not pool.assets[-1].data.root_vel_w.torch[1].any()
        torch.testing.assert_close(
            torch.stack([parcel.data.root_pose_w.torch[0] for parcel in pool.assets]), before_poses
        )
        torch.testing.assert_close(
            torch.stack([parcel.data.root_vel_w.torch[0] for parcel in pool.assets]), before_velocities
        )


def test_binary_policy_likelihoods():
    """Signed binary actions use Bernoulli likelihoods alongside the Gaussian arm."""
    distribution = ConveyorGaussianBernoulliDistribution(output_dim=8)
    distribution.update(torch.zeros(64, 8))
    samples = distribution.sample()
    previous = tuple(parameter.clone() for parameter in distribution.params)
    distribution.update(torch.full((64, 8), 0.2))
    means, deviations, logits = distribution.params
    expected = Normal(means, deviations).log_prob(samples[:, :-1]).sum(-1)
    expected += Bernoulli(logits=logits).log_prob((samples[:, -1:] + 1.0) / 2.0).sum(-1)
    torch.testing.assert_close(distribution.log_prob(samples), expected)
    assert set(samples[:, -1].unique().tolist()) == {-1, 1}
    divergence = distribution.kl_divergence(previous, distribution.params)
    assert torch.isfinite(divergence).all() and (divergence >= 0).all()
