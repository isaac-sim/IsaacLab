# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical receipt counting across target handoffs and independent environment resets."""

from contextlib import closing

import gymnasium as gym
import pytest
import torch

from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_receipts_record_success_and_continue_without_reset(device):
    """Receipts reward once, record the sampled row once, and preserve history through handoffs and failures."""
    task = "IsaacContrib-Conveyor-Suction-Sorting"
    cfg = parse_env_cfg(task, device=device, num_envs=2)
    cfg.sim.visualizer_cfgs = []
    cfg.sim.save_logs_to_file = False
    cfg.seed = 41
    cfg.terminations.failure = None
    cfg.terminations.invalid_action = None
    with closing(gym.make(task, cfg=cfg).unwrapped) as env:
        reset = env.event_manager.get_term_cfg("reset_parcels").func
        rows = torch.stack(
            [((reset.parcel_ids == parcel) & (reset.phase_ids == 0)).nonzero()[0, 0] for parcel in (0, 4)]
        )
        reset.reset_sampler.sampler.sample = lambda probabilities, count: rows[:count]
        env.reset()
        command = env.command_manager.get_term("sort")
        monitor = reset.reset_sampler.success_monitor
        assert not command.delivered.any(), "Every reset must start with all parcels unsorted."
        assert not command.new_success.any() and not env.episode_success.any()
        assert command.metrics["correct_deliveries"].tolist() == [0, 0]
        assert monitor.success_size[rows].tolist() == [0, 0]
        actions = torch.zeros((2, 4), device=device)
        actions[:, 3] = 1.0
        # An outfeed-speed impact must settle in the bin without crossing its floor.
        impact_ids = torch.tensor([0], device=device)
        pose = torch.tensor(
            [[2.07563591, 0.28751278, -0.71146984, -0.97748345, -0.07764439, -0.19566739, -0.01455657]],
            device=device,
        )
        pose[:, :3] += env.scene.env_origins[impact_ids]
        velocity = torch.tensor(
            [[0.31988335, 0.05606911, -3.89049959, -1.84165192, 10.48158455, 0.01690909]], device=device
        )
        env.parcels[0].write_root_pose_to_sim_index(root_pose=pose, env_ids=impact_ids)
        env.parcels[0].write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=impact_ids)
        env.sim.forward()
        env.scene.update(env.step_dt)
        received = [False, False]
        for _ in range(720):
            lengths = env.episode_length_buf.clone()
            _, rewards, terminated, timed_out, extras = env.step(actions)
            assert not terminated.any() and not timed_out.any(), "One receipt must not reset the physical scene."
            torch.testing.assert_close(env.episode_length_buf, lengths + 1)
            assert torch.isfinite(rewards).all()
            for env_id, parcel_id in enumerate((0, 4)):
                if command.delivered[env_id, parcel_id] and not received[env_id]:
                    assert env.episode_success[env_id] and command.metrics["success_rate"][env_id] == 1
                    assert extras["successes"][env_id]
                    assert monitor.success_size[rows[env_id]] == monitor.success_rate[rows[env_id]] == 1
                    delivery = dict(env.reward_manager.get_active_iterable_terms(env_id))["delivery"][0]
                    assert delivery == 600.0
                    received[env_id] = True
            if all(received):
                break
        else:
            pytest.fail("The released parcels never reached their physical bins.")
        assert not command.failed.any()
        assert env.active_ids[1] != 4, "The release fixture must also reach the next-target handoff."
        expected = torch.zeros(len(env.parcels), dtype=torch.bool, device=device)
        expected[4] = True
        torch.testing.assert_close(command.delivered[1], expected)

        # A handoff and an unrelated reset cannot repeat the receipt or its sampled-row outcome.
        _, _, _, _, extras = env.step(actions)
        assert not extras["successes"].any()
        assert dict(env.reward_manager.get_active_iterable_terms(1))["delivery"][0] == 0.0
        env.reset(env_ids=impact_ids)
        command.evaluate()
        torch.testing.assert_close(command.delivered[1], expected)
        assert command.metrics["correct_deliveries"][1] == 1
        assert env.episode_success[1]
        assert monitor.success_size[rows].tolist() == [1, 1]

        # A positive-length attempt without a receipt records one failure when it resets.
        env.step(actions)
        assert not command.delivered[0].any() and not env.episode_success[0]
        env.reset(env_ids=impact_ids)
        assert monitor.success_size[rows].tolist() == [2, 1]
        assert monitor.success_rate[rows].tolist() == [0.5, 1.0]

        def place_in_bin(env_id, parcel_id, bin_id):
            ids = torch.tensor([env_id], device=device)
            parcel = env.parcels[parcel_id]
            pose = parcel.data.default_root_pose.torch[ids].clone()
            pose[:, :3] = env.bin_positions[bin_id] + env.scene.env_origins[ids]
            pose[:, 2] -= 0.28 - env.parcel_sizes[parcel_id, 2] / 2 - 0.006
            parcel.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            parcel.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=device), env_ids=ids)
            env.sim.forward()
            env.scene.update(env.step_dt)

        # Identical released poses require observable qualification before they can earn a receipt.
        second = env.active_ids[1].item()
        place_in_bin(1, second, env.parcel_classes[second])
        command.lifted[1] = False
        unqualified = env.observation_manager.compute()["policy"][1].clone()
        command.lifted[1] = True
        qualified = env.observation_manager.compute()["policy"][1]
        assert not torch.equal(unqualified, qualified), "Pickup qualification must be observable to the policy."
        assert qualified.shape == (52,)
        command.lifted[1] = False
        for _ in range(15):
            env.step(actions)
        assert command.failed[1] and not command.delivered[1, second]
        command.failed[1] = False
        command.lifted[1] = True
        place_in_bin(1, second, env.parcel_classes[second])
        for _ in range(2):
            env.step(actions)
        assert not command.delivered[1, second]

        # An invalid action on a would-be receipt step earns no credit, without erasing earlier success.
        actions[1, 0] = torch.nan
        _, rewards, _, _, extras = env.step(actions)
        assert command.failed[1] and not command.delivered[1, second]
        assert not extras["successes"][1]
        assert dict(env.reward_manager.get_active_iterable_terms(1))["delivery"][0] == 0.0
        assert env.episode_success[1] and command.metrics["success_rate"][1] == 1
        assert monitor.success_size[rows[1]] == monitor.success_rate[rows[1]] == 1
        assert torch.isfinite(rewards).all()

        actions[1, 0] = 0.0
        command.failed[1] = False
        place_in_bin(1, second, env.parcel_classes[second])
        pulses = 0
        for _ in range(15):
            _, _, terminated, timed_out, extras = env.step(actions)
            assert not terminated.any() and not timed_out.any()
            pulse = extras["successes"][1].item()
            pulses += pulse
            assert dict(env.reward_manager.get_active_iterable_terms(1))["delivery"][0] == 600.0 * pulse
        assert pulses == 1
        assert env.episode_success[1]
        assert command.delivered[1].sum() == command.metrics["correct_deliveries"][1] == 2
        assert monitor.success_size[rows[1]] == monitor.success_rate[rows[1]] == 1

        # An unselected parcel cannot disappear into a wrong bin without failing its world.
        inactive = next(
            i for i in range(len(env.parcels)) if i != env.active_ids[0].item() and not command.delivered[0, i]
        )
        place_in_bin(0, inactive, (env.parcel_classes[inactive].item() + 1) % 5)
        for _ in range(15):
            env.step(actions)
        assert command.failed.tolist() == [True, False]
        assert env.episode_success.tolist() == [False, True]
        env.reset(env_ids=impact_ids)
        assert monitor.success_size[rows].tolist() == [3, 1]
        assert monitor.success_rate[rows[0]].item() == pytest.approx(1.0 / 3.0)
