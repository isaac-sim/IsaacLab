# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NeRD's native capture, offline/checkpoint and graph/reset boundary on two real articulated tasks."""

import json

import gymnasium as gym
import h5py
import pytest
import torch
import warp as wp

from isaaclab.app import launch_simulation
from isaaclab.managers import DatasetExportMode
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.utils import resolve_task_config


def never(env):
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


@pytest.mark.parametrize("task", ("Isaac-Cartpole-NeRD", "Isaac-Velocity-Flat-Anymal-C-NeRD"))
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_record_fit_resume_and_graph_resets(task, device, tmp_path):
    pytest.importorskip("physicsnemo_robotics")
    from isaaclab_newton.physics.nerd_manager import SolverNeRD

    from isaaclab_rl.nerd import NeRDTrainCfg, OfflineNeRD

    torch.backends.cuda.matmul.allow_tf32 = False
    cfg, _ = resolve_task_config(task + "-Explore", None, overrides=[])
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 4, device, 7
    cfg.recorders.transition.record_to_episodes = True
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    cfg.recorders.dataset_export_dir_path = str(tmp_path)
    cfg.recorders.export_in_close = True
    cfg.recorders.buffer_initial_capacity = 2
    cfg.episode_length_s = 6 * cfg.sim.dt * cfg.decimation
    with launch_simulation(cfg):
        env = gym.make(task + "-Explore", cfg=cfg).unwrapped
        try:
            env.reset()
            manager = env.sim.backend_manager
            capture = env.recorder_manager.get_term("transition")
            descriptor = capture.dynamics.descriptor
            metadata = {"dynamics": descriptor}
            (tmp_path / "dataset.json").write_text(json.dumps(metadata))
            # Stagger autoresets so prefixes and retained fields exercise different world lengths.
            env.episode_length_buf[:] = torch.arange(4, device=device)
            seen_terminal = False
            with torch.inference_mode():
                for step in range(16):
                    raw = manager.get_state_0()
                    before = torch.cat(
                        (wp.to_torch(raw.joint_q).reshape(4, -1), wp.to_torch(raw.joint_qd).reshape(4, -1)), -1
                    ).clone()
                    actions = torch.full((4, env.action_manager.total_action_dim), 0.1 * (step % 3 - 1), device=device)
                    env.step(actions)
                    torch.testing.assert_close(capture.latest["state"], before, rtol=0, atol=0)
                    if descriptor["control"] == "joint_target_q":
                        # In these floating-base worlds, the final coordinates are the 12 leg targets.
                        actual = capture.latest["control"][:, 7:]
                        names = [
                            name.rsplit("/", 1)[-1]
                            for name in manager.get_model().joint_label[: manager.get_model().joint_count // 4]
                        ]
                        names = [name for name in names if name in env.scene["robot"].joint_names]
                        expected = env.scene["robot"].actuators.target_command.position.torch[
                            :, [env.scene["robot"].joint_names.index(name) for name in names]
                        ]
                        # Reset overwrites asset targets, so compare continuing rows.
                        live = ~env.reset_buf
                        torch.testing.assert_close(actual[live], expected[live], rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(capture.latest["control"][:, 0], actions[:, 0] * 100, rtol=0, atol=0)
                    if env.reset_buf.any():
                        after = torch.cat(
                            (wp.to_torch(raw.joint_q).reshape(4, -1), wp.to_torch(raw.joint_qd).reshape(4, -1)), -1
                        )
                        assert not torch.equal(capture.latest["next_state"][env.reset_buf], after[env.reset_buf])
                        seen_terminal = True
            assert seen_terminal and manager._graph is not None
            # Last continuing sample agrees with cold episode materialization, byte for byte.
            for world in range(4):
                episode = env.recorder_manager.get_episode(world)
                if not episode.is_empty():
                    for name, tensor in capture.latest.items():
                        assert torch.equal(episode.data["transition"][name][-1], tensor[world])
            env.reset()
            assert (capture.latest["terminated"] | capture.latest["truncated"]).all()
            env.recorder_manager.export_episodes()
        finally:
            env.close()
    dataset = tmp_path / "dataset.hdf5"
    with h5py.File(dataset) as data:
        assert len(data["data"]) >= 4
        assert sum(len(episode["transition/state"]) for episode in data["data"].values()) == 64

    train_cfg = NeRDTrainCfg(
        max_updates=3,
        batch_size=8,
        compile=False,
        network={
            "num_heads": 1,
            "head_dim": 16,
            "intermediate_size": 32,
            "head_hidden": 16,
            "temporal_attention_spans": (2, 2),
            "linear_probe": True,
        },
    )
    learner = OfflineNeRD([dataset], train_cfg, device)
    assert all(torch.isfinite(torch.tensor(v)) for v in learner.update().values())
    checkpoint = tmp_path / "model.pt"
    torch.save(learner.checkpoint(), checkpoint)
    resumed = OfflineNeRD(
        [dataset], train_cfg, device, checkpoint=torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    assert torch.equal(learner.generator.get_state(), resumed.generator.get_state())
    assert torch.equal(learner.input_mean, resumed.input_mean)
    assert torch.equal(learner.target_scale, resumed.target_scale)
    learner.update()
    resumed.update()
    for name, tensor in learner.model.state_dict().items():
        # GPU/JIT reduction ordering can change final mantissa bits; resume preserves the update numerically.
        torch.testing.assert_close(tensor, resumed.model.state_dict()[name], rtol=1.0e-6, atol=1.0e-8)
    assert all(torch.isfinite(torch.tensor(v)) for v in learner.validate().values())
    torch.save(learner.checkpoint(), checkpoint)

    cfg, _ = resolve_task_config(task + "-Learned", None, overrides=[])
    cfg.play_mode()
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 8, device, 7
    # Playback imports visual-only geometry absent from headless collection; it is not physics.
    cfg.sim.physics.load_visual_shapes = True
    cfg.sim.physics.solver_cfg.checkpoint = str(checkpoint)
    cfg.sim.physics.solver_cfg.compile = "Anymal" in task
    for name, term in vars(cfg.terminations).items():
        if term is not None:
            term.func, term.params = never, {}
    with launch_simulation(cfg):
        env = gym.make(task + "-Learned", cfg=cfg).unwrapped
        try:
            env.reset()
            manager = env.sim.backend_manager
            reference = SolverNeRD(manager.get_model(), checkpoint, compile=cfg.sim.physics.solver_cfg.compile)
            with torch.no_grad():
                probe = torch.randn_like(reference.history)
                torch.testing.assert_close(reference.predict(probe), reference.network.predict_last(probe))
            state = manager.get_model().state()
            state.assign(manager.get_state_0())
            action = torch.zeros(8, env.action_manager.total_action_dim, device=device)
            addresses = (manager._solver.history.data_ptr(), manager._solver.pending_reset.data_ptr())
            with torch.inference_mode():
                for step in range(21):
                    if step in (4, 9, 13):
                        ids = torch.tensor([step % 4], device=device)
                        env.reset(ids)
                        state.assign(manager.get_state_0())
                        mask = wp.zeros(9, dtype=wp.bool, device=device)
                        wp.to_torch(mask)[ids] = True
                        reference.reset(state, world_mask=mask)
                    if step < 2 or step % 2:
                        env.step(action)
                    else:
                        # Direct graph replay must advance history without executing Python cursor updates.
                        wp.capture_launch(manager._graph)
                    reference.step(state, state, manager.get_control(), None, env.step_dt)
                    wp.synchronize_device(device)
                    torch.testing.assert_close(
                        wp.to_torch(manager.get_state_0().joint_q),
                        wp.to_torch(state.joint_q),
                        rtol=2e-5,
                        atol=2e-6,
                        msg=f"Joint positions at replay step {step}",
                    )
                    torch.testing.assert_close(
                        wp.to_torch(manager.get_state_0().joint_qd),
                        wp.to_torch(state.joint_qd),
                        rtol=2e-5,
                        atol=2e-6,
                        msg=f"Joint velocities at replay step {step}",
                    )
            assert manager._graph is not None
            assert addresses == (manager._solver.history.data_ptr(), manager._solver.pending_reset.data_ptr())
            with pytest.raises(ValueError, match="frame"):
                reference.step(state, state, manager.get_control(), None, env.step_dt * 2)
            with pytest.raises(ValueError, match="decimation"):
                manager.set_decimation(2)
            # A checkpoint must reject altered physics even when tensor shapes still agree.
            masses = wp.to_torch(manager.get_model().body_mass)
            saved_masses = masses.clone()
            masses.mul_(1.1)
            with pytest.raises(ValueError, match="invalidate"):
                reference.notify_model_changed(1)
            masses.copy_(saved_masses)
        finally:
            env.close()
