# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NeRD capture, fitting, checkpoint and graph/reset contracts across articulated tasks and contact models."""

import json
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import h5py
import newton
import pytest
import torch
import warp as wp

from isaaclab.app import launch_simulation
from isaaclab.managers import DatasetExportMode
from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import to_dict

from isaaclab_tasks.utils import resolve_task_config


def never(env):
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


@pytest.mark.parametrize(
    "task",
    (
        "Isaac-Cartpole-NeRD",
        "Isaac-Velocity-Flat-Anymal-C-NeRD",
        "Isaac-Velocity-Flat-Anymal-C-NeRD-Effort",
        "Isaac-Velocity-Rough-Anymal-C-NeRD",
    ),
)
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_record_fit_resume_and_graph_resets(task, device, tmp_path):
    pytest.importorskip("physicsnemo_robotics")
    from isaaclab_newton.physics.nerd import JointDynamics

    from isaaclab_rl.nerd import NeRDTrainCfg, OfflineNeRD

    torch.backends.cuda.matmul.allow_tf32 = False
    cfg, _ = resolve_task_config(task + "-Explore", None, overrides=[])
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 4, device, 7
    if "Rough" in task:
        from isaaclab.envs.mdp import reset_root_state_uniform
        from isaaclab.terrains import HfRandomUniformTerrainCfg

        cfg.scene.terrain.terrain_generator.num_rows = cfg.scene.terrain.terrain_generator.num_cols = 1
        cfg.scene.terrain.terrain_generator.sub_terrains = {
            "rough": HfRandomUniformTerrainCfg(noise_range=(-0.02, 0.02), noise_step=0.005)
        }
        cfg.events.reset_base.func = reset_root_state_uniform
        cfg.events.reset_base.params["pose_range"].update(x=(0.0, 0.0), y=(0.0, 0.0), z=(-0.35, -0.35))
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
            if "Cartpole" in task:
                with pytest.raises(ValueError, match="version 2"):
                    JointDynamics(dict(descriptor, version=1))
                gains = wp.to_torch(manager.get_model().joint_target_ke)
                saved_gains = gains.clone()
                gains.fill_(100.0)
                with pytest.raises(ValueError, match="position drive"):
                    JointDynamics(model=manager.get_model(), frame_dt=env.step_dt, control="joint_f")
                gains.copy_(saved_gains)
            metadata = {"dynamics": descriptor, "task": task + "-Explore"}
            (tmp_path / "dataset.json").write_text(json.dumps(metadata))
            # Stagger autoresets so prefixes and retained fields exercise different world lengths.
            env.episode_length_buf[:] = torch.arange(4, device=device)
            seen_terminal = False
            seen_contacts = False
            with torch.inference_mode():
                for step in range(16):
                    raw = manager.get_state_0()
                    before = torch.cat(
                        (wp.to_torch(raw.joint_q).reshape(4, -1), wp.to_torch(raw.joint_qd).reshape(4, -1)), -1
                    ).clone()
                    actions = torch.full((4, env.action_manager.total_action_dim), 0.1 * (step % 3 - 1), device=device)
                    env.step(actions)
                    if "Rough" in task:
                        contacts = capture.latest["contacts"]
                        valid = contacts[..., 0] == 1
                        seen_contacts |= bool(valid.any())
                        assert (
                            (contacts[..., 1][valid] >= 0)
                            & (contacts[..., 1][valid] < len(descriptor["contact_entities"]))
                        ).all()
                        torch.testing.assert_close(
                            contacts[..., 7:10][valid].norm(dim=-1), torch.ones_like(contacts[..., 0][valid])
                        )
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
                    elif "Cartpole" in task:
                        torch.testing.assert_close(capture.latest["control"][:, 0], actions[:, 0] * 100, rtol=0, atol=0)
                    else:
                        robot = env.scene["robot"]
                        names = [
                            name.rsplit("/", 1)[-1]
                            for name in manager.get_model().joint_label[: manager.get_model().joint_count // 4]
                        ]
                        names = [name for name in names if name in robot.joint_names]
                        expected = robot.actuators.applied_effort.torch[
                            :, [robot.joint_names.index(name) for name in names]
                        ]
                        live = ~env.reset_buf
                        torch.testing.assert_close(capture.latest["control"][live, 6:], expected[live], rtol=0, atol=0)
                    if env.reset_buf.any():
                        after = torch.cat(
                            (wp.to_torch(raw.joint_q).reshape(4, -1), wp.to_torch(raw.joint_qd).reshape(4, -1)), -1
                        )
                        assert not torch.equal(capture.latest["next_state"][env.reset_buf], after[env.reset_buf])
                        seen_terminal = True
            assert seen_terminal and manager._graph is not None
            if "Rough" in task:
                assert seen_contacts
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
    if "Rough" in task:
        train_cfg.network.update(contact_slot_count=2, contact_slot_dim=8)
        train_cfg.precision = "bfloat16"
    elif task.endswith("-Effort"):
        train_cfg.network_type = "physicsnemo_robotics.models.nerd:NewtonNeRDVectorTransformer"
        train_cfg.network = {
            "num_heads": 2,
            "hidden_size": 32,
            "head_hidden": 16,
            "temporal_depth": 2,
            "temporal_attention_spans": (2, 2),
            "position_encoding": "relative_lag",
        }
    elif "Anymal" in task:
        train_cfg.precision = "float16"
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
    assert learner.scaler.state_dict() == resumed.scaler.state_dict()
    learner.update()
    resumed.update()
    for name, tensor in learner.model.state_dict().items():
        # GPU/JIT reduction ordering can change final mantissa bits; resume preserves the update numerically.
        torch.testing.assert_close(tensor, resumed.model.state_dict()[name], rtol=1.0e-6, atol=1.0e-8)
    assert all(torch.isfinite(torch.tensor(v)) for v in learner.validate().values())
    torch.save(learner.checkpoint(), checkpoint)

    if "Cartpole" in task:
        # A two-rank CPU run must match one optimizer step over the combined samples.
        # This checks gradient averaging independently of DDP and exercises the actual CLI/save root.
        options = tmp_path / "training.json"
        options.write_text(json.dumps(to_dict(train_cfg)))
        output = tmp_path / "distributed"
        script = Path(ISAACLAB_ROOT) / "scripts/learning/nerd/train.py"
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
        command += [str(script), "--dataset", str(dataset), "--output", str(output)]
        command += ["--config", str(options), "--device", "cpu", "--log_interval", "1"]
        with (tmp_path / "distributed.log").open("w") as log:
            subprocess.run(command + ["--max_updates", "1"], stdout=log, stderr=log, check=True, timeout=120)
        distributed = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        oracle = OfflineNeRD([dataset], train_cfg, "cpu")
        samples = []
        for rank in range(2):
            oracle.generator.manual_seed(train_cfg.seed + rank)
            samples.append(oracle.sample(oracle.train_ids, train_cfg.batch_size // 2))
        inputs, targets = (torch.cat([sample[i] for sample in samples]) for i in range(2))
        (oracle.model.predict_last(inputs) - targets).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(oracle.model.parameters(), train_cfg.gradient_clip_norm)
        oracle.optimizer.step()
        for name, tensor in oracle.model.state_dict().items():
            torch.testing.assert_close(tensor, distributed["model"][name], rtol=1e-5, atol=1e-7)
        assert len(distributed["generators"]) == 2
        assert not torch.equal(*distributed["generators"])
        with (tmp_path / "distributed.log").open("a") as log:
            subprocess.run(
                command + ["--max_updates", "2", "--checkpoint", str(output / "last.pt")],
                stdout=log,
                stderr=log,
                check=True,
                timeout=120,
            )
        continued = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        assert continued["updates"] == 2
        assert all(not torch.equal(a, b) for a, b in zip(distributed["generators"], continued["generators"]))
        assert len((output / "metrics.jsonl").read_text().splitlines()) == 2

    _check_graph_resets(task, device, checkpoint)


def _check_graph_resets(task, device, checkpoint):
    from isaaclab_newton.physics.nerd_manager import SolverNeRD

    cfg, _ = resolve_task_config(task + "-Learned", None, overrides=[])
    cfg.play_mode()
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 8, device, 7
    if "Rough" in task:
        # Unseen terrain and different world count must preserve the robot/material contract.
        cfg.scene.terrain.terrain_generator.seed = 99
        cfg.scene.terrain.terrain_generator.num_rows = cfg.scene.terrain.terrain_generator.num_cols = 1
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
            collision = newton.CollisionPipeline(manager.get_model()) if "Rough" in task else None
            contacts = collision.contacts() if collision else None
            with torch.no_grad():
                probe = torch.randn_like(reference.history)
                if "Rough" in task:
                    tokens = reference.dynamics.contact_tokens(
                        reference.contact_history, reference.contact_mean, reference.contact_scale
                    )
                    torch.testing.assert_close(
                        reference.predict(probe, tokens), reference.network.predict_last(probe, tokens)
                    )
                else:
                    torch.testing.assert_close(reference.predict(probe), reference.network.predict_last(probe))
            state = manager.get_model().state()
            state.assign(manager.get_state_0())
            action = torch.zeros(8, env.action_manager.total_action_dim, device=device)
            if cfg.decimation > 1:
                # Save every applied effort, then independently replay four
                # single solver steps. ANYdrive's recurrent state stays owned by the real actuator.
                efforts = torch.empty(cfg.decimation, manager.get_model().joint_dof_count, device=device)
                cursor = torch.zeros(1, dtype=torch.long, device=device)
                replay_control = manager.get_model().control()

                def retain_effort():
                    with torch.cuda.stream(wp.stream_to_torch(wp.get_stream(device))):
                        efforts.index_copy_(
                            0, cursor.remainder(cfg.decimation), wp.to_torch(manager.get_control().joint_f)[None]
                        )
                        cursor.add_(1)

                manager.register_post_actuator_callback(retain_effort)
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
                    if step < 2 or step % 2 or not manager.handles_decimation():
                        env.step(action)
                    else:
                        # Direct graph replay must advance history without executing Python cursor updates.
                        wp.capture_launch(manager._graph)
                    if cfg.decimation > 1:
                        for substep in range(cfg.decimation):
                            wp.copy(replay_control.joint_f, wp.from_torch(efforts[substep]))
                            if collision:
                                collision.collide(state, contacts)
                            reference.step(state, state, replay_control, contacts, cfg.sim.dt)
                    else:
                        reference.step(state, state, manager.get_control(), None, env.step_dt)
                    wp.synchronize_device(device)
                    torch.testing.assert_close(
                        wp.to_torch(manager.get_state_0().joint_q),
                        wp.to_torch(state.joint_q),
                        rtol=2e-5,
                        atol=2e-6,
                    )
                    torch.testing.assert_close(
                        wp.to_torch(manager.get_state_0().joint_qd),
                        wp.to_torch(state.joint_qd),
                        rtol=2e-5,
                        atol=2e-6,
                    )
            assert manager._graph is not None
            assert addresses == (manager._solver.history.data_ptr(), manager._solver.pending_reset.data_ptr())
            with pytest.raises(ValueError, match="frame"):
                reference.step(state, state, manager.get_control(), None, env.step_dt * 2)
            if reference.dynamics.descriptor["control"] == "joint_target_q":
                with pytest.raises(ValueError, match="decimation"):
                    manager.set_decimation(2)
            # A checkpoint must reject altered physics even when tensor shapes still agree.
            masses = wp.to_torch(manager.get_model().body_mass)
            saved_masses = masses.clone()
            masses.mul_(1.1)
            with pytest.raises(ValueError, match="invalidate"):
                reference.notify_model_changed(1)
            masses.copy_(saved_masses)
            if "Rough" in task:
                materials = wp.to_torch(manager.get_model().shape_material_mu)
                saved_materials = materials.clone()
                materials.mul_(0.8)
                with pytest.raises(ValueError, match="invalidate"):
                    reference.notify_model_changed(1)
                materials.copy_(saved_materials)
            if "Cartpole" in task:
                anchors = wp.to_torch(manager.get_model().joint_X_p)
                saved_anchors = anchors.clone()
                roots = wp.to_torch(manager.get_model().joint_parent) == -1
                anchors[roots, 3:] = anchors.new_tensor([0.0, 2**-0.5, 0.0, 2**-0.5])
                with pytest.raises(ValueError, match="invalidate"):
                    reference.notify_model_changed(1)
                anchors.copy_(saved_anchors)
                flags = wp.to_torch(manager.get_model().body_flags)
                saved_flags = flags.clone()
                flags.zero_()
                with pytest.raises(ValueError, match="invalidate"):
                    reference.notify_model_changed(1)
                flags.copy_(saved_flags)
                model = manager.get_model()
                pairs = model.shape_collision_filter_pairs_array()
                assert len(pairs)
                model._set_shape_collision_filter_pairs(())
                with pytest.raises(ValueError, match="invalidate"):
                    reference.notify_model_changed(1)
                model._set_shape_collision_filter_pairs(pairs)
        finally:
            env.close()
