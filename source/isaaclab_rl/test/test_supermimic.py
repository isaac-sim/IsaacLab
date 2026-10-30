# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Portions derived from SuperMimic (MIT), Copyright (c) 2026 ETH Zurich, Chong Zhang.
# See docs/licenses/dependencies/supermimic-license.txt.

"""Contracts for native SuperMimic ownership, replay and evaluation."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_rl.supermimic.learner import ReplayBuffer


@pytest.mark.parametrize("architecture", ("mlp", "nerd-one-frame"))
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_compiled_world_model_propagates_finite_action_gradients(architecture, device):
    """Imagined control needs action gradients through frozen dynamics, including identity rotations."""
    from isaaclab_rl.supermimic.networks import WM_IN, WM_OUT, WorldModel, state_difference

    torch.manual_seed(0)
    network = None
    if architecture == "nerd-one-frame":
        dependency = pytest.importorskip("physicsnemo_robotics.models.parallel_nerd")
        network = dependency.ParallelNeRD(WM_IN, WM_OUT, temporal_attention_spans=(1, 1, 1, 1))
        torch.nn.init.normal_(network.head_output.weight, std=0.001)
    world = WorldModel(hidden=(32, 32), network=network).to(device)
    if network is None:
        torch.nn.init.normal_(world.net[-1].weight, std=0.001)
    world.requires_grad_(False)
    state = (
        torch.zeros(4, 3, device=device),
        torch.eye(3, device=device).expand(4, -1, -1),
        torch.zeros(4, 3, device=device),
        torch.zeros(4, 3, device=device),
        torch.zeros(4, 29, device=device),
        torch.zeros(4, 29, device=device),
    )
    action = torch.zeros(4, 29, device=device, requires_grad=True)
    context = torch.zeros(4, 33, device=device)
    successor = torch.compile(world.step)(state, action, context)
    loss = (state_difference(state, successor) - 0.1).square().mean()
    gradient = torch.autograd.grad(loss, action)[0]
    assert torch.isfinite(loss) and gradient.isfinite().all() and gradient.abs().sum() > 0
    assert all(parameter.grad is None for parameter in world.parameters())


def test_native_ownership_has_no_external_training_package():
    """Reject external forwarders, duplicate task stepping and task-to-learner dependency inversions."""
    task = Path(ISAACLAB_ROOT) / "source/isaaclab_tasks/isaaclab_tasks/contrib/g1_tracking"
    learning = Path(ISAACLAB_ROOT) / "source/isaaclab_rl/isaaclab_rl/supermimic"
    backends = learning.parent / "entrypoints/backends"
    roots = (backends / "train_supermimic.py", backends / "play_supermimic.py")
    for path in (*task.rglob("*.py"), *learning.rglob("*.py"), *roots):
        tree = ast.parse(path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imports.append(node.module or "")
        assert not any(name == "supermimic" or name.startswith("supermimic.") for name in imports), path
        symbols = {node.name for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        assert symbols.isdisjoint(
            {"TrackingSim", "TransitionCapture", "VelocityLearner", "VelocityWorldModel", "VelocityPolicy"}
        ), path
        if path.is_relative_to(task):
            assert not any(name.startswith(("isaaclab_rl", "physicsnemo", "wandb")) for name in imports), path
            assert "step" not in symbols and path.name not in ("train.py", "learner.py"), path
        elif path.is_relative_to(learning):
            assert path.name not in ("train.py", "evaluate.py", "sim.py", "robot.py"), path
            assert not any(name.startswith(("argparse", "gymnasium", "wandb", "isaaclab.envs")) for name in imports), (
                path
            )


def test_native_task_registration_and_optional_capture():
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, agent_cfg = resolve_task_config(
        "IsaacContrib-Motion-Tracking-G1", "supermimic_cfg_entry_point", overrides=[]
    )
    assert env_cfg.recorders.transition is None
    assert type(env_cfg).__module__.startswith("isaaclab_tasks.")
    assert type(agent_cfg).__module__.startswith("isaaclab_rl.")


def test_replay_ring_masks_boundaries_and_owns_samples():
    transition = {
        "s": torch.zeros(1, 71),
        "s1": torch.zeros(1, 71),
        "a": torch.zeros(1, 29),
        "a_prev": torch.zeros(1, 29),
        "xi": torch.zeros(1, 33),
        "r": torch.zeros(1),
        "f": torch.zeros(1, dtype=torch.long),
        "ep_step": torch.zeros(1, dtype=torch.long),
        "done": torch.zeros(1, dtype=torch.bool),
    }
    ring = ReplayBuffer(4, transition)
    assert "r" not in ring.data  # This learner evaluates analytical rewards; capture still supplies them to others.
    for i in range(6):
        transition["s"].fill_(i)
        transition["s1"].fill_(i + 1)
        transition["f"].fill_(i)
        transition["ep_step"].fill_(i)
        transition["done"].fill_(i == 3)
        ring.add(transition)
    transition["s"].fill_(99)
    window = ring.sample_windows(4, 4)
    torch.testing.assert_close(window["s"][..., 0], torch.tensor([[2.0, 3.0, 4.0, 5.0]]).expand(4, -1))
    torch.testing.assert_close(window["mask"], torch.tensor([[1.0, 1.0, 0.0, 0.0]]).expand(4, -1))
    # A state intervention or manual reset is a boundary even when no sampled done flag was set.
    ring.data["done"].zero_()
    ring.data["s"][ring._row(1), 0, 0] = -1
    torch.testing.assert_close(ring.sample_windows(4, 4)["mask"], torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(4, -1))
    ring.data["ep_step"].zero_()
    start, history = ring.sample_starts(32, 10)
    torch.testing.assert_close(history["s"], start["s"][:, None].expand(-1, 10, -1))


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_joint_learning_routes_gradients_and_seeds_by_interface(tmp_path, device):
    """Both actors/critics learn, while velocity gradients cannot update the decoder or motion encoder."""
    from types import SimpleNamespace

    from isaaclab_rl.supermimic.learner import Learner
    from isaaclab_rl.supermimic.networks import WorldModel
    from isaaclab_rl.supermimic.supermimic_cfg import SuperMimicCfg

    from isaaclab_tasks.contrib.g1_tracking.motion import MotionLib
    from isaaclab_tasks.contrib.g1_tracking.tracking import TrackingObjective, VelocityCfg

    torch.manual_seed(13)
    raw = np.zeros((32, 36), dtype=np.float32)
    raw[:, 2], raw[:, 6] = 0.8, 1.0
    np.save(tmp_path / "standing.npy", raw)
    # Geometry is independent of routing: use a rigid set of 14 co-located tracked bodies.
    kin = SimpleNamespace(
        q_lo=torch.full((29,), -2.0, device=device),
        q_hi=torch.full((29,), 2.0, device=device),
        forward=lambda p, R, v, w, q, qd: (
            p.unsqueeze(-2).expand(*p.shape[:-1], 14, 3),
            R.unsqueeze(-3).expand(*R.shape[:-2], 14, 3, 3),
            v.unsqueeze(-2).expand(*v.shape[:-1], 14, 3),
            w.unsqueeze(-2).expand(*w.shape[:-1], 14, 3),
        ),
    )
    motion = MotionLib(str(tmp_path), kin, device, src_fps=50)
    objective = TrackingObjective(
        kin, motion, torch.zeros(29, device=device), velocity_cfg=VelocityCfg(fraction=0.5, resampling_steps=2)
    )
    state = torch.zeros(8, 71, device=device)
    state[:, 2], state[:, 6] = 0.8, 1.0
    transition = {
        "s": state,
        "s1": state.clone(),
        "a": torch.zeros(8, 29, device=device),
        "a_prev": torch.zeros(8, 29, device=device),
        "xi": torch.zeros(8, 33, device=device),
        "f": torch.zeros(8, dtype=torch.long, device=device),
        "ep_step": torch.zeros(8, dtype=torch.long, device=device),
        "done": torch.zeros(8, dtype=torch.bool, device=device),
        "is_velocity": torch.arange(8, device=device) >= 4,
        "velocity_command": torch.full((8, 3), 0.5, device=device),
        "velocity_reference": torch.ones(8, dtype=torch.bool, device=device),
    }
    transition["xi"][:, 0] = 1
    transition["s"][:, 0] = torch.arange(8, device=device)  # Identifies each world's replay seeds.
    transition["s1"].copy_(transition["s"])
    cfg = SuperMimicCfg(
        compile=False,
        amp="none",
        horizon=2,
        policy_batch_size=8,
        buffer_steps=8,
        critic_epochs=1,
        critic_minibatches=1,
        velocity_alignment_weight=0.0,
    )
    world = WorldModel(hidden=(32,)).to(device)
    torch.nn.init.normal_(world.net[-1].weight, std=0.01)
    learner = Learner(
        cfg, objective, transition, ((slice(None), None, None),), 0.02, world, goal_frames=2, history=3, gamma=0.99
    )
    for frame in range(4):
        transition["f"].fill_(frame)
        transition["ep_step"].fill_(frame)
        learner.observe(transition)
    for velocity in (False, True):
        starts, _ = learner.buffer.sample_starts(64, 3, velocity)
        assert ((starts["s"][:, 0] >= 4) == velocity).all()
    models = (learner.pi.enc, learner.pi.dec, learner.pi.velocity_enc, learner.critic, learner.velocity_critic, world)
    before = [[p.detach().clone() for p in model.parameters()] for model in models]
    counts = [0, 0, 0]

    def count(index):
        def hook(gradient):
            assert gradient.isfinite().all() and gradient.abs().sum() > 0
            counts[index] += 1

        return hook

    handles = [model[-1].weight.register_hook(count(i)) for i, model in enumerate(models[:3])]
    result = learner.policy_update()
    assert counts == [1, 1, 1]  # Return gradients reach the velocity encoder without an auxiliary loss.
    cfg.velocity_alignment_weight = 0.1
    result.update(learner.policy_update())
    for handle in handles:
        handle.remove()
    assert counts == [2, 2, 2]  # Neither velocity return nor alignment may update shared-model weights.
    assert all(torch.isfinite(value) for value in result.values())
    for model, initial in zip(models[:-1], before[:-1], strict=True):
        assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), initial, strict=True))
    for parameter, initial in zip(world.parameters(), before[-1], strict=True):
        torch.testing.assert_close(parameter, initial, rtol=0, atol=0)


def test_evaluation_batches_cover_last_partial_batch_and_failed_tails():
    from types import SimpleNamespace

    from isaaclab_rl.entrypoints.backends.play_supermimic import evaluate

    lengths = torch.tensor([3, 4, 2, 5, 3, 4, 2])
    motion = SimpleNamespace(names=[f"clip{i}" for i in range(7)], clip_len=lengths)
    command = SimpleNamespace(motion=motion, cfg=SimpleNamespace(evaluation_clip_offset=0))
    capture = SimpleNamespace(latest={})

    class Env:
        num_envs, device, step_dt = 3, "cpu", 0.02
        command_manager = SimpleNamespace(get_term=lambda name: command)
        recorder_manager = SimpleNamespace(get_term=lambda name: capture)
        reward_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(weight=1.0))

        @property
        def unwrapped(self):
            return self

        def reset(self):
            self.ids = (torch.arange(3) + command.cfg.evaluation_clip_offset) % 7
            self.elapsed = 0
            return {"policy": None, "goal": None}, {}

        def step(self, action):
            self.elapsed += 1
            term = (self.ids == 3) & (self.elapsed == 1)
            capture.latest = {
                "r": (self.ids + 1) * 0.01,
                "done": term | (self.elapsed >= lengths[self.ids] - 1),
                "term": term,
            }
            return {"policy": None, "goal": None}, None, None, None, None

    result = evaluate(Env(), lambda *args: None)
    assert result["failed"] == ["clip3"]
    assert list(result["per_clip_steps"].values()) == [2, 3, 1, 1, 2, 3, 1]
    np.testing.assert_allclose(list(result["per_clip"].values()), [0.1, 0.2, 0.3, 0.1, 0.5, 0.6, 0.7], rtol=1e-6)


def test_training_root_rejects_different_analytical_task_semantics(tmp_path, monkeypatch):
    from isaaclab.envs import mdp

    from isaaclab_rl.entrypoints.backends import train_supermimic as train
    from isaaclab_rl.supermimic.supermimic_cfg import SuperMimicCfg

    from isaaclab_tasks.contrib.g1_tracking.g1_tracking_env_cfg import G1TrackingEnvCfg

    class ReachedLauncher(Exception):
        pass

    def launch(*args):
        raise ReachedLauncher

    monkeypatch.setattr(train, "launch_simulation", launch)
    invalid = (
        (("policy", "joint_vel", "func"), mdp.joint_pos),  # Same width, different physical quantity.
        (("policy", "joint_vel", "params"), {"asset_cfg": "different_asset"}),
        (("policy", "joint_vel", "delay_max_lag"), 1),
        (("policy", "flatten_history_dim"), True),
        (("goal", "reference", "scale"), 2.0),
        (("interface", "command", "scale"), 2.0),
    )
    argv = ["--task", "IsaacContrib-Motion-Tracking-G1", "--log_dir", str(tmp_path)]
    for path, value in invalid:
        cfg = G1TrackingEnvCfg()
        target = cfg.observations
        for name in path[:-1]:
            target = getattr(target, name)
        setattr(target, path[-1], value)
        monkeypatch.setattr(train, "resolve_task_config", lambda *args, **kwargs: (cfg, SuperMimicCfg()))
        with pytest.raises(ValueError, match="declared functions and layout"):
            train.run(argv)
    cfg = G1TrackingEnvCfg()
    term = cfg.observations.policy.__dict__.pop("joint_vel")
    cfg.observations.policy.joint_vel = term
    monkeypatch.setattr(train, "resolve_task_config", lambda *args, **kwargs: (cfg, SuperMimicCfg()))
    with pytest.raises(ValueError, match="declared functions and layout"):
        train.run(argv)
    for name, attribute, value in (
        ("failed", "func", mdp.time_out),
        ("failed", "time_out", True),
        ("clip_end", None, None),
    ):
        cfg = G1TrackingEnvCfg()
        if attribute is None:
            setattr(cfg.terminations, name, value)
        else:
            setattr(getattr(cfg.terminations, name), attribute, value)
        monkeypatch.setattr(train, "resolve_task_config", lambda *args, **kwargs: (cfg, SuperMimicCfg()))
        with pytest.raises(ValueError, match="Analytical termination"):
            train.run(argv)
    cfg = G1TrackingEnvCfg()
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.joint_vel.noise.n_max = 0.25
    cfg.commands.motion.objective.term_height = 0.3
    cfg.episode_length_s = 4.0
    monkeypatch.setattr(train, "resolve_task_config", lambda *args, **kwargs: (cfg, SuperMimicCfg()))
    with pytest.raises(ReachedLauncher):
        train.run(argv)  # Supported task parameters reach construction without a simulator.
    cfg.observations.policy.history_length = 0
    with pytest.raises(ValueError, match="must be positive"):
        train.run(argv)
