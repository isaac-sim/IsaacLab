# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differential contracts against a pinned, independently loaded reference checkout.

Set ISAACLAB_SUPERMIMIC_REFERENCE and ISAACLAB_MJLAB_REFERENCE to source checkouts.
Only pure tensor modules are imported; no replacement mathematical implementation
or simulator mock supplies the expected results.
"""

import importlib
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_rl.model_based import (
    bptt,
    models,
    quaternion,
    replay_buffer,
    representations,
    reset_sampling,
    world_model_learner,
)
from isaaclab_rl.model_based.config import REFERENCE_REVISION
from isaaclab_rl.model_based.rollout import MultiTaskWorldModelEnv
from isaaclab_rl.model_based.runner import Ratio

from isaaclab_tasks.core.motion_tracking.config.g1.model_based_env_cfg import G1ModelBasedEnvCfg
from isaaclab_tasks.core.motion_tracking.mdp.model_based import rewards
from isaaclab_tasks.core.motion_tracking.mdp.model_based.goals import TasksCfg, resolve_task_specs
from isaaclab_tasks.core.motion_tracking.mdp.model_based.reference import MotionReference


@pytest.fixture(scope="module")
def reference():
    source_path = os.environ.get("ISAACLAB_SUPERMIMIC_REFERENCE")
    math_path = os.environ.get("ISAACLAB_MJLAB_REFERENCE")
    if source_path is None or math_path is None:
        pytest.skip("Set source-checkout paths to run reference differential tests.")
    source, math_source = Path(source_path), Path(math_path)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    assert revision == REFERENCE_REVISION
    previous = set(sys.modules)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(source / "src"))
        # Namespace-only packages bypass simulator registration; numerical modules are unmodified.
        packages = {
            "supermimic.tasks": source / "src/supermimic/tasks",
            "mjlab": math_source / "src/mjlab",
            "mjlab.utils": math_source / "src/mjlab/utils",
            "mjlab.utils.lab_api": math_source / "src/mjlab/utils/lab_api",
        }
        for name, path in packages.items():
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            patch.setitem(sys.modules, name, package)
        result = types.SimpleNamespace()
        for name, module in (
            ("models", "models"),
            ("representation", "representations"),
            ("wm", "learners.wm_learner"),
            ("policy", "learners.policy_learner"),
            ("buffer", "replay_buffer"),
            ("sampler", "runners.reset_sampling"),
            ("reward", "utils.tracking_reward"),
        ):
            setattr(result, name, importlib.import_module("supermimic." + module))
        result.math = importlib.import_module("mjlab.utils.lab_api.math")
        yield result
    for name in set(sys.modules) - previous:
        if name.startswith(("supermimic", "mjlab")):
            sys.modules.pop(name, None)


def test_quaternion_adapter_values_and_gradients(reference):
    """WXYZ/XYZW conversion must preserve rotations and their action-gradient path."""
    torch.manual_seed(91)
    q = torch.nn.functional.normalize(torch.randn(7, 4, dtype=torch.float64), dim=-1)
    v = torch.randn(7, 3, dtype=torch.float64)
    for operation in ("matrix_from_quat", "quat_apply", "quat_apply_inverse", "quat_mul", "yaw_quat", "quat_inv"):
        outputs, gradients = [], []
        for module in (reference.math, quaternion):
            value = q.clone().requires_grad_()
            args = (
                (value, v)
                if operation in ("quat_apply", "quat_apply_inverse")
                else (value, q)
                if operation == "quat_mul"
                else (value,)
            )
            output = getattr(module, operation)(*args)
            outputs.append(output)
            gradients.append(torch.autograd.grad(output.square().sum(), value)[0])
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("kind", ["gencoord", "maxcoord"])
def test_dynamics_loss_and_action_gradient_match_reference(reference, kind):
    """Protect integration, the seven-transition loss, and differentiation through dynamics."""
    torch.manual_seed(12)
    num_joints, num_bodies, batch, frames = 3, 4, 5, 8
    if kind == "gencoord":
        full = torch.randn(batch, frames, 13 + 2 * num_joints) * 0.1
        full[..., 3:7] = torch.nn.functional.normalize(torch.randn(batch, frames, 4), dim=-1)
    else:
        full = torch.randn(batch, frames, num_bodies * 13) * 0.1
        full[..., num_bodies * 3 : num_bodies * 7] = torch.nn.functional.normalize(
            torch.randn(batch, frames, num_bodies, 4), dim=-1
        ).flatten(-2)
    reps = [
        module.create_representation(kind, full.shape[-1], num_joints, 0.02, 4, torch.zeros(num_joints), "rep_state")
        for module in (reference.representation, representations)
    ]
    dynamics = [
        module.Dynamics(rep, num_joints, 4, torch.ones(num_joints), torch.zeros(num_joints), (16, 16, 16))
        for module, rep in zip((reference.models, models), reps, strict=True)
    ]
    dynamics[1].load_state_dict(dynamics[0].state_dict())
    actions = torch.randn(batch, frames, num_joints)
    context = torch.randn(batch, 4)
    outputs, gradients = [], []
    for module, model, rep in zip((reference.wm, world_model_learner), dynamics, reps, strict=True):
        learner = module.WorldModelLearner(model, rep, module.WorldModelLearnerCfg())
        action = actions.clone().requires_grad_()
        represented, aux = rep.represent(full), rep.extract_aux_state(full)
        prediction, pred_aux = learner._rollout(represented[:, 0], action, aux[:, 0], context)
        loss, mse = learner._compute_losses(prediction, pred_aux, represented[:, 1:], aux[:, 1:])
        outputs.append((prediction, pred_aux, loss, mse))
        gradients.append(torch.autograd.grad(loss, action)[0])
        assert prediction.shape[1] == 7
        assert gradients[-1][:, -1].count_nonzero() == 0
        assert gradients[-1][:, :-1].abs().sum() > 0
    for expected, actual in zip(*outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(gradients[1], gradients[0], rtol=5e-5, atol=2e-5)


def test_actor_and_gae_match_reference(reference):
    """Protect encoder routing, bounded latents, history order, and soft-done GAE."""
    torch.manual_seed(29)
    specs = [types.SimpleNamespace(name="motion", goal_dim=8), types.SimpleNamespace(name="velocity", goal_dim=3)]
    actors = [
        module.MultiEncoderAE(7, 3, specs, {"motion": 0.5, "velocity": 0.5}, 4, (16, 16, 8), 0.1, 2)
        for module in (reference.models, models)
    ]
    actors[1].load_state_dict(actors[0].state_dict())
    obs = TensorDict(
        {"history": torch.randn(6, 2, 7), "goal_motion": torch.randn(6, 8), "goal_velocity": torch.randn(6, 3)},
        batch_size=[6],
    )
    outputs, gradients = [], []
    for actor in actors:
        actor.update_normalization(obs)
        output = actor(obs)
        loss = output.square().mean() + actor.auxiliary_loss().mean()
        outputs.append(output)
        gradients.append(torch.autograd.grad(loss, tuple(actor.parameters())))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for expected, actual in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    args = (torch.randn(5, 6, 1), torch.randn(6, 1), torch.rand(5, 6, 1), torch.rand(5, 6, 1), 0.99, 0.95)
    torch.testing.assert_close(bptt.gae_returns(*args), reference.policy.gae_returns(*args), rtol=0, atol=0)


def test_replay_and_return_sampling_match_reference(reference):
    """Protect episode boundaries, circular-buffer age, history padding, and curriculum updates."""
    buffers = []
    for module in (reference.buffer, replay_buffer):
        buffer = module.ReplayBuffer(
            {
                "state": module.FieldSpec(1),
                "terminated": module.FieldSpec(dtype=torch.bool),
                "truncated": module.FieldSpec(dtype=torch.bool),
            },
            8,
            3,
        )
        for step in range(12):
            buffer.insert(
                TensorDict(
                    {
                        "state": torch.full((3, 1), float(step)),
                        "terminated": torch.tensor([step % 3 == 0, False, False]),
                        "truncated": torch.tensor([False, step % 4 == 0, False]),
                    },
                    batch_size=[3],
                )
            )
        buffers.append(buffer)
    for causal in (False, True):
        results = []
        for buffer in buffers:
            torch.manual_seed(3)
            results.append(buffer.sample_causal(4, 32) if causal else (buffer.sample(3, 32), None))
        for key in results[0][0].keys():
            torch.testing.assert_close(results[0][0][key], results[1][0][key], rtol=0, atol=0)
        if causal:
            torch.testing.assert_close(results[0][1], results[1][1])
        else:
            assert not (results[1][0]["terminated"][:, :-1] | results[1][0]["truncated"][:, :-1]).any()
    samplers = [
        module.ReturnSampler(module.ReturnSamplerCfg(horizon=3), 4, 20, 3, 0.1)
        for module in (reference.sampler, reset_sampling)
    ]
    for step in range(12):
        args = (
            torch.tensor([step, step + 1, step + 2]),
            torch.tensor([0.01, 0.05, 0.1]),
            torch.tensor([step % 3 == 0, False, False]),
            torch.zeros(3, dtype=torch.bool),
            torch.tensor([False, step % 4 == 0, False]),
        )
        expected, actual = [sampler.record(*args) for sampler in samplers]
        if actual is not None:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    schedule = Ratio(0.4)
    assert sum(schedule(step) for step in range(40000) if step >= 0) == 16000


def test_tracking_reward_value_and_gradient_match_reference(reference):
    """Protect the differentiable reward, including root re-anchoring and penalties."""
    torch.manual_seed(7)
    q = torch.nn.functional.normalize(torch.randn(4, 3, 4), dim=-1)
    pos = torch.randn(4, 3, 3)
    target_pos = pos + torch.randn_like(pos) * 0.05
    velocity = torch.randn_like(pos)
    joint = torch.randn(4, 6)
    action = torch.randn(4, 3)
    results = []
    for module in (reference.reward, rewards):
        predicted = pos.clone().requires_grad_()
        reward = module.tracking_reward(
            torch.cat((predicted[:, 0], q[:, 0]), -1),
            predicted,
            quaternion.matrix_from_quat(q),
            velocity,
            velocity,
            joint,
            action,
            action * 0.9,
            target_pos,
            q,
            velocity * 0.95,
            velocity * 1.05,
            torch.tensor([[-1.0, 1.0]]).expand(3, 2),
            0.02,
        )
        results.append((reward, torch.autograd.grad(reward.sum(), predicted)[0]))
    for expected, actual in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_actor_critic_optimizer_step_matches_reference(reference, device):
    """One real BPTT/critic update must match the upstream weights, not only loss formulas."""
    data = os.environ.get("ISAACLAB_G1_TRACKING_DATA")
    if data is None:
        pytest.skip("Set ISAACLAB_G1_TRACKING_DATA for the G1 kinematic reference.")
    motion = MotionReference(
        types.SimpleNamespace(
            joint_names=G1ModelBasedEnvCfg().commands.motion.joint_names,
            tracked_bodies=G1ModelBasedEnvCfg().commands.motion.tracked_bodies,
            source_fps=30,
            motion_dir=str(Path(data) / "lafan1_g1"),
            kinematics_path=str(Path(data) / "g1_description/g1_29dof.xml"),
        ),
        device,
    )
    batch, frames = 4, motion.motion_starts[:4]
    full = torch.cat(
        (
            motion.root_pos_w[frames],
            motion.root_quat_w[frames],
            motion.joint_pos[frames],
            motion.root_lin_vel_w[frames],
            motion.root_ang_vel_w[frames],
            motion.joint_vel[frames],
        ),
        -1,
    )
    start = TensorDict(
        {
            "state": full,
            "prev_action": torch.zeros(batch, 29, device=device),
            "motion_time": frames,
            "sim_ctx": torch.zeros(batch, 33, device=device),
        },
        batch_size=[batch],
        device=device,
    )
    original_rollout = importlib.import_module("supermimic.envs.multi_task_world_model_env").MultiTaskWorldModelEnv
    learners = []
    for network_module, rep_module, policy_module, rollout_class in (
        (reference.models, reference.representation, reference.policy, original_rollout),
        (models, representations, bptt, MultiTaskWorldModelEnv),
    ):
        torch.manual_seed(55)
        rep = rep_module.create_representation(
            "gencoord", 71, 29, 0.02, 4, torch.zeros(29, device=device), "rep_state"
        ).to(device)
        rep.fk = motion.fk
        rep._reward_fn = rewards.tracking_reward
        rep._soft_joint_pos_limits = torch.stack((motion.fk.kinematics.q_lo, motion.fk.kinematics.q_hi), -1)
        dynamics = network_module.Dynamics(
            rep, 29, 33, torch.ones(29, device=device), torch.zeros(29, device=device), (16, 16, 16)
        ).to(device)
        specs = resolve_task_specs(TasksCfg(), rep, 670, device)
        actor = network_module.MultiEncoderAE(
            68, 29, specs, {"motion": 1.0, "velocity": 0.0}, 4, (16, 16, 16), 0.1, 1
        ).to(device)
        critic = network_module.build_mlp(68 + 670 + 33, (16, 16), 1).to(device)
        kwargs = dict(
            dynamics=dynamics,
            representation=rep,
            batch_size=batch,
            replay_buffer=None,
            task_specs=specs[:1],
            history_length=1,
            termination_softness=0.05,
            device=device,
        )
        if rollout_class is original_rollout:
            # Configuration/asset bindings only; all transition, reward, and gradient logic is upstream code.
            command = types.SimpleNamespace(motion=motion, motion_anchor_body_index=0)
            env = types.SimpleNamespace(
                num_actions=29,
                step_dt=0.02,
                cfg=types.SimpleNamespace(
                    rewards={"tracking": types.SimpleNamespace(weight=5)},
                    terminations={
                        "anchor_pos": types.SimpleNamespace(params={"threshold": 0.25}),
                        "anchor_ori": types.SimpleNamespace(params={"threshold": 0.8}),
                    },
                ),
                command_manager=types.SimpleNamespace(get_term=lambda name: command),
            )
            env.unwrapped = env
            kwargs["env"] = env
        else:
            kwargs.update(
                motion=motion, num_actions=29, max_step_reward=0.1, anchor_pos_threshold=0.25, anchor_ori_threshold=0.8
            )
        rollout = rollout_class(**kwargs)
        learners.append(
            policy_module.ActorCriticPolicyLearner(
                actor, critic, rollout, policy_module.ActorCriticPolicyLearnerCfg(horizon=4, batch_size=batch), device
            )
        )
    outputs = []
    for learner in learners:
        learner.wm_env.dynamics.requires_grad_(False)
        torch.manual_seed(81)
        outputs.append(learner._train_batch(start.clone()))
    for key in outputs[0]:
        torch.testing.assert_close(outputs[1][key], outputs[0][key], rtol=2e-4, atol=2e-6)
    for name in ("actor", "critic"):
        for key, expected in getattr(learners[0], name).state_dict().items():
            torch.testing.assert_close(getattr(learners[1], name).state_dict()[key], expected, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_adam_schedule_survives_cuda_graph_replay(device):
    """Captured Adam must update every layer and read changing rates like eager Adam."""
    import copy

    from isaaclab_rl.model_based.optim import CapturedUpdate, build_optimizers, graph_state

    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)
    ).to(device)
    expected = copy.deepcopy(model)
    optimizers = build_optimizers([model], 2e-4, device=device, optimizer="adam")
    eager = torch.optim.Adam(expected.parameters(), lr=2e-4, fused=True)
    inputs = TensorDict({"x": torch.randn(5, 4, device=device), "y": torch.randn(5, 2, device=device)}, batch_size=[5])

    def update(batch):
        optimizers[0].zero_grad(set_to_none=True)
        loss = (model(batch["x"]) - batch["y"]).square().mean()
        loss.backward()
        optimizers[0].step()
        return {"loss": loss.detach()}

    captured = CapturedUpdate(update, lambda: graph_state([model], optimizers), warmup=1)
    for rate in (2e-4, 1.5e-4, 1e-4, 6e-6, 6e-6):
        optimizers[0].param_groups[0]["lr"].fill_(rate)
        eager.param_groups[0]["lr"] = rate
        captured(inputs)
        eager.zero_grad(set_to_none=True)
        (expected(inputs["x"]) - inputs["y"]).square().mean().backward()
        eager.step()
        for actual, reference_parameter in zip(model.parameters(), expected.parameters(), strict=True):
            torch.testing.assert_close(actual, reference_parameter, rtol=1e-5, atol=1e-6)
