# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU contracts for navigation distributions, reflection and native mirror-loss gradients."""

import math

import pytest
import torch
from rsl_rl.extensions import Symmetry
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import BetaDistribution
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaaclab_tasks.contrib.microduck_obstacle.agents.navigation_distribution import MicroDuckBrakingBetaDistribution
from isaaclab_tasks.contrib.microduck_obstacle.mdp.symmetry import compute_symmetric_states

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES


@pytest.mark.parametrize("history", [1, 10])
def test_navigation_reflection_preserves_physical_feature_and_history_contract(history):
    """Reflect named joints and vectors, without reversing ToF rows or time."""
    policy = torch.arange(1, 107, dtype=torch.float32).reshape(2, 53)
    critic = torch.arange(101, 247, dtype=torch.float32).reshape(2, 73)
    planes = torch.arange(2 * history * 128, dtype=torch.float32).reshape(2, history, 2, 8, 8)
    planes[:, :, 1] = (planes[:, :, 1] % 3 == 0).float()
    ages = torch.arange(2 * history, dtype=torch.float32).reshape(2, history) / 100.0
    obs = TensorDict(
        {"policy": policy, "critic": critic, "tof": torch.cat((planes.flatten(1), ages), dim=-1)}, batch_size=[2]
    )
    actions = torch.tensor([[0.2, 0.3, -0.4], [-0.8, -0.1, 0.7]])
    original = obs.clone()
    original_actions = actions.clone()
    augmented, augmented_actions = compute_symmetric_states(None, obs, actions)
    assert augmented.batch_size == (4,)
    for group in obs.keys():
        torch.testing.assert_close(obs[group], original[group])
        torch.testing.assert_close(augmented[group][:2], original[group])
    torch.testing.assert_close(actions, original_actions)
    torch.testing.assert_close(augmented_actions[:2], actions)
    torch.testing.assert_close(augmented_actions[2:], torch.tensor([[0.2, -0.3, 0.4], [-0.8, 0.1, -0.7]]))

    mirrored = augmented[2:]
    torch.testing.assert_close(mirrored["policy"][:, :6], policy[:, :6] * torch.tensor([-1, 1, -1, 1, -1, 1]))
    torch.testing.assert_close(mirrored["critic"][:, :9], critic[:, :9] * torch.tensor([1, -1, 1, -1, 1, -1, 1, -1, 1]))
    for group, starts in (("policy", (6, 20)), ("critic", (9, 23))):
        for start in starts:
            for index, name in enumerate(MICRODUCK_JOINT_NAMES):
                partner = (
                    name.replace("left_", "right_") if name.startswith("left_") else name.replace("right_", "left_")
                )
                sign = 1 if name in ("neck_pitch", "head_pitch") else -1
                expected = original[group][:, start + MICRODUCK_JOINT_NAMES.index(partner)] * sign
                torch.testing.assert_close(mirrored[group][:, start + index], expected)
    for group, starts in (("policy", (34, 37, 50)), ("critic", (37, 40))):
        for start in starts:
            torch.testing.assert_close(
                mirrored[group][:, start : start + 3], original[group][:, start : start + 3] * torch.tensor([1, -1, -1])
            )
    for group, head, body in (("policy", 40, 44), ("critic", 55, 59)):
        torch.testing.assert_close(
            mirrored[group][:, head : head + 4], original[group][:, head : head + 4] * torch.tensor([1, 1, -1, -1])
        )
        torch.testing.assert_close(
            mirrored[group][:, body : body + 6],
            original[group][:, body : body + 6] * torch.tensor([1, -1, 1, -1, 1, -1]),
        )
    for start in (43, 45, 47):
        torch.testing.assert_close(mirrored["critic"][:, start], critic[:, start + 1])
        torch.testing.assert_close(mirrored["critic"][:, start + 1], critic[:, start])
    torch.testing.assert_close(mirrored["critic"][:, 49:52], critic[:, 52:55] * torch.tensor([1, -1, 1]))
    torch.testing.assert_close(mirrored["critic"][:, 52:55], critic[:, 49:52] * torch.tensor([1, -1, 1]))
    torch.testing.assert_close(mirrored["critic"][:, 65:], critic[:, 65:])

    reflected_planes = mirrored["tof"][:, : history * 128].reshape(2, history, 2, 8, 8)
    for column in range(8):
        torch.testing.assert_close(reflected_planes[..., column], planes[..., 7 - column])
    torch.testing.assert_close(mirrored["tof"][:, history * 128 :], ages)
    twice, twice_actions = compute_symmetric_states(None, mirrored, augmented_actions[2:])
    for group in original.keys():
        torch.testing.assert_close(twice[group][2:], original[group])
    torch.testing.assert_close(twice_actions[2:], original_actions)

    obs_only, no_actions = compute_symmetric_states(None, obs=obs)
    no_obs, actions_only = compute_symmetric_states(None, actions=actions)
    assert no_actions is None and no_obs is None
    torch.testing.assert_close(obs_only["tof"], augmented["tof"])
    torch.testing.assert_close(actions_only, augmented_actions)
    assert compute_symmetric_states(None) == (None, None)


@pytest.mark.parametrize("forward_braking_only", [False, True])
def test_native_mirror_loss_keeps_ppo_batch_and_backpropagates_through_braking_actor(forward_braking_only, tmp_path):
    """Native mirror-only loss updates the actor; saved config preserves deployment support."""
    torch.manual_seed(19)
    count = 4
    obs = TensorDict({"policy": torch.randn(count, 53), "tof": torch.randn(count, 258)}, batch_size=[count])
    actor_cfg = {
        "hidden_dims": (8,),
        "activation": "tanh",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": (
                "isaaclab_tasks.contrib.microduck_obstacle.agents.navigation_distribution:MicroDuckBrakingBetaDistribution"
            )
        },
    }
    if forward_braking_only:
        actor_cfg["distribution_cfg"]["forward_braking_only"] = True
    actor = MLPModel(obs, {"actor": ["policy", "tof"]}, "actor", 3, **actor_cfg)
    # Asymmetric statistics make reflecting normalized coordinates observably wrong.
    calibration = TensorDict(
        {group: values * 2.0 + torch.arange(values.shape[-1]) / 20.0 for group, values in obs.items()},
        batch_size=[count],
    )
    actor.update_normalization(calibration)
    actor.eval()
    normalization_before = {name: value.clone() for name, value in actor.obs_normalizer.state_dict().items()}
    batch = RolloutStorage.Batch(
        observations=obs.clone(),
        actions=torch.randn(count, 3),
        values=torch.randn(count, 1),
        advantages=torch.randn(count, 1),
        returns=torch.randn(count, 1),
        old_actions_log_prob=torch.randn(count, 1),
    )
    before = {
        name: value.clone() for name, value in vars(batch).items() if isinstance(value, (torch.Tensor, TensorDict))
    }
    symmetry = Symmetry(
        env=None,
        data_augmentation_func="isaaclab_tasks.contrib.microduck_obstacle.mdp.symmetry:compute_symmetric_states",
        use_data_augmentation=False,
        use_mirror_loss=True,
        mirror_loss_coeff=0.1,
    )
    symmetry.augment_batch(batch, count)
    for name, value in before.items():
        if name == "observations":
            assert batch.observations.batch_size == (count,)
            for group in obs.keys():
                torch.testing.assert_close(batch.observations[group], value[group])
        else:
            torch.testing.assert_close(getattr(batch, name), value)

    outputs = []
    raw_normalizer_inputs = []

    def retain_outputs(module, inputs, output):
        output.retain_grad()
        outputs.append(output)

    actor_hook = actor.register_forward_hook(retain_outputs)
    normalizer_hook = actor.obs_normalizer.register_forward_pre_hook(
        lambda module, inputs: raw_normalizer_inputs.append(inputs[0].detach().clone())
    )
    loss = symmetry.compute_loss(actor, batch, count)
    actor_hook.remove()
    normalizer_hook.remove()
    assert torch.isfinite(loss) and loss > 0
    assert batch.observations.batch_size == (2 * count,) and batch.actions.shape == (count, 3)
    torch.testing.assert_close(raw_normalizer_inputs[0][:count], torch.cat((obs["policy"], obs["tof"]), dim=-1))
    torch.testing.assert_close(raw_normalizer_inputs[0][count:, 0], -obs["policy"][:, 0])
    torch.testing.assert_close(raw_normalizer_inputs[0][count:, 53:61], obs["tof"][:, :8].flip(-1))
    torch.testing.assert_close(raw_normalizer_inputs[0][count:, -2:], obs["tof"][:, -2:])
    assert (outputs[0][:, 0] > -1.0).all()  # Exercise the ordinary differentiable Beta means.
    (symmetry.mirror_loss_coeff * loss).backward()
    assert (outputs[0].grad[:count] == 0.0).all()
    assert outputs[0].grad[count:].abs().sum() > 0
    gradients = [parameter.grad for parameter in actor.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0
    for name, value in normalization_before.items():
        torch.testing.assert_close(actor.obs_normalizer.state_dict()[name], value)

    # Weight-only checkpoints need the saved constructor config to preserve the opt-in support.
    with torch.no_grad():
        actor.mlp[-2].weight.zero_()
        actor.mlp[-2].bias.fill_(math.log(math.expm1(1.0)))  # All three heads are Beta(2, 2).
    checkpoint_path = tmp_path / "actor.pt"
    torch.save({"actor_cfg": actor_cfg, "state_dict": actor.state_dict()}, checkpoint_path)
    saved = torch.load(checkpoint_path, weights_only=True)
    restored = MLPModel(obs, {"actor": ["policy", "tof"]}, "actor", 3, **saved["actor_cfg"])
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored.eval()
    expected = torch.zeros(count, 3)
    if forward_braking_only:
        expected[:, 0] = -0.5
    torch.testing.assert_close(restored(obs), expected)
    exported_path = tmp_path / "actor_jit.pt"
    torch.jit.script(restored.as_jit()).save(str(exported_path))
    exported = torch.jit.load(str(exported_path))
    torch.testing.assert_close(exported(torch.cat((obs["policy"], obs["tof"]), dim=-1)), expected)


def test_forward_braking_beta_matches_analytic_density_moments_and_legacy_defaults():
    """Protect affine density, moments, native entropy convention and existing failure guards."""
    # Beta(2, 3) has mean 2/5, std 1/5, density 12*u*(1-u)^2 and entropy 9/4-log(12).
    logits = torch.tensor(
        [[[math.log(math.expm1(1.0))] * 3, [math.log(math.expm1(2.0))] * 3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    native = BetaDistribution(3)
    default = MicroDuckBrakingBetaDistribution(3)
    explicit_default = MicroDuckBrakingBetaDistribution(3, forward_braking_only=False)
    braking = MicroDuckBrakingBetaDistribution(3, forward_braking_only=True)
    for distribution in (native, default, explicit_default, braking):
        distribution.update(logits)
    probe = torch.tensor([[-0.75, -0.4, 0.5]], dtype=torch.float64)
    untouched = probe.clone()
    for distribution in (default, explicit_default):
        for attribute in ("mean", "std", "entropy"):
            torch.testing.assert_close(getattr(distribution, attribute), getattr(native, attribute), rtol=0, atol=0)
        torch.testing.assert_close(distribution.log_prob(probe), native.log_prob(probe), rtol=0, atol=0)
        torch.testing.assert_close(distribution.deterministic_output(logits), native.deterministic_output(logits))
        torch.manual_seed(23)
        samples = distribution.sample()
        torch.manual_seed(23)
        torch.testing.assert_close(samples, native.sample(), rtol=0, atol=0)
    torch.testing.assert_close(braking.mean, torch.tensor([[-0.6, -0.2, -0.2]], dtype=torch.float64))
    torch.testing.assert_close(braking.std, torch.tensor([[0.2, 0.4, 0.4]], dtype=torch.float64))
    torch.testing.assert_close(braking.deterministic_output(logits), braking.mean)
    expected_log_density = math.log(12 * 0.25 * 0.75**2) + math.log(12 * 0.3 * 0.7**2 / 2)
    expected_log_density += math.log(12 * 0.75 * 0.25**2 / 2)
    torch.testing.assert_close(braking.log_prob(probe), torch.tensor([expected_log_density], dtype=torch.float64))
    torch.testing.assert_close(probe, untouched, rtol=0, atol=0)
    expected_entropy = 3 * (2.25 - math.log(12)) - math.log(2)
    torch.testing.assert_close(braking.entropy, torch.tensor([expected_entropy], dtype=torch.float64))
    # KL between Beta(2, 3) and Beta(3, 2) is 1/2 per coordinate, independent of a shared affine map.
    alpha, beta = braking.params
    torch.testing.assert_close(
        braking.kl_divergence((alpha, beta), (beta, alpha)), torch.tensor([1.5], dtype=torch.float64)
    )
    (-braking.log_prob(probe)).sum().backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad.abs() > 0).all()

    braking.update(logits.detach().expand(32768, -1, -1))
    torch.manual_seed(29)
    samples = braking.sample()
    assert ((samples[:, 0] >= -1.0) & (samples[:, 0] <= 0.0)).all()
    assert (samples[:, 1:].abs() <= 1.0).all()
    torch.testing.assert_close(
        samples.mean(0), torch.tensor([-0.6, -0.2, -0.2], dtype=torch.float64), atol=0.008, rtol=0
    )
    torch.testing.assert_close(samples.std(0), torch.tensor([0.2, 0.4, 0.4], dtype=torch.float64), atol=0.006, rtol=0)

    guards = torch.zeros(4, 2, 3)
    guards[0] = -100.0
    guards[1, 0, 1] = float("nan")
    guards[2, 1, 2] = float("inf")
    expected = torch.tensor([[-1.0, 0.0, 0.0]] * 3 + [[-0.5, 0.0, 0.0]])
    torch.testing.assert_close(braking.deterministic_output(guards), expected)
    torch.testing.assert_close(torch.jit.script(braking.as_deterministic_output_module())(guards), expected)
