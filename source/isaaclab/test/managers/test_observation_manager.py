# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

import pytest
import torch

from isaaclab.managers import ManagerTermBase, ObservationGroupCfg, ObservationManager, ObservationTermCfg
from isaaclab.utils import configclass, modifiers

pytestmark = pytest.mark.unit

IMAGE_SHAPE = (8, 16)


def grilled_chicken(env):
    return torch.ones(env.num_envs, 4, device=env.device)


def grilled_chicken_with_bbq(env, bbq: bool):
    return bbq * torch.ones(env.num_envs, 1, device=env.device)


def grilled_chicken_with_curry(env, hot: bool):
    return hot * 2 * torch.ones(env.num_envs, 1, device=env.device)


def grilled_chicken_with_yoghurt(env, hot: bool, bland: float):
    return hot * bland * torch.ones(env.num_envs, 5, device=env.device)


def grilled_chicken_with_yoghurt_and_bbq(env, hot: bool, bland: float, bbq: bool = False):
    return hot * bland * bbq * torch.ones(env.num_envs, 3, device=env.device)


def grilled_chicken_image(env, bland: float, channel: int = 1):
    return bland * torch.ones(env.num_envs, *IMAGE_SHAPE, channel, device=env.device)


def pos_w_data(env) -> torch.Tensor:
    return env.pos_w


def lin_vel_w_data(env) -> torch.Tensor:
    return env.lin_vel_w


class complex_function_class(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: object):
        super().__init__(cfg, env)
        self._time_passed = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._time_passed[env_ids] = 0.0

    def __call__(self, env: object, interval: float) -> torch.Tensor:
        self._time_passed += interval
        return self._time_passed.clone().unsqueeze(-1)


class non_callable_complex_function_class(ManagerTermBase):
    def call_me(self, env: object) -> torch.Tensor:
        return torch.ones(env.num_envs, 2, device=env.device)


class StatefulBiasModifier(modifiers.ModifierBase):
    """Stateful modifier used to verify lazy callable resolution."""

    def __init__(self, cfg: modifiers.ModifierCfg, data_dim: tuple[int, ...], device: str) -> None:
        super().__init__(cfg, data_dim, device)
        self.value = cfg.params["value"]
        self.reset_count = 0

    def reset(self, env_ids=None) -> None:
        self.reset_count += 1

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data + self.value


class InvalidModifier:
    """Class with the modifier constructor contract but the wrong base type."""

    def __init__(self, cfg, data_dim, device):
        pass


@configclass
class SampleGroupCfg(ObservationGroupCfg):
    term_1 = ObservationTermCfg(func=grilled_chicken, scale=10)
    term_2 = ObservationTermCfg(func=grilled_chicken, scale=2)
    term_3 = ObservationTermCfg(func=grilled_chicken_with_bbq, scale=5, params={"bbq": True})
    term_4 = ObservationTermCfg(func=grilled_chicken_with_yoghurt, scale=1.0, params={"hot": False, "bland": 2.0})
    term_5 = ObservationTermCfg(
        func=grilled_chicken_with_yoghurt_and_bbq, scale=1.0, params={"hot": False, "bland": 2.0}
    )


@configclass
class PolicyOnlyCfg:
    policy: ObservationGroupCfg = ObservationGroupCfg()


def make_cfg(**groups: ObservationGroupCfg) -> PolicyOnlyCfg:
    """Build an observation manager configuration from the given groups."""
    cfg = PolicyOnlyCfg()
    for name, group in groups.items():
        setattr(cfg, name, group)
    return cfg


@pytest.fixture
def env(make_env):
    env = make_env()
    env.pos_w = torch.rand((env.num_envs, 3), device=env.device)
    env.lin_vel_w = torch.rand((env.num_envs, 3), device=env.device)
    return env


@pytest.mark.parametrize(("history_length", "expected_shape"), [(0, "(4,)"), (5, "(20,)")])
def test_str(env, history_length, expected_shape):
    """The string representation lists every term with its (history-expanded) shape."""
    cfg = make_cfg(
        policy=SampleGroupCfg(term_1=ObservationTermCfg(func=grilled_chicken, history_length=history_length))
    )
    obs_man = ObservationManager(cfg, env)
    assert len(obs_man.active_terms["policy"]) == 5
    cells = [cell.strip() for cell in str(obs_man).split("|")]
    assert cells[cells.index("term_1") + 1] == expected_shape


def test_config_equivalence(env):
    """Annotated and un-annotated group configurations produce the same manager."""

    @configclass
    class MyObservationManagerCfg:
        policy = SampleGroupCfg()
        critic = SampleGroupCfg(concatenate_terms=False, term_4=None)

    @configclass
    class MyObservationManagerAnnotatedCfg:
        policy: ObservationGroupCfg = SampleGroupCfg()
        critic: ObservationGroupCfg = SampleGroupCfg(concatenate_terms=False, term_4=None)

    obs_man_from_cfg = ObservationManager(MyObservationManagerCfg(), env)
    obs_man_from_annotated_cfg = ObservationManager(MyObservationManagerAnnotatedCfg(), env)

    assert obs_man_from_cfg.active_terms == obs_man_from_annotated_cfg.active_terms
    assert obs_man_from_cfg.group_obs_term_dim == obs_man_from_annotated_cfg.group_obs_term_dim
    assert obs_man_from_cfg.group_obs_dim == obs_man_from_annotated_cfg.group_obs_dim
    assert obs_man_from_cfg._group_obs_term_cfgs == obs_man_from_annotated_cfg._group_obs_term_cfgs
    assert obs_man_from_cfg._group_obs_concatenate == obs_man_from_annotated_cfg._group_obs_concatenate


def test_config_terms(env):
    """Terms set to None are dropped and mixed shapes can only be kept in non-concatenated groups."""

    @configclass
    class MyObservationManagerCfg:
        @configclass
        class SampleGroupCfg(ObservationGroupCfg):
            term_1 = ObservationTermCfg(func=grilled_chicken, scale=10)
            term_2 = ObservationTermCfg(func=grilled_chicken_with_curry, scale=0.0, params={"hot": False})

        @configclass
        class SampleMixedGroupCfg(ObservationGroupCfg):
            concatenate_terms = False
            term_1 = ObservationTermCfg(func=grilled_chicken, scale=2.0)
            term_2 = ObservationTermCfg(func=grilled_chicken_image, scale=1.5, params={"bland": 0.5})

        @configclass
        class SampleImageGroupCfg(ObservationGroupCfg):
            term_1 = ObservationTermCfg(func=grilled_chicken_image, scale=1.5, params={"bland": 0.5, "channel": 1})
            term_2 = ObservationTermCfg(func=grilled_chicken_image, scale=0.5, params={"bland": 0.1, "channel": 3})

        policy: ObservationGroupCfg = SampleGroupCfg()
        critic: ObservationGroupCfg = SampleGroupCfg(term_2=None)
        mixed: ObservationGroupCfg = SampleMixedGroupCfg()
        image: ObservationGroupCfg = SampleImageGroupCfg()

    obs_man = ObservationManager(MyObservationManagerCfg(), env)
    assert {name: len(terms) for name, terms in obs_man.active_terms.items()} == {
        "policy": 2,
        "critic": 1,
        "mixed": 2,
        "image": 2,
    }

    cfg = MyObservationManagerCfg()
    cfg.mixed.concatenate_terms = True
    with pytest.raises(RuntimeError):
        ObservationManager(cfg, env)


def test_compute(env):
    """Scales are applied per term and the same term yields the same data within and across groups."""
    pos_scale_tuple = (2.0, 3.0, 1.0)

    @configclass
    class MyObservationManagerCfg:
        @configclass
        class PolicyCfg(ObservationGroupCfg):
            term_1 = ObservationTermCfg(func=grilled_chicken, scale=10)
            term_2 = ObservationTermCfg(func=grilled_chicken_with_curry, scale=0.0, params={"hot": False})
            term_3 = ObservationTermCfg(func=pos_w_data, scale=pos_scale_tuple)
            term_4 = ObservationTermCfg(func=lin_vel_w_data, scale=1.5)

        @configclass
        class CriticCfg(ObservationGroupCfg):
            term_1 = ObservationTermCfg(func=pos_w_data, scale=pos_scale_tuple)
            term_2 = ObservationTermCfg(func=lin_vel_w_data, scale=1.5)
            term_3 = ObservationTermCfg(func=pos_w_data, scale=pos_scale_tuple)
            term_4 = ObservationTermCfg(func=lin_vel_w_data, scale=1.5)

        @configclass
        class ImageCfg(ObservationGroupCfg):
            term_1 = ObservationTermCfg(func=grilled_chicken_image, scale=1.5, params={"bland": 0.5, "channel": 1})
            term_2 = ObservationTermCfg(func=grilled_chicken_image, scale=0.5, params={"bland": 0.1, "channel": 3})

        policy: ObservationGroupCfg = PolicyCfg()
        critic: ObservationGroupCfg = CriticCfg()
        image: ObservationGroupCfg = ImageCfg()

    observations = ObservationManager(MyObservationManagerCfg(), env).compute()
    obs_policy, obs_critic, obs_image = observations["policy"], observations["critic"], observations["image"]

    assert obs_policy.shape == (env.num_envs, 11)
    assert obs_critic.shape == (env.num_envs, 12)
    assert obs_image.shape == (env.num_envs, *IMAGE_SHAPE, 4)
    assert torch.equal(env.pos_w * torch.tensor(pos_scale_tuple, device=env.device), obs_critic[:, :3])
    assert torch.equal(env.lin_vel_w * 1.5, obs_critic[:, 3:6])
    # same terms give the same data within and across groups
    assert torch.equal(obs_critic[:, 0:3], obs_critic[:, 6:9])
    assert torch.equal(obs_critic[:, 3:6], obs_critic[:, 9:12])
    assert torch.equal(obs_policy[:, 5:8], obs_critic[:, 0:3])
    assert torch.equal(obs_policy[:, 8:11], obs_critic[:, 3:6])


@pytest.mark.parametrize("group_history_length", [None, 10], ids=["term_history", "group_history"])
def test_compute_with_history(env, group_history_length):
    """History buffers fill from the first sample, roll over time and reset per environment.

    A group-level history length overrides the term-level one for every term in the group.
    """
    term_history_length = 5
    history_length = group_history_length or term_history_length

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        history_length = group_history_length
        term_1 = ObservationTermCfg(func=grilled_chicken, history_length=term_history_length)
        term_2 = ObservationTermCfg(func=lin_vel_w_data)

    obs_man = ObservationManager(make_cfg(policy=PolicyCfg()), env)
    term_2_repeats = history_length if group_history_length else 1
    expected_t0 = torch.cat(
        (torch.ones(env.num_envs, 4 * history_length, device=env.device), env.lin_vel_w.repeat(1, term_2_repeats)),
        dim=-1,
    )

    obs_policy = obs_man.compute()["policy"]
    assert obs_policy.shape == (env.num_envs, 4 * history_length + 3 * term_2_repeats)
    torch.testing.assert_close(obs_policy, expected_t0)
    # constant terms keep the same history after rolling the buffer
    for _ in range(history_length):
        obs_policy = obs_man.compute(update_history=True)["policy"]
    torch.testing.assert_close(obs_policy, expected_t0)
    # full and partial resets refill the history from the next sample
    obs_man.reset()
    torch.testing.assert_close(obs_man.compute(update_history=True)["policy"], expected_t0)
    reset_env_ids = [2, 4, 16]
    obs_man.reset(reset_env_ids)
    obs_policy = obs_man.compute(update_history=True)["policy"]
    torch.testing.assert_close(obs_policy[reset_env_ids], expected_t0[reset_env_ids])


def test_compute_with_2d_history(env):
    """Image history is flattened per environment unless ``flatten_history_dim`` is disabled."""
    history_length = 5
    image_term = ObservationTermCfg(
        func=grilled_chicken_image, params={"bland": 1.0, "channel": 1}, history_length=history_length
    )

    @configclass
    class FlattenedPolicyCfg(ObservationGroupCfg):
        term_1 = image_term

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        term_1 = ObservationTermCfg(
            func=grilled_chicken_image,
            params={"bland": 1.0, "channel": 1},
            history_length=history_length,
            flatten_history_dim=False,
        )

    observations = ObservationManager(make_cfg(flat_obs_policy=FlattenedPolicyCfg(), policy=PolicyCfg()), env).compute()
    assert observations["flat_obs_policy"].shape == (env.num_envs, history_length * IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
    assert observations["policy"].shape == (env.num_envs, history_length, *IMAGE_SHAPE, 1)


def test_compute_updates_history_only_when_requested(env):
    """Observation history changes only when ``update_history`` is enabled."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        history_length = 5
        term_1 = ObservationTermCfg(func=pos_w_data)

    manager = ObservationManager(make_cfg(policy=PolicyCfg()), env)
    history = manager._group_obs_term_history_buffer["policy"]["term_1"]
    zeros = torch.zeros(env.num_envs, dtype=torch.int64)

    torch.testing.assert_close(history.current_length, zeros)
    manager.compute()
    torch.testing.assert_close(history.current_length, zeros)

    manager.compute(update_history=True)
    torch.testing.assert_close(history.current_length, zeros + 1)
    history_after_update = history.buffer.clone()

    env.pos_w.add_(10.0)
    policy_observation = manager.compute()["policy"]
    torch.testing.assert_close(history.current_length, zeros + 1)
    torch.testing.assert_close(history.buffer, history_after_update)
    torch.testing.assert_close(policy_observation, history_after_update.reshape(env.num_envs, -1))

    manager.compute(update_history=True)
    torch.testing.assert_close(history.current_length, zeros + 2)
    torch.testing.assert_close(history.buffer[:, -1], env.pos_w)


def test_callable_class_term(env):
    """Class terms keep state across computations and reset it per environment."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        term_1 = ObservationTermCfg(func=grilled_chicken, scale=10)
        term_2 = ObservationTermCfg(func=complex_function_class, scale=0.2, params={"interval": 0.5})

    obs_man = ObservationManager(make_cfg(policy=PolicyCfg()), env)
    observations = obs_man.compute()
    assert observations["policy"].shape == (env.num_envs, 5)
    assert observations["policy"][0, -1].item() == pytest.approx(0.2 * 0.5)

    num_exec_count = 10
    for _ in range(num_exec_count):
        observations = obs_man.compute()
    assert observations["policy"][0, -1].item() == pytest.approx(0.2 * 0.5 * (num_exec_count + 1))

    obs_man.reset(env_ids=[0, 4, 9, 14, 19])
    observations = obs_man.compute()
    assert observations["policy"][0, -1].item() == pytest.approx(0.2 * 0.5)
    assert observations["policy"][1, -1].item() == pytest.approx(0.2 * 0.5 * (num_exec_count + 2))


def test_modifier_compute(env):
    """Modifiers are applied in order before the term is returned."""
    modifier_1 = modifiers.ModifierCfg(func=modifiers.bias, params={"value": 1.0})
    modifier_2 = modifiers.ModifierCfg(func=modifiers.scale, params={"multiplier": 2.0})
    modifier_3 = modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (-0.5, 0.5)})
    modifier_4 = modifiers.IntegratorCfg(dt=env.dt)

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        concatenate_terms = False
        term_1 = ObservationTermCfg(func=pos_w_data, modifiers=[])
        term_2 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_1])
        term_3 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_1, modifier_4])
        term_4 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_1, modifier_2])
        term_5 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_1, modifier_2, modifier_3])

    obs_policy = ObservationManager(make_cfg(policy=PolicyCfg()), env).compute()["policy"]

    assert torch.equal(obs_policy["term_1"] + 1.0, obs_policy["term_2"])
    assert torch.equal(2.0 * (obs_policy["term_1"] + 1.0), obs_policy["term_4"])
    assert torch.min(obs_policy["term_5"]) >= -0.5
    assert torch.max(obs_policy["term_5"]) <= 0.5


@pytest.mark.parametrize(
    "modifier_cfg",
    [
        modifiers.ModifierCfg(func=StatefulBiasModifier, params={"value": 2.0}),
        modifiers.ModifierCfg(func=modifiers.bias, params={"value": 2.0}),
    ],
    ids=["class", "function"],
)
def test_modifier_cfg_roundtrip(env, modifier_cfg):
    """Modifiers serialized to strings by a configuration round-trip are resolved back to callables (#6067)."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        term_1 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_cfg])

    cfg = make_cfg(policy=PolicyCfg())
    cfg.from_dict(cfg.to_dict())
    roundtripped_cfg = cfg.policy.term_1.modifiers[0]
    assert isinstance(roundtripped_cfg.func, str)
    assert roundtripped_cfg.params == {"value": 2.0}

    manager = ObservationManager(cfg, env)
    torch.testing.assert_close(manager.compute()["policy"], env.pos_w + 2.0)

    prepared_modifier = manager.cfg.policy.term_1.modifiers[0].func
    if modifier_cfg.func is StatefulBiasModifier:
        assert isinstance(prepared_modifier, StatefulBiasModifier)
        manager.reset()
        assert prepared_modifier.reset_count == 1
    else:
        assert prepared_modifier is modifiers.bias


def _invalid_params_cfg():
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        term_1 = ObservationTermCfg(func=grilled_chicken_with_bbq, scale=0.1, params={"hot": False})
        term_2 = ObservationTermCfg(func=grilled_chicken_with_yoghurt, scale=2.0, params={"hot": False})

    return make_cfg(policy=PolicyCfg())


def _non_callable_class_cfg():
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        term_1 = ObservationTermCfg(func=grilled_chicken, scale=10)
        term_2 = ObservationTermCfg(func=non_callable_complex_function_class, scale=0.2)

    return make_cfg(policy=PolicyCfg())


def _invalid_modifier_params_cfg():
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        concatenate_terms = False
        term_1 = ObservationTermCfg(
            func=pos_w_data, modifiers=[modifiers.ModifierCfg(func=modifiers.clip, params={"min": -0.5, "max": 0.5})]
        )

    return make_cfg(policy=PolicyCfg())


def _invalid_modifier_class_cfg():
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        term_1 = ObservationTermCfg(func=pos_w_data, modifiers=[modifiers.ModifierCfg(func=InvalidModifier)])

    cfg = make_cfg(policy=PolicyCfg())
    cfg.from_dict(cfg.to_dict())
    return cfg


@pytest.mark.parametrize(
    ("make_invalid_cfg", "error"),
    [
        (_invalid_params_cfg, ValueError),
        (_non_callable_class_cfg, NotImplementedError),
        (_invalid_modifier_params_cfg, ValueError),
        (_invalid_modifier_class_cfg, TypeError),
    ],
    ids=["term_params", "non_callable_class", "modifier_params", "modifier_class"],
)
def test_invalid_config(env, make_invalid_cfg, error):
    """Invalid term, class and modifier configurations are rejected on construction."""
    with pytest.raises(error):
        ObservationManager(make_invalid_cfg(), env)


def test_serialize(env):
    """Class terms serialize through their own ``serialize`` method."""
    serialize_data = {"test": 0}

    class test_serialize_term(ManagerTermBase):
        def __call__(self, env) -> torch.Tensor:
            return grilled_chicken(env)

        def serialize(self) -> dict:
            return serialize_data

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        concatenate_terms = False
        term_1 = ObservationTermCfg(func=test_serialize_term)

    assert ObservationManager(make_cfg(policy=PolicyCfg()), env).serialize() == {"policy": {"term_1": serialize_data}}


def test_concatenate_dim(env):
    """Terms concatenate along the configured (batch-offset) dimension."""

    @configclass
    class ImageGroupCfg(ObservationGroupCfg):
        term_1 = ObservationTermCfg(func=grilled_chicken_image, scale=1.0, params={"bland": 1.0, "channel": 1})
        term_2 = ObservationTermCfg(func=grilled_chicken_image, scale=1.0, params={"bland": 1.0, "channel": 1})

    @configclass
    class MyObservationManagerCfg:
        policy: ObservationGroupCfg = ImageGroupCfg(concatenate_dim=1)
        critic: ObservationGroupCfg = ImageGroupCfg(concatenate_dim=2)
        critic_neg_dim: ObservationGroupCfg = ImageGroupCfg(concatenate_dim=-1)

    observations = ObservationManager(MyObservationManagerCfg(), env).compute()
    obs_policy, obs_critic = observations["policy"], observations["critic"]
    height, width = IMAGE_SHAPE

    assert obs_policy.shape == (env.num_envs, height, 2 * width, 1)
    assert obs_critic.shape == (env.num_envs, height, width, 2)
    torch.testing.assert_close(obs_policy[:, :, :width, :], obs_policy[:, :, width:, :])
    torch.testing.assert_close(obs_critic[..., 0], obs_critic[..., 1])
    torch.testing.assert_close(observations["critic_neg_dim"], obs_critic)
