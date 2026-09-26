# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for observation managers."""

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none
from collections import namedtuple
from typing import TYPE_CHECKING, cast

import pytest
import torch

from isaaclab.managers import (
    ManagerTermBase,
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
    RewardTermCfg,
)
from isaaclab.utils import DelayBuffer, configclass, modifiers, noise

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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
    return bland * torch.ones(env.num_envs, 128, 256, channel, device=env.device)


class complex_function_class(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: object):
        self.cfg = cfg
        self.env = env
        # define some variables
        self._time_passed = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._time_passed[env_ids] = 0.0

    def __call__(self, env: object, interval: float) -> torch.Tensor:
        self._time_passed += interval
        return self._time_passed.clone().unsqueeze(-1)


class non_callable_complex_function_class(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: object):
        self.cfg = cfg
        self.env = env
        # define some variables
        self._cost = 2 * self.env.num_envs

    def call_me(self, env: object) -> torch.Tensor:
        return torch.ones(env.num_envs, 2, device=env.device) * self._cost


class MyDataClass:
    def __init__(self, num_envs: int, device: str):
        self.pos_w = torch.rand((num_envs, 3), device=device)
        self.lin_vel_w = torch.rand((num_envs, 3), device=device)


def pos_w_data(env) -> torch.Tensor:
    return env.data.pos_w


def lin_vel_w_data(env) -> torch.Tensor:
    return env.data.lin_vel_w


class DummySimulation:
    """Minimal playing simulation double."""

    def is_playing(self) -> bool:
        """Return whether the simulated timeline is playing."""
        return True


@pytest.fixture
def setup_env():
    dt = 0.01
    num_envs = 20
    device = "cpu"
    # create dummy environment with a playing simulation, so the manager resolves terms immediately
    return namedtuple("ManagerBasedEnv", ["num_envs", "device", "data", "dt", "sim"])(
        num_envs, device, MyDataClass(num_envs, device), dt, DummySimulation()
    )


def test_str(setup_env):
    env = setup_env
    """Test the string representation of the observation manager."""

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class SampleGroupCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            term_1 = ObservationTermCfg(func=grilled_chicken, scale=10)
            term_2 = ObservationTermCfg(func=grilled_chicken, scale=2)
            term_3 = ObservationTermCfg(func=grilled_chicken_with_bbq, scale=5, params={"bbq": True})
            term_4 = ObservationTermCfg(
                func=grilled_chicken_with_yoghurt, scale=1.0, params={"hot": False, "bland": 2.0}
            )
            term_5 = ObservationTermCfg(
                func=grilled_chicken_with_yoghurt_and_bbq, scale=1.0, params={"hot": False, "bland": 2.0}
            )

        policy: ObservationGroupCfg = SampleGroupCfg()
        # a term history flattens into the term shape
        hist: ObservationGroupCfg = SampleGroupCfg(
            term_1=ObservationTermCfg(func=grilled_chicken, scale=10, history_length=5)
        )

    # create observation manager
    cfg = MyObservationManagerCfg()
    obs_man = ObservationManager(cfg, env)
    assert len(obs_man.active_terms["policy"]) == 5
    obs_man_str = str(obs_man)
    for group_name, expected_shape in (("policy", "(4,)"), ("hist", "(20,)")):
        group_str = obs_man_str.split(f"Group: '{group_name}'")[1]
        group_str_split = group_str.split("|")
        term_1_str_index = next(i for i, s in enumerate(group_str_split) if s.strip() == "term_1")
        term_1_str_shape = group_str_split[term_1_str_index + 1].strip()
        assert term_1_str_shape == expected_shape


def test_config_equivalence(setup_env):
    env = setup_env
    """Test the equivalence of observation manager created from different config types."""

    # create from config class
    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class SampleGroupCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            your_term = ObservationTermCfg(func=grilled_chicken, scale=10)
            his_term = ObservationTermCfg(func=grilled_chicken, scale=2)
            my_term = ObservationTermCfg(func=grilled_chicken_with_bbq, scale=5, params={"bbq": True})
            her_term = ObservationTermCfg(
                func=grilled_chicken_with_yoghurt, scale=1.0, params={"hot": False, "bland": 2.0}
            )

        policy = SampleGroupCfg()
        critic = SampleGroupCfg(concatenate_terms=False, her_term=None)

    cfg = MyObservationManagerCfg()
    obs_man_from_cfg = ObservationManager(cfg, env)

    # create from config class
    @configclass
    class MyObservationManagerAnnotatedCfg:
        """Test config class for observation manager with annotations on terms."""

        @configclass
        class SampleGroupCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            your_term: ObservationTermCfg = ObservationTermCfg(func=grilled_chicken, scale=10)
            his_term: ObservationTermCfg = ObservationTermCfg(func=grilled_chicken, scale=2)
            my_term: ObservationTermCfg = ObservationTermCfg(
                func=grilled_chicken_with_bbq, scale=5, params={"bbq": True}
            )
            her_term: ObservationTermCfg = ObservationTermCfg(
                func=grilled_chicken_with_yoghurt, scale=1.0, params={"hot": False, "bland": 2.0}
            )

        policy: ObservationGroupCfg = SampleGroupCfg()
        critic: ObservationGroupCfg = SampleGroupCfg(concatenate_terms=False, her_term=None)

    cfg = MyObservationManagerAnnotatedCfg()
    obs_man_from_annotated_cfg = ObservationManager(cfg, env)

    # check equivalence
    # parsed terms
    assert obs_man_from_cfg.active_terms == obs_man_from_annotated_cfg.active_terms
    assert obs_man_from_cfg.group_obs_term_dim == obs_man_from_annotated_cfg.group_obs_term_dim
    assert obs_man_from_cfg.group_obs_dim == obs_man_from_annotated_cfg.group_obs_dim
    # parsed term configs
    assert obs_man_from_cfg._group_obs_term_cfgs == obs_man_from_annotated_cfg._group_obs_term_cfgs
    assert obs_man_from_cfg._group_obs_concatenate == obs_man_from_annotated_cfg._group_obs_concatenate


def test_config_terms(setup_env):
    env = setup_env
    """Test the number of terms in the observation manager."""

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class SampleGroupCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            term_1 = ObservationTermCfg(func=grilled_chicken, scale=10)
            term_2 = ObservationTermCfg(func=grilled_chicken_with_curry, scale=0.0, params={"hot": False})

        @configclass
        class SampleMixedGroupCfg(ObservationGroupCfg):
            """Test config class for policy observation group with a mix of vector and matrix terms."""

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

    # create observation manager
    cfg = MyObservationManagerCfg()
    obs_man = ObservationManager(cfg, env)

    assert len(obs_man.active_terms["policy"]) == 2
    assert len(obs_man.active_terms["critic"]) == 1
    assert len(obs_man.active_terms["mixed"]) == 2
    assert len(obs_man.active_terms["image"]) == 2

    # create a new obs manager but where mixed group has invalid config
    cfg = MyObservationManagerCfg()
    cfg.mixed.concatenate_terms = True

    with pytest.raises(RuntimeError):
        ObservationManager(cfg, env)


def test_compute(setup_env):
    env = setup_env
    """Test the observation computation."""

    pos_scale_tuple = (2.0, 3.0, 1.0)

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class PolicyCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

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

    # create observation manager
    cfg = MyObservationManagerCfg()
    obs_man = ObservationManager(cfg, env)
    # compute observation using manager
    observations = obs_man.compute()

    # obtain the group observations
    obs_policy: torch.Tensor = observations["policy"]
    obs_critic: torch.Tensor = observations["critic"]
    obs_image: torch.Tensor = observations["image"]

    # check the observation shape
    assert obs_policy.shape == (env.num_envs, 11)
    assert obs_critic.shape == (env.num_envs, 12)
    assert obs_image.shape == (env.num_envs, 128, 256, 4)
    # check that the scales are applied correctly
    assert torch.equal(env.data.pos_w * torch.tensor(pos_scale_tuple, device=env.device), obs_critic[:, :3])
    assert torch.equal(env.data.lin_vel_w * 1.5, obs_critic[:, 3:6])
    # make sure that the data are the same for same terms
    # -- within group
    assert torch.equal(obs_critic[:, 0:3], obs_critic[:, 6:9])
    assert torch.equal(obs_critic[:, 3:6], obs_critic[:, 9:12])
    # -- between groups
    assert torch.equal(obs_policy[:, 5:8], obs_critic[:, 0:3])
    assert torch.equal(obs_policy[:, 8:11], obs_critic[:, 3:6])


def test_compute_with_2d_history(setup_env):
    env = setup_env
    """Test the observation computation with history buffers for 2D observations."""
    HISTORY_LENGTH = 5

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class FlattenedPolicyCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            term_1 = ObservationTermCfg(
                func=grilled_chicken_image, params={"bland": 1.0, "channel": 1}, history_length=HISTORY_LENGTH
            )
            # total observation size: term_dim (128, 256) * history_len (5) = 163840

        @configclass
        class PolicyCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            term_1 = ObservationTermCfg(
                func=grilled_chicken_image,
                params={"bland": 1.0, "channel": 1},
                history_length=HISTORY_LENGTH,
                flatten_history_dim=False,
            )
            # total observation size: (5, 128, 256, 1)

        flat_obs_policy: ObservationGroupCfg = FlattenedPolicyCfg()
        policy: ObservationGroupCfg = PolicyCfg()

    # create observation manager
    cfg = MyObservationManagerCfg()
    obs_man = ObservationManager(cfg, env)
    # compute observation using manager
    observations = obs_man.compute()
    # obtain the group observations
    obs_policy_flat: torch.Tensor = observations["flat_obs_policy"]
    obs_policy: torch.Tensor = observations["policy"]
    # check the observation shapes
    assert obs_policy_flat.shape == (env.num_envs, 163840)
    assert obs_policy.shape == (env.num_envs, HISTORY_LENGTH, 128, 256, 1)


def test_invalid_observation_config(setup_env):
    env = setup_env
    """Test the invalid observation config."""

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class PolicyCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            term_1 = ObservationTermCfg(func=grilled_chicken_with_bbq, scale=0.1, params={"hot": False})
            term_2 = ObservationTermCfg(func=grilled_chicken_with_yoghurt, scale=2.0, params={"hot": False})

        policy: ObservationGroupCfg = PolicyCfg()

    # create observation manager
    cfg = MyObservationManagerCfg()
    # check the invalid config
    with pytest.raises(ValueError):
        ObservationManager(cfg, env)


def test_callable_class_term(setup_env):
    env = setup_env
    """Test the observation computation with callable class term."""

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class PolicyCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            term_1 = ObservationTermCfg(func=grilled_chicken, scale=10)
            term_2 = ObservationTermCfg(func=complex_function_class, scale=0.2, params={"interval": 0.5})

        policy: ObservationGroupCfg = PolicyCfg()

    # create observation manager
    cfg = MyObservationManagerCfg()
    obs_man = ObservationManager(cfg, env)
    # compute observation using manager
    observations = obs_man.compute()
    # check the observation
    assert observations["policy"].shape == (env.num_envs, 5)
    assert observations["policy"][0, -1].item() == pytest.approx(0.2 * 0.5)

    # check memory in term
    num_exec_count = 10
    for _ in range(num_exec_count):
        observations = obs_man.compute()
    assert observations["policy"][0, -1].item() == pytest.approx(0.2 * 0.5 * (num_exec_count + 1))

    # check reset works
    obs_man.reset(env_ids=[0, 4, 9, 14, 19])
    observations = obs_man.compute()
    assert observations["policy"][0, -1].item() == pytest.approx(0.2 * 0.5)
    assert observations["policy"][1, -1].item() == pytest.approx(0.2 * 0.5 * (num_exec_count + 2))


def test_non_callable_class_term(setup_env):
    env = setup_env
    """Test the observation computation with non-callable class term."""

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class PolicyCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            term_1 = ObservationTermCfg(func=grilled_chicken, scale=10)
            term_2 = ObservationTermCfg(func=non_callable_complex_function_class, scale=0.2)

        policy: ObservationGroupCfg = PolicyCfg()

    # create observation manager config
    cfg = MyObservationManagerCfg()
    # create observation manager
    with pytest.raises(NotImplementedError):
        ObservationManager(cfg, env)


def test_modifier_compute(setup_env):
    env = setup_env
    """Test the observation computation with modifiers."""

    modifier_1 = modifiers.ModifierCfg(func=modifiers.bias, params={"value": 1.0})
    modifier_2 = modifiers.ModifierCfg(func=modifiers.scale, params={"multiplier": 2.0})
    modifier_3 = modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (-0.5, 0.5)})
    modifier_4 = modifiers.IntegratorCfg(dt=env.dt)

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class PolicyCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            concatenate_terms = False
            term_1 = ObservationTermCfg(func=pos_w_data, modifiers=[])
            term_2 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_1])
            term_3 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_1, modifier_4])

        @configclass
        class CriticCfg(ObservationGroupCfg):
            """Test config class for critic observation group"""

            concatenate_terms = False
            term_1 = ObservationTermCfg(func=pos_w_data, modifiers=[])
            term_2 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_1])
            term_3 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_1, modifier_2])
            term_4 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_1, modifier_2, modifier_3])

        policy: ObservationGroupCfg = PolicyCfg()
        critic: ObservationGroupCfg = CriticCfg()

    # create observation manager
    cfg = MyObservationManagerCfg()
    obs_man = ObservationManager(cfg, env)
    # compute observation using manager
    observations = obs_man.compute()

    # obtain the group observations
    obs_policy: dict[str, torch.Tensor] = observations["policy"]
    obs_critic: dict[str, torch.Tensor] = observations["critic"]

    # check correct application of modifications
    assert torch.equal(obs_policy["term_1"] + 1.0, obs_policy["term_2"])
    # the first integration step averages the biased signal with a zero initial value
    torch.testing.assert_close(obs_policy["term_3"], 0.5 * (obs_policy["term_1"] + 1.0) * env.dt)
    assert torch.equal(obs_critic["term_1"] + 1.0, obs_critic["term_2"])
    assert torch.equal(2.0 * (obs_critic["term_1"] + 1.0), obs_critic["term_3"])
    assert torch.min(obs_critic["term_4"]) >= -0.5
    assert torch.max(obs_critic["term_4"]) <= 0.5


def test_serialize(setup_env):
    """Test serialize call for ManagerTermBase terms."""
    env = setup_env

    serialize_data = {"test": 0}

    class test_serialize_term(ManagerTermBase):
        def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
            super().__init__(cfg, env)

        def __call__(self, env: ManagerBasedEnv) -> torch.Tensor:
            return grilled_chicken(env)

        def serialize(self) -> dict:
            return serialize_data

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class PolicyCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            concatenate_terms = False
            term_1 = ObservationTermCfg(func=test_serialize_term)

        policy: ObservationGroupCfg = PolicyCfg()

    # create observation manager
    cfg = MyObservationManagerCfg()
    obs_man = ObservationManager(cfg, env)

    # check expected output
    assert obs_man.serialize() == {"policy": {"term_1": serialize_data}}


def test_modifier_invalid_config(setup_env):
    env = setup_env
    """Test modifier initialization with invalid config."""

    modifier = modifiers.ModifierCfg(func=modifiers.clip, params={"min": -0.5, "max": 0.5})

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class PolicyCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            concatenate_terms = False
            term_1 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier])

        policy: ObservationGroupCfg = PolicyCfg()

    # create observation manager
    cfg = MyObservationManagerCfg()

    with pytest.raises(ValueError):
        ObservationManager(cfg, env)


def test_concatenate_dim(setup_env):
    """Test concatenation of observations along different dimensions."""
    env = setup_env

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class PolicyCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            concatenate_terms = True
            concatenate_dim = 1  # Concatenate along dimension 1
            term_1 = ObservationTermCfg(func=grilled_chicken_image, scale=1.0, params={"bland": 1.0, "channel": 1})
            term_2 = ObservationTermCfg(func=grilled_chicken_image, scale=1.0, params={"bland": 1.0, "channel": 1})

        @configclass
        class CriticCfg(ObservationGroupCfg):
            """Test config class for critic observation group."""

            concatenate_terms = True
            concatenate_dim = 2  # Concatenate along dimension 2
            term_1 = ObservationTermCfg(func=grilled_chicken_image, scale=1.0, params={"bland": 1.0, "channel": 1})
            term_2 = ObservationTermCfg(func=grilled_chicken_image, scale=1.0, params={"bland": 1.0, "channel": 1})

        @configclass
        class CriticCfg_neg_dim(ObservationGroupCfg):
            """Test config class for critic observation group."""

            concatenate_terms = True
            concatenate_dim = -1  # Concatenate along last dimension
            term_1 = ObservationTermCfg(func=grilled_chicken_image, scale=1.0, params={"bland": 1.0, "channel": 1})
            term_2 = ObservationTermCfg(func=grilled_chicken_image, scale=1.0, params={"bland": 1.0, "channel": 1})

        policy: ObservationGroupCfg = PolicyCfg()
        critic: ObservationGroupCfg = CriticCfg()
        critic_neg_dim: ObservationGroupCfg = CriticCfg_neg_dim()

    # create observation manager
    cfg = MyObservationManagerCfg()
    obs_man = ObservationManager(cfg, env)
    # compute observation using manager
    observations = obs_man.compute()

    # obtain the group observations
    obs_policy: torch.Tensor = observations["policy"]
    obs_critic: torch.Tensor = observations["critic"]
    obs_critic_neg_dim: torch.Tensor = observations["critic_neg_dim"]

    # check the observation shapes
    # For policy: concatenated along dim 1, so width should be doubled
    assert obs_policy.shape == (env.num_envs, 128, 512, 1)
    # For critic: concatenated along last dim, so channels should be doubled
    assert obs_critic.shape == (env.num_envs, 128, 256, 2)
    # For critic_neg_dim: concatenated along last dim, so channels should be doubled
    assert obs_critic_neg_dim.shape == (env.num_envs, 128, 256, 2)

    # verify the data is concatenated correctly
    # For policy: check that the second half matches the first half
    torch.testing.assert_close(obs_policy[:, :, :256, :], obs_policy[:, :, 256:, :])
    # For critic: check that the second channel matches the first channel
    torch.testing.assert_close(obs_critic[:, :, :, 0], obs_critic[:, :, :, 1])

    # For critic_neg_dim: check that it is the same as critic
    torch.testing.assert_close(obs_critic_neg_dim, obs_critic)


def dummy_observation(env: DummyEnv) -> torch.Tensor:
    """Return the dummy environment observation."""
    return env.observation


class DummyEnv:
    """Minimal environment double used by :class:`ObservationManager`."""

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.sim = DummySimulation()
        self.observation = torch.arange(num_envs, dtype=torch.float32).unsqueeze(-1)


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
class HistoryObservationsCfg:
    """Observation configuration with group-level history."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observation group configuration."""

        dummy: ObservationTermCfg = ObservationTermCfg(func=dummy_observation)

        def __post_init__(self):
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()


def test_class_modifier_roundtrip_preserves_func_and_params():
    """Reproduce #6067 with a class modifier and non-empty parameters."""
    cfg = HistoryObservationsCfg()
    cfg.policy.history_length = None
    cfg.policy.dummy.modifiers = [modifiers.ModifierCfg(func=StatefulBiasModifier, params={"value": 2.0})]
    cfg.from_dict(cfg.to_dict())
    term_cfg = cfg.policy.dummy
    assert term_cfg.modifiers is not None
    modifier_cfg = term_cfg.modifiers[0]
    assert isinstance(modifier_cfg, modifiers.ModifierCfg)
    assert isinstance(modifier_cfg.func, str)
    assert modifier_cfg.params == {"value": 2.0}

    env = DummyEnv()
    manager = ObservationManager(cfg, cast("ManagerBasedEnv", env))
    prepared_term_cfg = manager.cfg.policy.dummy
    assert prepared_term_cfg.modifiers is not None
    prepared_modifier_cfg = prepared_term_cfg.modifiers[0]
    assert isinstance(prepared_modifier_cfg.func, StatefulBiasModifier)
    observations = manager.compute()["policy"]
    torch.testing.assert_close(observations, env.observation + 2.0)

    manager.reset()
    assert prepared_modifier_cfg.func.reset_count == 1


def test_stateless_modifier_cfg_roundtrip_preserves_signature_validation():
    """A stateless modifier remains callable and inspectable after a configuration round-trip."""
    cfg = HistoryObservationsCfg()
    cfg.policy.history_length = None
    cfg.policy.dummy.modifiers = [modifiers.ModifierCfg(func=modifiers.bias, params={"value": 2.0})]
    cfg.from_dict(cfg.to_dict())

    env = DummyEnv()
    manager = ObservationManager(cfg, cast("ManagerBasedEnv", env))
    observations = manager.compute()["policy"]
    torch.testing.assert_close(observations, env.observation + 2.0)


def test_class_modifier_validates_constructed_instance():
    """Class modifier validation checks the constructed object."""
    cfg = HistoryObservationsCfg()
    cfg.policy.history_length = None
    cfg.policy.dummy.modifiers = [modifiers.ModifierCfg(func=InvalidModifier)]
    cfg.from_dict(cfg.to_dict())

    with pytest.raises(TypeError, match="is not an instance of 'ModifierBase'"):
        ObservationManager(cfg, cast("ManagerBasedEnv", DummyEnv()))


@pytest.mark.parametrize(
    ("lag", "history_length", "term_history"), [(0, 2, False), (1, 0, False), (1, 2, False), (0, 2, True)]
)
def test_compute_updates_history_only_when_requested(lag, history_length, term_history):
    """History alone, delay alone, and their combination advance only on recorded samples.

    A group history length overrides the term's own history length; without one, the term history applies.
    """
    cfg = HistoryObservationsCfg()
    if term_history:
        cfg.policy.history_length = None
        cfg.policy.dummy.history_length = history_length
    else:
        cfg.policy.history_length = history_length
        cfg.policy.dummy.history_length = 3
    cfg.policy.enable_corruption = True
    cfg.policy.dummy.delay_min_lag = cfg.policy.dummy.delay_max_lag = lag
    cfg.policy.dummy.delay_hold_prob = 0.5 if history_length == 0 else 1.0
    cfg.policy.dummy.noise = noise.ConstantNoiseCfg(bias=0.0)
    cfg.policy.dummy.scale = 2.0
    env = DummyEnv()
    manager = ObservationManager(cfg, cast("ManagerBasedEnv", env))
    delay = manager._group_obs_term_delay_buffer["policy"].get("dummy")
    history = manager._group_obs_term_history_buffer["policy"].get("dummy")
    manager.compute()
    if delay:
        assert isinstance(delay, DelayBuffer)
        assert torch.all(delay.num_pushes == 0)
    if history:
        assert torch.all(history.current_length == 0)

    for step in range(6):
        if step == 3:
            manager.reset([1])
        env.observation.fill_(step)
        # Delay retains each sample's noise; history stacks the delayed, scaled outputs.
        manager.cfg.policy.dummy.noise.bias = float(step)
        output = manager.compute(update_history=True)["policy"]
        sample_steps = torch.arange(step - max(1, history_length) + 1, step + 1)
        expected = 4.0 * (sample_steps - lag).clamp_min(0).expand(env.num_envs, -1).clone()
        if step >= 3:
            expected[1].clamp_(min=12.0)
        torch.testing.assert_close(output, expected)
        env.observation.fill_(-100.0)
        rng_state = torch.get_rng_state()
        torch.testing.assert_close(manager.compute()["policy"], expected)
        torch.testing.assert_close(manager.compute_group("policy"), expected)
        assert torch.equal(torch.get_rng_state(), rng_state)


@pytest.mark.parametrize(
    ("params", "error"),
    [
        ({"delay_min_lag": -1}, ValueError),
        ({"delay_min_lag": 2, "delay_max_lag": 1}, ValueError),
        ({"delay_min_lag": 0.5}, TypeError),
        ({"delay_hold_prob": -0.1}, ValueError),
        ({"delay_hold_prob": 1.1}, ValueError),
    ],
)
def test_observation_delay_config_validation(params, error):
    """Delay requires ordered nonnegative integer bounds and a probability in [0, 1]."""
    cfg = ObservationTermCfg(func=dummy_observation, **params)
    with pytest.raises(error, match="delay"):
        cfg.validate()
