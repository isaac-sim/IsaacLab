# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
import pytest
from collections import namedtuple

from isaaclab.managers import ManagerTermBase, ObservationGroupCfg, ObservationManager, ObservationTermCfg
from isaaclab.utils import configclass, modifiers


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


@pytest.fixture
def test_env():
    """Set up the environment for testing."""
    # set up the environment
    dt = 0.01
    num_envs = 20
    device = "cuda:0"
    # create dummy environment
    env = namedtuple("ManagerBasedEnv", ["num_envs", "device", "data", "dt"])(
        num_envs, device, MyDataClass(num_envs, device), dt
    )
    return env


def test_str(test_env):
    """Test the string representation of the observation manager."""

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class SampleGroupCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            term_1 = ObservationTermCfg(func="__main__:grilled_chicken", scale=10)
            term_2 = ObservationTermCfg(func=grilled_chicken, scale=2)
            term_3 = ObservationTermCfg(func=grilled_chicken_with_bbq, scale=5, params={"bbq": True})
            term_4 = ObservationTermCfg(
                func=grilled_chicken_with_yoghurt, scale=1.0, params={"hot": False, "bland": 2.0}
            )
            term_5 = ObservationTermCfg(
                func=grilled_chicken_with_yoghurt_and_bbq, scale=1.0, params={"hot": False, "bland": 2.0}
            )

        policy: ObservationGroupCfg = SampleGroupCfg()

    # create observation manager
    cfg = MyObservationManagerCfg()
    obs_man = ObservationManager(cfg, test_env)
    assert len(obs_man.active_terms["policy"]) == 5
    # print the expected string
    obs_man_str = str(obs_man)
    print()
    print(obs_man_str)
    obs_man_str_split = obs_man_str.split("|")
    term_1_str_index = obs_man_str_split.index(" term_1           ")
    term_1_str_shape = obs_man_str_split[term_1_str_index + 1].strip()
    assert term_1_str_shape == "(4,)"


def test_str_with_history(test_env):
    """Test the string representation of the observation manager with history terms."""

    TERM_1_HISTORY = 5

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class SampleGroupCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            term_1 = ObservationTermCfg(func="__main__:grilled_chicken", scale=10, history_length=TERM_1_HISTORY)
            term_2 = ObservationTermCfg(func=grilled_chicken, scale=2)
            term_3 = ObservationTermCfg(func=grilled_chicken_with_bbq, scale=5, params={"bbq": True})
            term_4 = ObservationTermCfg(
                func=grilled_chicken_with_yoghurt, scale=1.0, params={"hot": False, "bland": 2.0}
            )
            term_5 = ObservationTermCfg(
                func=grilled_chicken_with_yoghurt_and_bbq, scale=1.0, params={"hot": False, "bland": 2.0}
            )

        policy: ObservationGroupCfg = SampleGroupCfg()

    # create observation manager
    cfg = MyObservationManagerCfg()
    obs_man = ObservationManager(cfg, test_env)
    assert len(obs_man.active_terms["policy"]) == 5
    # print the expected string
    obs_man_str = str(obs_man)
    print()
    print(obs_man_str)
    obs_man_str_split = obs_man_str.split("|")
    term_1_str_index = obs_man_str_split.index(" term_1           ")
    term_1_str_shape = obs_man_str_split[term_1_str_index + 1].strip()
    assert term_1_str_shape == "(20,)"


def test_config_equivalence(test_env):
    """Test the equivalence of observation manager created from different config types."""

    # create from config class
    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class SampleGroupCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            your_term = ObservationTermCfg(func="__main__:grilled_chicken", scale=10)
            his_term = ObservationTermCfg(func=grilled_chicken, scale=2)
            my_term = ObservationTermCfg(func=grilled_chicken_with_bbq, scale=5, params={"bbq": True})
            her_term = ObservationTermCfg(
                func=grilled_chicken_with_yoghurt, scale=1.0, params={"hot": False, "bland": 2.0}
            )

        policy = SampleGroupCfg()
        critic = SampleGroupCfg(concatenate_terms=False, her_term=None)

    cfg = MyObservationManagerCfg()
    obs_man_from_cfg = ObservationManager(cfg, test_env)

    # create from config class
    @configclass
    class MyObservationManagerAnnotatedCfg:
        """Test config class for observation manager with annotations on terms."""

        @configclass
        class SampleGroupCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            your_term: ObservationTermCfg = ObservationTermCfg(func="__main__:grilled_chicken", scale=10)
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
    obs_man_from_annotated_cfg = ObservationManager(cfg, test_env)

    # check equivalence
    # parsed terms
    assert obs_man_from_cfg.active_terms == obs_man_from_annotated_cfg.active_terms
    assert obs_man_from_cfg.group_obs_term_dim == obs_man_from_annotated_cfg.group_obs_term_dim
    assert obs_man_from_cfg.group_obs_dim == obs_man_from_annotated_cfg.group_obs_dim
    # parsed term configs
    assert obs_man_from_cfg._group_obs_term_cfgs == obs_man_from_annotated_cfg._group_obs_term_cfgs
    assert obs_man_from_cfg._group_obs_concatenate == obs_man_from_annotated_cfg._group_obs_concatenate


def test_config_terms(test_env):
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
    obs_man = ObservationManager(cfg, test_env)

    # check number of terms
    assert len(obs_man.active_terms["policy"]) == 2
    assert len(obs_man.active_terms["critic"]) == 1
    assert len(obs_man.active_terms["mixed"]) == 2
    assert len(obs_man.active_terms["image"]) == 2

    # check term dimensions
    assert obs_man.group_obs_term_dim["policy"]["term_1"] == 4
    assert obs_man.group_obs_term_dim["policy"]["term_2"] == 1
    assert obs_man.group_obs_term_dim["critic"]["term_1"] == 4
    assert obs_man.group_obs_term_dim["mixed"]["term_1"] == 4
    assert obs_man.group_obs_term_dim["mixed"]["term_2"] == (128, 256, 1)
    assert obs_man.group_obs_term_dim["image"]["term_1"] == (128, 256, 1)
    assert obs_man.group_obs_term_dim["image"]["term_2"] == (128, 256, 3)

    # check group dimensions
    assert obs_man.group_obs_dim["policy"] == 5
    assert obs_man.group_obs_dim["critic"] == 4
    assert obs_man.group_obs_dim["mixed"] == None  # noqa: E711
    assert obs_man.group_obs_dim["image"] == None  # noqa: E711


def test_compute(test_env):
    """Test the computation of observations."""

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
    obs_man = ObservationManager(cfg, test_env)

    # compute observations
    obs = obs_man.compute()

    # check policy observations
    assert obs["policy"].shape == (test_env.num_envs, 11)  # 4 + 1 + 3 + 3
    assert torch.allclose(obs["policy"][:, :4], 10 * torch.ones(test_env.num_envs, 4, device=test_env.device))
    assert torch.allclose(obs["policy"][:, 4], torch.zeros(test_env.num_envs, device=test_env.device))
    assert torch.allclose(obs["policy"][:, 5:8], test_env.data.pos_w)
    assert torch.allclose(obs["policy"][:, 8:], 1.5 * test_env.data.lin_vel_w)

    # check critic observations
    assert obs["critic"].shape == (test_env.num_envs, 12)  # 3 + 3 + 3 + 3
    assert torch.allclose(obs["critic"][:, :3], test_env.data.pos_w)
    assert torch.allclose(obs["critic"][:, 3:6], 1.5 * test_env.data.lin_vel_w)
    assert torch.allclose(obs["critic"][:, 6:9], test_env.data.pos_w)
    assert torch.allclose(obs["critic"][:, 9:], 1.5 * test_env.data.lin_vel_w)

    # check image observations
    assert obs["image"].shape == (test_env.num_envs, 128, 256, 4)  # (128, 256, 1) + (128, 256, 3)
    assert torch.allclose(obs["image"][:, :, :, 0], 1.5 * 0.5 * torch.ones(test_env.num_envs, 128, 256, device=test_env.device))
    assert torch.allclose(obs["image"][:, :, :, 1:], 0.5 * 0.1 * torch.ones(test_env.num_envs, 128, 256, 3, device=test_env.device))


def test_compute_with_history(test_env):
    """Test the computation of observations with history."""

    HISTORY_LENGTH = 5

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class PolicyCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            term_1 = ObservationTermCfg(func=grilled_chicken, history_length=HISTORY_LENGTH)
            # total observation size: term_dim (4) * history_len (5) = 20
            term_2 = ObservationTermCfg(func=lin_vel_w_data)
            # total observation size: term_dim (3) = 3

        policy: ObservationGroupCfg = PolicyCfg()

    # create observation manager
    cfg = MyObservationManagerCfg()
    obs_man = ObservationManager(cfg, test_env)

    # compute observations
    obs = obs_man.compute()

    # check policy observations
    assert obs["policy"].shape == (test_env.num_envs, 23)  # 20 + 3
    assert torch.allclose(obs["policy"][:, :20], torch.ones(test_env.num_envs, 20, device=test_env.device))
    assert torch.allclose(obs["policy"][:, 20:], test_env.data.lin_vel_w)


def test_compute_with_2d_history(test_env):
    """Test the computation of observations with 2D history."""

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
    obs_man = ObservationManager(cfg, test_env)

    # compute observations
    obs = obs_man.compute()

    # check flattened policy observations
    assert obs["flat_obs_policy"].shape == (test_env.num_envs, 163840)  # 128 * 256 * 5
    assert torch.allclose(obs["flat_obs_policy"], torch.ones(test_env.num_envs, 163840, device=test_env.device))

    # check policy observations
    assert obs["policy"].shape == (test_env.num_envs, 5, 128, 256, 1)
    assert torch.allclose(obs["policy"], torch.ones(test_env.num_envs, 5, 128, 256, 1, device=test_env.device))


def test_compute_with_group_history(test_env):
    """Test the computation of observations with group history."""

    GROUP_HISTORY_LENGTH = 10
    TERM_HISTORY_LENGTH = 5

    @configclass
    class MyObservationManagerCfg:
        """Test config class for observation manager."""

        @configclass
        class PolicyCfg(ObservationGroupCfg):
            """Test config class for policy observation group."""

            history_length = GROUP_HISTORY_LENGTH
            # group level history length will override all terms
            term_1 = ObservationTermCfg(func=grilled_chicken, history_length=TERM_HISTORY_LENGTH)
            # total observation size: term_dim (4) * history_len (5) = 20
            # with override total obs size: term_dim (4) * history_len (10) = 40
            term_2 = ObservationTermCfg(func=lin_vel_w_data)
            # total observation size: term_dim (3) = 3
            # with override total obs size: term_dim (3) * history_len (10) = 30

        policy: ObservationGroupCfg = PolicyCfg()

    # create observation manager
    cfg = MyObservationManagerCfg()
    obs_man = ObservationManager(cfg, test_env)

    # compute observations
    obs = obs_man.compute()

    # check policy observations
    assert obs["policy"].shape == (test_env.num_envs, 70)  # 40 + 30
    assert torch.allclose(obs["policy"][:, :40], torch.ones(test_env.num_envs, 40, device=test_env.device))
    assert torch.allclose(obs["policy"][:, 40:], test_env.data.lin_vel_w.repeat(1, 10))


def test_invalid_observation_config(test_env):
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
        obs_man = ObservationManager(cfg, test_env)


def test_callable_class_term(test_env):
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

    # create observation manager config
    cfg = MyObservationManagerCfg()
    # create observation manager
    obs_man = ObservationManager(cfg, test_env)

    # compute observations
    obs = obs_man.compute()

    # check policy observations
    assert obs["policy"].shape == (test_env.num_envs, 5)  # 4 + 1
    assert torch.allclose(obs["policy"][:, :4], 10 * torch.ones(test_env.num_envs, 4, device=test_env.device))
    assert torch.allclose(obs["policy"][:, 4], 0.2 * 0.5 * torch.ones(test_env.num_envs, 1, device=test_env.device))


def test_non_callable_class_term(test_env):
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
        obs_man = ObservationManager(cfg, test_env)


def test_modifier_compute(test_env):
    """Test the computation of observations with modifiers."""

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
    obs_man = ObservationManager(cfg, test_env)

    # compute observations
    obs = obs_man.compute()

    # check policy observations
    assert obs["policy"].shape == (test_env.num_envs, 3, 3)  # 3 terms, each with shape (3,)
    assert torch.allclose(obs["policy"][:, 0], test_env.data.pos_w)
    assert torch.allclose(obs["policy"][:, 1], modifier_1(test_env.data.pos_w))
    assert torch.allclose(obs["policy"][:, 2], modifier_4(modifier_1(test_env.data.pos_w)))

    # check critic observations
    assert obs["critic"].shape == (test_env.num_envs, 4, 3)  # 4 terms, each with shape (3,)
    assert torch.allclose(obs["critic"][:, 0], test_env.data.pos_w)
    assert torch.allclose(obs["critic"][:, 1], modifier_1(test_env.data.pos_w))
    assert torch.allclose(obs["critic"][:, 2], modifier_2(modifier_1(test_env.data.pos_w)))
    assert torch.allclose(obs["critic"][:, 3], modifier_3(modifier_2(modifier_1(test_env.data.pos_w))))


def test_modifier_invalid_config(test_env):
    """Test the invalid modifier config."""

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
    # check the invalid config
    with pytest.raises(ValueError):
        obs_man = ObservationManager(cfg, test_env)


if __name__ == "__main__":
    run_tests()
