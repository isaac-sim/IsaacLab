# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for observation managers."""

from __future__ import annotations

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none
import inspect
from typing import TYPE_CHECKING, cast

import pytest
import torch

from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from isaaclab.utils import DelayBuffer, configclass, modifiers, noise

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def dummy_observation(env: DummyEnv) -> torch.Tensor:
    """Return the dummy environment observation."""
    return env.observation


class DummySimulation:
    """Minimal playing simulation double."""

    def is_playing(self) -> bool:
        """Return whether the simulated timeline is playing."""
        return True


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


def test_modifier_resolution_stays_out_of_observation_manager():
    """Observation-specific code receives resolved modifier callables from ``ManagerBase``."""
    source = inspect.getsource(ObservationManager._prepare_terms)
    assert "inspect.isclass(mod_cfg.func)" in source
    assert "string_to_callable" not in source


def test_modifier_base_cfg_marker_does_not_exist():
    """Stateful modifiers must not require a marker configuration subtype."""
    assert not hasattr(modifiers, "ModifierBaseCfg")
    assert not hasattr(modifiers, "DelayCfg"), "Observation delay uses the shared buffer directly."


@pytest.mark.parametrize(("lag", "history_length"), [(0, 2), (1, 0), (1, 2)])
def test_compute_updates_history_only_when_requested(lag, history_length):
    """History alone, delay alone, and their combination advance only on recorded samples."""
    cfg = HistoryObservationsCfg()
    cfg.policy.history_length = history_length
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

    outputs = []
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
        outputs.append((output, expected))
        env.observation.fill_(-100.0)
        rng_state = torch.get_rng_state()
        torch.testing.assert_close(manager.compute()["policy"], expected)
        torch.testing.assert_close(manager.compute_group("policy"), expected)
        assert torch.equal(torch.get_rng_state(), rng_state)
    # returned observations must not alias the manager's history or delay storage
    for output, expected in outputs:
        torch.testing.assert_close(output, expected)


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
