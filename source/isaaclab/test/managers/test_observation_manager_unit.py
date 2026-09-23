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
from isaaclab.test.utils import test_devices
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


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("lag_bounds", [(0, 0), (2, 2), (0, 3)])
@pytest.mark.parametrize("history_length", [0, 3])
def test_compute_updates_history_only_when_requested(device, lag_bounds, history_length):
    """Delay and stacked history share a recording clock, including partial resets and extra reads."""
    env = DummyEnv()
    env.device = device
    env.observation = env.observation.to(device)
    cfg = HistoryObservationsCfg()
    cfg.policy.history_length = history_length
    cfg.policy.enable_corruption = True
    cfg.policy.dummy.delay_min_lag, cfg.policy.dummy.delay_max_lag = lag_bounds
    cfg.policy.dummy.modifiers = [modifiers.ModifierCfg(func=modifiers.bias, params={"value": 1.0})]
    cfg.policy.dummy.noise = noise.ConstantNoiseCfg(bias=0.0)
    cfg.policy.dummy.clip = (-1000.0, 1000.0)
    cfg.policy.dummy.scale = 2.0
    original = cfg.to_dict()
    cfg.from_dict(original)
    manager = ObservationManager(cfg, cast("ManagerBasedEnv", env))
    buffers = manager._group_obs_term_delay_buffer["policy"]
    delay = buffers.get("dummy")
    assert isinstance(delay, DelayBuffer) if lag_bounds[1] else not buffers
    assert not hasattr(modifiers, "DelayCfg"), "Delay sampling belongs to the observation manager."
    lags = delay.time_lags.clone() if delay else torch.zeros(env.num_envs, device=device, dtype=torch.int)
    assert torch.all((lag_bounds[0] <= lags) & (lags <= lag_bounds[1]))
    samples, delivered = [[] for _ in range(env.num_envs)], [[] for _ in range(env.num_envs)]
    history = manager._group_obs_term_history_buffer["policy"].get("dummy")

    manager.compute()
    if delay:
        assert torch.all(delay.num_pushes == 0)
    if history:
        assert torch.all(history.current_length == 0)

    for step in range(11):
        if step == 5:
            manager.reset([1])
            samples[1].clear()
            delivered[1].clear()
            if delay:
                torch.testing.assert_close(delay.time_lags[0], lags[0])
                lags = delay.time_lags.clone()
                assert lag_bounds[0] <= lags[1] <= lag_bounds[1]
                assert delay.num_pushes[0] == step and delay.num_pushes[1] == 0
        env.observation.fill_(step * 10.0)
        # Vary the corruption to ensure delay stores the processed sample, not fresh noise on old data.
        manager.cfg.policy.dummy.noise.bias = float(step)
        output = manager.compute(update_history=True)["policy"]
        for index in range(env.num_envs):
            samples[index].append(2.0 * (step * 11.0 + 1.0))
            delivered[index].append(samples[index][max(0, len(samples[index]) - 1 - int(lags[index]))])
        length = max(1, history_length)
        expected = torch.tensor(
            [[values[max(0, len(values) - length + index)] for index in range(length)] for values in delivered],
            device=device,
        )
        torch.testing.assert_close(output, expected)
        if delay:
            torch.testing.assert_close(delay.time_lags, lags)
            torch.testing.assert_close(
                delay.num_pushes, torch.tensor([len(values) for values in samples], device=device)
            )
        if history:
            torch.testing.assert_close(
                history.current_length,
                torch.tensor([min(history_length, len(values)) for values in delivered], device=device),
            )
        # Both entry points must read the last recorded samples without advancing delay or history.
        if delay or history:
            env.observation.fill_(-100.0)
            torch.testing.assert_close(manager.compute()["policy"], expected)
            torch.testing.assert_close(manager.compute_group("policy"), expected)

    assert cfg.to_dict() == original
    serialized = manager.serialize()["policy"]["dummy"]["cfg"]
    assert (serialized["delay_min_lag"], serialized["delay_max_lag"]) == lag_bounds


@pytest.mark.parametrize("lag_bounds", [(-1, 2), (2, 1), (1, 0), (0.5, 2), (0, True)])
def test_observation_delay_config_validation(lag_bounds):
    """Invalid delay bounds fail configuration validation and standalone manager construction."""
    cfg = HistoryObservationsCfg()
    cfg.policy.dummy.delay_min_lag, cfg.policy.dummy.delay_max_lag = lag_bounds
    error = ValueError if all(type(value) is int for value in lag_bounds) else TypeError
    with pytest.raises(error, match="delay"):
        cfg.validate()
    with pytest.raises(error, match="delay"):
        ObservationManager(cfg, cast("ManagerBasedEnv", DummyEnv()))
