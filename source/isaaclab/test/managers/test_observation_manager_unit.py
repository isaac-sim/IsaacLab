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

from isaaclab.managers import (
    Delay,
    DelayCfg,
    ManagerTermBase,
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
)
from isaaclab.test.utils import test_devices
from isaaclab.utils import configclass, modifiers

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

    def __init__(self, num_envs: int = 2, device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.device = device
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


class CounterTerm(ManagerTermBase):
    """Stateful source used to check reset propagation through delay."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.count = torch.zeros(env.num_envs, 1, device=env.device)

    def __call__(self, env, increment: float = 1.0):
        self.count.add_(increment)
        return self.count

    def reset(self, env_ids=None):
        self.count[slice(None) if env_ids is None else env_ids] = 0


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


def test_delay_cfg_in_func_slot():
    """Wrap a callable using the existing func slot, with serializable config and partial resets."""
    cfg = HistoryObservationsCfg()
    cfg.policy.history_length = None
    cfg.policy.dummy.func = DelayCfg(term=dummy_observation, params={}, min_lag=2, max_lag=2)
    original = cfg.to_dict()
    cfg.from_dict(original)
    env = DummyEnv()
    manager = ObservationManager(cfg, env)
    manager.reset()
    assert "delay" not in ObservationTermCfg.__dataclass_fields__
    assert not hasattr(modifiers, "DelayCfg"), "Delay configuration must not belong to observation modifiers."
    for step in range(9):
        if step == 5:
            manager.reset([1])
        env.observation.fill_(step + 10)
        output = manager.compute()["policy"]
        expected = torch.tensor([[max(0, step - 2)], [max(5 if step >= 5 else 0, step - 2)]])
        torch.testing.assert_close(output, expected.float() + 10)
    assert cfg.to_dict() == original
    assert manager.serialize()["policy"]["dummy"]["cfg"]["func"] == original["policy"]["dummy"]["func"]


def test_delay_cfg_resets_wrapped_stateful_term():
    """The manager owns one resettable term that resets its source and history together."""
    cfg = HistoryObservationsCfg()
    cfg.policy.history_length = None
    cfg.policy.dummy.func = DelayCfg(term=CounterTerm, params={"increment": 2.0}, min_lag=1, max_lag=1)
    original = cfg.to_dict()
    env = DummyEnv()
    manager = ObservationManager(cfg, env)
    manager.reset()
    for _ in range(3):
        manager.compute()
    manager.reset([1])
    output = manager.compute()["policy"]
    torch.testing.assert_close(output, torch.tensor([[6.0], [2.0]]))
    assert manager.serialize()["policy"]["dummy"]["cfg"]["func"] == original["policy"]["dummy"]["func"]


def test_delay_cfg_validates_wrapped_parameters():
    """Validate the wrapped callable's signature and reject ambiguous outer parameters."""
    cfg = HistoryObservationsCfg()
    cfg.policy.dummy.func = DelayCfg(term=CounterTerm, params={"unknown": 2.0})
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ObservationManager(cfg, DummyEnv())
    cfg.policy.dummy.func.params = {}
    cfg.policy.dummy.params = {"increment": 2.0}
    with pytest.raises(ValueError, match="inside its nested configuration"):
        ObservationManager(cfg, DummyEnv())


def test_compute_updates_history_only_when_requested():
    """Observation history changes only when ``update_history`` is enabled."""
    env = DummyEnv()
    manager = ObservationManager(HistoryObservationsCfg(), cast("ManagerBasedEnv", env))
    history = manager._group_obs_term_history_buffer["policy"]["dummy"]

    torch.testing.assert_close(history.current_length, torch.zeros(env.num_envs, dtype=torch.int64))

    manager.compute()
    torch.testing.assert_close(history.current_length, torch.zeros(env.num_envs, dtype=torch.int64))

    manager.compute(update_history=True)
    torch.testing.assert_close(history.current_length, torch.ones(env.num_envs, dtype=torch.int64))
    history_after_update = history.buffer.clone()

    env.observation.add_(10.0)
    observations = manager.compute()
    policy_observation = observations["policy"]
    assert isinstance(policy_observation, torch.Tensor)
    torch.testing.assert_close(history.current_length, torch.ones(env.num_envs, dtype=torch.int64))
    torch.testing.assert_close(history.buffer, history_after_update)
    torch.testing.assert_close(policy_observation, history_after_update.reshape(env.num_envs, -1))

    manager.compute(update_history=True)
    torch.testing.assert_close(history.current_length, torch.full((env.num_envs,), 2, dtype=torch.int64))
    torch.testing.assert_close(history.buffer[:, -1], env.observation)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("shape", [(3,), (3, 2, 2)])
@pytest.mark.parametrize("settings", [{"min_lag": 2, "max_lag": 2}, {"update_period": 3}, {"hold_prob": 1.0}])
def test_delay_delivery_and_reset(device, shape, settings):
    """Fixed latency, sensor cadence, and holds share shape-preserving reset semantics."""
    cfg = DelayCfg(term=dummy_observation, per_env_phase=False, **settings)
    env = DummyEnv(shape[0], device)
    delay = Delay(cfg, env)
    for step in range(12):
        if step == 5:
            delay.reset([1])
        data = torch.full(shape, step + 10, dtype=torch.float64, device=device)
        env.observation = data
        result = delay(env)
        expected = []
        for env_id in range(shape[0]):
            start = 5 if env_id == 1 and step >= 5 else 0
            if cfg.hold_prob == 1.0:
                sample = start
            elif cfg.update_period > 1:
                sample = start + (step - start) // cfg.update_period * cfg.update_period
            else:
                sample = max(start, step - cfg.max_lag)
            expected.append(sample + 10)
        expected = torch.tensor(expected, dtype=data.dtype, device=device).view(3, *([1] * (len(shape) - 1)))
        torch.testing.assert_close(result, expected.expand_as(data))
        result.fill_(-999)  # Downstream in-place processing must not corrupt held frames.


@pytest.mark.parametrize("device", test_devices())
def test_delay_stochastic_delivery(device):
    """Jitter never delivers an older sample, and bounded GPU lag generation never calls the setter."""
    cfg = DelayCfg(term=dummy_observation, min_lag=1, max_lag=4, update_period=3, hold_prob=0.2)
    env = DummyEnv(16, device)
    delay = Delay(cfg, env)
    assert "_step" not in vars(delay), "The buffer owns the per-environment clock."
    delay._buffer.set_time_lag = lambda *args: pytest.fail("The hot path must not validate lags on the host.")
    previous = torch.full((16, 1), -1.0, device=device)
    for step in range(40):
        env.observation = torch.full_like(previous, step)
        output = delay(env)
        assert torch.all(output >= previous)
        assert torch.all(output <= max(0, step - cfg.min_lag))
        assert torch.all(delay._buffer.time_lags >= cfg.min_lag)
        assert torch.all(delay._buffer.time_lags <= cfg.max_lag)
        previous = output

    shared = Delay(DelayCfg(term=dummy_observation, max_lag=4, per_env=False), env)
    for step in range(12):
        env.observation = torch.full_like(previous, step)
        output = shared(env)
        assert torch.all(output == output[0])


@pytest.mark.parametrize(
    "settings", [{"min_lag": -1}, {"min_lag": 2, "max_lag": 1}, {"update_period": 0}, {"hold_prob": 1.1}]
)
def test_delay_cfg_validation(settings):
    """Reject invalid scheduling parameters in the config before allocating any buffer."""
    cfg = DelayCfg(term=dummy_observation, **settings)
    with pytest.raises(ValueError):
        cfg.validate()
    with pytest.raises(ValueError):
        Delay(cfg, DummyEnv())


@pytest.mark.parametrize("device", test_devices())
def test_delay_schedule_cuda_graph(device):
    """Cadence and held output advance on-device during graph replay."""
    if not device.startswith("cuda"):
        pytest.skip("CUDA graph replay requires CUDA.")
    with torch.cuda.device(device):
        env = DummyEnv(2, device)
        delay = Delay(DelayCfg(term=dummy_observation, update_period=3, per_env_phase=False), env)
        data = torch.zeros(2, 1, device=device)
        env.observation = data
        delay(env)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = delay(env)
        delay.reset()
        for step in range(10):
            if step == 5:
                delay.reset([1])
            data.fill_(step)
            graph.replay()
            expected = torch.full_like(data, step // 3 * 3)
            if step >= 5:
                expected[1] = 5 + (step - 5) // 3 * 3
            torch.testing.assert_close(output, expected)
