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
    ManagerTermBase,
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
    SceneEntityCfg,
)
from isaaclab.utils import configclass, modifiers

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def dummy_observation(env: DummyEnv) -> torch.Tensor:
    """Return the dummy environment observation."""
    return env.observation


class DummySimulation:
    """Minimal playing simulation double."""

    playing = True

    def is_playing(self) -> bool:
        """Return whether the simulated timeline is playing."""
        return self.playing


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


class PreparedObservation(ManagerTermBase):
    """Term whose preparation state must survive manager construction."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.reset_ids = []
        self.close_count = 0
        env.prepared.append(self)

    @classmethod
    def prepare_scene(cls, cfg, env):
        assert not env.sim.is_playing()
        return cls(cfg, env)

    def __deepcopy__(self, memo):
        raise AssertionError("Runtime observation terms must not be copied.")

    def __call__(self, env, sensor_cfg, gain=1.0):
        assert sensor_cfg is self.cfg.params["sensor_cfg"]
        return env.observation * gain

    def reset(self, env_ids=None):
        self.reset_ids.append(env_ids)

    def close(self):
        self.close_count += 1


@pytest.mark.parametrize("string_func", [False, True])
def test_prepared_observation_adoption_preserves_state_and_partial_reset(string_func):
    cfg = HistoryObservationsCfg()
    cfg.policy.history_length = None
    cfg.policy.dummy = ObservationTermCfg(
        func=f"{__name__}:PreparedObservation" if string_func else PreparedObservation,
        params={"sensor_cfg": SceneEntityCfg("camera"), "gain": 2.0},
    )
    env = DummyEnv()
    env.sim.playing = False
    env.scene = {"camera": object()}
    env.prepared = []
    prepared = ObservationManager.prepare_scene(cfg, env)
    instance = prepared["policy/dummy"]
    env.sim.playing = True

    manager = ObservationManager(cfg, env, prepared_terms=prepared)
    assert prepared == {}
    assert env.prepared == [instance]
    assert manager.cfg.policy.dummy.func is instance
    assert instance.cfg is manager.cfg.policy.dummy
    assert instance.cfg.params["sensor_cfg"] is not cfg.policy.dummy.params["sensor_cfg"]
    torch.testing.assert_close(manager.compute()["policy"], env.observation * 2.0)
    manager.reset(env_ids=[1])
    assert instance.reset_ids == [None, [1]]

    manager.close()
    manager.close()
    assert instance.close_count == 1


@pytest.mark.parametrize("failure", ["prepare", "signature", "initialize"])
def test_prepared_observations_close_after_startup_failure(failure):
    class FailingObservation(ManagerTermBase):
        @classmethod
        def prepare_scene(cls, cfg, env):
            if failure == "prepare":
                raise RuntimeError("preparation failed")
            return

        def __call__(self, env):
            raise RuntimeError("initialization failed")

    cfg = HistoryObservationsCfg()
    cfg.policy.history_length = None
    cfg.policy.dummy = ObservationTermCfg(func=PreparedObservation, params={"sensor_cfg": SceneEntityCfg("camera")})
    cfg.policy.failing = ObservationTermCfg(
        func=dummy_observation if failure == "signature" else FailingObservation,
        params={"unexpected": True} if failure == "signature" else {},
    )
    env = DummyEnv()
    env.sim.playing = False
    env.scene = {"camera": object()}
    env.prepared = []

    with pytest.raises((RuntimeError, ValueError, TypeError)):
        prepared = ObservationManager.prepare_scene(cfg, env)
        env.sim.playing = True
        ObservationManager(cfg, env, prepared_terms=prepared)
    assert len(env.prepared) == 1
    assert env.prepared[0].close_count == 1


def test_observation_close_continues_after_term_error():
    class FailingCloseObservation(PreparedObservation):
        def close(self):
            super().close()
            raise RuntimeError("close failed")

    cfg = HistoryObservationsCfg()
    cfg.policy.history_length = None
    cfg.policy.dummy = ObservationTermCfg(func=PreparedObservation, params={"sensor_cfg": SceneEntityCfg("camera")})
    cfg.policy.failing = ObservationTermCfg(
        func=FailingCloseObservation, params={"sensor_cfg": SceneEntityCfg("camera")}
    )
    env = DummyEnv()
    env.sim.playing = False
    env.scene = {"camera": object()}
    env.prepared = []
    prepared = ObservationManager.prepare_scene(cfg, env)
    env.sim.playing = True
    manager = ObservationManager(cfg, env, prepared_terms=prepared)

    with pytest.raises(RuntimeError, match="close an observation term"):
        manager.close()
    manager.close()
    assert [term.close_count for term in env.prepared] == [1, 1]


def test_default_observation_preparation_keeps_normal_construction():
    class NormalObservation(ManagerTermBase):
        def __call__(self, env):
            return env.observation

    cfg = HistoryObservationsCfg()
    cfg.policy.history_length = None
    cfg.policy.dummy.func = NormalObservation
    env = DummyEnv()
    assert ObservationManager.prepare_scene(cfg, env) == {}
    manager = ObservationManager(cfg, env)
    torch.testing.assert_close(manager.compute()["policy"], env.observation)
    manager.close()


def test_scene_preparation_does_not_mutate_user_configuration():
    class NormalizingObservation(PreparedObservation):
        @classmethod
        def prepare_scene(cls, cfg, env):
            cfg.params["gain"] = 3.0
            return super().prepare_scene(cfg, env)

    cfg = HistoryObservationsCfg()
    cfg.policy.dummy = ObservationTermCfg(
        func=NormalizingObservation, params={"sensor_cfg": SceneEntityCfg("camera"), "gain": 1.0}
    )
    env = DummyEnv()
    env.sim.playing = False
    env.prepared = []
    prepared = ObservationManager.prepare_scene(cfg, env)
    assert cfg.policy.dummy.params["gain"] == 1.0
    assert prepared["policy/dummy"].cfg.params["gain"] == 3.0
    prepared["policy/dummy"].close()
