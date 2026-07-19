# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Warp-first event manager launch replay."""

from __future__ import annotations

from collections.abc import Hashable
from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab_experimental.managers import EventManager, EventTermCfg, ManagerTermBase


class _LaunchRecorder:
    """Record cached-launch sites while executing the kernels eagerly."""

    def __init__(self, device: str):
        self._device = device
        self.sites: list[Hashable | None] = []
        self.reset_count = 0

    def launch(
        self,
        kernel: wp.Kernel,
        dim: int,
        inputs=(),
        outputs=(),
        *,
        stream: wp.Stream | None = None,
        site: Hashable | None = None,
    ) -> None:
        self.sites.append(site)
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=self._device, stream=stream)

    def reset(self) -> None:
        self.reset_count += 1


class _Env:
    """Minimal owner for event manager tests."""

    def __init__(self, num_envs: int = 4, *, is_playing: bool = True):
        self.num_envs = num_envs
        self.device = "cpu"
        self.is_playing = is_playing
        self.sim = SimpleNamespace(is_playing=lambda: self.is_playing)
        self.scene = SimpleNamespace(cfg=SimpleNamespace(replicate_physics=False))
        self.rng_state_wp = wp.array(range(11, 11 + num_envs), dtype=wp.uint32, device=self.device)
        self._warp_launch = _LaunchRecorder(self.device)
        self.class_reset_masks: list[wp.array] = []
        self.class_reset_tags: list[str] = []
        self.cache_invalidations = 0

    def invalidate_wp_graphs(self) -> None:
        self.cache_invalidations += 1
        self._warp_launch.reset()


def _record_event(env: _Env, env_mask: wp.array(dtype=wp.bool)) -> None:
    del env, env_mask


class _StatefulResetTerm(ManagerTermBase):
    """Stateful reset term used to verify class-term registration."""

    def reset(self, env_mask: wp.array(dtype=wp.bool) | None = None) -> None:
        self._env.class_reset_masks.append(env_mask)

    def __call__(self, env: _Env, env_mask: wp.array(dtype=wp.bool)) -> None:
        del env, env_mask


class _TaggedResetTerm(ManagerTermBase):
    """Stateful reset term used to verify configuration replacement."""

    def __init__(self, cfg: EventTermCfg, env: _Env):
        super().__init__(cfg, env)
        self._tag: str = cfg.params["tag"]

    def reset(self, env_mask: wp.array(dtype=wp.bool) | None = None) -> None:
        del env_mask
        self._env.class_reset_tags.append(self._tag)

    def __call__(self, env: _Env, env_mask: wp.array(dtype=wp.bool), tag: str) -> None:
        del env, env_mask, tag


def test_recurring_bookkeeping_uses_stable_launch_sites() -> None:
    """Only recurring event bookkeeping should use the environment launch cache."""
    env = _Env()
    manager = EventManager(
        {
            "global_interval": EventTermCfg(
                func=_record_event,
                mode="interval",
                interval_range_s=(0.1, 0.1),
                is_global_time=True,
            ),
            "local_interval": EventTermCfg(
                func=_record_event,
                mode="interval",
                interval_range_s=(0.1, 0.1),
            ),
            "reset_first": EventTermCfg(func=_record_event, mode="reset"),
            "reset_second": EventTermCfg(
                func=_record_event,
                mode="reset",
                min_step_count_between_reset=2,
            ),
        },
        env,
    )

    # Timer allocation, initialization, and RNG seeding are one-shot eager launches.
    assert env._warp_launch.sites == []

    env_mask = wp.array([True, False, True, False], dtype=wp.bool, device=env.device)
    global_step_count = wp.array([3], dtype=wp.int32, device=env.device)
    manager.apply("interval", dt=0.1)
    manager.reset(env_mask=env_mask)
    manager.apply("reset", env_mask_wp=env_mask, global_env_step_count=global_step_count)
    wp.synchronize()

    assert env._warp_launch.sites == [
        ("event_manager", "interval_step_global", 0, 0.1, 0.1, 0.1),
        ("event_manager", "interval_step_per_env", 1, 0.1, 0.1, 0.1),
        ("event_manager", "interval_reset", 1, env_mask, 0.1, 0.1),
        ("event_manager", "reset_valid_mask", 0, env_mask, 0),
        ("event_manager", "reset_valid_mask", 1, env_mask, 2),
    ]


def test_instantiated_class_term_remains_registered_for_reset() -> None:
    """A class term instantiated while parsing should still receive manager resets."""
    env = _Env()
    manager = EventManager(
        {"stateful_reset": EventTermCfg(func=_StatefulResetTerm, mode="reset")},
        env,
    )
    env_mask = wp.array([False, True, False, True], dtype=wp.bool, device=env.device)

    assert isinstance(manager.get_term_cfg("stateful_reset").func, _StatefulResetTerm)
    assert manager._mode_class_term_cfgs["reset"] == [manager.get_term_cfg("stateful_reset")]

    manager.reset(env_mask=env_mask)

    assert env.class_reset_masks == [env_mask]


def test_set_term_cfg_replaces_instantiated_class_term() -> None:
    """Replacing a class term should update reset ownership and invalidate cached work."""
    env = _Env()
    manager = EventManager(
        {"stateful_reset": EventTermCfg(func=_TaggedResetTerm, mode="reset", params={"tag": "old"})},
        env,
    )
    old_term = manager.get_term_cfg("stateful_reset").func
    replacement = EventTermCfg(func=_TaggedResetTerm, mode="reset", params={"tag": "new"})

    manager.set_term_cfg("stateful_reset", replacement)

    new_cfg = manager.get_term_cfg("stateful_reset")
    assert isinstance(new_cfg.func, _TaggedResetTerm)
    assert new_cfg.func is not old_term
    assert manager._mode_class_term_cfgs["reset"] == [new_cfg]
    assert env.cache_invalidations == 1
    assert env._warp_launch.reset_count == 1

    manager.reset(env_mask=wp.array([True] * env.num_envs, dtype=wp.bool, device=env.device))

    assert env.class_reset_tags == ["new"]


@pytest.mark.parametrize("use_dict_cfg", [False, True], ids=["config-object", "dictionary"])
def test_set_term_cfg_before_play_updates_deferred_resolution_source(use_dict_cfg: bool) -> None:
    """A pre-play replacement should be the term instantiated during deferred resolution."""
    env = _Env(is_playing=False)
    initial = EventTermCfg(func=_TaggedResetTerm, mode="reset", params={"tag": "old"})
    cfg = {"stateful_reset": initial} if use_dict_cfg else SimpleNamespace(stateful_reset=initial)
    manager = EventManager(cfg, env)
    replacement = EventTermCfg(func=_TaggedResetTerm, mode="reset", params={"tag": "new"})

    manager.set_term_cfg("stateful_reset", replacement)
    env.is_playing = True
    manager._resolve_terms_callback(None)

    resolved_cfg = manager.get_term_cfg("stateful_reset")
    source_cfg = manager.cfg["stateful_reset"] if use_dict_cfg else manager.cfg.stateful_reset
    assert source_cfg is resolved_cfg
    assert isinstance(resolved_cfg.func, _TaggedResetTerm)

    env_mask = wp.array([True] * env.num_envs, dtype=wp.bool, device=env.device)
    manager.apply(
        "reset",
        env_mask_wp=env_mask,
        global_env_step_count=wp.zeros(1, dtype=wp.int32, device=env.device),
    )
    manager.reset(env_mask=env_mask)

    assert env.class_reset_tags == ["new"]


def test_set_interval_term_rejects_missing_range_without_side_effects() -> None:
    """An invalid interval replacement should not mutate the term or invalidate caches."""
    env = _Env()
    manager = EventManager(
        {
            "interval": EventTermCfg(
                func=_record_event,
                mode="interval",
                interval_range_s=(0.1, 0.2),
            )
        },
        env,
    )
    old_cfg = manager.get_term_cfg("interval")

    with pytest.raises(ValueError, match="interval_range_s"):
        manager.set_term_cfg("interval", EventTermCfg(func=_record_event, mode="interval"))

    assert manager.get_term_cfg("interval") is old_cfg
    assert env.cache_invalidations == 0
    assert env._warp_launch.reset_count == 0


def test_set_reset_term_rejects_negative_step_count_without_side_effects() -> None:
    """An invalid reset replacement should not mutate the term or invalidate caches."""
    env = _Env()
    manager = EventManager(
        {"reset": EventTermCfg(func=_record_event, mode="reset")},
        env,
    )
    old_cfg = manager.get_term_cfg("reset")

    with pytest.raises(ValueError, match="min_step_count_between_reset"):
        manager.set_term_cfg(
            "reset",
            EventTermCfg(func=_record_event, mode="reset", min_step_count_between_reset=-1),
        )

    assert manager.get_term_cfg("reset") is old_cfg
    assert env.cache_invalidations == 0
    assert env._warp_launch.reset_count == 0
