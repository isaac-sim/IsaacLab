# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from collections.abc import Callable
from copy import deepcopy
from gymnasium.envs.registration import registry as gym_registry
from typing import Any

from .black_box_wrapper import BlackBoxWrapper
from .factories import MP_DEFAULTS, get_basis_generator, get_controller, get_phase_generator, get_trajectory_generator


def _nested_update(base: dict, update: dict) -> dict:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _nested_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base


def _normalize_devices(config: dict, device: str | torch.device | None) -> dict:
    """Ensure all device fields and tensor values align with the target device.

    Args:
        config (dict): Arbitrary nested MP config.
        device (str | torch.device | None): Target device; when `None`, values are left
            unchanged.

    Returns:
        dict: Config with any `"device"` entries set to `device` and tensor values moved.

    Notes:
        This is called after merge so late overrides cannot force a different device.
        Nested dicts are processed recursively; non-tensor leaves are untouched.
    """

    def _apply(val):
        if torch.is_tensor(val) and device is not None:
            return val.to(device)
        return val

    for k, v in config.items():
        if isinstance(v, dict):
            # overwrite explicit device entries to avoid hardcoding cuda:0
            if device is not None and "device" in v:
                v["device"] = device
            config[k] = _normalize_devices(v, device)
        else:
            config[k] = _apply(v)
    return config


def _merge_mp_config(
    mp_type: str,
    mp_wrapper_cls,
    mp_config_override: dict[str, Any] | None,
    env_device: str | torch.device | None,
):
    """Merge MP configuration in the order: `MP_DEFAULTS -> wrapper.mp_config -> override`.

    Args:
        mp_type (str): Motion primitive type key (e.g., `"ProDMP"`, `"ProMP"`, `"DMP"`).
        mp_wrapper_cls: Wrapper class exposing `mp_config` overrides per `mp_type`.
        mp_config_override (dict | None): User override that wins over defaults and class
            config.
        env_device (str | torch.device | None): Target device propagated into every
            component's kwargs and used to relocate any tensor values.

    Returns:
        dict: Merged config ready for factory construction. Ensures `black_box_kwargs`
        includes a `reward_aggregation` function (defaults to `torch.sum`).

    Notes:
        Each kwargs block gains a `device` entry when missing so downstream factories
        produce modules on the same device as the wrapped environment. `_normalize_devices`
        enforces consistency even when overrides contain tensors.
    """

    config = deepcopy(MP_DEFAULTS.get(mp_type, {}))
    _nested_update(config, deepcopy(getattr(mp_wrapper_cls, "mp_config", {}).get(mp_type, {})))
    _nested_update(config, mp_config_override or {})

    # Propagate device into every kwargs block and move tensors accordingly.
    for key in (
        "trajectory_generator_kwargs",
        "phase_generator_kwargs",
        "basis_generator_kwargs",
        "controller_kwargs",
        "black_box_kwargs",
    ):
        block = config.get(key, {})
        if isinstance(block, dict) and env_device is not None and "device" not in block:
            block["device"] = env_device
    config = _normalize_devices(config, env_device)

    black_box_kwargs = config.setdefault("black_box_kwargs", {})
    if "reward_aggregation" not in black_box_kwargs:
        black_box_kwargs["reward_aggregation"] = torch.sum

    return config


def make_mp_env(
    base_id: str,
    mp_wrapper_cls: Callable,
    mp_type: str = "ProDMP",
    device: str | torch.device | None = None,
    mp_config_override: dict[str, Any] | None = None,
    env_make_kwargs: dict[str, Any] | None = None,
):
    """Construct an MP-enabled environment on top of an existing step-based env.

    Args:
        base_id (str): Gym id of the base environment to wrap.
        mp_wrapper_cls (Callable): Wrapper class deriving from `RawMPInterface`.
        mp_type (str): Key into `MP_DEFAULTS` and wrapper `mp_config` (`"ProDMP"` by default).
        device (str | torch.device | None): Device used for all MP components and for
            relocating tensors; falls back to `env.device` or CPU.
        mp_config_override (dict | None): User overrides applied after defaults and class
            config. Keys mirror `MP_DEFAULTS`.
        env_make_kwargs (dict | None): Extra kwargs forwarded to `gym.make(base_id, ...)`.

    Returns:
        BlackBoxWrapper: Wrapped environment ready for Gym RL or IsaacLab managers.

    Flow:
        1. Instantiate the base env and wrap it with `mp_wrapper_cls`.
        2. Merge config (`MP_DEFAULTS -> wrapper.mp_config -> mp_config_override`) and
           propagate `device`.
        3. Build phase, basis, trajectory generator, and controller via factories.
        4. Derive `duration` from overrides, `env.unwrapped.max_episode_length_s`, or
           `max_episode_length * step_dt`; fall back to 1.0s.
        5. Construct `BlackBoxWrapper` with the components and merged kwargs.

    Notes:
        The returned wrapper preserves Fancy Gym parity by reporting the MP parameter
        space as `action_space` while internally stepping the base env with controller
        outputs.
    """
    env = gym.make(base_id, **(env_make_kwargs or {}))
    env_device = device or getattr(env, "device", None) or getattr(env.unwrapped, "device", None) or "cpu"
    env = mp_wrapper_cls(env)

    config = _merge_mp_config(mp_type, mp_wrapper_cls, mp_config_override, env_device)

    traj_kwargs = config.get("trajectory_generator_kwargs", {})
    phase_kwargs = config.get("phase_generator_kwargs", {})
    basis_kwargs = config.get("basis_generator_kwargs", {})
    controller_kwargs = config.get("controller_kwargs", {})
    black_box_kwargs = config.get("black_box_kwargs", {})

    phase_gen = get_phase_generator(**phase_kwargs)
    basis_gen = get_basis_generator(phase_generator=phase_gen, **basis_kwargs)

    if hasattr(env, "single_action_space"):
        action_dim = int(env.single_action_space.shape[0])
    else:
        action_dim = int(env.action_space.shape[-1])
    traj_gen = get_trajectory_generator(action_dim=action_dim, basis_generator=basis_gen, **traj_kwargs)

    controller = get_controller(**controller_kwargs)

    duration = traj_kwargs.get("duration")
    if duration is None:
        duration = getattr(env.unwrapped, "max_episode_length_s", None)
    if duration is None and hasattr(env.unwrapped, "max_episode_length") and hasattr(env.unwrapped, "step_dt"):
        duration = env.unwrapped.max_episode_length * env.unwrapped.step_dt
    if duration is None:
        duration = 1.0

    wrapped = BlackBoxWrapper(
        env,
        trajectory_generator=traj_gen,
        tracking_controller=controller,
        duration=duration,
        **black_box_kwargs,
    )
    return wrapped


def upgrade(
    mp_id: str,
    base_id: str,
    mp_wrapper_cls: Callable,
    mp_type: str = "ProDMP",
    device: str | torch.device = "cuda:0",
    mp_config_override: dict[str, Any] | None = None,
    env_make_kwargs: dict[str, Any] | None = None,
):
    """Register a gym id for an MP variant of an existing environment.

    Args:
        mp_id (str): Target Gym id for the MP environment; returned unchanged when
            already registered.
        base_id (str): Gym id of the underlying step-based environment.
        mp_wrapper_cls (Callable): Wrapper implementing `RawMPInterface`.
        mp_type (str): MP key used for config merging (see `MP_DEFAULTS`).
        device (str | torch.device): Device propagated into all MP components.
        mp_config_override (dict | None): User overrides applied last.
        env_make_kwargs (dict | None): Default kwargs forwarded to `gym.make` at creation.

    Returns:
        str: Registered `mp_id`.

    Notes:
        The entry point defers to `make_mp_env` to build the final wrapper. Registry is
        idempotent: calling `upgrade` again with the same `mp_id` returns immediately.
    """
    if mp_id in gym_registry:
        return mp_id

    def _entry_point(**kwargs):
        merged_kwargs = dict(env_make_kwargs or {})
        merged_kwargs.update(kwargs)
        return make_mp_env(
            base_id=base_id,
            mp_wrapper_cls=mp_wrapper_cls,
            mp_type=mp_type,
            device=device,
            mp_config_override=mp_config_override,
            env_make_kwargs=merged_kwargs,
        )

    gym.register(id=mp_id, entry_point=_entry_point)
    return mp_id
