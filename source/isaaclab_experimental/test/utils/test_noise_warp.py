# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Warp-native observation noise record and replay."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_experimental.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab_experimental.managers.observation_manager import ObservationManager
from isaaclab_experimental.utils import noise

from isaaclab.utils import class_to_dict
from isaaclab.utils.noise import UniformNoiseCfg as StableUniformNoiseCfg
from isaaclab.utils.warp import WarpLaunchCache

wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

_DEVICE = "cuda:0"
_NUM_ENVS = 8
_OBS_DIM = 3


def _make_rng_state() -> wp.array:
    """Create deterministic per-environment RNG state."""
    return wp.array(np.arange(_NUM_ENVS, dtype=np.uint32) + 42, dtype=wp.uint32, device=_DEVICE)


def _bind_noise(cfg: noise.NoiseCfg, cache: WarpLaunchCache, rng_state: wp.array) -> None:
    """Bind a standalone noise config as the observation manager does."""
    cfg.rng_state_wp = rng_state
    cfg._warp_launch = cache


def _fill_zeros(env, out: wp.array, out_dim: int) -> None:
    """Fill one synthetic observation term."""
    del env, out_dim
    out.zero_()


def _fill_twos(env, out: wp.array, out_dim: int) -> None:
    """Fill one synthetic observation term above its clip range."""
    del env, out_dim
    out.fill_(2.0)


def _make_observation_manager(
    mode: str, noise_cfg: noise.NoiseCfg | StableUniformNoiseCfg | None = None
) -> tuple[ObservationManager, SimpleNamespace]:
    """Create a minimal observation manager with uniform corruption enabled."""
    if noise_cfg is None:
        noise_cfg = noise.UniformNoiseCfg(n_min=-0.25, n_max=0.5)
    term_cfg = ObservationTermCfg(
        func=_fill_zeros,
        params={"out_dim": _OBS_DIM},
        noise=noise_cfg,
    )
    group_cfg = ObservationGroupCfg()
    group_cfg.enable_corruption = True
    group_cfg.sample = term_cfg
    env = SimpleNamespace(
        num_envs=_NUM_ENVS,
        device=_DEVICE,
        sim=SimpleNamespace(is_playing=lambda: True),
        scene={},
        rng_state_wp=_make_rng_state(),
        _warp_launch=WarpLaunchCache(mode=mode, debug=True, device=_DEVICE),
    )
    return ObservationManager({"policy": group_cfg}, env), env


def test_constant_noise_replays_static_variants():
    """Constant noise should replay and re-record changed static values."""
    cfg = noise.ConstantNoiseCfg(bias=0.5, operation="add")
    cache = WarpLaunchCache(mode="replay", debug=True, device=_DEVICE)
    _bind_noise(cfg, cache, _make_rng_state())
    data = wp.ones((_NUM_ENVS, _OBS_DIM), dtype=wp.float32, device=_DEVICE)

    cfg.func(data, cfg)
    torch.testing.assert_close(wp.to_torch(data), torch.full((_NUM_ENVS, _OBS_DIM), 1.5, device=_DEVICE))

    data.fill_(2.0)
    cfg.func(data, cfg)
    torch.testing.assert_close(wp.to_torch(data), torch.full((_NUM_ENVS, _OBS_DIM), 2.5, device=_DEVICE))

    cfg.bias = -1.0
    cfg.func(data, cfg)
    torch.testing.assert_close(wp.to_torch(data), torch.full((_NUM_ENVS, _OBS_DIM), 1.5, device=_DEVICE))


@pytest.mark.parametrize(
    "cfg_factory",
    [
        lambda: noise.UniformNoiseCfg(n_min=-0.5, n_max=0.75),
        lambda: noise.GaussianNoiseCfg(mean=0.2, std=0.4),
    ],
    ids=["uniform", "gaussian"],
)
def test_random_noise_replay_matches_eager(cfg_factory):
    """Random-noise replay should advance persistent RNG exactly like eager launches."""
    replay_cfg = cfg_factory()
    eager_cfg = cfg_factory()
    replay_rng = _make_rng_state()
    eager_rng = _make_rng_state()
    _bind_noise(replay_cfg, WarpLaunchCache(mode="replay", debug=True, device=_DEVICE), replay_rng)
    _bind_noise(eager_cfg, WarpLaunchCache(mode="eager", debug=True, device=_DEVICE), eager_rng)
    replay_data = wp.zeros((_NUM_ENVS, _OBS_DIM), dtype=wp.float32, device=_DEVICE)
    eager_data = wp.zeros((_NUM_ENVS, _OBS_DIM), dtype=wp.float32, device=_DEVICE)

    for _ in range(3):
        replay_data.zero_()
        eager_data.zero_()
        replay_cfg.func(replay_data, replay_cfg)
        eager_cfg.func(eager_data, eager_cfg)
        torch.testing.assert_close(wp.to_torch(replay_data), wp.to_torch(eager_data))
        torch.testing.assert_close(wp.to_torch(replay_rng), wp.to_torch(eager_rng))


def test_observation_manager_binds_noise_and_replays_inside_capture():
    """Manager-prepared uniform noise should replay inside CUDA graph capture."""
    replay_manager, replay_env = _make_observation_manager("replay")
    eager_manager, _ = _make_observation_manager("eager")
    serialized_cfg = replay_manager._group_obs_term_cfgs["policy"][0].noise
    prepared_cfg = replay_manager._group_obs_noise_runtime["policy"]["sample"]

    assert prepared_cfg.rng_state_wp is replay_env.rng_state_wp
    assert prepared_cfg._warp_launch is replay_env._warp_launch
    assert not hasattr(serialized_cfg, "rng_state_wp")
    assert not hasattr(serialized_cfg, "_warp_launch")
    assert class_to_dict(serialized_cfg)["n_min"] == -0.25

    replay_first = replay_manager.compute()["policy"].clone()
    eager_first = eager_manager.compute()["policy"].clone()
    torch.testing.assert_close(replay_first, eager_first)

    replay_second = replay_manager.compute()["policy"].clone()
    eager_second = eager_manager.compute()["policy"].clone()
    torch.testing.assert_close(replay_second, eager_second)

    prepared_term = replay_manager._group_obs_term_cfgs["policy"][0]
    with wp.ScopedCapture() as capture:
        prepared_term.out_wp.zero_()
        prepared_cfg.func(prepared_term.out_wp, prepared_cfg)
    expected = eager_manager.compute()["policy"].clone()
    wp.capture_launch(capture.graph)
    torch.testing.assert_close(replay_manager._group_out_torch["policy"], expected)


def test_observation_manager_rejects_stable_torch_noise_cfg():
    """Stable Torch noise configs should fail fast instead of silently skipping corruption."""
    with pytest.raises(TypeError, match="which the Warp observation manager would silently ignore"):
        _make_observation_manager("replay", StableUniformNoiseCfg(n_min=-0.25, n_max=0.5))


def test_observation_clip_replay_uses_changed_bounds():
    """Observation clipping should specialize replay when configured bounds change."""
    term_cfg = ObservationTermCfg(func=_fill_twos, params={"out_dim": _OBS_DIM}, clip=(-1.0, 1.0))
    group_cfg = ObservationGroupCfg()
    group_cfg.sample = term_cfg
    env = SimpleNamespace(
        num_envs=_NUM_ENVS,
        device=_DEVICE,
        sim=SimpleNamespace(is_playing=lambda: True),
        scene={},
        rng_state_wp=_make_rng_state(),
        _warp_launch=WarpLaunchCache(mode="replay", debug=True, device=_DEVICE),
    )
    manager = ObservationManager({"policy": group_cfg}, env)
    prepared_term_cfg = manager._group_obs_term_cfgs["policy"][0]

    first = manager.compute_group("policy").clone()
    torch.testing.assert_close(first, torch.ones((_NUM_ENVS, _OBS_DIM), device=_DEVICE))

    prepared_term_cfg.clip = (-1.0, 0.25)
    second = manager.compute_group("policy").clone()
    torch.testing.assert_close(second, torch.full((_NUM_ENVS, _OBS_DIM), 0.25, device=_DEVICE))
