# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the device a launch runs on: the distributed rank's GPU and ``launch_simulation`` precedence.

No actual GPUs required: ``torch.cuda.device_count``, the CUDA device-selection helper and the Kit launcher
are mocked.
"""

from __future__ import annotations

import argparse
import sys
import types
from unittest.mock import patch

import pytest

import isaaclab.app.sim_launcher as sim_launcher
from isaaclab.sim.simulation_cfg import SimulationCfg

_RANK_ENV_VARS = ("LOCAL_RANK", "JAX_LOCAL_RANK", "WORLD_SIZE", "RANK", "JAX_RANK")


class _DummyEnvCfg:
    """Minimal env config stub; ``scan`` needs a real SimulationCfg."""

    def __init__(self, device: str):
        self.sim = SimulationCfg(device=device)


def _set_rank_env(monkeypatch, env: dict[str, str]):
    """Replace the distributed launcher environment variables with *env*."""
    for name in _RANK_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    "args, env, gpu_count, expected",
    [
        pytest.param(
            argparse.Namespace(distributed=True), {"LOCAL_RANK": "3", "WORLD_SIZE": "4"}, 4, "cuda:3", id="multi_gpu"
        ),
        # CUDA_VISIBLE_DEVICES leaves each rank one GPU, so rank 1 must not select cuda:1
        pytest.param(
            argparse.Namespace(distributed=True),
            {"LOCAL_RANK": "1", "WORLD_SIZE": "2"},
            1,
            "cuda:0",
            id="restricted_visible_devices",
        ),
        pytest.param(
            argparse.Namespace(distributed=True),
            {"LOCAL_RANK": "0", "JAX_LOCAL_RANK": "1", "WORLD_SIZE": "2"},
            2,
            "cuda:1",
            id="jax_local_rank_added",
        ),
        pytest.param({"distributed": True}, {"LOCAL_RANK": "2", "WORLD_SIZE": "4"}, 4, "cuda:2", id="dict_args"),
        pytest.param(argparse.Namespace(distributed=True), {}, 4, "cuda:0", id="missing_env_defaults_to_rank0"),
        # 2 nodes x 4 GPUs: the local rank is compared with the local GPU count, not WORLD_SIZE
        pytest.param(
            argparse.Namespace(distributed=True),
            {"LOCAL_RANK": "3", "WORLD_SIZE": "8", "RANK": "7"},
            4,
            "cuda:3",
            id="multi_node",
        ),
        pytest.param({"distributed": False}, {"LOCAL_RANK": "1"}, 2, None, id="not_distributed"),
        pytest.param({}, {"LOCAL_RANK": "1"}, 2, None, id="no_distributed_key"),
    ],
)
def test_resolve_distributed_device(monkeypatch, args, env, gpu_count, expected):
    """A distributed run selects this rank's GPU in the launcher args and CUDA; other runs are left alone."""
    _set_rank_env(monkeypatch, env)
    args_dict = vars(args) if isinstance(args, argparse.Namespace) else args

    with (
        patch("torch.cuda.device_count", return_value=gpu_count),
        patch.object(sim_launcher, "set_cuda_device") as set_cuda_device,
    ):
        sim_launcher._resolve_distributed_device(args_dict)

    if expected is None:
        assert "device" not in args_dict
        set_cuda_device.assert_not_called()
    else:
        # the Kit launcher reads the resolved device from the launcher args
        assert args_dict["device"] == expected
        set_cuda_device.assert_called_once_with(expected)


@pytest.mark.parametrize(
    "kit_device, cfg_device, args, env, expected, expected_kit_arg",
    [
        # a started Kit runtime refines the device
        pytest.param("cuda:3", "cuda:0", argparse.Namespace(), {}, "cuda:3", "cuda:0", id="kit_refines_device"),
        # without --device, Kit starts on the config's device, e.g. a CPU-only task
        pytest.param(None, "cpu", argparse.Namespace(device=None), {}, "cpu", "cpu", id="kit_gets_config_device"),
        # every distributed rank runs on its own GPU, overriding the config
        *(
            pytest.param(
                "kitless",
                "cpu",
                argparse.Namespace(distributed=True),
                {"LOCAL_RANK": str(rank), "WORLD_SIZE": "2"},
                f"cuda:{rank}",
                None,
                id=f"kitless_distributed_rank{rank}",
            )
            for rank in (0, 1)
        ),
        pytest.param("kitless", "cuda:0", argparse.Namespace(device="cuda:1"), {}, "cuda:1", None, id="kitless_cuda1"),
        # a bare ``cuda`` is pinned to the physics GPU index
        pytest.param("kitless", "cuda:1", argparse.Namespace(device="cuda"), {}, "cuda:0", None, id="kitless_cuda"),
        pytest.param("kitless", "cuda:0", argparse.Namespace(device="cpu"), {}, "cpu", None, id="kitless_cpu"),
    ],
)
def test_launch_simulation_device(monkeypatch, kit_device, cfg_device, args, env, expected, expected_kit_arg):
    """The run's device is written to the config and the args, and the rank's GPU is selected in CUDA.

    ``kit_device`` is the device the Kit launcher reports, or ``"kitless"`` for a launch without Kit.
    """
    _set_rank_env(monkeypatch, env)
    kit_args = {}

    class _FakeKitLauncher:
        device = kit_device

        def __init__(self, launcher_args):
            kit_args["device"] = launcher_args["device"]

        def close(self, exit_code=0):
            pass

    monkeypatch.setitem(sys.modules, "isaaclab_physx.app", types.SimpleNamespace(KitLauncher=_FakeKitLauncher))
    if kit_device == "kitless":
        real_scan = sim_launcher.scan

        def kitless_scan(*scan_args, **scan_kwargs):
            result = real_scan(*scan_args, **scan_kwargs)
            result.needs_kit = False
            return result

        monkeypatch.setattr(sim_launcher, "scan", kitless_scan)
    env_cfg = _DummyEnvCfg(device=cfg_device)

    with (
        patch("torch.cuda.device_count", return_value=2),
        patch.object(sim_launcher, "set_cuda_device") as set_cuda_device,
    ):
        with sim_launcher.launch_simulation(env_cfg, args):
            assert env_cfg.sim.device == expected
            assert args.device == expected

    assert kit_args.get("device") == expected_kit_arg
    if getattr(args, "distributed", False):
        set_cuda_device.assert_called_once_with(expected)
    else:
        set_cuda_device.assert_not_called()
