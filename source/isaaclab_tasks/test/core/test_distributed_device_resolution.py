# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device selection through the public launcher, without starting a runtime or touching a GPU."""

import argparse
import sys
from types import SimpleNamespace

import isaaclab_physx.app as physx_app
import pytest
import torch
from isaaclab_newton.physics import NewtonCfg

import isaaclab.app.sim_launcher as sim_launcher
from isaaclab.app import SimulationLauncher, launch_simulation
from isaaclab.sim import SimulationCfg


@pytest.fixture
def selected_devices(monkeypatch):
    """Isolate rank variables, storage configuration, and CUDA device selection."""
    for name in ("LOCAL_RANK", "WORLD_SIZE", "RANK", "JAX_LOCAL_RANK", "JAX_RANK", "LIVESTREAM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.assets", SimpleNamespace(configure_storage_profile=lambda: None))
    devices = []
    monkeypatch.setattr(sim_launcher, "set_cuda_device", devices.append)
    return devices


@pytest.mark.parametrize("args_type", [dict, argparse.Namespace])
@pytest.mark.parametrize("cfg_type", ["simulation", "environment", "physics"])
@pytest.mark.parametrize(
    ("visible_gpus", "local_rank", "world_size", "jax_local_rank", "expected"),
    [
        (4, 3, 4, 0, "cuda:3"),
        (1, 1, 2, 0, "cuda:0"),
        (2, 0, 2, 1, "cuda:1"),
        (4, 3, 8, 0, "cuda:3"),
        (4, None, None, None, "cuda:0"),
    ],
    ids=["multi_gpu", "restricted_visibility", "jax_rank", "multi_node", "default_rank"],
)
def test_distributed_device(
    monkeypatch, selected_devices, args_type, cfg_type, visible_gpus, local_rank, world_size, jax_local_rank, expected
):
    """Rank selection updates caller-owned arguments and both supported simulation-config locations."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: visible_gpus)
    for name, value in (("LOCAL_RANK", local_rank), ("WORLD_SIZE", world_size), ("JAX_LOCAL_RANK", jax_local_rank)):
        if value is not None:
            monkeypatch.setenv(name, str(value))
    sim_cfg = SimulationCfg(physics=NewtonCfg(), visualizer_cfgs=[], device="cuda:7")
    cfg = SimpleNamespace(sim=sim_cfg) if cfg_type == "environment" else sim_cfg
    if cfg_type == "physics":
        cfg = sim_cfg.physics
    args = args_type(distributed=True)

    with launch_simulation(cfg, args) as physics_cfg:
        assert physics_cfg is sim_cfg.physics
        if cfg_type != "physics":
            assert sim_cfg.device == expected
    assert (args if isinstance(args, dict) else vars(args))["device"] == expected
    assert selected_devices == [expected]


@pytest.mark.parametrize("args", [None, {}, {"distributed": False}, argparse.Namespace(distributed=False)])
def test_non_distributed_device_is_unchanged(selected_devices, args):
    """Optional launcher arguments preserve the configured device and remain caller-owned."""
    cfg = SimulationCfg(physics=NewtonCfg(), visualizer_cfgs=[], device="cuda:7")
    with launch_simulation(cfg, args):
        assert cfg.device == "cuda:7"
    assert selected_devices == []
    if args is not None:
        values = args if isinstance(args, dict) else vars(args)
        assert values["kit_visualizer"] is False


@pytest.mark.parametrize("args_type", [dict, argparse.Namespace])
@pytest.mark.parametrize("nested", [False, True], ids=["simulation", "environment"])
def test_runtime_device_overrides_distributed_selection(monkeypatch, selected_devices, args_type, nested):
    """A runtime may refine the device, after rank selection but before simulation construction."""
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    received, closed = [], []

    class KitRuntime(SimulationLauncher):
        device = "cuda:3"

        def __init__(self, launcher_args):
            values = vars(launcher_args) if isinstance(launcher_args, argparse.Namespace) else launcher_args
            received.append(values["device"])

        def close(self, exit_code=0):
            closed.append(exit_code)

    monkeypatch.setattr(physx_app, "KitLauncher", KitRuntime)
    sim_cfg = SimulationCfg(physics=NewtonCfg(), visualizer_cfgs=[], device="cuda:7")
    cfg = SimpleNamespace(sim=sim_cfg) if nested else sim_cfg
    with launch_simulation(cfg, args_type(distributed=True, require_kit=True)):
        assert sim_cfg.device == "cuda:3"
    assert selected_devices == received == ["cuda:1"]
    assert closed == [0]
