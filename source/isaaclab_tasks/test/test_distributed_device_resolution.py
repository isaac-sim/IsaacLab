# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for distributed multi-GPU device resolution logic.

These tests verify that ``_resolve_distributed_device`` (in sim_launcher)
correctly handles:

- Normal multi-GPU: each rank sees all GPUs (local_rank maps directly)
- CUDA_VISIBLE_DEVICES restricted: each rank sees 1 GPU (fallback to cuda:0)
- Multi-node: WORLD_SIZE > local GPU count (local_rank still maps correctly)
- JAX_LOCAL_RANK: added to local_rank for JAX distributed training
- Non-distributed: no device override applied
- launch_simulation device propagation from AppLauncher

No actual GPUs required — ``torch.cuda.device_count`` and
``torch.cuda.set_device`` are mocked throughout.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from unittest.mock import patch

import isaaclab_tasks.utils.sim_launcher as sim_launcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummySimCfg:
    """Minimal sim config stub with a mutable ``device`` attribute."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device


class _DummyEnvCfg:
    """Minimal env config stub wrapping a sim config."""

    def __init__(self, device: str = "cuda:0"):
        self.sim = _DummySimCfg(device)


def _make_distributed_args(**overrides) -> argparse.Namespace:
    """Create an argparse.Namespace with ``distributed=True`` plus any overrides."""
    defaults = {"distributed": True}
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_env_vars(
    local_rank: int = 0,
    world_size: int = 2,
    rank: int = 0,
    jax_local_rank: int | None = None,
    jax_rank: int | None = None,
) -> dict[str, str]:
    """Build a dict of environment variables for distributed training."""
    env = {
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "RANK": str(rank),
    }
    if jax_local_rank is not None:
        env["JAX_LOCAL_RANK"] = str(jax_local_rank)
    if jax_rank is not None:
        env["JAX_RANK"] = str(jax_rank)
    return env


# ---------------------------------------------------------------------------
# _resolve_distributed_device — Namespace launcher_args
# ---------------------------------------------------------------------------


class TestResolveDistributedDeviceNamespace:
    """Tests for _resolve_distributed_device with argparse.Namespace args."""

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=4)
    def test_normal_multi_gpu_rank0(self, mock_count, mock_set_device):
        """4 visible GPUs, world_size=4, rank 0 → cuda:0."""
        env_cfg = _DummyEnvCfg()
        args = _make_distributed_args()
        env = _make_env_vars(local_rank=0, world_size=4)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:0"
        mock_set_device.assert_called_once_with("cuda:0")

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=4)
    def test_normal_multi_gpu_rank3(self, mock_count, mock_set_device):
        """4 visible GPUs, world_size=4, rank 3 → cuda:3."""
        env_cfg = _DummyEnvCfg()
        args = _make_distributed_args()
        env = _make_env_vars(local_rank=3, world_size=4)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:3"
        mock_set_device.assert_called_once_with("cuda:3")

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=1)
    def test_cuda_visible_devices_restricted_rank0(self, mock_count, mock_set_device):
        """1 visible GPU (CUDA_VISIBLE_DEVICES set), world_size=2, rank 0 → cuda:0."""
        env_cfg = _DummyEnvCfg()
        args = _make_distributed_args()
        env = _make_env_vars(local_rank=0, world_size=2)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:0"
        mock_set_device.assert_called_once_with("cuda:0")

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=1)
    def test_cuda_visible_devices_restricted_rank1(self, mock_count, mock_set_device):
        """1 visible GPU, world_size=2, rank 1 → falls back to cuda:0 (not cuda:1)."""
        env_cfg = _DummyEnvCfg()
        args = _make_distributed_args()
        env = _make_env_vars(local_rank=1, world_size=2)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:0"
        mock_set_device.assert_called_once_with("cuda:0")

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=2)
    def test_jax_local_rank_added(self, mock_count, mock_set_device):
        """JAX_LOCAL_RANK is added to LOCAL_RANK for correct device mapping."""
        env_cfg = _DummyEnvCfg()
        args = _make_distributed_args()
        # LOCAL_RANK=0, JAX_LOCAL_RANK=1 → effective local_rank=1
        env = _make_env_vars(local_rank=0, world_size=2, jax_local_rank=1)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:1"
        mock_set_device.assert_called_once_with("cuda:1")

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=1)
    def test_jax_local_rank_with_restricted_gpus(self, mock_count, mock_set_device):
        """JAX_LOCAL_RANK + restricted GPUs → fallback to cuda:0."""
        env_cfg = _DummyEnvCfg()
        args = _make_distributed_args()
        env = _make_env_vars(local_rank=0, world_size=2, jax_local_rank=1)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:0"
        mock_set_device.assert_called_once_with("cuda:0")


# ---------------------------------------------------------------------------
# _resolve_distributed_device — dict launcher_args
# ---------------------------------------------------------------------------


class TestResolveDistributedDeviceDict:
    """Tests for _resolve_distributed_device with dict-style args."""

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=4)
    def test_dict_args_distributed(self, mock_count, mock_set_device):
        """Dict launcher_args with distributed=True should work identically."""
        env_cfg = _DummyEnvCfg()
        args = {"distributed": True}
        env = _make_env_vars(local_rank=2, world_size=4)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:2"
        mock_set_device.assert_called_once_with("cuda:2")

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=1)
    def test_dict_args_restricted(self, mock_count, mock_set_device):
        """Dict args with restricted GPUs should fall back to cuda:0."""
        env_cfg = _DummyEnvCfg()
        args = {"distributed": True}
        env = _make_env_vars(local_rank=3, world_size=4)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:0"
        mock_set_device.assert_called_once_with("cuda:0")


# ---------------------------------------------------------------------------
# _resolve_distributed_device — non-distributed (no-op)
# ---------------------------------------------------------------------------


class TestResolveDistributedDeviceNoop:
    """Tests that non-distributed runs skip device resolution."""

    @patch("torch.cuda.set_device")
    def test_not_distributed_namespace(self, mock_set_device):
        """distributed=False → device unchanged, set_device not called."""
        env_cfg = _DummyEnvCfg(device="cuda:0")
        args = argparse.Namespace(distributed=False)

        sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:0"
        mock_set_device.assert_not_called()

    @patch("torch.cuda.set_device")
    def test_not_distributed_dict(self, mock_set_device):
        """Dict with distributed=False → no-op."""
        env_cfg = _DummyEnvCfg(device="cuda:0")
        args = {"distributed": False}

        sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:0"
        mock_set_device.assert_not_called()

    @patch("torch.cuda.set_device")
    def test_no_distributed_key(self, mock_set_device):
        """Dict without 'distributed' key → no-op."""
        env_cfg = _DummyEnvCfg(device="cuda:0")
        args = {}

        sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:0"
        mock_set_device.assert_not_called()

    @patch("torch.cuda.set_device")
    def test_none_launcher_args(self, mock_set_device):
        """launcher_args=None → no-op."""
        env_cfg = _DummyEnvCfg(device="cuda:0")

        sim_launcher._resolve_distributed_device(env_cfg, None)

        assert env_cfg.sim.device == "cuda:0"
        mock_set_device.assert_not_called()


# ---------------------------------------------------------------------------
# _resolve_distributed_device — edge cases
# ---------------------------------------------------------------------------


class TestResolveDistributedDeviceEdgeCases:
    """Edge cases for device resolution."""

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=2)
    def test_env_cfg_without_sim(self, mock_count, mock_set_device):
        """env_cfg with no 'sim' attribute → set_device still called, no crash."""

        class _BareEnvCfg:
            pass

        env_cfg = _BareEnvCfg()
        args = _make_distributed_args()
        env = _make_env_vars(local_rank=1, world_size=2)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        # Should still call set_device even without sim_cfg
        mock_set_device.assert_called_once_with("cuda:1")

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=2)
    def test_world_size_equals_visible_gpus(self, mock_count, mock_set_device):
        """Exact match: 2 visible GPUs, world_size=2 → use local_rank directly."""
        env_cfg = _DummyEnvCfg()
        args = _make_distributed_args()
        env = _make_env_vars(local_rank=1, world_size=2)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:1"

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=8)
    def test_more_gpus_than_world_size(self, mock_count, mock_set_device):
        """8 visible GPUs but only 2 ranks → use local_rank directly."""
        env_cfg = _DummyEnvCfg()
        args = _make_distributed_args()
        env = _make_env_vars(local_rank=1, world_size=2)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:1"

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=0)
    def test_zero_visible_gpus(self, mock_count, mock_set_device):
        """0 visible GPUs → fallback to cuda:0 (will fail later at CUDA init)."""
        env_cfg = _DummyEnvCfg()
        args = _make_distributed_args()
        env = _make_env_vars(local_rank=0, world_size=2)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:0"

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=4)
    def test_missing_env_vars_default_to_zero(self, mock_count, mock_set_device):
        """Missing LOCAL_RANK/WORLD_SIZE → defaults to 0/1."""
        env_cfg = _DummyEnvCfg()
        args = _make_distributed_args()

        # Remove distributed env vars if they exist
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("LOCAL_RANK", "WORLD_SIZE", "RANK", "JAX_LOCAL_RANK", "JAX_RANK")
        }

        with patch.dict(os.environ, clean_env, clear=True):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        # local_rank=0, 0 < 4 → cuda:0
        assert env_cfg.sim.device == "cuda:0"


# ---------------------------------------------------------------------------
# _resolve_distributed_device — multi-node scenarios
# ---------------------------------------------------------------------------


class TestResolveDistributedDeviceMultiNode:
    """Tests for multi-node setups where WORLD_SIZE > local GPU count."""

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=4)
    def test_multi_node_rank3_sees_4_gpus(self, mock_count, mock_set_device):
        """2 nodes × 4 GPUs, WORLD_SIZE=8, local_rank=3, 4 visible → cuda:3.

        Previously this would fail because 4 >= 8 is False, falling back to cuda:0.
        With the fix (local_rank < num_visible_gpus), 3 < 4 → cuda:3 ✅
        """
        env_cfg = _DummyEnvCfg()
        args = _make_distributed_args()
        env = _make_env_vars(local_rank=3, world_size=8, rank=7)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:3"
        mock_set_device.assert_called_once_with("cuda:3")

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=4)
    def test_multi_node_rank0_sees_4_gpus(self, mock_count, mock_set_device):
        """2 nodes × 4 GPUs, WORLD_SIZE=8, local_rank=0 → cuda:0."""
        env_cfg = _DummyEnvCfg()
        args = _make_distributed_args()
        env = _make_env_vars(local_rank=0, world_size=8, rank=4)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:0"
        mock_set_device.assert_called_once_with("cuda:0")

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.device_count", return_value=1)
    def test_multi_node_restricted_gpus(self, mock_count, mock_set_device):
        """Multi-node with CUDA_VISIBLE_DEVICES=<one GPU per rank>, local_rank=1 → cuda:0."""
        env_cfg = _DummyEnvCfg()
        args = _make_distributed_args()
        env = _make_env_vars(local_rank=1, world_size=8, rank=5)

        with patch.dict(os.environ, env, clear=False):
            sim_launcher._resolve_distributed_device(env_cfg, args)

        assert env_cfg.sim.device == "cuda:0"
        mock_set_device.assert_called_once_with("cuda:0")


# ---------------------------------------------------------------------------
# launch_simulation integration — verify device propagation from AppLauncher
# ---------------------------------------------------------------------------


class TestLaunchSimulationDevicePropagation:
    """Verify that launch_simulation propagates AppLauncher.device to env_cfg."""

    def test_kit_path_propagates_applauncher_device(self, monkeypatch):
        """When Kit is needed, AppLauncher.device should be written to env_cfg.sim.device."""

        class _FakeAppLauncher:
            def __init__(self, launcher_args):
                self.device = "cuda:3"  # Simulate resolved device
                self.app = types.SimpleNamespace(close=lambda: None)

        # Mock has_kit to return False so AppLauncher gets created
        mock_isaaclab_utils = types.ModuleType("isaaclab.utils")
        mock_isaaclab_utils.has_kit = lambda: False
        monkeypatch.setitem(sys.modules, "isaaclab.utils", mock_isaaclab_utils)

        monkeypatch.setitem(
            sys.modules,
            "isaaclab.app",
            types.SimpleNamespace(AppLauncher=_FakeAppLauncher),
        )
        monkeypatch.setattr(
            "importlib.util.find_spec",
            lambda name: object() if name == "omni.kit" else None,
        )
        # Force needs_kit=True, no cameras
        monkeypatch.setattr(
            sim_launcher,
            "compute_kit_requirements",
            lambda env_cfg, launcher_args: (True, False, set()),
        )
        # Mock _resolve_distributed_device to avoid torch.cuda calls
        monkeypatch.setattr(
            sim_launcher,
            "_resolve_distributed_device",
            lambda env_cfg, launcher_args: None,
        )

        env_cfg = _DummyEnvCfg(device="cuda:0")
        args = argparse.Namespace()

        with sim_launcher.launch_simulation(env_cfg, args):
            pass

        assert env_cfg.sim.device == "cuda:3"

    def test_kitless_path_uses_resolve_distributed_device(self, monkeypatch):
        """When Kit is NOT needed, _resolve_distributed_device sets the device."""
        resolved_devices = []

        def _fake_resolve(env_cfg, launcher_args):
            env_cfg.sim.device = "cuda:1"
            resolved_devices.append("cuda:1")

        monkeypatch.setattr(
            sim_launcher,
            "compute_kit_requirements",
            lambda env_cfg, launcher_args: (False, False, set()),
        )
        monkeypatch.setattr(
            sim_launcher,
            "_resolve_distributed_device",
            _fake_resolve,
        )

        env_cfg = _DummyEnvCfg(device="cuda:0")
        args = _make_distributed_args()

        with sim_launcher.launch_simulation(env_cfg, args):
            pass

        assert env_cfg.sim.device == "cuda:1"
        assert len(resolved_devices) == 1
