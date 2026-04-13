# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Test multi-GPU device assignment for Newton physics."""

import os
import subprocess
import unittest.mock

import pytest
import torch


def _get_num_gpus() -> int:
    """Return number of available CUDA GPUs."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


class TestResolveCudaDevice:
    """Tests for resolve_cuda_device across different GPU visibility scenarios."""

    def test_full_visibility(self):
        """When all GPUs are visible, each rank gets its own device."""
        from isaaclab.utils.distributed import resolve_cuda_device

        with unittest.mock.patch("torch.cuda.device_count", return_value=4):
            for local_rank in range(4):
                device, device_id = resolve_cuda_device(local_rank)
                assert device == f"cuda:{local_rank}"
                assert device_id == local_rank

    def test_restricted_visibility(self):
        """When CUDA_VISIBLE_DEVICES restricts each process to 1 GPU, all ranks use cuda:0."""
        from isaaclab.utils.distributed import resolve_cuda_device

        with unittest.mock.patch("torch.cuda.device_count", return_value=1):
            for local_rank in range(4):
                device, device_id = resolve_cuda_device(local_rank)
                assert device == "cuda:0"
                assert device_id == 0

    def test_multi_node(self):
        """Multi-node: 4 visible GPUs per node, world_size=8, each rank gets its own GPU."""
        from isaaclab.utils.distributed import resolve_cuda_device

        with unittest.mock.patch("torch.cuda.device_count", return_value=4):
            for local_rank in range(4):
                device, device_id = resolve_cuda_device(local_rank)
                assert device == f"cuda:{local_rank}", (
                    f"Multi-node: local_rank={local_rank} should map to cuda:{local_rank}"
                )

    def test_local_rank_exceeds_visible(self):
        """When local_rank >= num_visible, fall back to cuda:0."""
        from isaaclab.utils.distributed import resolve_cuda_device

        with unittest.mock.patch("torch.cuda.device_count", return_value=2):
            device, device_id = resolve_cuda_device(5)
            assert device == "cuda:0"
            assert device_id == 0


@pytest.mark.skipif(_get_num_gpus() < 2, reason="Requires at least 2 GPUs")
class TestMultiGPUDeviceAssignment:
    """Integration tests for multi-GPU training."""

    def test_cartpole_newton_multigpu(self):
        """Test that multi-GPU cartpole training with Newton physics runs without error."""
        num_gpus = min(_get_num_gpus(), 4)

        # Use torchrun from the same Python environment as the test runner
        import shutil
        import sys

        torchrun = shutil.which("torchrun", path=os.path.dirname(sys.executable))
        if torchrun is None:
            torchrun = shutil.which("torchrun")
        assert torchrun is not None, "torchrun not found — is torch installed?"

        cmd = [
            torchrun,
            f"--nproc_per_node={num_gpus}",
            "scripts/reinforcement_learning/rsl_rl/train.py",
            "--task", "Isaac-Cartpole-Direct-v0",
            "--num_envs", "64",
            "--max_iterations", "2",
            "--headless",
            "--distributed",
            "presets=newton",
        ]

        env = os.environ.copy()
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_IB_DISABLE"] = "1"

        test_dir = os.path.dirname(os.path.abspath(__file__))
        isaaclab_root = os.path.dirname(os.path.dirname(os.path.dirname(test_dir)))

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=isaaclab_root,
        )

        has_training_output = "Learning iteration 1" in result.stdout
        no_cuda_errors = "CUDA error" not in result.stderr
        assert result.returncode == 0 and has_training_output and no_cuda_errors, (
            f"Multi-GPU training failed (rc={result.returncode}):\n"
            f"stdout (last 2000): {result.stdout[-2000:]}\n"
            f"stderr (last 2000): {result.stderr[-2000:]}"
        )


@pytest.mark.skipif(_get_num_gpus() < 2, reason="Requires at least 2 GPUs")
class TestMultiGPUCameraRendering:
    """Tests for multi-GPU training with camera observations."""

    def test_preset_renderer_has_newton_renderer_field(self):
        """Test that MultiBackendRendererCfg has newton_renderer field."""
        from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

        cfg = MultiBackendRendererCfg()
        assert hasattr(cfg, "newton_renderer"), (
            "MultiBackendRendererCfg should have 'newton_renderer' field"
        )
        assert "NewtonWarp" in type(cfg.newton_renderer).__name__, (
            f"newton_renderer should be NewtonWarpRendererCfg, got {type(cfg.newton_renderer).__name__}"
        )

    def test_alias_fields_are_independent_instances(self):
        """Test that aliased fields are separate instances (not shared mutable references)."""
        from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

        cfg = MultiBackendRendererCfg()
        assert cfg.isaacsim_rtx_renderer is not cfg.default, (
            "isaacsim_rtx_renderer and default should be independent instances"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
