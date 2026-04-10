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


@pytest.mark.skipif(_get_num_gpus() < 2, reason="Requires at least 2 GPUs")
class TestMultiGPUDeviceAssignment:
    """Tests for multi-GPU device assignment in distributed training."""

    def test_resolve_cuda_device_full_visibility(self):
        """When all GPUs are visible, each rank gets its own device."""
        from isaaclab.utils.distributed import resolve_cuda_device

        # Simulate 4 visible GPUs, world_size=4
        with unittest.mock.patch("torch.cuda.device_count", return_value=4), \
             unittest.mock.patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            for local_rank in range(4):
                device, device_id = resolve_cuda_device(local_rank)
                assert device == f"cuda:{local_rank}"
                assert device_id == local_rank

    def test_resolve_cuda_device_restricted_visibility(self):
        """When CUDA_VISIBLE_DEVICES restricts each process to 1 GPU, all ranks use cuda:0."""
        from isaaclab.utils.distributed import resolve_cuda_device

        # Simulate 1 visible GPU (CUDA_VISIBLE_DEVICES restricted), world_size=4
        with unittest.mock.patch("torch.cuda.device_count", return_value=1), \
             unittest.mock.patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            for local_rank in range(4):
                device, device_id = resolve_cuda_device(local_rank)
                assert device == "cuda:0"
                assert device_id == 0

    def test_cartpole_newton_multigpu(self):
        """Test that multi-GPU cartpole training with Newton physics runs without error."""
        num_gpus = min(_get_num_gpus(), 4)

        # Run a quick 2-iteration training to verify setup works
        cmd = [
            "torchrun",
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
        # Required for containers with IOMMU where P2P transport fails,
        # and for environments without InfiniBand.
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_IB_DISABLE"] = "1"

        # Resolve IsaacLab root from test file location
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

        # Verify training actually ran (not just a clean exit from early skip)
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

    def test_preset_renderer_matching(self):
        """Test that newton preset correctly matches the renderer."""
        from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

        cfg = MultiBackendRendererCfg()
        assert hasattr(cfg, "newton"), "MultiBackendRendererCfg should have 'newton' field"
        assert "NewtonWarp" in type(cfg.newton).__name__, (
            f"newton field should be NewtonWarpRendererCfg, got {type(cfg.newton).__name__}"
        )

    def test_physx_preset_is_independent_instance(self):
        """Test that physx preset is a separate instance from default (not a shared alias)."""
        from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

        cfg = MultiBackendRendererCfg()
        assert cfg.physx is not cfg.default, (
            "physx and default should be independent instances, not shared aliases"
        )

    def test_camera_not_kit_with_preset_renderer(self):
        """Test that PresetCfg renderers are not detected as Kit cameras."""
        from isaaclab.sensors import TiledCameraCfg
        from isaaclab_tasks.utils.presets import MultiBackendRendererCfg
        from isaaclab_tasks.utils.sim_launcher import _is_kit_camera

        cam = TiledCameraCfg(
            prim_path="/test",
            data_types=["rgb"],
            width=64,
            height=64,
            renderer_cfg=MultiBackendRendererCfg(),
        )

        # PresetCfg renderers resolve to match the physics backend, so not necessarily Kit
        assert not _is_kit_camera(cam), (
            "Camera with PresetCfg renderer should not be detected as Kit camera"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
