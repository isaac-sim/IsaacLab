# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for the cartpole-camera-presets frame-stacking policy.

Cfg-level: for each ``(physics, renderer)`` preset combo, the resolved cfg combined with
:meth:`CartpoleCameraPresetsEnv._resolve_frame_stack_default` produces the expected
``frame_stack`` value; user-set values are respected; per-data-type single-frame channel
counts are correct.

End-to-end: constructs ``CartpoleCameraPresetsEnv`` for the PhysX baseline and
Newton+Warp combos, then verifies ``env.reset()`` / ``env.step()`` produce policy
observations of the expected stacked shape."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import pytest  # noqa: E402
from isaaclab_newton.physics import NewtonCfg  # noqa: E402
from isaaclab_newton.renderers import NewtonWarpRendererCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from isaaclab_physx.renderers import IsaacRtxRendererCfg  # noqa: E402

from isaaclab_tasks.direct.cartpole.cartpole_camera_presets_env import CartpoleCameraPresetsEnv  # noqa: E402
from isaaclab_tasks.direct.cartpole.cartpole_camera_presets_env_cfg import CartpoleCameraPresetsEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci


def _resolve(*presets: str):
    """Build a fresh CartpoleCameraPresetsEnvCfg and resolve with the given preset names.

    Returns the resolved root cfg (a ``BaseCartpoleCameraEnvCfg`` instance).
    """
    outer = CartpoleCameraPresetsEnvCfg()
    return resolve_presets(outer, selected=set(presets))


def _policy_default(cfg) -> int:
    """Run the task's policy helper on a resolved cfg."""
    return CartpoleCameraPresetsEnv._resolve_frame_stack_default(cfg.tiled_camera, cfg.sim.physics)


class TestFrameStackTruthTable:
    """One test per cell of the physics × renderer matrix."""

    def test_no_presets_resolves_to_default(self):
        cfg = _resolve()
        assert cfg.frame_stack == -1, "Cfg sentinel default must survive preset resolution"
        assert _policy_default(cfg) == 1

    def test_physx_default_renderer(self):
        cfg = _resolve("physx")
        assert isinstance(cfg.sim.physics, PhysxCfg)
        assert isinstance(cfg.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)
        assert _policy_default(cfg) == 1

    def test_physx_with_warp_renderer(self):
        """PhysX has implicit damping — no stacking needed even with Warp."""
        cfg = _resolve("physx", "newton_renderer")
        assert isinstance(cfg.sim.physics, PhysxCfg)
        assert isinstance(cfg.tiled_camera.renderer_cfg, NewtonWarpRendererCfg)
        assert _policy_default(cfg) == 1

    def test_newton_with_default_renderer(self):
        """Newton physics + RTX renderer — RTX provides temporal information."""
        cfg = _resolve("newton_mjwarp")
        assert isinstance(cfg.sim.physics, NewtonCfg)
        assert isinstance(cfg.tiled_camera.renderer_cfg, IsaacRtxRendererCfg)
        assert _policy_default(cfg) == 1

    def test_newton_with_warp_renderer_stacks(self):
        """Newton + Warp — the combo that needs explicit temporal stacking."""
        cfg = _resolve("newton_mjwarp", "newton_renderer")
        assert isinstance(cfg.sim.physics, NewtonCfg)
        assert isinstance(cfg.tiled_camera.renderer_cfg, NewtonWarpRendererCfg)
        assert _policy_default(cfg) == 2


class TestObsSpaceBumpArithmetic:
    """The env class bumps ``observation_space[-1] *= frame_stack`` when stacking — sanity-check
    that the arithmetic across data-type variants stays correct."""

    @pytest.mark.parametrize(
        "data_type_preset,expected_single_channels",
        [
            ("default", 3),  # RGB
            ("depth", 1),
            ("albedo", 3),
            ("semantic_segmentation", 4),
            ("simple_shading_constant_diffuse", 3),
            ("simple_shading_diffuse_mdl", 3),
            ("simple_shading_full_mdl", 3),
        ],
    )
    def test_observation_space_unstacked_channels(self, data_type_preset, expected_single_channels):
        """Each data-type variant declares the expected single-frame channel count."""
        cfg = _resolve(data_type_preset)
        assert cfg.observation_space[-1] == expected_single_channels


# ---------------------------------------------------------------------------
# End-to-end: construct the real env and verify the obs pipeline
# ---------------------------------------------------------------------------


class TestEnvConstructionEndToEnd:
    """Construct ``CartpoleCameraPresetsEnv`` for real and verify the obs pipeline.

    These tests catch wiring bugs that cfg-only tests miss: that the env class
    correctly resolves the policy, bumps obs_space, allocates the buffer, runs the
    buffer in ``_get_observations``, and resets it in ``_reset_idx``.
    """

    @pytest.mark.parametrize(
        "presets,user_frame_stack,expected_frame_stack,expected_channels",
        [
            # PhysX default (no presets): policy resolves to 1 → buffer skipped → 3 channels.
            (frozenset(), -1, 1, 3),
            # Newton + Warp: policy resolves to 2 → buffer active → 6 channels.
            (frozenset({"newton_mjwarp", "newton_renderer"}), -1, 2, 6),
            # User override (env.frame_stack=4) on Newton+Warp: policy is skipped → 12 channels.
            (frozenset({"newton_mjwarp", "newton_renderer"}), 4, 4, 12),
            # ``frame_stack=0`` is a synonym for "no stacking" → normalized to 1.
            (frozenset(), 0, 1, 3),
            # Explicit single-frame: ``frame_stack=1`` short-circuits the policy.
            (frozenset(), 1, 1, 3),
        ],
    )
    def test_env_obs_shape_matches_policy(self, presets, user_frame_stack, expected_frame_stack, expected_channels):
        # Build + resolve cfg; trim envs for test speed.
        outer = CartpoleCameraPresetsEnvCfg()
        env_cfg = resolve_presets(outer, selected=set(presets))
        env_cfg.scene.num_envs = 2
        env_cfg.frame_stack = user_frame_stack

        env = None
        try:
            env = CartpoleCameraPresetsEnv(cfg=env_cfg)
            assert env.cfg.frame_stack == expected_frame_stack, (
                f"presets={presets} user_fs={user_frame_stack}: expected"
                f" frame_stack={expected_frame_stack}, got {env.cfg.frame_stack}"
            )
            # Reset and verify obs shape.
            obs, _ = env.reset()
            expected_shape = (env.num_envs, env_cfg.tiled_camera.height, env_cfg.tiled_camera.width, expected_channels)
            assert obs["policy"].shape == expected_shape, (
                f"presets={presets}: reset obs shape {tuple(obs['policy'].shape)} != expected {expected_shape}"
            )
            # Step once and confirm the shape persists.
            import torch as _torch

            action = _torch.zeros(env.num_envs, 1, device=env.device)
            obs, _, _, _, _ = env.step(action)
            assert obs["policy"].shape == expected_shape, (
                f"presets={presets}: step obs shape {tuple(obs['policy'].shape)} != expected {expected_shape}"
            )
        finally:
            if env is not None:
                env.close()
            else:
                # Mid-init failure left a SimulationContext singleton; clear it for the next case.
                import contextlib

                import isaaclab.sim as sim_utils

                sim = sim_utils.SimulationContext.instance()
                if sim is not None:
                    with contextlib.suppress(Exception):
                        sim.clear_instance()

    def test_buffer_ring_shift_e2e(self):
        """Verify the buffer's ring shift is wired through the env's obs pipeline.

        Uses an identity that holds regardless of renderer behavior: the frame that was
        newest at reset must appear at the oldest position after one step. This catches
        slot-order bugs in the buffer's narrow+copy_ rebuild without depending on the
        camera producing different content frame-to-frame.
        """
        import torch as _torch

        outer = CartpoleCameraPresetsEnvCfg()
        env_cfg = resolve_presets(outer, selected={"newton_mjwarp", "newton_renderer"})
        env_cfg.scene.num_envs = 2

        env = None
        try:
            env = CartpoleCameraPresetsEnv(cfg=env_cfg)
            assert env.cfg.frame_stack == 2, "Newton+Warp must auto-resolve to frame_stack=2 for this test"

            c = env_cfg.observation_space[-1] // env.cfg.frame_stack
            obs, _ = env.reset()
            reset_newest = obs["policy"][..., -c:].clone()

            action = _torch.zeros(env.num_envs, 1, device=env.device)
            obs, _, _, _, _ = env.step(action)
            step_oldest = obs["policy"][..., :c]

            assert _torch.allclose(step_oldest, reset_newest), (
                "Ring shift broken: the frame that was newest at reset did not appear at the oldest "
                "position after one step."
            )
        finally:
            if env is not None:
                env.close()
            else:
                import contextlib

                import isaaclab.sim as sim_utils

                sim = sim_utils.SimulationContext.instance()
                if sim is not None:
                    with contextlib.suppress(Exception):
                        sim.clear_instance()
