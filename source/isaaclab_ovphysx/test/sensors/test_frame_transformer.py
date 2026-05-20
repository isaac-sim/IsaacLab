# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Real-backend tests for the OVPhysX FrameTransformer.

Mirrors :mod:`isaaclab_physx.test.sensors.test_frame_transformer` 1-to-1 for the
test cases that already exist in PhysX. One additional case exercises the
OVPhysX-specific multi-binding gather path that combines an articulation link
target and a standalone RigidObject target.

Run via ``./isaaclab.sh -p -m pytest`` (the standard Kit Python entrypoint).
"""

from __future__ import annotations

import pytest
import warp as wp

# CI pipelines that pattern-match ``isaaclab_ov*`` may try to collect these
# tests without the ovphysx wheel installed. Skip gracefully in that case.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import BaseFrameTransformer, FrameTransformerCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # noqa: E402

wp.init()


# ---------------------------------------------------------------------------
# Device-lock autouse fixture (mirrors test_contact_sensor.py)                #
# ---------------------------------------------------------------------------

_LOCKED_DEVICE: list[str | None] = [None]
"""Device the session pins to on the first parametrized test that runs."""


@pytest.fixture(autouse=True)
def _ovphysx_skip_other_device(request):
    """Skip parametrized tests on the device the session is not pinned to.

    ``ovphysx<=0.3.7`` binds device mode at the C++ layer on the first
    ``ovphysx.PhysX(device=...)`` construction and cannot swap without a
    process restart.  Pin the session to whichever device is first used.
    """
    callspec = getattr(request.node, "callspec", None)
    device = callspec.params.get("device") if callspec is not None else None
    if device is None:
        return
    locked = _LOCKED_DEVICE[0]
    if locked is None:
        _LOCKED_DEVICE[0] = device
        return
    if device != locked:
        pytest.skip(
            f"ovphysx process-global device lock is held by '{locked}'; cannot run '{device}' "
            "tests in the same session.  Run pytest twice (once per device) for full coverage."
        )


# ---------------------------------------------------------------------------
# Simulation context helper                                                  #
# ---------------------------------------------------------------------------


def _ovphysx_sim_context(device: str, **kwargs):
    """Wrapper around :func:`build_simulation_context` that injects OVPhysX cfg.

    OVPhysX needs ``physics=OvPhysxCfg()`` set on the cfg so the
    :class:`~isaaclab.sim.SimulationContext` dispatches to OVPhysX rather than
    PhysX. ``add_ground_plane``, ``auto_add_lighting``, etc. flow through to
    :func:`build_simulation_context` unchanged.
    """
    dt = kwargs.pop("dt", 0.005)
    gravity_enabled = kwargs.pop("gravity_enabled", True)
    gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, dt=dt, gravity=gravity)
    return build_simulation_context(device=device, sim_cfg=sim_cfg, **kwargs)


# ---------------------------------------------------------------------------
# Scene configuration                                                        #
# ---------------------------------------------------------------------------


@configclass
class _SceneCfg(InteractiveSceneCfg):
    """Scene cfg shared across FrameTransformer tests; ``frame_transformer`` is filled per-test."""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    frame_transformer: FrameTransformerCfg = None  # filled per-test


# ---------------------------------------------------------------------------
# Tests                                                                      #
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_frame_transformer_factory_dispatch(device):
    """Smoke test: ``FrameTransformer(cfg)`` resolves to the OVPhysX backend.

    ``FrameTransformer`` is a :class:`~isaaclab.utils.backend_utils.FactoryBase`,
    so the returned instance is *not* an instance of the factory class itself.
    Verify via :attr:`~isaaclab.sensors.frame_transformer.BaseFrameTransformer.__backend_name__`.
    """
    with _ovphysx_sim_context(device=device) as sim:
        sim._app_control_on_stop_handle = None
        cfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/LF_SHANK")],
        )
        scene_cfg = _SceneCfg(num_envs=2, env_spacing=2.0)
        scene_cfg.frame_transformer = cfg
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        sensor = scene.sensors["frame_transformer"]
        assert isinstance(sensor, BaseFrameTransformer)
        assert sensor.__backend_name__ == "ovphysx"
