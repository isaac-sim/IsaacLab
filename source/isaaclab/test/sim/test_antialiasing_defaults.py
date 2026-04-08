# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app


"""Rest everything follows."""

import logging

import pytest

from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.sim.simulation_cfg import RenderCfg, SimulationCfg
from isaaclab.sim.simulation_context import SimulationContext


# Capture warnings from the simulation_context logger
_warning_capture: list[str] = []
_original_warning = None


@pytest.fixture(autouse=True)
def _clear_sim_and_warnings():
    """Clear SimulationContext and warning capture before each test."""
    SimulationContext.clear_instance()
    _warning_capture.clear()
    yield
    SimulationContext.clear_instance()


@pytest.fixture(autouse=True, scope="session")
def _install_warning_capture():
    """Install a logging handler to capture temporal AA warnings."""

    class _Handler(logging.Handler):
        def handle(self, record):
            _warning_capture.append(record.getMessage())
            return True

    handler = _Handler(level=logging.WARNING)
    logger = logging.getLogger("isaaclab.sim.simulation_context")
    logger.addHandler(handler)
    yield
    logger.removeHandler(handler)


def _create_sim(render_cfg=None):
    """Create a SimulationContext with the given RenderCfg."""
    if render_cfg is not None:
        cfg = SimulationCfg(render=render_cfg)
    else:
        cfg = SimulationCfg()
    return SimulationContext(cfg)


def _get_aa_op():
    return get_settings_manager().get("/rtx/post/aa/op")


def _get_limited_ops():
    return get_settings_manager().get("/rtx-transient/post/aa/limitedOps")


def _has_temporal_warning():
    return any("temporal" in w for w in _warning_capture)


# -- Tests -------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_default_rendercfg_is_fxaa():
    """RenderCfg() should default to FXAA anti-aliasing."""
    assert RenderCfg().antialiasing_mode == "FXAA"


@pytest.mark.isaacsim_ci
def test_default_aa_op_is_fxaa():
    """Default SimulationContext should apply FXAA (aa/op=2) via Replicator."""
    _create_sim()
    assert _get_aa_op() == 2, f"Expected aa/op=2 (FXAA), got {_get_aa_op()}"


@pytest.mark.isaacsim_ci
def test_default_limited_ops_disabled():
    """Default FXAA should set limitedOps=False so the C++ renderer doesn't override to DLSS."""
    _create_sim()
    assert _get_limited_ops() is False, f"Expected limitedOps=False, got {_get_limited_ops()}"


@pytest.mark.isaacsim_ci
def test_fxaa_no_temporal_warning():
    """FXAA (default) should NOT trigger the temporal AA warning."""
    _create_sim()
    assert not _has_temporal_warning(), "FXAA should not trigger temporal warning"


@pytest.mark.isaacsim_ci
def test_off_no_temporal_warning():
    """AA Off should NOT trigger the temporal AA warning."""
    _create_sim(RenderCfg(antialiasing_mode="Off"))
    assert _get_aa_op() == 0, f"Expected aa/op=0 (Off), got {_get_aa_op()}"
    assert _get_limited_ops() is False
    assert not _has_temporal_warning(), "Off should not trigger temporal warning"


@pytest.mark.isaacsim_ci
def test_dlss_triggers_temporal_warning():
    """DLSS should trigger the temporal AA warning."""
    _create_sim(RenderCfg(antialiasing_mode="DLSS"))
    assert _get_aa_op() == 3, f"Expected aa/op=3 (DLSS), got {_get_aa_op()}"
    assert _has_temporal_warning(), "DLSS should trigger temporal warning"


@pytest.mark.isaacsim_ci
def test_taa_triggers_temporal_warning():
    """TAA should trigger the temporal AA warning and set limitedOps=False."""
    _create_sim(RenderCfg(antialiasing_mode="TAA"))
    assert _get_aa_op() == 1, f"Expected aa/op=1 (TAA), got {_get_aa_op()}"
    assert _get_limited_ops() is False
    assert _has_temporal_warning(), "TAA should trigger temporal warning"


@pytest.mark.isaacsim_ci
def test_dlaa_triggers_temporal_warning():
    """DLAA should trigger the temporal AA warning."""
    _create_sim(RenderCfg(antialiasing_mode="DLAA"))
    assert _get_aa_op() == 4, f"Expected aa/op=4 (DLAA), got {_get_aa_op()}"
    assert _has_temporal_warning(), "DLAA should trigger temporal warning"


@pytest.mark.isaacsim_ci
def test_none_skips_replicator():
    """antialiasing_mode=None should skip the Replicator call entirely (no warning)."""
    _create_sim(RenderCfg(antialiasing_mode=None))
    assert not _has_temporal_warning(), "None should not trigger temporal warning"


@pytest.mark.isaacsim_ci
def test_explicit_dlss_overrides_default():
    """Explicit DLSS should override the FXAA default."""
    _create_sim(RenderCfg(antialiasing_mode="DLSS"))
    assert _get_aa_op() == 3, f"Expected aa/op=3 (DLSS), got {_get_aa_op()}"
