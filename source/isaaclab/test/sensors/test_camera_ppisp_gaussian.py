# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the camera PPISP wrapper applied to a 3D Gaussian (NuRec /
ParticleField) scene through the ``isaac_rtx`` renderer.

The scene is synthesised at test time by :mod:`generate_synthetic_gaussian_asset` — a few
large fully-opaque Gaussians of known colors arranged in front of the camera,
bound to ``ParticleFieldEmissive.mdl`` with ``apply_inverse_tonemap=0`` and
``apply_srgb_linear=0`` so the wrapper PPISP is the only ISP authority.

The wrapper PPISP cfg
(:func:`generate_synthetic_gaussian_asset.make_aggressive_ppisp_cfg`) engages every PPISP
feature past its subtle-correction defaults so the integration test can check
*semantic invariants* of the PPISP pipeline (the renderer produces HDR, PPISP
maps it to a non-degenerate LDR range, vignetting darkens corners, and output
stays in [0, 255]) instead of doing a fidelity-against-baked comparison —
which would have to absorb renderer-internal HDR-magnitude calibration drift
between renderers.

Renderer parametrization:
  * ``isaac_rtx`` is the only renderer exercised by this test.
  * ``ovrtx`` coverage lives in ``test_camera_ppisp_gaussian_ovrtx.py``, which
    uses the same ``InteractiveScene`` setup.
  * ``newton_warp`` coverage lives in ``test_camera_ppisp_gaussian_newton.py``,
    which builds an ``InteractiveScene`` with a Newton-backed
    ``SimulationCfg`` to give Newton the model it needs.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import tempfile

import pytest
from generate_synthetic_gaussian_asset import (
    SYNTHETIC_GAUSSIAN_CAMERA_REGEX,
    assert_ppisp_invariants,
    assert_ppisp_lifts_exposure,
    make_synthetic_gaussian_usd,
    render_synthetic_gaussian_scene,
)

from isaaclab.sim import SimulationCfg


def _collect_renderer_cfg_params() -> list:
    """Return pytest.param entries for installed RTX-backed renderer packages.

    Each renderer lives in its own optional package; a missing package is silently
    excluded so tests run on partial installs. ``ovrtx`` and ``newton_warp`` are
    not listed here — their dedicated tests
    (``test_camera_ppisp_gaussian_ovrtx.py`` / ``test_camera_ppisp_gaussian_newton.py``)
    use ``InteractiveScene`` so cameras land under ``/World/envs/env_0/``.
    """
    params: list = []
    try:
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        params.append(pytest.param(IsaacRtxRendererCfg, id="isaac_rtx"))
    except ImportError:
        pass
    return params


_RENDERER_CFG_PARAMS = _collect_renderer_cfg_params()

SIM_DT = 0.01
MULTI_TILE_COUNT = 4
ISAAC_RTX_RESPONSIVITY = 1.2


def _isaac_rtx_sim_cfg(device: str) -> SimulationCfg:
    return SimulationCfg(dt=SIM_DT, device=device)


if not _RENDERER_CFG_PARAMS:
    pytest.skip(
        "No renderer packages installed (isaaclab_physx).",
        allow_module_level=True,
    )


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("renderer_cfg_cls", _RENDERER_CFG_PARAMS)
@pytest.mark.isaacsim_ci
def test_camera_ppisp_wrapper_signatures_on_synthetic_gaussians(renderer_cfg_cls, device):
    """Wrapper PPISP via ``isaac_rtx`` must show every PPISP-feature signature.

    Renders a synthetic RGBW gaussian grid through ``isaac_rtx`` + the aggressive
    wrapper PPISP cfg and asserts:

    1. **Non-degenerate frame** — content is rendered (not pure black / pure white).
    2. **HDR source** — ``rgb_hdr`` is present and bright enough for PPISP.
    3. **PPISP LDR mapping** — the center patch lands in a useful, non-saturated
       LDR range after the calibrated responsivity/exposure pair.
    4. **Vignetting** — each corner patch mean is meaningfully below the center mean.
    5. **CRF/clamping** — output stays in [0, 255] with no overflow.
    """
    with tempfile.TemporaryDirectory(prefix="isaaclab-synth-gauss-") as tmpdir:
        asset_path = make_synthetic_gaussian_usd(f"{tmpdir}/synthetic_gaussians.usda")
        output = render_synthetic_gaussian_scene(
            asset_path,
            sim_cfg=_isaac_rtx_sim_cfg(device),
            renderer_cfg=renderer_cfg_cls(),
            data_types=["rgb", "rgb_hdr"],
            sim_dt=SIM_DT,
            responsivity=ISAAC_RTX_RESPONSIVITY,
        )
    assert_ppisp_lifts_exposure(output["rgb_hdr"][0], output["rgb"][0], label="isaac_rtx")
    assert_ppisp_invariants(output["rgb"][0], label="isaac_rtx")


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("renderer_cfg_cls", _RENDERER_CFG_PARAMS)
@pytest.mark.isaacsim_ci
def test_camera_ppisp_wrapper_signatures_on_synthetic_gaussians_multitile(renderer_cfg_cls, device):
    """Multi-tile wrapper PPISP via ``isaac_rtx`` must hold the same invariants
    independently for every tile.

    Builds an :class:`InteractiveScene` with :data:`MULTI_TILE_COUNT` envs so
    the camera regex resolves to one camera per env. Both ``rgb`` and
    ``rgb_hdr`` are batched over the matched cameras, and each tile is checked
    independently for HDR presence, useful PPISP LDR mapping, vignetting, and
    bounded output.
    """
    with tempfile.TemporaryDirectory(prefix="isaaclab-synth-gauss-") as tmpdir:
        asset_path = make_synthetic_gaussian_usd(f"{tmpdir}/synthetic_gaussians.usda")
        output = render_synthetic_gaussian_scene(
            asset_path,
            sim_cfg=_isaac_rtx_sim_cfg(device),
            renderer_cfg=renderer_cfg_cls(),
            data_types=["rgb", "rgb_hdr"],
            num_envs=MULTI_TILE_COUNT,
            sim_dt=SIM_DT,
            responsivity=ISAAC_RTX_RESPONSIVITY,
        )

    rgb = output["rgb"]
    rgb_hdr = output["rgb_hdr"]
    assert rgb.shape[0] == MULTI_TILE_COUNT, (
        f"Expected {MULTI_TILE_COUNT} tiles, got shape={tuple(rgb.shape)}. "
        f"Check that the camera regex {SYNTHETIC_GAUSSIAN_CAMERA_REGEX} resolves to one camera per env."
    )
    for i in range(MULTI_TILE_COUNT):
        assert_ppisp_lifts_exposure(rgb_hdr[i], rgb[i], label=f"isaac_rtx tile {i}")
        assert_ppisp_invariants(rgb[i], label=f"isaac_rtx tile {i}")
