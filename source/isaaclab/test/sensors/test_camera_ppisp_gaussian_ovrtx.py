# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the camera PPISP wrapper applied to a 3D Gaussian (NuRec /
ParticleField) scene through the ``ovrtx`` renderer.

The asset is synthesised at test time by :mod:`generate_synthetic_gaussian_asset`
and rendered via :func:`generate_synthetic_gaussian_asset.render_synthetic_gaussian_scene`.
The aggressive wrapper PPISP cfg
(:func:`generate_synthetic_gaussian_asset.make_aggressive_ppisp_cfg`) intentionally
engages every feature past its subtle-correction defaults so each can be
asserted independently:

* **Exposure/responsivity** tuned so each renderer's HDR scale maps into a
  non-degenerate, non-saturated LDR center patch.
* **Vignetting** with ``alpha1 = -1.8`` per channel plus per-channel higher
  order coefficients — corners are much darker than the center.
* **Color homography** that shifts red and green chromaticity anchors.
* **CRF** with stronger shoulder than default — bright input compresses before
  the final [0, 1] kernel clamp.

The integration test checks *semantic invariants* of the PPISP pipeline
(OVRTX produces HDR, PPISP maps it to useful LDR, vignetting darkens corners,
and output stays in [0, 255]) instead of a fidelity-against-baked comparison,
which would have to absorb renderer-internal HDR-magnitude calibration drift
between renderers.

Notes:
  * Runs **kit-less**: this test does not call
    :class:`~isaaclab.app.AppLauncher`. ``ovrtx`` and Isaac Sim Kit ship the
    same RTX hydra libraries (``librtx.hydra.so``, ``liblegacy.hydra.so``)
    under conflicting USD namespaces; loading both into the same process
    causes a dynamic-linker crash. See
    :func:`isaaclab.app.sim_launcher.launch_simulation` for the
    documented incompatibility.
  * Uses Newton physics because ``ovrtx`` is incompatible with Kit/Isaac Sim
    and the PhysX backend requires Kit (``carb``) to bootstrap.
  * Requests ``"rgb_hdr"`` in ``data_types`` because the test asserts the raw
    HDR source with :func:`assert_ppisp_lifts_exposure`. The PPISP render path
    itself also allocates an internal HDR buffer when ``isp_cfg`` is set, so
    ``"rgb_hdr"`` is not required just to enable PPISP.
"""

import importlib.util
import tempfile

import pytest
from generate_synthetic_gaussian_asset import (
    SYNTHETIC_GAUSSIAN_CAMERA_REGEX,
    assert_images_meaningfully_different,
    assert_ppisp_controller_matches_static,
    assert_ppisp_invariants,
    assert_ppisp_lifts_exposure,
    make_aggressive_ppisp_cfg,
    make_neutral_ppisp_cfg,
    make_synthetic_gaussian_usd,
    render_synthetic_gaussian_scene,
    render_synthetic_gaussian_scene_with_controller_ppisp_attrs,
    render_synthetic_gaussian_scene_with_static_ppisp_attrs,
)

from isaaclab.sim import SimulationCfg

# OVRTX renderer + Newton physics are required (kit-less + non-PhysX). Use a
# collection-time skip marker instead of module-level ``importorskip`` so CI's
# per-file runner does not see pytest's "no tests collected" exit code.
_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "isaaclab_newton")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]
_SKIP_MISSING_OVRTX = pytest.mark.skipif(
    bool(_MISSING_MODULES),
    reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
)

if not _MISSING_MODULES:
    from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg  # noqa: E402
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg  # noqa: E402
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
else:
    MJWarpSolverCfg = None
    NewtonCfg = None
    OVRTXRendererCfg = None

SIM_DT = 1.0 / 60.0
MULTI_TILE_COUNT = 4

# Mark the gaussian-on-OVRTX tests xfail by default: the wrapper-side
# ``carb.settings.set_bool`` call that disables RTX-side tonemapping is a
# no-op for the kit-less ovrtx backend, and the equivalent must be applied
# externally before launching pytest.
_XFAIL_OVRTX_GAUSSIAN_PPISP = pytest.mark.xfail(
    reason=(
        "kit-less ovrtx cannot toggle RTX-side tonemapping at runtime; the equivalent must be "
        "applied externally before launching pytest"
    ),
    strict=False,
)


def _ovrtx_sim_cfg(device: str) -> SimulationCfg:
    return SimulationCfg(
        dt=SIM_DT,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1),
        device=device,
    )


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.isaacsim_ci
@_SKIP_MISSING_OVRTX
@_XFAIL_OVRTX_GAUSSIAN_PPISP
def test_camera_ppisp_wrapper_signatures_on_synthetic_gaussians_ovrtx(device):
    """Wrapper PPISP via ``ovrtx`` must show every PPISP-feature signature.

    Renders a synthetic RGBW gaussian grid through ``ovrtx`` plus the
    aggressive wrapper PPISP cfg and asserts:

    1. **Non-degenerate frame** — content is rendered (not pure black / pure white).
    2. **HDR source** — ``rgb_hdr`` is present and bright enough for PPISP.
    3. **PPISP LDR mapping** — the center patch lands in a useful, non-saturated
       LDR range after the calibrated responsivity/exposure pair.
    4. **Vignetting** — each corner patch mean is meaningfully below the center patch mean.
    5. **CRF/clamping** — no value exceeds 255.
    """
    with tempfile.TemporaryDirectory(prefix="isaaclab-synth-gauss-") as tmpdir:
        asset_path = make_synthetic_gaussian_usd(f"{tmpdir}/synthetic_gaussians.usda")
        output = render_synthetic_gaussian_scene(
            asset_path,
            sim_cfg=_ovrtx_sim_cfg(device),
            renderer_cfg=OVRTXRendererCfg(),
            data_types=["rgb", "rgb_hdr"],
            sim_dt=SIM_DT,
        )
    assert_ppisp_lifts_exposure(output["rgb_hdr"][0], output["rgb"][0], label="ovrtx")
    assert_ppisp_invariants(output["rgb"][0], label="ovrtx")


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.isaacsim_ci
@_SKIP_MISSING_OVRTX
@_XFAIL_OVRTX_GAUSSIAN_PPISP
def test_camera_ppisp_authored_static_attrs_are_applied_on_synthetic_gaussians_ovrtx(device):
    """OVRTX must apply camera-authored static PPISP attributes."""
    with tempfile.TemporaryDirectory(prefix="isaaclab-synth-gauss-") as tmpdir:
        asset_path = make_synthetic_gaussian_usd(f"{tmpdir}/synthetic_gaussians.usda")
        aggressive_cfg = make_aggressive_ppisp_cfg()

        neutral = render_synthetic_gaussian_scene_with_static_ppisp_attrs(
            asset_path,
            sim_cfg=_ovrtx_sim_cfg(device),
            renderer_cfg=OVRTXRendererCfg(),
            ppisp_cfg=make_neutral_ppisp_cfg(),
            data_types=["rgb", "rgb_hdr"],
            sim_dt=SIM_DT,
        )
        aggressive = render_synthetic_gaussian_scene_with_static_ppisp_attrs(
            asset_path,
            sim_cfg=_ovrtx_sim_cfg(device),
            renderer_cfg=OVRTXRendererCfg(),
            ppisp_cfg=aggressive_cfg,
            data_types=["rgb", "rgb_hdr"],
            sim_dt=SIM_DT,
        )

    assert_images_meaningfully_different(neutral["rgb"][0], aggressive["rgb"][0], label="ovrtx authored PPISP")
    assert_ppisp_lifts_exposure(aggressive["rgb_hdr"][0], aggressive["rgb"][0], label="ovrtx authored PPISP")
    assert_ppisp_invariants(aggressive["rgb"][0], label="ovrtx authored PPISP")


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.isaacsim_ci
@_SKIP_MISSING_OVRTX
@_XFAIL_OVRTX_GAUSSIAN_PPISP
def test_camera_ppisp_controller_matches_static_attrs_on_synthetic_gaussians_ovrtx(device):
    """OVRTX controller output must match the equivalent static PPISP cfg."""
    with tempfile.TemporaryDirectory(prefix="isaaclab-synth-gauss-") as tmpdir:
        asset_path = make_synthetic_gaussian_usd(f"{tmpdir}/synthetic_gaussians.usda")
        ppisp_cfg = make_aggressive_ppisp_cfg()

        static = render_synthetic_gaussian_scene_with_static_ppisp_attrs(
            asset_path,
            sim_cfg=_ovrtx_sim_cfg(device),
            renderer_cfg=OVRTXRendererCfg(),
            ppisp_cfg=ppisp_cfg,
            data_types=["rgb", "rgb_hdr"],
            sim_dt=SIM_DT,
        )
        controller = render_synthetic_gaussian_scene_with_controller_ppisp_attrs(
            asset_path,
            sim_cfg=_ovrtx_sim_cfg(device),
            renderer_cfg=OVRTXRendererCfg(),
            ppisp_cfg=ppisp_cfg,
            data_types=["rgb", "rgb_hdr"],
            sim_dt=SIM_DT,
        )

    assert_ppisp_controller_matches_static(static["rgb"][0], controller["rgb"][0], label="ovrtx controller")
    assert_ppisp_invariants(controller["rgb"][0], label="ovrtx controller")


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.isaacsim_ci
@_SKIP_MISSING_OVRTX
@_XFAIL_OVRTX_GAUSSIAN_PPISP
def test_camera_ppisp_wrapper_signatures_on_synthetic_gaussians_ovrtx_multitile(device):
    """Multi-tile wrapper PPISP via ``ovrtx`` must hold the same invariants
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
            sim_cfg=_ovrtx_sim_cfg(device),
            renderer_cfg=OVRTXRendererCfg(),
            data_types=["rgb", "rgb_hdr"],
            num_envs=MULTI_TILE_COUNT,
            sim_dt=SIM_DT,
        )

    rgb = output["rgb"]
    rgb_hdr = output["rgb_hdr"]
    assert rgb.shape[0] == MULTI_TILE_COUNT, (
        f"Expected {MULTI_TILE_COUNT} tiles, got shape={tuple(rgb.shape)}. "
        f"Check that the camera regex {SYNTHETIC_GAUSSIAN_CAMERA_REGEX} resolves to one camera per env."
    )
    for i in range(MULTI_TILE_COUNT):
        assert_ppisp_lifts_exposure(rgb_hdr[i], rgb[i], label=f"ovrtx tile {i}")
        assert_ppisp_invariants(rgb[i], label=f"ovrtx tile {i}")
