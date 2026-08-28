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
    assert_gaussian_contribution,
    assert_images_meaningfully_different,
    assert_ppisp_controller_matches_static,
    assert_ppisp_invariants,
    assert_ppisp_lifts_exposure,
    assert_tiled_views_match,
    make_aggressive_ppisp_cfg,
    make_neutral_ppisp_cfg,
    make_offscreen_gaussian_scene,
    make_synthetic_gaussian_usd,
    render_synthetic_gaussian_scene,
    render_synthetic_gaussian_scene_with_controller_ppisp_attrs,
    render_synthetic_gaussian_scene_with_static_ppisp_attrs,
)

from isaaclab.sim import SimulationCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering]

# Use collection-time skip markers so unavailable optional modules remain
# visible per test in reports.
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

# OVRTX renders no gaussian contribution in any tile once more than one view tile is
# active, for the ``sortingModeHint = "zDepth"`` sort mode that NuRec exports author (and
# that the renderer also defaults to when the token is absent). Measured against a control
# render whose gaussians sit outside the frustum, at 128/256/512 px per tile:
#
#   num_envs=1, zDepth          -> mean abs diff 8.6   (gaussians rendered)
#   num_envs=2, zDepth          -> mean abs diff 0.35  (nothing rendered)
#   num_envs=2, cameraDistance  -> mean abs diff 17.5  (gaussians rendered)
#
# ``cameraDistance`` is the only gaussian sort mode that does not read the per-view-tile
# ``WorldToView`` matrix (``cameraForward`` in ``rtx/raytracing/Gaussians.is.hlsl``), and
# that matrix switches from the constant buffer to the view-tile buffers exactly when the
# view tile count exceeds one. isaac_rtx renders zDepth correctly at num_envs=2 through the
# same shader, so this is an OVRTX-side view-tile setup issue rather than an asset problem.
_XFAIL_OVRTX_MULTI_TILE_GAUSSIANS = pytest.mark.xfail(
    reason="ovrtx drops all gaussian contribution for zDepth sorting when more than one view tile is active",
    strict=True,
)


def _ovrtx_sim_cfg(device: str) -> SimulationCfg:
    return SimulationCfg(
        dt=SIM_DT,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1),
        device=device,
    )


@pytest.mark.parametrize("device", ["cuda:0"])
@_SKIP_MISSING_OVRTX
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
            stabilisation_steps=15,
        )
    assert_ppisp_lifts_exposure(output["rgb_hdr"][0], output["rgb"][0], label="ovrtx")
    assert_ppisp_invariants(output["rgb"][0], label="ovrtx")


@pytest.mark.parametrize("device", ["cuda:0"])
@_SKIP_MISSING_OVRTX
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
@_SKIP_MISSING_OVRTX
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
@_SKIP_MISSING_OVRTX
@_XFAIL_OVRTX_MULTI_TILE_GAUSSIANS
def test_camera_ppisp_wrapper_signatures_on_synthetic_gaussians_ovrtx_multitile(device):
    """Multi-tile wrapper PPISP via ``ovrtx`` must hold the same invariants
    independently for every tile.

    Builds an :class:`InteractiveScene` with :data:`MULTI_TILE_COUNT` envs so
    the camera regex resolves to one camera per env. Both ``rgb`` and
    ``rgb_hdr`` are batched over the matched cameras, and each tile is checked
    independently for HDR presence, useful PPISP LDR mapping, vignetting, and
    bounded output.

    Every tile is also compared against a control render whose gaussians sit outside the
    frustum. The PPISP signatures above hold on a gaussian-free render as well — the
    background alone satisfies them — so without that comparison this test passes even when
    the renderer drops every splat, which is exactly what OVRTX does here (see
    :data:`_XFAIL_OVRTX_MULTI_TILE_GAUSSIANS`).
    """
    with tempfile.TemporaryDirectory(prefix="isaaclab-synth-gauss-") as tmpdir:
        render_kwargs = dict(
            sim_cfg=_ovrtx_sim_cfg(device),
            renderer_cfg=OVRTXRendererCfg(),
            data_types=["rgb", "rgb_hdr"],
            num_envs=MULTI_TILE_COUNT,
            sim_dt=SIM_DT,
        )
        output = render_synthetic_gaussian_scene(
            make_synthetic_gaussian_usd(f"{tmpdir}/synthetic_gaussians.usda"), **render_kwargs
        )
        control = render_synthetic_gaussian_scene(
            make_synthetic_gaussian_usd(f"{tmpdir}/offscreen_gaussians.usda", scene=make_offscreen_gaussian_scene()),
            **render_kwargs,
        )

    rgb = output["rgb"]
    rgb_hdr = output["rgb_hdr"]
    assert rgb.shape[0] == MULTI_TILE_COUNT, (
        f"Expected {MULTI_TILE_COUNT} tiles, got shape={tuple(rgb.shape)}. "
        f"Check that the camera regex {SYNTHETIC_GAUSSIAN_CAMERA_REGEX} resolves to one camera per env."
    )
    assert_tiled_views_match(rgb, label="ovrtx rgb")
    # Tile HDR means spread by ~6% across the four tiles even though every env renders the
    # same content (measured 11.91 / 12.36 / 12.24 / 12.67 with splats present, against
    # 0.5% spread on a gaussian-free control), so the default tolerance is too tight here.
    # A tile that rendered no gaussians at all differs far more than this bound allows, and
    # assert_gaussian_contribution below tests that case directly rather than via tolerance.
    assert_tiled_views_match(rgb_hdr, max_relative_mean_abs_diff=0.1, label="ovrtx rgb_hdr")
    for i in range(MULTI_TILE_COUNT):
        assert_ppisp_lifts_exposure(rgb_hdr[i], rgb[i], label=f"ovrtx tile {i}")
        assert_ppisp_invariants(rgb[i], label=f"ovrtx tile {i}")
        assert_gaussian_contribution(rgb[i], control["rgb"][i], label=f"ovrtx tile {i}")
