# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the camera ISP wrapper applied to a 3D Gaussian (NuRec /
ParticleField) scene through the ``ovrtx`` renderer.

The asset is synthesised at test time by :mod:`generate_synthetic_gaussian_asset`
and rendered via :func:`generate_synthetic_gaussian_asset.render_synthetic_gaussian_scene`.
The aggressive wrapper ISP cfg
(:func:`generate_synthetic_gaussian_asset.make_aggressive_ppisp_cfg`) intentionally
engages every feature past its subtle-correction defaults so each can be
asserted independently:

* **Exposure** offset of +2 stops (input ×4).
* **Vignetting** with ``alpha1 = -1.5`` per channel — corners attenuate to
  ~25-40% of center.
* **Color homography** that shifts red and green chromaticity anchors.
* **CRF** with stronger shoulder than default — saturated input compresses
  rather than clips.

The integration test checks *semantic invariants* of the ISP pipeline
(vignetting darkens corners, exposure increases mean, no overflow above 255)
instead of a fidelity-against-baked comparison, which would have to absorb
renderer-internal HDR-magnitude calibration drift between renderers.

Notes:
  * Runs **kit-less**: this test does not call
    :class:`~isaaclab.app.AppLauncher`. ``ovrtx`` and Isaac Sim Kit ship the
    same RTX hydra libraries (``librtx.hydra.so``, ``liblegacy.hydra.so``)
    under conflicting USD namespaces; loading both into the same process
    causes a dynamic-linker crash. See
    :func:`isaaclab_tasks.utils.sim_launcher.launch_simulation` for the
    documented incompatibility.
  * Uses Newton physics because ``ovrtx`` is incompatible with Kit/Isaac Sim
    and the PhysX backend requires Kit (``carb``) to bootstrap.
  * Requests ``"rgb_hdr"`` in ``data_types`` because
    ``isaaclab_ov.get_render_var_configs`` only authors ``HdrColor`` when
    ``"rgb_hdr"`` is in ``data_types``; in kit-less mode there is no
    Replicator AnnotatorRegistry to auto-add it.
"""

import importlib.util
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
    """Wrapper ISP via ``ovrtx`` must show every ISP-feature signature.

    Renders a synthetic RGBW gaussian grid through ``ovrtx`` plus the
    aggressive wrapper ISP cfg and asserts (via :func:`assert_ppisp_invariants`):

    1. **Non-degenerate frame** — content is rendered (not pure black / pure white).
    2. **Vignetting** — each corner patch mean is meaningfully below the center patch mean.
    3. **Exposure** — center patch is bright (the +2 stop boost lifts the
       gaussian colors well into the upper half of the 0-255 range).
    4. **CRF clamping** — no value exceeds 255 (kernel ``wp.clamp`` plus CRF
       compression keeps the output bounded).
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
def test_camera_ppisp_wrapper_signatures_on_synthetic_gaussians_ovrtx_multitile(device):
    """Multi-tile wrapper ISP via ``ovrtx`` must hold the same invariants
    independently for every tile.

    Builds an :class:`InteractiveScene` with :data:`MULTI_TILE_COUNT` envs so
    the camera regex resolves to one camera per env;
    :attr:`Camera.data.output["rgb"]` then carries one frame per matched
    camera and each is asserted independently.
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
