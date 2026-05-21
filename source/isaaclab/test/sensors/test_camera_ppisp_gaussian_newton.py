# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the camera ISP wrapper applied to a 3D Gaussian (NuRec /
ParticleField) scene through the Newton (warp) renderer.

The scene is synthesised at test time by :mod:`generate_synthetic_gaussian_asset`
and rendered via :func:`generate_synthetic_gaussian_asset.render_synthetic_gaussian_scene`.
The aggressive wrapper ISP cfg
(:func:`generate_synthetic_gaussian_asset.make_aggressive_ppisp_cfg`) engages every
ISP feature past its subtle-correction defaults so the integration test can
check semantic invariants — vignetting darkens corners, exposure increases
mean, CRF clamps output to [0, 255] — without needing to do a fidelity
comparison against a baked-SH reference.

Newton's Warp ray tracer is not physically based, so absolute brightness for
the same scene differs from RTX-backed renderers, but the ISP-side signatures
(vignetting falloff ratio, no overflow, etc.) are renderer-agnostic and the
same thresholds apply.

Notes:
  * Uses ``InteractiveScene`` because ``newton_warp`` requires
    ``/World/envs/env_0/...`` and a Newton model owned by ``NewtonManager``;
    bare ``create_prim`` won't populate either.
  * The shared helper adds an invisible rigid-body anchor under env_0 because
    Newton fails to build a model when the scene has no rigid bodies (just
    ParticleField).
"""

import importlib.util

import pytest

_REQUIRED_MODULES = ("isaaclab_newton", "newton")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]
_SKIP_MISSING_NEWTON = pytest.mark.skipif(
    bool(_MISSING_MODULES),
    reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
)

if not _MISSING_MODULES:
    # Launch Isaac Sim before importing modules that depend on an active app.

    from isaaclab.app import AppLauncher  # noqa: E402

    simulation_app = AppLauncher(headless=True, enable_cameras=True).app

    import tempfile  # noqa: E402

    from generate_synthetic_gaussian_asset import (  # noqa: E402
        SYNTHETIC_GAUSSIAN_CAMERA_REGEX,
        assert_ppisp_invariants,
        assert_ppisp_lifts_exposure,
        make_synthetic_gaussian_usd,
        render_synthetic_gaussian_scene,
    )
    from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg  # noqa: E402
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg  # noqa: E402
    from isaaclab_newton.renderers import NewtonWarpRendererCfg  # noqa: E402

    from isaaclab.sim import SimulationCfg  # noqa: E402
else:
    tempfile = None
    SYNTHETIC_GAUSSIAN_CAMERA_REGEX = None
    assert_ppisp_invariants = None
    assert_ppisp_lifts_exposure = None
    make_synthetic_gaussian_usd = None
    render_synthetic_gaussian_scene = None
    SimulationCfg = None
    MJWarpSolverCfg = None
    NewtonCfg = None
    NewtonWarpRendererCfg = None

SIM_DT = 1.0 / 60.0
MULTI_TILE_COUNT = 4


def _newton_sim_cfg(device: str) -> SimulationCfg:
    return SimulationCfg(
        dt=SIM_DT,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1),
        device=device,
    )


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.isaacsim_ci
@_SKIP_MISSING_NEWTON
def test_camera_ppisp_wrapper_signatures_on_synthetic_gaussians_newton(device):
    """Wrapper ISP via ``newton_warp`` must show every ISP-feature signature.

    Renders a synthetic RGBW gaussian grid through Newton's Warp ray tracer
    plus the aggressive wrapper ISP cfg and asserts the same semantic
    invariants (via :func:`assert_ppisp_invariants`) as the ovrtx counterpart:
    vignetting falloff, exposure-boosted center, output bounded in [0, 255].
    """
    with tempfile.TemporaryDirectory(prefix="isaaclab-synth-gauss-") as tmpdir:
        asset_path = make_synthetic_gaussian_usd(f"{tmpdir}/synthetic_gaussians.usda")
        output = render_synthetic_gaussian_scene(
            asset_path,
            sim_cfg=_newton_sim_cfg(device),
            renderer_cfg=NewtonWarpRendererCfg(),
            data_types=["rgb", "rgb_hdr"],
            sim_dt=SIM_DT,
            # Newton's emissive HDR is ~50x lower than the RTX backends' for
            # the same scene; lift the effective signal so the aggressive cfg
            # (tuned for the RTX scale) lands in the same LDR range.
            responsivity=50.0,
        )
    assert_ppisp_lifts_exposure(output["rgb_hdr"][0], output["rgb"][0], label="newton_warp")
    assert_ppisp_invariants(output["rgb"][0], label="newton_warp")


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.isaacsim_ci
@_SKIP_MISSING_NEWTON
def test_camera_ppisp_wrapper_signatures_on_synthetic_gaussians_newton_multitile(device):
    """Multi-tile wrapper ISP via ``newton_warp`` must hold the same invariants
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
            sim_cfg=_newton_sim_cfg(device),
            renderer_cfg=NewtonWarpRendererCfg(),
            data_types=["rgb", "rgb_hdr"],
            num_envs=MULTI_TILE_COUNT,
            sim_dt=SIM_DT,
            responsivity=50.0,
        )

    rgb = output["rgb"]
    rgb_hdr = output["rgb_hdr"]
    assert rgb.shape[0] == MULTI_TILE_COUNT, (
        f"Expected {MULTI_TILE_COUNT} tiles, got shape={tuple(rgb.shape)}. "
        f"Check that the camera regex {SYNTHETIC_GAUSSIAN_CAMERA_REGEX} resolves to one camera per env."
    )
    for i in range(MULTI_TILE_COUNT):
        assert_ppisp_lifts_exposure(rgb_hdr[i], rgb[i], label=f"newton_warp tile {i}")
        assert_ppisp_invariants(rgb[i], label=f"newton_warp tile {i}")
