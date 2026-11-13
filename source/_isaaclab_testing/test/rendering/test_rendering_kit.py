# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Golden scenes rendered through Kit-compatible renderer/backend pairs."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

import pytest  # noqa: E402
import torch  # noqa: E402
from isaaclab_physx.renderers import IsaacRtxRendererCfg  # noqa: E402
from rendering_cases import KIT_RENDERING_CASES, RenderCase  # noqa: E402
from rendering_runner import run_rendering_case  # noqa: E402
from rendering_runtime import build_rendering_scene  # noqa: E402
from rendering_scene_cfgs import EndeavorGaussianSplatsSceneCfg  # noqa: E402

from isaaclab.app.settings_manager import get_settings_manager  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci


@pytest.mark.parametrize("case", [pytest.param(case, id=case.id, marks=case.marks) for case in KIT_RENDERING_CASES])
def test_rendering(case: RenderCase, request: pytest.FixtureRequest) -> None:
    """Compare every compatible camera output from one scene construction."""
    run_rendering_case(case, request)


def test_gaussian_raw_hdr_without_isp() -> None:
    """Isaac RTX returns finite raw HDR for Gaussian splats on a camera without an ISP."""
    settings = get_settings_manager()
    gaussian_setting = "/rtx/rtpt/gaussian/skipTonemapping/enabled"
    previous_gaussian_setting = settings.get(gaussian_setting)
    scene_cfg = EndeavorGaussianSplatsSceneCfg(num_envs=1, env_spacing=50.0)
    scene_cfg.camera.renderer_cfg = IsaacRtxRendererCfg()
    scene_cfg.camera.isp_cfg = None
    try:
        # A preceding PPISP case can leave this process-wide setting False and hide the regression.
        # Start from the renderer's default; the camera's backend must establish its own HDR policy.
        settings.set(gaussian_setting, True)
        with build_rendering_scene(scene_cfg, "physx", data_types=("rgb_hdr",)) as runtime:
            for _ in range(8):
                runtime.render_camera()
            outputs, _ = runtime.camera_outputs()
            hdr = outputs["rgb_hdr"][..., :3]
            assert hdr.is_floating_point(), "Raw HDR must retain floating-point camera data."
            assert torch.isfinite(hdr).all(), "Gaussian raw HDR contains non-finite RGB values."
            assert hdr.count_nonzero() > 0, "Gaussian raw HDR contains no scene color."
    finally:
        settings.set(gaussian_setting, previous_gaussian_setting)
