# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ovrtx renders in a process that has loaded usd-exchange, in either order.

usd-exchange 3.0.0 and ovrtx 0.5.0 shipped differently built MaterialX DLLs under the same names, and on Windows
whichever loaded second failed (nvbug 6817794). ovrtx loads its runtime only when a renderer is created, so every
case creates one, each in a fresh interpreter so that the load order is the case's own.
"""

import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.windows_ci,
    pytest.mark.skipif(sys.platform != "win32", reason="the DLL name clash is specific to the Windows loader"),
]


def _load_usd():
    from pxr import Usd  # noqa: F401


def _load_materialx():
    import usdex.core  # noqa: F401

    from pxr import UsdMtlx  # noqa: F401


def _create_renderer():
    import ovrtx

    ovrtx.Renderer()


def _render_cartpole_camera():
    import torch

    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env import CartpoleCameraEnv
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, _ = resolve_task_config(
        "Isaac-Cartpole-Camera-Direct", "", overrides=("physics=newton_mjwarp", "renderer=ovrtx")
    )
    env_cfg.scene.num_envs = 2
    env_cfg.frame_stack = 1
    env = CartpoleCameraEnv(cfg=env_cfg)
    try:
        obs, _ = env.reset()
        image = obs["policy"]
        assert torch.isfinite(image).all(), "Camera observation contains NaN or infinity"
        assert image.amax() > image.amin(), "Camera observation is constant"
    finally:
        env.close()


_CASES = {
    "ovrtx-alone": (_create_renderer,),
    "usd-first": (_load_usd, _create_renderer),
    "materialx-first": (_load_materialx, _create_renderer),
    "ovrtx-first": (_create_renderer, _load_materialx),
    "isaaclab-render": (_render_cartpole_camera,),
}


@pytest.mark.parametrize("case", _CASES)
def test_ovrtx_coexists_with_usd_exchange(case):
    result = subprocess.run([sys.executable, __file__, case], capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, f"{case} exited with {result.returncode}:\n{result.stdout}\n{result.stderr}"


if __name__ == "__main__":
    for step in _CASES[sys.argv[1]]:
        step()
