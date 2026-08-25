# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression guard for OMPE-105749: a deforming cable culled by its own scene partition.

Kit RTX measures the bounding box of a ``UsdGeom.BasisCurves`` prim once and never refreshes it as
the curve deforms, so a per-environment scene partition sized from that box culls the cable as soon
as it moves outside it. ``Isaac-Lift-Cable-Franka-Camera`` pins its partition to the whole workspace
with two marker cubes. The golden AOV tests in ``test_rendering_franka_cable.py`` capture a settled
cable, so they never leave the spawn bounding box and cannot catch this.
"""

# Launch Isaac Sim Simulator first for kit-based combinations.
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sensors import Camera  # noqa: E402
from isaaclab.utils.renderers import isaac_rtx_per_env_scene_partition_enabled  # noqa: E402

from isaaclab_tasks.core.lift.config.franka_soft.franka_cable_env_cfg import (  # noqa: E402
    FrankaCableCameraEnvCfg,
)
from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci

# Distance the cable is moved along +y [m]. Swept in 0.1 m steps on Kit 110.1.2: without the marker
# cubes the cable is culled from 0.6 m onward, and past 0.9 m it leaves the frustum whatever the
# partition does. 0.7 m is clear of both. A probe shorter than 0.6 m cannot see the bug at all.
_PROBE_OFFSET_Y = 0.7

# The shipped camera is 128x128 and clips at 3 m, too tight to keep the cable framed at the probe
# pose, which would make a zero count ambiguous. Widen both so zero can only mean culling.
_PROBE_CLIPPING_RANGE = (0.01, 60.0)
_PROBE_RESOLUTION = 512


def _cable_pixels(camera: Camera) -> int:
    """Count pixels whose instance id resolves to a prim path under a cable.

    Reads ``instance_id_segmentation_fast`` rather than RGB, so the check is exact and needs no
    golden image: a culled cable contributes no pixels to any AOV.
    """
    output = camera.data.output["instance_id_segmentation_fast"]
    ids = (output if isinstance(output, torch.Tensor) else output.torch).cpu().numpy()
    labels = (camera.data.info or {}).get("instance_id_segmentation_fast", {}).get("idToLabels", {})
    cable_ids = [int(key) for key, label in labels.items() if "/Cable" in str(label)]
    return int(np.isin(ids, cable_ids).sum()) if cable_ids else 0


def test_cable_visible_beyond_partition_bounds():
    """The cable still renders after moving past the bounds an unpinned partition would have."""
    if not isaac_rtx_per_env_scene_partition_enabled():
        # The env drops the markers when partitioning is off, so there is no workaround to guard.
        pytest.skip("partitioning disabled by ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION")

    env_cfg = resolve_presets(FrankaCableCameraEnvCfg(), {"newton_mjwarp_vbd_proxy", "isaacsim_rtx"})
    env_cfg.scene.num_envs = 1

    camera_cfg = env_cfg.scene.base_camera
    # ``rgb`` keeps the env's own base_image observation term fed; the assertions read the
    # segmentation AOV. Both are static AOVs, so they share a single render product.
    camera_cfg.data_types = ["rgb", "instance_id_segmentation_fast"]
    camera_cfg.width = camera_cfg.height = _PROBE_RESOLUTION
    camera_cfg.spawn.clipping_range = _PROBE_CLIPPING_RANGE
    # int32 ids keyed by id in ``idToLabels``; the colorized form keys by RGBA tuple instead.
    camera_cfg.renderer_cfg.colorize_instance_id_segmentation = False
    # Exercise the partitioned path regardless of the ambient env-var default.
    camera_cfg.renderer_cfg.enable_scene_partitioning = True

    # Spawn the cable deterministically, and stop the MDP resetting it once the probe pose puts it
    # outside the task's workspace bounds.
    env_cfg.events.reset_cable.params["position_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    env_cfg.terminations.cable_out_of_bounds = None
    env_cfg.terminations.ee_below_table = None
    env_cfg.terminations.joint_vel_out_of_limit = None
    # Training ramps gravity from ~0 to -9.81; without this the cable floats instead of draping.
    if env_cfg.curriculum is not None:
        env_cfg.curriculum.gravity = None

    env = ManagerBasedRLEnv(env_cfg)
    try:
        camera = env.scene.sensors["base_camera"]
        cable = env.scene["cable"]
        env_ids = torch.arange(env.num_envs, device=env.device)
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

        # Settle onto the table, and confirm the cable renders at all before moving it.
        for _ in range(2):
            env.step(actions)
        settled = _cable_pixels(camera)
        assert settled > 0, "cable is not visible at its spawn pose; the probe cannot detect culling"

        pose = cable.data.default_segment_pose_w.torch.clone()
        pose[..., 1] += _PROBE_OFFSET_Y
        cable.write_segment_pose_to_sim_index(segment_pose=pose, env_ids=env_ids)
        cable.write_segment_velocity_to_sim_index(
            segment_velocity=cable.data.default_segment_velocity_w.torch.clone(), env_ids=env_ids
        )
        env.step(actions)

        moved = _cable_pixels(camera)
        assert moved > 0, (
            f"cable vanished after moving {_PROBE_OFFSET_Y} m in y ({settled} px at spawn, {moved} px"
            " after): the scene partition is culling the deforming curve. Check that the"
            " partition-bounds markers still spawn and still envelop the probe pose (OMPE-105749)."
        )
    finally:
        # Explicit cleanup before pytest teardown, otherwise OV native code can complain about leaks.
        env.close()
