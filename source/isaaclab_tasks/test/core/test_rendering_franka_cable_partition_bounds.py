# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression guard for OMPE-105749: a deforming cable culled by its own scene partition.

Kit RTX computes the bounding box of a ``UsdGeom.BasisCurves`` prim once and never refreshes it as
the curve deforms. A per-environment scene partition sized from that stale box culls the cable as
soon as it animates outside the box, so the cable vanishes from the camera images while still being
in frame. ``Isaac-Lift-Cable-Franka-Camera`` pins its partition to the whole workspace with two
marker cubes so the cable cannot leave the bounds; this test moves the cable past the point where an
unpinned partition would cull it and asserts it still renders.

The golden AOV tests in ``test_rendering_franka_cable.py`` capture a settled cable, so they never
leave the bounding box computed at spawn and cannot catch this.
"""

# Launch Isaac Sim Simulator first for kit-based combinations.
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci

# Env-local offset the cable is teleported by, in metres along +y. Calibrated by sweeping the cable
# out in 0.1 m steps on Kit 110.1.2 and comparing three configurations:
#
#   offset  partitioning off   markers absent   markers present
#   0.50    414 px             414 px           414 px
#   0.60    265 px               0 px           265 px   <- cull threshold without the markers
#   0.70    151 px               0 px           151 px
#   0.80     56 px               0 px            56 px
#   0.90      0 px               0 px             0 px   <- cable leaves the frustum
#
# 0.7 m sits clear of the 0.6 m cull threshold and clear of the 0.9 m framing limit. Below 0.6 m the
# probe cannot detect the bug at all, which is why the margin on both sides matters.
_PROBE_OFFSET_Y = 0.7

# The shipped camera renders 128x128 and clips at 3 m, which is too tight to follow the cable out to
# the cull threshold and back. Widen both for this probe so a zero count means culling, not framing.
_PROBE_CLIPPING_RANGE = (0.01, 60.0)
_PROBE_RESOLUTION = 512


def _cable_pixel_count(camera: Any) -> int:
    """Count pixels whose instance id resolves to a prim path under a cable.

    Uses ``instance_id_segmentation_fast`` rather than RGB so the check is exact and needs no golden
    image: a culled cable contributes no pixels to any AOV.
    """
    output = camera.data.output["instance_id_segmentation_fast"]
    ids = (output if isinstance(output, torch.Tensor) else output.torch).cpu().numpy().reshape(-1)
    id_to_labels = (camera.data.info or {}).get("instance_id_segmentation_fast", {}).get("idToLabels", {})
    cable_ids = [int(key) for key, label in id_to_labels.items() if "/Cable" in str(label)]
    if not cable_ids:
        return 0
    return int(np.isin(ids, cable_ids).sum())


def _make_env_cfg():
    """Build the Franka cable camera env configured for the culling probe."""
    from isaaclab_tasks.core.lift.config.franka_soft.franka_cable_env_cfg import FrankaCableCameraEnvCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    env_cfg = FrankaCableCameraEnvCfg()
    env_cfg = resolve_presets(env_cfg, {"newton_mjwarp_vbd_proxy", "isaacsim_rtx"})

    env_cfg.scene.num_envs = 1
    # ``rgb`` keeps the env's own base_image observation term satisfied; the assertions read the
    # segmentation AOV. Both belong to the static capture group, so they share one render product.
    env_cfg.scene.base_camera.data_types = ["rgb", "instance_id_segmentation_fast"]
    # int32 ids keyed by id in ``idToLabels``; the colorized form keys by RGBA tuple instead.
    env_cfg.scene.base_camera.renderer_cfg.colorize_instance_id_segmentation = False
    # Exercise the partitioned path regardless of the ambient env-var default.
    env_cfg.scene.base_camera.renderer_cfg.enable_scene_partitioning = True
    env_cfg.scene.base_camera.spawn.clipping_range = _PROBE_CLIPPING_RANGE
    env_cfg.scene.base_camera.width = _PROBE_RESOLUTION
    env_cfg.scene.base_camera.height = _PROBE_RESOLUTION

    # Spawn the cable deterministically, and stop the MDP resetting it once the probe pose puts it
    # outside the task's workspace bounds.
    env_cfg.events.reset_cable.params["position_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    env_cfg.terminations.cable_out_of_bounds = None
    env_cfg.terminations.ee_below_table = None
    env_cfg.terminations.joint_vel_out_of_limit = None
    # Training ramps gravity from ~0 to -9.81; without this the cable floats instead of draping.
    if env_cfg.curriculum is not None:
        env_cfg.curriculum.gravity = None

    return env_cfg


def test_cable_visible_beyond_partition_bounds():
    """The cable still renders after deforming past the bounds an unpinned partition would have."""
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab.utils.renderers import isaac_rtx_per_env_scene_partition_enabled

    if not isaac_rtx_per_env_scene_partition_enabled():
        # The env drops the markers when partitioning is off, so there is no workaround to guard.
        pytest.skip("scene partitioning disabled by ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION")

    env_cfg = _make_env_cfg()

    env = None
    try:
        env = ManagerBasedRLEnv(env_cfg)
        camera = env.scene.sensors["base_camera"]
        cable = env.scene["cable"]
        env_ids = torch.arange(env.num_envs, device=env.device)
        zero_actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

        # Settle onto the table, then confirm the cable renders at all before moving it.
        for _ in range(2):
            env.step(zero_actions)
        settled_pixels = _cable_pixel_count(camera)
        assert settled_pixels > 0, "Cable is not visible at its spawn pose; the probe cannot detect culling."

        default_pose = cable.data.default_segment_pose_w.torch
        initial_span_y = float(default_pose[..., 1].amax() - default_pose[..., 1].amin())
        assert _PROBE_OFFSET_Y > initial_span_y, (
            f"Probe offset {_PROBE_OFFSET_Y} m no longer leaves the cable's initial y extent of"
            f" {initial_span_y} m, so it cannot trip the stale-bounding-box cull."
        )

        moved_pose = default_pose.clone()
        moved_pose[..., 1] += _PROBE_OFFSET_Y
        cable.write_segment_pose_to_sim_index(segment_pose=moved_pose, env_ids=env_ids)
        cable.write_segment_velocity_to_sim_index(
            segment_velocity=cable.data.default_segment_velocity_w.torch.clone(), env_ids=env_ids
        )
        env.step(zero_actions)

        moved_pixels = _cable_pixel_count(camera)
        assert moved_pixels > 0, (
            f"Cable vanished after moving {_PROBE_OFFSET_Y} m in y ({settled_pixels} px at spawn,"
            f" {moved_pixels} px after the move). The scene partition is culling the deforming curve"
            " - check that the partition-bounds markers are still spawned and still envelop the probe"
            " pose (OMPE-105749)."
        )
    finally:
        if env is not None:
            env.close()

            # This invokes camera sensor and renderer cleanup explicitly before pytest teardown, otherwise OV
            # native code could probably complain about leaks and trigger segmentation fault.
            env = None
