# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime calibration must change pixels without changing USD, across renderer processes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


@pytest.mark.parametrize("backend", ["kit", "ovrtx", "ovstage", "newton"])
def test_runtime_intrinsics_reach_rendering(backend):
    # Kit and standalone OVRTX need separate processes; the CI runner owns the cold-shader timeout.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), backend],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_contract(backend):
    app = None
    if backend == "kit":
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, enable_cameras=True).app
    else:
        os.environ["ISAAC_LAB_OVRTX_USE_OVSTAGE"] = "1" if backend == "ovstage" else "0"

    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60, gravity=(0.0, 0.0, 0.0), device="cuda:0")
    if backend == "kit":
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        renderer_cfg = IsaacRtxRendererCfg()
    else:
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

        sim_cfg.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1)
        if backend == "newton":
            from isaaclab_newton.renderers import NewtonWarpRendererCfg

            renderer_cfg = NewtonWarpRendererCfg()
        else:
            from isaaclab_ov.renderers import OVRTXRendererCfg

            renderer_cfg = OVRTXRendererCfg()

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Target",
            spawn=sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                mass_props=sim_utils.MassCfg(mass=1.0),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            ),
        )
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            width=160,
            height=120,
            update_period=0,
            update_latest_camera_pose=True,
            data_types=["distance_to_image_plane"],
            renderer_cfg=renderer_cfg,
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 5.0), convention="opengl"),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                horizontal_aperture=24.0,
                vertical_aperture=18.0,
                clipping_range=(0.1, 100.0),
            ),
        )
        second_camera = camera.replace(prim_path="{ENV_REGEX_NS}/CameraB") if backend in ("ovrtx", "ovstage") else None

    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(SceneCfg(num_envs=2, env_spacing=20.0))
        camera = scene["camera"]
        sim.reset()

        def capture():
            for _ in range(4):
                sim.step()
                camera.update(sim_cfg.dt, force_recompute=True)
                depth = camera.data.output["distance_to_image_plane"].torch[..., 0]
            mask = torch.isfinite(depth) & (depth > 4.0) & (depth < 6.0)
            return mask.any(dim=1).sum(dim=1).cpu().numpy()

        before = capture()
        assert (before > 20).all(), before
        original = camera.data.intrinsic_matrices.torch.clone()
        public_pointer = camera.data.intrinsic_matrices.warp.ptr
        # Record only authored camera properties: physics may independently update other scene state.
        usd_attributes = [
            [attribute.Get() for attribute in prim.GetPrim().GetAttributes()] for prim in camera._sensor_prims
        ]
        if backend == "newton":
            ray_pointer = camera._render_data.camera_rays.ptr

        wider = original.clone()
        wider[:, 0, 0] *= 0.5
        wider[:, 1, 1] *= 0.5
        for focal_length in (None, 12.0):
            camera.set_intrinsic_matrices(wider, focal_length=focal_length)
            assert [
                [attribute.Get() for attribute in prim.GetPrim().GetAttributes()] for prim in camera._sensor_prims
            ] == usd_attributes, "Runtime setter authored USD"
            after = capture()
            np.testing.assert_allclose(after, before * 0.5, atol=2.0, rtol=0.0)
            torch.testing.assert_close(camera.data.intrinsic_matrices.torch, wider)
            assert camera.data.intrinsic_matrices.warp.ptr == public_pointer
            indices = torch.tensor([1], device=camera.device)
            if backend == "newton":
                with pytest.raises(ValueError, match="identical camera intrinsics"):
                    camera.set_intrinsic_matrices(original[1:2], env_ids=indices)
                torch.testing.assert_close(camera.data.intrinsic_matrices.torch, wider)
                assert camera._render_data.camera_rays.ptr == ray_pointer
                np.testing.assert_array_equal(capture(), after)
            else:
                camera.set_intrinsic_matrices(original[1:2], env_ids=indices)
                capture()
                expected = wider.clone()
                expected[1] = original[1]
                torch.testing.assert_close(camera.data.intrinsic_matrices.torch, expected)
                # RTX's tiled render product currently shares its first camera's projection, even
                # with USD authoring. Check per-view device writes separately from uniform pixels.
                renderer = camera._renderer
                cameras = [(camera, "Camera")]
                if backend in ("ovrtx", "ovstage"):
                    second = scene["second_camera"]
                    assert second._renderer is renderer
                    second.set_intrinsic_matrices(original)
                    np.testing.assert_array_equal(capture(), after)
                    cameras.append((second, "CameraB"))
                for checked, camera_name in cameras:
                    paths = [f"/World/envs/env_{i}/{camera_name}" for i in range(2)]
                    for row, name in enumerate(("focalLength", "horizontalAperture", "verticalAperture")):
                        if backend == "kit":
                            fabric = sim_utils.get_current_stage(fabric=True)
                            actual = [fabric.GetPrimAtPath(path).GetAttribute(name).Get() for path in paths]
                        elif backend == "ovrtx":
                            actual = np.from_dlpack(renderer._renderer.read_attribute(name, paths))
                        else:
                            import ovstage

                            actual = []
                            token = renderer._stage_paths.intern_token(name)
                            for path in paths:
                                path_list = renderer._stage_paths.create_path_list_from_strings([path])
                                with renderer._stage.query_from_path_list(path_list) as query:
                                    with renderer._stage.read_attributes(
                                        query,
                                        [token],
                                        ovstage.OrdinalRange.latest(renderer._current_ordinal - 1),
                                    ) as read:
                                        for group in read.groups():
                                            try:
                                                actual.append(torch.from_dlpack(group.dlpack(0)).cpu().item())
                                            finally:
                                                renderer._stage.release_group(group)
                                renderer._stage_paths.destroy_path_list(path_list)
                        np.testing.assert_allclose(actual, checked._intrinsic_parameters.numpy()[row])
            camera.set_intrinsic_matrices(original)
            np.testing.assert_array_equal(capture(), before)
        if backend == "kit":
            camera.set_intrinsic_matrices(wider)
            np.testing.assert_array_equal(capture(), after)
            fabric = camera._render_data.intrinsic_stage
            row_attribute = camera._render_data.intrinsic_row_attribute
            paths = camera._render_data.spec.camera_prim_paths
            sim.stop()
            assert all(not fabric.GetPrimAtPath(path).GetAttribute(row_attribute).IsValid() for path in paths)
            sim.reset()
            torch.testing.assert_close(camera.data.intrinsic_matrices.torch, original)
            np.testing.assert_array_equal(capture(), before)
        elif backend in ("ovrtx", "ovstage"):
            assert not hasattr(camera._renderer, "_camera_intrinsic_bindings")
        assert [
            [attribute.Get() for attribute in prim.GetPrim().GetAttributes()] for prim in camera._sensor_prims
        ] == usd_attributes
        print(f"{backend}: projected widths {before.tolist()} -> {after.tolist()}; USD unchanged", flush=True)
    if app is not None:
        app.close()


if __name__ == "__main__":
    _run_contract(sys.argv[1])
