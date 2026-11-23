# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendered regression: tracking markers belong in operator views, not sensor images."""

from types import SimpleNamespace

import numpy as np
import ovstage
import pytest
import torch
import warp as wp

pytest.importorskip("ovrtx")
pytest.importorskip("ovphysx")
pytest.importorskip("isaaccapture.ovxr_session_manager")

from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_teleop.ov_libraries_xr import _IsaacLabOvxrApplicationInterface
from isaaclab_teleop.ovxr_cfg import OvLibrariesXrCameraFeedCfg, OvLibrariesXrCfg
from isaaclab_teleop.ovxr_renderer import OvxrRendererCfg
from isaaclab_teleop.ovxr_session import OvxrSessionBootstrap
from ovrtx import Device

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import InclusionSet
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.test.utils import resolve_test_sim_device

pytestmark = [pytest.mark.ov, pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires RTX rendering")]


def test_tracking_markers_render_in_eyes_but_not_sensor_after_motion_and_reset(monkeypatch):
    """Read native depth so identical camera views distinguish a visible marker from the task object."""
    device = resolve_test_sim_device()
    renderer_cfg = OvxrRendererCfg()
    camera_cfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Sensor",
        width=96,
        height=96,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24, horizontal_aperture=20.955, clipping_range=(0.1, 30)),
        offset=CameraCfg.OffsetCfg(pos=(0, 0, 5), rot=(0, 0, 0, 1), convention="opengl"),
        renderer_cfg=renderer_cfg,
        update_latest_camera_pose=True,
    )
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=4)
    scene_cfg.body = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Body", spawn=sim_utils.CuboidCfg(size=(1, 1, 1)))
    scene_cfg.light = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Light", spawn=sim_utils.DomeLightCfg(intensity=500))
    scene_cfg.clone_cfg.clone_combinations = [InclusionSet(assets=["light", "light"])]
    scene_cfg.sensor = camera_cfg
    scene_cfg.left = camera_cfg.replace(prim_path="/Eyes/Left")
    scene_cfg.right = camera_cfg.replace(prim_path="/Eyes/Right")
    cameras = (scene_cfg.sensor, scene_cfg.left, scene_cfg.right)
    sim_cfg = sim_utils.SimulationCfg(device=device, physics=OvPhysxCfg())
    bootstrap = OvxrSessionBootstrap(sim_cfg, renderer_cfg)
    sim = bootstrap.simulation_context
    host = scene = markers = None
    try:
        for cfg in cameras:
            bootstrap.configure_auxiliary_camera(cfg)
        bootstrap.create_native_pair()
        scene = InteractiveScene(scene_cfg)
        sim.register_interactive_scene(scene)
        sim.reset()
        scene.reset()
        env = SimpleNamespace(sim=sim, scene=scene, cfg=SimpleNamespace(sim=sim_cfg), num_envs=1, device=device)
        env.unwrapped = env
        cfg = OvLibrariesXrCfg(
            auto_launch_cloudxr=False,
            camera_paths=("/Eyes/Left", "/Eyes/Right"),
            camera_feeds=tuple(
                OvLibrariesXrCameraFeedCfg(name=name, camera_cfg=camera)
                for name, camera in zip(("sensor", "left", "right"), cameras, strict=True)
            ),
        )
        host = _IsaacLabOvxrApplicationInterface(bootstrap, env, cfg)
        binding = host.attach_on_render_thread(host.offer_render_owner_transfer())
        products = bootstrap._lease.auxiliary_render_product_paths
        # The second light exists only in the native replicated scene, not the source USD.
        light_paths = ("/World/envs/env_0/Light", "/World/envs/env_0/Light_1")
        assert not scene.stage.GetPrimAtPath(light_paths[1])
        native_scene = sim.get_or_create_backend(renderer_cfg).scene
        markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Tracking", markers={"joint": sim_utils.CuboidCfg(size=(0.4, 0.4, 0.4))}
            )
        )
        for frame_index, marker_z in enumerate((1.0, 1.5, 0.8)):
            if frame_index == 2:
                sim.reset()
                scene.reset()
            markers.visualize(translations=torch.tensor([[1.3, 0.0, marker_z]], device=device))
            sim.step(render=False)
            scene.update(sim_cfg.dt)
            with host.publication_guard():
                ordinal = host.publish_complete_scene(frame_index)
                with host.owner_lane():
                    # Both authored and copied lights must illuminate every camera, including after reset.
                    partitions = {}
                    with native_scene.query(light_paths) as query:
                        query.wait()
                        with native_scene.stage.read_attributes(
                            query,
                            [native_scene.paths.intern_token("omni:scenePartition")],
                            ovstage.OrdinalRange.latest(ordinal),
                        ) as read:
                            read.wait()
                            for group in read.groups():
                                try:
                                    column = group.columns[0]
                                    values = np.asarray(column.array(0).copy()).reshape(column.data_count, -1)
                                    names = native_scene.paths.get_path_strings(group.prim_list)
                                    for index in range(min(group.prim_count, column.data_count)):
                                        partitions[names[group.prim_index(index)]] = int(
                                            values[column.data_row_index(index), 0]
                                        )
                                finally:
                                    native_scene.stage.release_group(group)
                    assert partitions == dict.fromkeys(light_paths, 0), "Environment lights are not common to all views"
                    outputs = binding.renderer.step(set(products), delta_time=sim_cfg.dt, ordinal=ordinal)
                    depths = []
                    for product in products:
                        frame = max(outputs[product].frames, key=lambda item: item.end_time)
                        depth_var = next(var for name, var in frame.render_vars.items() if name.endswith("/depth"))
                        with depth_var.map(device=Device.CUDA) as mapping:
                            mapping.wait()
                            array = wp.from_dlpack(mapping)
                            depth = array.numpy().copy().reshape(96, 96)
                            del array
                        del mapping
                        if depth.dtype == np.uint32:
                            depth = depth.view(np.float32)
                        depths.append(depth)
                    del frame, depth_var, outputs
            for index, (product, depth) in enumerate(zip(products, depths, strict=True)):
                assert depth[48, 48] == pytest.approx(4.5, abs=0.03), "Task object disappeared"
                visible = bool(np.any(np.isclose(depth, 5.0 - marker_z - 0.2, atol=0.03)))
                assert visible is (index != 0), f"Marker visibility is wrong for {product}"
    finally:
        markers = None
        if host is not None:
            host.close_publication()
            host.close_on_render_thread()
        else:
            bootstrap.abort()
        if scene is not None:
            for name in ("sensor", "left", "right"):
                scene[name]._invalidate_initialize_callback(None)
        sim.register_interactive_scene(None)
        sim.clear_instance()
