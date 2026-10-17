# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendered contract for independent light resets in cloned environments."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.renderers import RendererCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


def run_light_isolation_contract(
    sim: SimulationContext,
    renderer_cfg: RendererCfg,
    *,
    per_environment_domes: bool = True,
    env_spacing: float = 0.0,
) -> None:
    """Reset one dome and one analytic light, checking every untouched camera frame."""

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        dome = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Dome" if per_environment_domes else "/World/Dome",
            spawn=sim_utils.DomeLightCfg(intensity=300.0, color=(0.8, 0.8, 0.8)),
        )
        ceiling = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Ceiling",
            spawn=sim_utils.DiskLightCfg(intensity=1000.0, radius=0.5),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 1.3), rot=(0.0, 0.3826834324, 0.0, 0.9238795325)),
        )
        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.6, 0.6, 0.6),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                mass_props=sim_utils.MassCfg(mass=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
        )
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            height=96,
            width=96,
            data_types=["rgb", "depth"],
            renderer_cfg=renderer_cfg,
            offset=CameraCfg.OffsetCfg(
                pos=(2.0, 0.0, 1.0),
                rot=(0.4091605966, 0.4091605966, 0.5767040889, 0.5767040889),
                convention="opengl",
            ),
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 100.0)),
        )

    scene = InteractiveScene(SceneCfg(num_envs=3, env_spacing=env_spacing, lazy_sensor_update=False))
    sim.register_interactive_scene(scene)
    sim.reset()
    scene.reset()
    camera = scene["camera"]
    for _ in range(3):
        sim.step()
        scene.update(sim.cfg.dt)

    def capture() -> np.ndarray:
        frames = []
        for _ in range(48):
            sim.render()
            camera.update(sim.cfg.dt, force_recompute=True)
            frames.append(camera.data.output["rgb"].torch[..., :3].cpu().numpy().copy())
        return np.asarray(frames, dtype=np.float32)

    baseline = capture()[-12:].mean(axis=0)
    ordinary = capture()
    before = ordinary[-12:].mean(axis=0)
    noise = np.abs(ordinary - baseline).mean(axis=(2, 3, 4)).max(axis=0)
    if per_environment_domes:
        sim.render_context.write_light_properties(
            ["/World/envs/env_0/Dome"], {"color": np.asarray([[0.9, 0.02, 0.02]], dtype=np.float32)}
        )
        changed = capture()
        delta = np.abs(changed - before).mean(axis=(2, 3, 4))
        assert delta[-12:, 0].mean() > 15.0, f"Selected dome update was not rendered: {delta[-12:].mean(axis=0)}"
        assert np.all(delta[:, 1:] <= np.maximum(3.0, 4.0 * noise[1:])), (
            f"Dome edit changed untouched views: peak={delta.max(axis=0)}, noise={noise}"
        )
    else:
        changed = ordinary

    # Inspect the object rather than diluting its illumination change with the dome background.
    before = changed[-12:, :, 32:64, 32:64].mean(axis=0)
    noise = np.abs(changed[:, :, 32:64, 32:64] - before).mean(axis=(2, 3, 4)).max(axis=0)
    sim.render_context.write_light_properties(["/World/envs/env_1/Ceiling"], {"intensity": np.zeros(1)})
    changed = capture()[:, :, 32:64, 32:64]
    delta = np.abs(changed - before).mean(axis=(2, 3, 4))
    assert delta[-12:, 1].mean() > 3.0, f"Selected analytic light update was not rendered: {delta[-12:].mean(axis=0)}"
    assert np.all(delta[:, [0, 2]] <= np.maximum(3.0, 4.0 * noise[[0, 2]])), (
        f"Analytic light edit changed untouched views: peak={delta.max(axis=0)}, "
        f"settled={delta[-12:].mean(axis=0)}, noise={noise}"
    )

    # Moving the selected light away must preserve the off appearance, despite restoring its intensity.
    off = changed[-12:].mean(axis=0)
    transform = np.eye(4, dtype=np.float64)[None]
    c = np.sqrt(0.5)
    transform[0, :3, :3] = ((c, 0, -c), (0, 1, 0), (c, 0, c))
    origin = scene.env_origins[1].cpu().numpy()
    transform[0, 3, :3] = origin + (100.0, 0.0, 1.3)
    sim.render_context.write_light_properties(
        ["/World/envs/env_1/Ceiling"], {"intensity": np.full(1, 1000.0), "transform": transform}
    )
    moved = capture()[-12:, :, 32:64, 32:64].mean(axis=0)
    assert np.all(np.abs(moved - off).mean(axis=(1, 2, 3)) < 3.0), "Light world transform was not rendered."
    transform[0, 3, :3] = origin + (1.0, 0.0, 1.3)
    sim.render_context.write_light_properties(["/World/envs/env_1/Ceiling"], {"transform": transform})
    restored = capture()[-12:, :, 32:64, 32:64].mean(axis=0)
    assert np.all(np.abs(restored - before).mean(axis=(1, 2, 3)) < 3.0), "Light did not return to its initial pose."

    # Identical coincident geometry could hide a partition leak until one object moves.
    before = capture()[-12:].mean(axis=0)
    depth = camera.data.output["depth"].torch[..., 0]
    visible_before = ((depth > 0.05) & (depth < 5.0)).sum(dim=(1, 2))
    assert torch.all(visible_before > 100), f"Expected visible cubes: {visible_before}"
    pose = torch.tensor([[5.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]], device=sim.device)
    pose[0, :3] += scene.env_origins[2]
    scene["cube"].write_root_pose_to_sim_index(root_pose=pose, env_ids=[2])
    for _ in range(3):
        sim.step()
        scene.update(sim.cfg.dt)
    changed = capture()[-12:].mean(axis=0)
    depth = camera.data.output["depth"].torch[..., 0]
    visible_after = ((depth > 0.05) & (depth < 5.0)).sum(dim=(1, 2))
    assert visible_after[2] == 0, f"Selected object motion was not rendered: {visible_after}"
    torch.testing.assert_close(visible_after[:2], visible_before[:2], rtol=0.0, atol=0)
    delta = np.abs(changed - before).mean(axis=(1, 2, 3))
    assert np.all(delta[:2] < 3.0), f"Object motion changed untouched views: {delta}"
