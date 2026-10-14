# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment setup for the G1 terrain traversals."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs import ManagerBasedRLEnv

from .terrain import create_terrain
from .terrain_env_cfg import G1PlatformsEnvCfg

if TYPE_CHECKING:
    from isaaclab_visualizers.kit import KitVisualizer


class G1TerrainEnv(ManagerBasedRLEnv):
    """A fixed course with explicit resets and absolute joint position actions."""

    def __init__(self, cfg: G1PlatformsEnvCfg, **kwargs):
        if cfg.scene.num_envs != 1:
            raise ValueError("The reconstructed courses support one robot per scene.")
        is_physx = isinstance(cfg.sim.physics, PhysxCfg)
        if not is_physx and not isinstance(cfg.sim.physics, NewtonCfg):
            raise ValueError("Select physics=isaacsim_physx or physics=newton_mjwarp.")
        cfg.sim.dt = 0.0025 if is_physx else 0.005
        cfg.decimation = 8 if is_physx else 4
        cfg.sim.render_interval = cfg.decimation
        if cfg.terrain_name == "granular":
            from .granular import configure_granular  # noqa: PLC0415

            configure_granular(cfg, is_physx)
        self._terrain_directory = TemporaryDirectory(prefix="g1-terrain-")
        cfg.scene.terrain.usd_path = str(
            create_terrain(Path(self._terrain_directory.name), cfg.terrain_name, cfg.course, is_physx)
        )
        super().__init__(cfg, **kwargs)
        if cfg.terrain_name == "granular" and is_physx:
            self.sim.set_setting("/physics/updateParticlesToUsd", True)

    def close(self):
        """Release the generated terrain after the simulator closes."""
        super().close()
        if hasattr(self, "_terrain_directory"):
            self._terrain_directory.cleanup()

    def reset(self, **kwargs):
        """Initialize solver and sensor buffers with a neutral airborne step before resetting."""
        result = super().reset(**kwargs)
        if self.cfg.warmup_height is None:
            return result
        robot = self.scene["robot"]
        pose = robot.data.root_pose_w.torch.clone()
        pose[:, 2] = self.cfg.warmup_height
        robot.write_root_pose_to_sim(pose)
        self.step(robot.data.default_joint_pos.torch.clone())
        return super().reset(**kwargs)

    def prepare_view(self, visualizer: KitVisualizer) -> None:
        """Configure the spectator camera after initializing physics and depth geometry."""
        import omni.kit.app  # noqa: PLC0415
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        from isaaclab.sim import bind_visual_material  # noqa: PLC0415

        from .terrain import separate_ramp_base  # noqa: PLC0415

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
        import omni.replicator.core as rep  # noqa: PLC0415

        rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
        stage = self.sim.stage
        for name, material_cfg in self.cfg.robot_visual_materials.items():
            material_cfg.func(f"/World/G1Looks/{name}", material_cfg)
        dark_parts = {
            "head_link",
            "pelvis",
            "left_hip_pitch_link",
            "right_hip_pitch_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        }
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot")):
            if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetName().startswith("imu_in_"):
                UsdGeom.Imageable(prim).MakeInvisible()
            if prim.GetName() == "visuals":
                prim.SetInstanceable(False)
                for part in prim.GetChildren():
                    if part.GetName() == "logo_link":
                        UsdGeom.Imageable(part).MakeInvisible()
                    else:
                        material = "/World/G1Looks/" + ("dark" if part.GetName() in dark_parts else "light")
                        bind_visual_material(part.GetPath(), material)
        # Kit's viewport camera opinions in the session layer override the scene layer.
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            camera_path = "/OmniverseKit_Persp"
            camera = UsdGeom.Camera.Define(stage, camera_path)
            transform = UsdGeom.Xformable(camera.GetPrim())
            transform.ClearXformOpOrder()
            transform.SetResetXformStack(True)
            view = Gf.Matrix4d(1.0).SetLookAt(
                Gf.Vec3d(*self.cfg.camera_eye), Gf.Vec3d(*self.cfg.camera_target), Gf.Vec3d(0.0, 0.0, 1.0)
            )
            transform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(view.GetInverse())
            aperture = 20.955 * visualizer.cfg.window.size[1] / visualizer.cfg.window.size[0]
            camera.CreateHorizontalApertureAttr(20.955)
            camera.CreateVerticalApertureAttr(aperture)
            camera.CreateFocalLengthAttr(aperture / (2 * math.tan(math.radians(self.cfg.camera_vfov) / 2)))
        for _ in range(60):
            self.sim.render()
            visualizer.render_rgb_array()
        self.sim.forward()
        self.scene.update(0.0)
        if self.cfg.scene.depth_camera is not None:
            self.scene["depth_camera"].update(0.0, force_recompute=True)
            separate_ramp_base(stage)
        for _ in range(30):
            self.sim.render()
            visualizer.render_rgb_array()
