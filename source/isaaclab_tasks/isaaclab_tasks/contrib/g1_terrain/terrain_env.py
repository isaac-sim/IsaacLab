# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment setup for the G1 terrain traversals."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporter

from .terrain_env_cfg import G1PlatformsEnvCfg

if TYPE_CHECKING:
    from isaaclab_visualizers.kit import KitVisualizer


class ParticleTerrainImporter(TerrainImporter):
    """Load the particle USD and enable GPU simulation with host state buffers."""

    def import_usd(self, name: str, usd_path: str):
        from pxr import PhysxSchema  # noqa: PLC0415

        super().import_usd(name, usd_path)
        sim = SimulationContext.instance()
        sim.set_setting("/physics/cudaDevice", torch.cuda.current_device())
        sim.set_setting("/physics/suppressReadback", False)
        scene = PhysxSchema.PhysxSceneAPI(sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path))
        scene.CreateEnableGPUDynamicsAttr(True)
        scene.CreateBroadphaseTypeAttr("GPU")


class G1TerrainEnv(ManagerBasedRLEnv):
    """A fixed course with explicit resets and absolute joint position actions."""

    def __init__(self, cfg: G1PlatformsEnvCfg, **kwargs):
        """Use host buffers for PhysX particles only after the launcher selects the CUDA device."""
        if cfg.scene.num_envs != 1:
            raise ValueError("The reconstructed courses support one robot per scene.")
        is_physx = isinstance(cfg.sim.physics, PhysxCfg)
        if not is_physx and not isinstance(cfg.sim.physics, NewtonCfg):
            raise ValueError("Select physics=isaacsim_physx or physics=newton_mjwarp.")
        if cfg.terrain_name == "granular" and is_physx:
            cfg.sim.device = "cpu"
        super().__init__(cfg, **kwargs)
        if cfg.terrain_name == "granular" and is_physx:
            self.sim.set_setting("/physics/updateParticlesToUsd", True)

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

    def prepare_view(self, kit_visualizer: KitVisualizer | None = None) -> None:
        """Prepare scene appearance and the optional Kit spectator after initializing physics."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        from isaaclab.sim import bind_visual_material  # noqa: PLC0415

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
        for prim in Usd.PrimRange(stage.GetPrimAtPath(self.scene["robot"].cfg.spawn.spawn_path)):
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
        if kit_visualizer is not None:
            import omni.kit.app  # noqa: PLC0415

            omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
            import omni.replicator.core as rep  # noqa: PLC0415

            rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
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
                aperture = 20.955 * kit_visualizer.cfg.window.size[1] / kit_visualizer.cfg.window.size[0]
                camera.CreateHorizontalApertureAttr(20.955)
                camera.CreateVerticalApertureAttr(aperture)
                camera.CreateFocalLengthAttr(aperture / (2 * math.tan(math.radians(self.cfg.camera_vfov) / 2)))
        # Only Kit's RTX needs warm-up frames; on other backends queued renders would overlap the depth
        # camera's first CUDA graph capture below.
        if kit_visualizer is not None:
            for _ in range(60):
                self.sim.render()
                kit_visualizer.render_rgb_array()
        self.sim.forward()
        self.scene.update(0.0)
        if self.cfg.scene.depth_camera is not None:
            self.scene["depth_camera"].update(0.0, force_recompute=True)
            terrain = stage.GetPrimAtPath(f"{self.cfg.scene.terrain.prim_path}/terrain")
            terrain.GetVariantSet("appearance").SetVariantSelection("visual")
        if kit_visualizer is not None:
            for _ in range(30):
                self.sim.render()
                kit_visualizer.render_rgb_array()
