# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Persistent Gaussian geometry, full SH3, and material-frame bruise shading."""

import math

import numpy as np
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton.viewer import ViewerRTX

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from .. import profiling
from ..scene.background import add_ebc_background
from ..scene.tableware import add_tableware_visuals
from .gaussian_stream import BerryGaussianStream
from .settings import apply_sampling_settings, require_live_gaussian_renderer


class BerryViewer(ViewerRTX):
    """Render prepared Gaussians with optional interior hiding for appearance checks."""

    def __init__(
        self,
        env,
        pipeline=True,
        partitions=1,
        antialiasing="default",
        rtpt_spp=None,
        hide_interior=False,
        sh_rotation=True,
        view="auto",
        **kwargs,
    ):
        require_live_gaussian_renderer()
        self.env = env
        self.berry = env.berry
        self.aperture = 0.08
        self.pipeline = pipeline
        self.sh_rotation = sh_rotation
        self.antialiasing = antialiasing
        self.rtpt_spp = rtpt_spp
        self.sampling_overrides = {}
        self.pending_frame = None
        self.follow_berry = True
        self.background_path = None
        self.reset_requested = False
        self.last_center = self.berry.rest.mean(0).copy()
        self.streams = [
            BerryGaussianStream(
                berry,
                f"/World/Berries/{name}" if len(env.berries) > 1 else "/World/Berry",
                partitions,
                hide_interior,
            )
            for name, berry in env.berries.items()
        ]
        super().__init__(**kwargs, environment="studio", fps=30, async_rendering=False)
        self.set_model(NewtonManager.get_model())
        if view == "auto":
            view = "workcell" if env.cfg.background == "ebc" or len(env.berries) > 1 else "berry"
        self.set_view(view)
        self.register_ui_callback(self.controls, position="side")

    def set_view(self, view: str) -> None:
        """Choose a fixed workcell overview or the close, following berry view."""
        if view == "scene":
            eye = np.array([1.5, -2.0, 1.1])
            target = np.array([0.25, 0.05, -0.18])
        elif view == "workcell":
            eye = np.array([0.75, -0.24, 0.36])
            target = np.array([0.46, 0.07, 0.035])
        elif view == "berry":
            center = self.berry.positions().mean(0)
            if self.env.cfg.background == "studio":
                center = center - self.berry.rest.mean(0) + np.array([0, 0, 0.018])
            target = self.berry.offset + center
            eye = target + np.array([0.08, -0.025, 0.027])
        else:
            raise ValueError(f"Unknown camera view: {view}")
        self.follow_berry = view == "berry"
        delta = target - eye
        self.set_camera(
            wp.vec3(*eye),
            math.degrees(math.asin(delta[2] / np.linalg.norm(delta))),
            math.degrees(math.atan2(delta[1], delta[0])),
        )
        self.camera.near = 0.0001
        self.camera.pivot = type(self.camera.pos)(*target)

    def controls(self, ui):
        count = len(self.env.berries)
        if self.env.cfg.berry == "all":
            title = "All four berries"
        else:
            title = f"{count} {self.env.cfg.berry}s" if count > 1 else self.env.cfg.berry.title()
        ui.text(f"{title} | continuous grasp")
        ui.text("LB enable | RT close | LT open | release to hold")
        ui.text("Keyboard: WASDQE / ZX TG CV; K close, J open; R reset")
        aperture = float(self.env.action_manager.get_term("gripper_action").processed_actions[0].sum())
        ui.text(f"Commanded aperture: {aperture * 1000:.1f} mm")
        _, self.follow_berry = ui.checkbox("Follow berry", self.follow_berry)
        _, self.sh_rotation = ui.checkbox("Rotate SH with material", self.sh_rotation)
        if ui.button("Room / robot view"):
            self.set_view("scene")
        if ui.button("Plate and bowl view"):
            self.set_view("workcell")
        if ui.button("Berry close-up"):
            self.set_view("berry")
        if len(self.env.berries) > 1:
            for name, berry in self.env.berries.items():
                if ui.button(f"View {name}"):
                    self.berry = berry
                    self.last_center = berry.positions().mean(0)
                    self.set_view("berry")
        if ui.button("Reset robot and berries"):
            self.reset_requested = True

    def _init_ovrtx(self):
        for stream in self.streams:
            stream.author(self.stage)
        if self.env.cfg.background == "ebc":
            add_tableware_visuals(self.stage)
            self.background_path = add_ebc_background(
                self.stage, UsdShade.Shader.Get(self.stage, f"{self.streams[0].root_path}/Materials/Radiance/Shader")
            )
        super()._init_ovrtx()
        for stream in self.streams:
            stream.bind(self._rtx)

    def _add_studio_lights(self):
        super()._add_studio_lights()
        if self.env.cfg.background == "ebc":
            # A soft overhead task light keeps the transparent receiving bowl legible.
            key = UsdLux.DistantLight.Get(self.stage, "/root/_RTXDistantLight/_RTXDistantLight")
            key.GetIntensityAttr().Set(1500)
            key.GetAngleAttr().Set(8)
            light = UsdLux.RectLight.Define(self.stage, "/World/TablewareLight")
            light.CreateWidthAttr(0.6)
            light.CreateHeightAttr(0.5)
            light.CreateIntensityAttr(1500)
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(0.48, 0.12, 0.65))

    def _add_camera_lights_and_render_product(self):
        super()._add_camera_lights_and_render_product()
        for prim in self.stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                UsdGeom.Camera(prim).CreateClippingRangeAttr(Gf.Vec2f(0.0001, 100))
        product = self.stage.GetPrimAtPath(self._render_product_path)
        product.CreateAttribute("omni:rtx:dlss:frameGeneration", Sdf.ValueTypeNames.Bool).Set(False)
        product.RemoveProperty("omni:rtx:quality")
        product.RemoveProperty("omni:rtx:waitForEvents")
        self.sampling_overrides = apply_sampling_settings(product, self.antialiasing, self.rtpt_spp)
        if self.env.cfg.background == "ebc":
            # Rays through the bowl can cross both walls and its solid glass base.
            product.CreateAttribute("omni:rtx:rtpt:maxBounces", Sdf.ValueTypeNames.Int).Set(8)
            product.CreateAttribute("omni:rtx:rtpt:maxSpecularAndTransmissionBounces", Sdf.ValueTypeNames.Int).Set(8)

    def _render_and_display(self):
        with profiling.zone("gaussians: publish"):
            for stream, prepared in zip(self.streams, self.prepared):
                stream.update(prepared)
        from ovrtx import Device

        if not self.pipeline:
            with profiling.zone("ovrtx: step"):
                self._render_products = self._rtx.step(render_products={self._render_product_path}, delta_time=1 / 30)
        if self._window is not None and self._window.context is not None:
            for product in (self._render_products or {}).values():
                for frame in product.frames:
                    for var in frame.render_vars.values():
                        if var.source_name == "LdrColor":
                            with var.map(device=Device.CUDA) as mapping:
                                pixels = wp.from_dlpack(mapping, dtype=wp.vec4ub)
                                self._blit_to_window(pixels)
                                mapping.unmap(stream=pixels.device.stream.cuda_stream)
        if self.pipeline:
            with profiling.zone("ovrtx: step_async"):
                self.pending_frame = self._rtx.step_async(
                    render_products={self._render_product_path}, delta_time=1 / 30
                )

    def capture_image(self):
        # OVRTX 0.6 keys render vars by prim path, not by source name.
        from ovrtx import Device

        self.finish_frame()
        for product in (self._render_products or {}).values():
            for frame in product.frames:
                for var in frame.render_vars.values():
                    if var.source_name == "LdrColor":
                        with var.map(device=Device.CPU) as mapping:
                            return np.from_dlpack(mapping).copy()
        raise RuntimeError(
            f"No color output: {[(str(k), str(v), str(v.frames)) for k, v in self._render_products.items()]}"
        )

    def save_screenshot(self, path):
        from PIL import Image

        Image.fromarray(self.capture_image()).save(path)

    def draw(self, time_s):
        with profiling.zone("gaussians: deform and shade"):
            self.prepared = [stream.prepare(self.sh_rotation) for stream in self.streams]
        center = self.berry.positions().mean(0)
        if self.follow_berry:
            shift = center - self.last_center
            self.set_camera(wp.vec3(*(np.asarray(self.camera.pos) + shift)), self.camera.pitch, self.camera.yaw)
            self.camera.pivot += type(self.camera.pos)(*shift)
        self.last_center = center
        # Complete the previous frame before publishing any changed scene arrays.
        # The robot/MPM step overlapped that render using independent state buffers.
        self.finish_frame()
        self.begin_frame(time_s)
        with profiling.zone("viewer: log state"):
            self.log_state(NewtonManager.get_state_0())
        with profiling.zone("viewer: end frame"):
            self.end_frame()
        if getattr(self, "gui", None) is not None:
            self.gui.update_camera_from_keys = lambda *args: None

    def finish_frame(self):
        if self.pending_frame is not None:
            with profiling.zone("ovrtx: wait for frame"):
                self._render_products = self.pending_frame.wait().fetch()
            self.pending_frame = None

    def verify_geometry(self):
        """Verify every berry’s native Gaussian arrays, including static SH and opacity."""
        self.finish_frame()
        return all(stream.verify(prepared) for stream, prepared in zip(self.streams, self.prepared))

    def close(self):
        self.finish_frame()
        renderer = self._rtx
        for stream in self.streams:
            stream.close()
        super().close()
        if renderer is not None:
            renderer.destroy()
