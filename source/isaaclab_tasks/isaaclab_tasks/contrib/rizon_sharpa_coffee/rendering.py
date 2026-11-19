# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional native grain rendering and colour transport over the MPM liquid mixture."""

from __future__ import annotations

import math
from builtins import ExceptionGroup
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from isaaclab_newton.physics.mpm_solver import implicit_mpm_solvers
from isaaclab_visualizers.kit import KitVisualizer
from isaaclab_visualizers.newton import NewtonGLVisualizer, NewtonRTXVisualizer
from newton.geometry import ParticleSurface

import isaaclab.sim as sim_utils

from .physics import SolverCoffeeMPM

if TYPE_CHECKING:
    from newton.viewer import ViewerGL

    from pxr import Usd, UsdGeom, UsdShade

    from isaaclab.envs import ManagerBasedEnv

    from .demo_env_cfg import CoffeePresentationCfg

COFFEE_COLOR = (0.095, 0.032, 0.012)
WATER_COLOR = (0.66, 0.83, 0.88)
BREW_COLOR = (0.12, 0.041, 0.014)


class CoffeeGLVisualizer(NewtonGLVisualizer):
    """Publish task geometry while the native GL frame is active."""

    rendering: CoffeeRendering | None = None

    def _update_state(self):
        state = super()._update_state()
        if self.rendering is not None:
            self.rendering.draw(self._viewer)
        return state


class CoffeeRTXVisualizer(NewtonRTXVisualizer):
    """Publish task geometry into the camera's shared RTX scene."""

    rendering: CoffeeRendering | None = None

    def initialize(self, sim, *, cameras):
        if self._is_initialized:
            return
        super().initialize(sim, cameras=cameras)
        self._geometry_queries = {}
        for renderer in self._renderers:
            renderer.geometry_callbacks[self] = self._update_geometry

    def _update_geometry(self, renderer) -> None:
        if self.rendering is not None:
            import ovstage
            from isaaclab_ov.stage import OVSTAGE_POINT_DTYPE
            from ovrtx import DataAccess

            from pxr import Usd, UsdGeom

            self.rendering.update()
            queries = self._geometry_queries.setdefault(renderer, {})
            errors = renderer.drain_pending_renders()
            if errors:
                raise ExceptionGroup("RTX renders failed before coffee geometry update", errors)
            root = f"/Coffee_{id(self)}"
            attributes = {f"{root}/Grains": {"points": self.rendering._points}}
            if self.rendering._dissolution is not None or not queries:
                attributes[f"{root}/Grains"]["widths"] = self.rendering._widths
            if self.rendering._surface is not None:
                points, indices, normals, colors, _ = self.rendering._water_surface_arrays()
                attributes[f"{root}/Water"] = dict(
                    points=self.rendering._empty_points if points is None else points,
                    normals=self.rendering._empty_points if normals is None else normals,
                    faceVertexIndices=self.rendering._empty_indices if indices is None else indices,
                    faceVertexCounts=np.full(0 if indices is None else indices.size // 3, 3, np.int32),
                    **{"primvars:displayColor": np.empty((0, 3), np.float32) if colors is None else colors},
                )
            if not queries:
                stage = Usd.Stage.CreateInMemory()
                stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Coffee").GetPrim())
                grains = create_coffee_points(stage, "/Coffee/Grains")
                grains.GetPointsAttr().Set(self.rendering._points.numpy())
                grains.GetWidthsAttr().Set(self.rendering._widths.numpy())
                grains.GetDisplayColorPrimvar().Set(self.rendering._colors.numpy())
                if self.rendering._surface is not None:
                    create_water_mesh(stage, "/Coffee/Water")
                usd = stage.GetRootLayer().ExportToString()
                renderer.geometry_resources.callback(self._geometry_queries.pop, renderer, None)
                if renderer._use_ovstage:
                    reference = ovstage.population.add_usd_reference_from_string(renderer.scene.stage, usd, root)
                    renderer.geometry_resources.callback(renderer._remove_usd_reference, reference)
                    ovstage.population.apply_usd_changes(renderer.scene.stage, ordinal=renderer.scene.ordinal)
                    for path in attributes:
                        queries[path] = renderer.geometry_resources.enter_context(renderer.scene.query([path]))
                else:
                    reference = renderer.backend.renderer.add_usd_reference_from_string(usd, root)
                    renderer.geometry_resources.callback(renderer.backend.renderer.remove_usd, reference)
                    queries.update(dict.fromkeys(attributes))
            stream = wp.get_stream(self.rendering._model.device).cuda_stream or 1
            semantics = {"points": ovstage.AttributeSemantic.POINT, "normals": ovstage.AttributeSemantic.NORMAL}
            for path, values in attributes.items():
                for name, array in values.items():
                    if renderer._use_ovstage:
                        renderer.scene.stage.write_attribute(
                            queries[path], name, ordinal=renderer.scene.ordinal,
                            tensors=[ovstage.make_dltensor(array, dtype=OVSTAGE_POINT_DTYPE)
                                     if name in ("points", "normals", "primvars:displayColor") else array],
                            is_array=True, semantic=semantics.get(name, ovstage.AttributeSemantic.NONE),
                            cuda_stream=stream,
                        ).wait()  # fmt: skip
                    else:
                        renderer.backend.renderer.write_array_attribute(
                            [path], name, [array], data_access=DataAccess.ASYNC, cuda_stream=stream
                        )


class CoffeeKitVisualizer(KitVisualizer):
    """Publish Newton grains and liquid as USD geometry before Kit rendering."""

    rendering: CoffeeRendering | None = None
    _update_subscription = None

    def initialize(self, sim, *, cameras) -> None:
        if self._is_initialized:
            return
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

        import omni.kit.app
        from pxr import UsdGeom

        apply_isaac_rtx_global_settings(self.cfg.backend.render_settings)
        super().initialize(sim, cameras=cameras)
        self._stage = sim.stage
        self._root_path = "/World/CoffeePresentation/Kit"
        UsdGeom.Xform.Define(self._stage, self._root_path)
        self._coffee = create_coffee_points(self._stage, f"{self._root_path}/Coffee")
        self._water, self._water_shader = create_water_mesh(self._stage, f"{self._root_path}/Water")
        self._coffee_attributes_ready = False
        self._update_subscription = (
            omni.kit.app.get_app()
            .get_pre_update_event_stream()
            .create_subscription_to_pop(self._update_rendering, name="Coffee presentation")
        )

    def _update_rendering(self, event) -> None:
        if self.rendering is None:
            return
        self.rendering.update()
        self._update_coffee(self.rendering._points, self.rendering._radii, self.rendering._colors)
        if self.rendering._surface is not None:
            points, indices, normals, colors, fraction = self.rendering._water_surface_arrays()
            self._update_water(points, indices, normals, colors, 0.42 + 0.40 * fraction)

    def _update_coffee(self, points: wp.array, radii: wp.array, colors: wp.array) -> None:
        from pxr import Vt

        positions = points.numpy()
        dynamic_radii = self.rendering._dissolution is not None
        if dynamic_radii or not self._coffee_attributes_ready:
            widths = 2.0 * radii.numpy()
            display_colors = colors.numpy()
            if dynamic_radii:
                visible = widths > 0.0
                positions, widths, display_colors = positions[visible], widths[visible], display_colors[visible]
            self._coffee.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(widths))
            self._coffee.GetDisplayColorPrimvar().Set(Vt.Vec3fArray.FromNumpy(display_colors))
            self._coffee_attributes_ready = True
        self._coffee.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(positions))

    def _update_water(self, points, indices, normals, colors, opacity: float) -> None:
        from pxr import Vt

        if points is None or colors is None:
            self._water.MakeInvisible()
            return
        self._water.MakeVisible()
        self._water.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.numpy()))
        triangles = indices.numpy().reshape(-1)
        self._water.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles) // 3, 3, dtype=np.int32)))
        self._water.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(triangles))
        self._water.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.numpy()))
        self._water.GetDisplayColorPrimvar().Set(Vt.Vec3fArray.FromNumpy(colors.numpy()))
        self._water_shader.GetInput("opacity").Set(opacity)

    def close(self) -> None:
        if self._is_closed:
            return
        if self._update_subscription is not None:
            self._update_subscription.unsubscribe()
            self._update_subscription = None
        self.rendering = None
        initialized = self._is_initialized
        try:
            super().close()
        finally:
            if initialized:
                self._stage.RemovePrim(self._root_path)

    def _set_usd_camera_pose(self, camera_path: str, position, target) -> bool:
        """Publish this demo's camera above Kit's session-layer defaults."""
        from pxr import Usd

        with Usd.EditContext(self._sim.stage, self._sim.stage.GetSessionLayer()):
            return super()._set_usd_camera_pose(camera_path, position, target)


class _GLWaterSurface:
    """Publish the reconstructed water as a textured Newton mesh instance."""

    def __init__(self, device):
        self._uvs = None
        self._palette = np.asarray(((WATER_COLOR, BREW_COLOR),), dtype=np.float32)
        self._pose = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        self._white = wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3, device=device)
        # Newton instance materials encode roughness, metallic, checker, and texture weight.
        self._material = wp.array([(0.08, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=device)
        self._opacity = wp.empty(1, dtype=float, device="cpu")
        self._published = False

    def draw(self, viewer, points, indices, normals, colors, fraction):
        hidden = points is None or not indices.size
        if not hidden:
            if self._uvs is None or self._uvs.size < points.size:
                self._uvs = wp.empty(points.size, dtype=wp.vec2, device=points.device)
            uvs = self._uvs[: points.size]
            wp.launch(
                _water_palette_uvs,
                dim=points.size,
                inputs=[colors, wp.vec3(WATER_COLOR), wp.vec3(BREW_COLOR)],
                outputs=[uvs],
                device=points.device,
            )
            viewer.log_mesh(
                "/coffee_water_mesh",
                points,
                indices,
                normals=normals,
                uvs=uvs,
                texture=self._palette,
                hidden=True,
                backface_culling=False,
                dynamic=True,
            )
            self._published = True
        if self._published:
            self._opacity.fill_(0.42 + 0.40 * fraction)
            viewer.log_instances(
                "/coffee_water_surface",
                "/coffee_water_mesh",
                self._pose,
                scales=None,
                colors=self._white,
                materials=self._material,
                opacities=self._opacity,
                hidden=hidden,
            )


class CoffeeRendering:
    """Render one demonstration scene with native grains and a reconstructed liquid.

    Visualizers call :meth:`update` before publishing geometry. Call :meth:`reset`
    after particle resets and :meth:`close` before the environment closes. The physics
    solver owns grain advection, including intermediate steps and CUDA capture.
    Physics owns coffee-to-liquid conversion even without presentation. Converted
    coffee joins the reconstructed liquid and stops displaying as grains. Reset
    preloads the dry coffee's colour tracer. Conversion activates the carrier,
    allowing particle motion to advect the tracer and neighbouring carriers to mix it.

    Args:
        env: Demo environment with ``coffee`` and optionally ``water`` MPM assets.
        voxel_size: Physics grid spacing [m].
        color_mass_scale: Uniformly mixed tracer mass [kg] producing 63% of the final tint.
            This is an adjustable display normalization, not a physical calibration.
        diffusion_rate: Phenomenological colour mixing rate [1/s].
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        *,
        voxel_size: float = 0.0025,
        color_mass_scale: float = 0.0005,
        diffusion_rate: float = 18.0,
    ):
        if env.num_envs != 1:
            raise ValueError("CoffeeRendering supports one demo scene; disable presentation for batched RL.")
        if not math.isfinite(color_mass_scale) or color_mass_scale <= 0:
            raise ValueError("color_mass_scale must be finite and positive.")
        if not math.isfinite(diffusion_rate) or diffusion_rate < 0:
            raise ValueError("diffusion_rate must be finite and nonnegative.")
        self._backend_manager = env.sim.backend_manager
        self._model = self._backend_manager.backend.model
        self._env = env
        self._visualizers = [
            visualizer
            for visualizer in env.sim.visualizers
            if isinstance(visualizer, (CoffeeGLVisualizer, CoffeeRTXVisualizer, CoffeeKitVisualizer))
        ]
        solvers = [
            solver
            for solver in implicit_mpm_solvers(self._backend_manager.backend.solver)
            if isinstance(solver, SolverCoffeeMPM)
        ]
        if len(solvers) != 1 or solvers[0].grains is None:
            raise ValueError("CoffeeRendering requires one SolverCoffeeMPM with grains_per_particle > 0.")
        if not self._visualizers:
            raise ValueError("CoffeeRendering requires a coffee GL, RTX or Kit visualizer.")
        self._solver = solvers[0]
        self._dissolution = self._solver.dissolution
        device = self._model.device
        self._coffee_ids = env.particle_ids("coffee")
        self._factor = self._solver.grains.shape[1]
        count = self._coffee_ids.size * self._factor
        self._points = wp.empty(count, dtype=wp.vec3, device=device)
        self._radii = wp.empty(count, dtype=float, device=device)
        self._widths = wp.empty_like(self._radii)
        self._colors = wp.empty(count, dtype=wp.vec3, device=device)
        self._color_mass_scale = color_mass_scale
        self._liquid_ids = None
        self._liquid_flags = None
        self._liquid_points = None
        self._liquid_radii = None
        self._surface = None
        self._surface_cache = None
        self._gl_surfaces: dict[ViewerGL, _GLWaterSurface] = {}
        self._hash_grid = None
        self._mixing_radius = 2.0 * voxel_size
        self._diffusion_rate = diffusion_rate
        self._tint_elapsed = 0.0
        self._appearance_time = None
        self._vertex_colors = None
        self._empty_points = wp.empty(0, dtype=wp.vec3, device=device)
        self._empty_indices = wp.empty(0, dtype=wp.int32, device=device)
        if self._dissolution is not None:
            # Fixed capacity lets newly converted particles join without moving
            # buffers. Inactive dry coffee is excluded by the liquid mask.
            self._liquid_ids = self._dissolution.liquid_ids
            self._liquid_flags = self._dissolution.liquid_flags
            self._liquid_points = wp.empty(self._liquid_ids.size, dtype=wp.vec3, device=device)
            self._liquid_radii = wp.empty(self._liquid_ids.size, dtype=float, device=device)
            liquid_masses = self._model.particle_mass.numpy()[self._liquid_ids.numpy()]
            self._liquid_mass = wp.array(liquid_masses, dtype=float, device=device)
            self._concentration_scale = color_mass_scale / float(
                liquid_masses[: self._dissolution.water_ids.size].sum()
            )
            self._dye_mass = wp.empty(self._liquid_ids.size, dtype=float, device=device)
            self._next_dye_mass = wp.empty_like(self._dye_mass)
            self._diffusion_support = wp.empty_like(self._dye_mass)
            self._hash_grid = wp.HashGrid(64, 64, 64, device=device)
            self._hash_grid.reserve(self._liquid_ids.size)
            # Resolve surface detail independently of the physics grid while
            # keeping enough kernel overlap for a connected liquid surface.
            surface_voxel_size = min(0.3 * voxel_size, 0.001)
            self._surface = ParticleSurface(
                voxel_size=surface_voxel_size,
                # Reuse native storage alongside captured physics, scaling
                # capacity with the visual grid resolution.
                max_grid_cells=math.ceil(1048576 * max(1.0, (0.0012 / surface_voxel_size) ** 3)),
                kernel_radius=2.5 * voxel_size,
                threshold=0.4,
                smooth_lambda=0.7,
                anisotropic=True,
                kernel_scale=0.5,
                anisotropy_ratio=2.0,
                anisotropy_scale=1.0,
                anisotropy_min_neighbors=12,
                anisotropy_binning=False,
                anisotropy_strength=0.15,
                field_smooth_iterations=3,
                field_smooth_radius=3,
                mesh_smooth_iterations=4,
                mesh_smooth_lambda=0.5,
                device=device,
            )
        for visualizer in self._visualizers:
            visualizer.rendering = self
        self.reset()

    def update(self) -> None:
        """Prepare appearance once per simulation state before native rendering."""
        now = self._env.sim.backend_manager.get_simulation_time()
        if now == self._appearance_time:
            return
        dt = 0.0 if self._appearance_time is None else max(now - self._appearance_time, 0.0)
        self._appearance_time = now
        self._surface_cache = None
        state = self._backend_manager.backend.state_0
        if self._dissolution is not None:
            wp.copy(self._liquid_points, wp.indexedarray(state.particle_q, self._liquid_ids))
            # The coupled MPM view owns the converted particles' actual radii.
            wp.copy(self._liquid_radii, wp.indexedarray(self._solver.model.particle_radius, self._liquid_ids))
            self._tint_elapsed += dt
            # Optical transport at 25 Hz avoids repeating neighbour searches at
            # the 100 Hz control rate. Physical MPM motion keeps its own rate.
            if self._tint_elapsed >= 0.04 - 1.0e-8:
                tint_dt, self._tint_elapsed = self._tint_elapsed, 0.0
                self._hash_grid.build(self._liquid_points, self._mixing_radius)
                wp.launch(
                    _water_diffusion_support,
                    dim=self._liquid_ids.size,
                    inputs=[
                        self._liquid_points,
                        self._liquid_mass,
                        self._liquid_flags,
                        self._hash_grid.id,
                        self._mixing_radius,
                    ],
                    outputs=[self._diffusion_support],
                    device=self._model.device,
                )
                wp.launch(
                    _diffuse_water_color,
                    dim=self._liquid_ids.size,
                    inputs=[
                        self._liquid_points,
                        self._liquid_mass,
                        self._liquid_flags,
                        self._hash_grid.id,
                        self._mixing_radius,
                        -math.expm1(-self._diffusion_rate * tint_dt),
                        self._diffusion_support,
                        self._dye_mass,
                    ],
                    outputs=[self._next_dye_mass],
                    device=self._model.device,
                )
                self._dye_mass, self._next_dye_mass = self._next_dye_mass, self._dye_mass
        wp.launch(
            _gather_grains,
            dim=(self._coffee_ids.size, self._factor),
            inputs=[
                self._coffee_ids,
                self._solver.grains,
                self._solver.model.particle_radius,
                wp.vec3(COFFEE_COLOR),
                self._liquid_flags,
                0 if self._dissolution is None else self._dissolution.water_ids.size,
            ],
            outputs=[self._points, self._radii, self._widths, self._colors],
            device=self._model.device,
        )

    def draw(self, viewer: ViewerGL) -> None:
        """Publish grains and the liquid surface while a Newton GL frame is active."""
        self.update()
        viewer.log_points("/coffee_grains", self._points, radii=self._radii, colors=self._colors)
        if self._surface is not None:
            surface = self._gl_surfaces.get(viewer)
            if surface is None:
                surface = self._gl_surfaces[viewer] = _GLWaterSurface(self._model.device)
            surface.draw(viewer, *self._water_surface_arrays())

    def _water_surface_arrays(self):
        """Reconstruct once per physics state and interpolate tint on shared vertices."""
        if self._surface_cache is not None:
            return self._surface_cache
        # Native ACTIVE flags exclude dry coffee without a host-side compaction.
        self._surface.update_field(self._liquid_points, self._liquid_radii, particle_flags=self._liquid_flags)
        points, indices, normals = self._surface.resurface().to_arrays()
        fraction = 1.0 - math.exp(-self._dissolution.converted_mass_kg / self._color_mass_scale)
        hidden = points is None
        colors = None
        if not hidden and indices.size:
            if self._vertex_colors is None or self._vertex_colors.size < points.size:
                self._vertex_colors = wp.empty(points.size, dtype=wp.vec3, device=self._model.device)
            self._hash_grid.build(self._liquid_points, self._mixing_radius)
            wp.launch(
                _color_water_vertices,
                dim=points.size,
                inputs=[
                    points,
                    self._liquid_points,
                    self._liquid_mass,
                    self._liquid_flags,
                    self._dye_mass,
                    self._hash_grid.id,
                    1.5 * self._mixing_radius,
                    self._concentration_scale,
                    wp.vec3(WATER_COLOR),
                    wp.vec3(BREW_COLOR),
                ],
                outputs=[self._vertex_colors],
                device=self._model.device,
            )
            colors = self._vertex_colors[: points.size]
        self._surface_cache = points, indices, normals, colors, fraction
        return self._surface_cache

    def reset(self, env_ids=None) -> None:
        """Clear appearance after the single demo world is reset."""
        if env_ids is not None and not isinstance(env_ids, slice) and len(env_ids) == 0:
            return
        self._tint_elapsed = 0.0
        self._appearance_time = None
        self._surface_cache = None
        if self._dissolution is not None:
            self._dye_mass.zero_()
            # Dry carriers retain their tracer until physics activates them as liquid.
            water_count = self._dissolution.water_ids.size
            wp.copy(self._dye_mass[water_count:], self._liquid_mass[water_count:])
        self.update()

    def close(self) -> None:
        """Detach presentation from viewers before environment teardown."""
        for visualizer in self._visualizers:
            visualizer.rendering = None
        self._visualizers.clear()
        self._gl_surfaces.clear()
        self._surface_cache = None


def spawn_presentation(prim_path: str, cfg: CoffeePresentationCfg, translation=None, orientation=None):
    """Spawn the shared RTX backdrop before scene cameras initialize."""
    from pxr import UsdGeom

    from .assets import COFFEE_ASSET_DIR

    stage = sim_utils.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    for path in ("/World/DomeLight", "/World/KeyLight"):
        stage.GetPrimAtPath(path).GetAttribute("inputs:intensity").Set(0.0)
    _add_coffee_studio_lights(stage, f"{prim_path}/Lights")
    if cfg.backdrop == "kitchen":
        kitchen_path = f"{prim_path}/Kitchen"
        kitchen_cfg = sim_utils.UsdFileCfg(usd_path=f"{COFFEE_ASSET_DIR}/kitchen.usda")
        kitchen_cfg.func(kitchen_path, kitchen_cfg)
        for prim in stage.Traverse():
            if prim.GetName() == "StudioFloor":
                UsdGeom.Imageable(prim).MakeInvisible()
    return root


def _add_coffee_studio_lights(stage, parent: str) -> None:
    """Author the same broad tabletop emitters for the USD renderers."""
    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, f"{parent}/Ambient")
    dome.CreateIntensityAttr(180.0)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    target = Gf.Vec3d(0.45, 0.0, 0.18)
    softboxes = (
        ("Key", (1.12, -0.48, 1.05), (0.65, 0.90), 3600.0, (1.0, 0.94, 0.85)),
        ("Fill", (0.20, 0.65, 1.10), (0.80, 1.10), 1800.0, (0.90, 0.95, 1.0)),
        ("Rim", (-0.30, 0.30, 1.30), (0.35, 0.90), 700.0, (1.0, 0.97, 0.92)),
    )
    for name, position, size, intensity, color in softboxes:
        light = UsdLux.RectLight.Define(stage, f"{parent}/{name}")
        light.CreateWidthAttr(size[0])
        light.CreateHeightAttr(size[1])
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr(Gf.Vec3f(*color))
        transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), target, Gf.Vec3d(0.0, 0.0, 1.0))
        UsdGeom.Xformable(light).AddTransformOp().Set(transform.GetInverse())


def _vertex_color_material(stage, path, color, *, roughness: float, opacity: float = 1.0):
    """Use the native material spawner and connect the coffee colour primvar."""
    from pxr import Sdf, UsdShade

    cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness, metallic=0.0, opacity=opacity)
    with sim_utils.use_stage(stage):
        shader = UsdShade.Shader(cfg.func(path, cfg))
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.33)
    reader = UsdShade.Shader.Define(stage, f"{path}/Colour")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    shader.GetInput("diffuseColor").ConnectToSource(reader.ConnectableAPI(), "result")
    return UsdShade.Material.Get(stage, path), shader


def create_coffee_points(stage: Usd.Stage, path: str) -> UsdGeom.Points:
    """Define the grain point cloud and bind its vertex-color material."""
    from pxr import UsdGeom, UsdShade

    grains = UsdGeom.Points.Define(stage, path)
    grains.CreateWidthsAttr()
    grains.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    grains.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
    material, _ = _vertex_color_material(stage, f"{path}/Material", COFFEE_COLOR, roughness=0.85)
    UsdShade.MaterialBindingAPI.Apply(grains.GetPrim()).Bind(material)
    return grains


def create_water_mesh(stage: Usd.Stage, path: str) -> tuple[UsdGeom.Mesh, UsdShade.Shader]:
    """Define the liquid mesh with PreviewSurface and MDL terminals on one material."""
    from pxr import Sdf, UsdGeom, UsdShade

    from isaaclab.utils.assets import retrieve_file_path

    from .assets import COFFEE_ASSET_DIR

    water = UsdGeom.Mesh.Define(stage, path)
    water.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    water.CreateDoubleSidedAttr(True)
    water.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    water.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
    material_path = f"{path}/Material"
    material, preview_shader = _vertex_color_material(stage, material_path, WATER_COLOR, roughness=0.08)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Liquid")
    shader.SetSourceAsset(Sdf.AssetPath(retrieve_file_path(f"{COFFEE_ASSET_DIR}/CoffeeWater.mdl")), "mdl")
    shader.SetSourceAssetSubIdentifier("CoffeeWater", "mdl")
    output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    output.SetRenderType("material")
    for terminal in (material.CreateSurfaceOutput, material.CreateVolumeOutput, material.CreateDisplacementOutput):
        terminal("mdl").ConnectToSource(output)
    UsdShade.MaterialBindingAPI.Apply(water.GetPrim()).Bind(material)
    return water, preview_shader


@wp.func
def _tint_weight(a: wp.vec3, b: wp.vec3, radius: float):
    t = wp.max(1.0 - wp.length_sq(a - b) / (radius * radius), 0.0)
    return t * t


@wp.kernel
def _water_diffusion_support(
    water: wp.array[wp.vec3],
    mass: wp.array[float],
    flags: wp.array[int],
    grid: wp.uint64,
    radius: float,
    support: wp.array[float],
):
    i = wp.tid()
    weight = float(0.0)
    if flags[i] != 0:
        for j in wp.hash_grid_query(grid, water[i], radius):
            if flags[j] != 0:
                weight += _tint_weight(water[i], water[j], radius) * mass[j]
    support[i] = weight


@wp.kernel
def _diffuse_water_color(
    water: wp.array[wp.vec3],
    mass: wp.array[float],
    flags: wp.array[int],
    grid: wp.uint64,
    radius: float,
    fraction: float,
    support: wp.array[float],
    dye: wp.array[float],
    result: wp.array[float],
):
    # Symmetric conductance conserves tracer even for unequal carrier masses.
    # The outgoing sum is bounded by one, making this a convex update for
    # fraction in [0,1]. Use a separate output so neighbour reads are stable.
    i = wp.tid()
    if flags[i] == 0:
        result[i] = dye[i]
        return
    concentration = dye[i] / mass[i]
    change = float(0.0)
    for j in wp.hash_grid_query(grid, water[i], radius):
        if flags[j] != 0:
            weight = _tint_weight(water[i], water[j], radius)
            conductance = mass[i] * mass[j] * weight / wp.max(wp.max(support[i], support[j]), 1.0e-20)
            change += conductance * (dye[j] / mass[j] - concentration)
    result[i] = dye[i] + fraction * change


@wp.kernel
def _color_water_vertices(
    points: wp.array[wp.vec3],
    water: wp.array[wp.vec3],
    mass: wp.array[float],
    flags: wp.array[int],
    dye: wp.array[float],
    grid: wp.uint64,
    radius: float,
    concentration_scale: float,
    clear_color: wp.vec3,
    brown_color: wp.vec3,
    colors: wp.array[wp.vec3],
):
    vertex = wp.tid()
    center = points[vertex]
    tracer = float(0.0)
    carrier = float(0.0)
    for j in wp.hash_grid_query(grid, center, radius):
        if flags[j] != 0:
            weight = _tint_weight(center, water[j], radius)
            tracer += weight * dye[j]
            carrier += weight * mass[j]
    concentration = tracer / wp.max(carrier, 1.0e-20)
    fraction = 1.0 - wp.exp(-concentration / concentration_scale)
    colors[vertex] = wp.lerp(clear_color, brown_color, fraction)


@wp.kernel
def _water_palette_uvs(colors: wp.array[wp.vec3], clear_color: wp.vec3, brown_color: wp.vec3, uvs: wp.array[wp.vec2]):
    vertex = wp.tid()
    fraction = wp.clamp((colors[vertex][2] - clear_color[2]) / (brown_color[2] - clear_color[2]), 0.0, 1.0)
    # Texel centers keep the two-color palette inside the native repeat boundary.
    uvs[vertex] = wp.vec2(0.25 + 0.5 * fraction, 0.5)


@wp.kernel
def _gather_grains(
    ids: wp.array[int],
    grains: wp.array2d[wp.vec3],
    radii: wp.array[float],
    color: wp.vec3,
    liquid_flags: wp.array[int],
    water_count: int,
    points: wp.array[wp.vec3],
    result_radii: wp.array[float],
    widths: wp.array[float],
    colors: wp.array[wp.vec3],
):
    i, j = wp.tid()
    index = i * grains.shape[1] + j
    points[index] = grains[ids[i], j]
    # Preserve aggregate visual volume as sample density increases.
    radius_scale = wp.pow(1.0 / float(grains.shape[1]), 1.0 / 3.0)
    result_radii[index] = 0.9 * radii[ids[i]] * radius_scale
    if liquid_flags and liquid_flags[water_count + i] != 0:
        result_radii[index] = 0.0
    widths[index] = 2.0 * result_radii[index]
    rng = wp.rand_init(index)
    colors[index] = color * (0.75 + 0.5 * wp.randf(rng))
