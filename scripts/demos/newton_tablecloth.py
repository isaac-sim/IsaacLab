# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the classic tablecloth trick at five pull speeds with Newton VBD.

The demo follows Isaac Lab's direct-workflow pattern: an
``InteractiveSceneCfg`` declares the ground, tables, cloths, and tableware;
``DeformableObject`` exposes the cloth's kinematic targets; and a graph-safe
callback updates the pulled edges at the VBD substep rate. Only the physics
configuration and substep controller are Newton-specific.

.. code-block:: bash

    # Newton GL is the default visualizer.
    uv run python scripts/demos/newton_tablecloth.py --device cuda:0

    # Run the complete comparison without rendering.
    uv run python scripts/demos/newton_tablecloth.py --device cuda:0 --visualizer none --max_steps 240

    # Record the complete comparison as a 60 FPS MP4 through Isaac Sim Kit.
    uv run --extra isaacsim --extra video python scripts/demos/newton_tablecloth.py \
        --device cuda:0 --visualizer kit --video
"""

from __future__ import annotations

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Compare five Newton VBD tablecloth pull speeds.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many frames; negative runs forever.")
parser.add_argument("--video", action="store_true", help="Record the complete demo to videos/newton_tablecloth/.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()

import warp as wp
from _newton_tablecloth_utils import (
    BOWL_LOCAL_HEIGHT,
    BOWL_LOCAL_RADIUS,
    BOWL_LOCAL_Z_MIN,
    BOWL_SCALE,
    BOWL_USD,
    FORK_CENTER_OF_MASS,
    FORK_DIAGONAL_INERTIA,
    FORK_LOCAL_Z_MIN,
    FORK_ROTATION,
    FORK_SCALE,
    FORK_USD,
    KITCHEN_ISLAND_SIZE,
    KITCHEN_ISLAND_USD,
    WINE_GLASS_CENTER_OF_MASS,
    WINE_GLASS_DIAGONAL_INERTIA,
    WINE_GLASS_LOCAL_Z_MIN,
    WINE_GLASS_USD,
    VisualTableUsdFileCfg,
    collision_properties,
    create_video_recorder,
    rigid_material,
    rigid_object_cfg,
    tabletop_collider_cfg,
)
from isaaclab_newton.physics import (
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    AssetBaseCfg,
    DeformableObject,
    DeformableObjectCfg,
    RigidObjectCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.physics import PhysicsEvent
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

FPS = 60
SUBSTEPS = 25
VIDEO_STEPS = 240
VIDEO_OUTPUT_DIR = "videos/newton_tablecloth"
VIDEO_RESOLUTION = (1920, 1080)
SETTLE_TIME = 0.5
PULL_DISTANCE = 1.25
PULL_RAMP_TIME = 0.40
PULL_DESCENT_START = 0.95
PULL_SPEEDS = (0.5, 1.0, 1.5, 2.0, 4.0)

# Match the scaled, rotated countertop rather than its slightly smaller nominal dimensions.
TABLE_HALF_WIDTH = 0.5168
TABLE_HALF_DEPTH = 0.2629
TABLE_TOP_Z = 0.75
TABLETOP_HALF_HEIGHT = 0.025

CLOTH_WIDTH = 1.08
CLOTH_DEPTH = 0.70
CLOTH_PARTICLE_RADIUS = 0.0005
TABLEWARE_CLEARANCE = 0.008
LANE_SPACING = 0.95

KITCHEN_ISLAND_SCALE = (TABLE_TOP_Z / KITCHEN_ISLAND_SIZE[2],) * 3
KITCHEN_ISLAND_ROTATION = FORK_ROTATION


@configclass
class TableclothSceneCfg(InteractiveSceneCfg):
    """One Isaac Lab scene containing five side-by-side tablecloth trials."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    def __post_init__(self) -> None:
        """Declare the repeated trials along one row while keeping every asset configurable."""
        super().__post_init__()
        lane_centers = [(lane - 0.5 * (len(PULL_SPEEDS) - 1)) * LANE_SPACING for lane in range(len(PULL_SPEEDS))]
        tableware: dict[str, RigidObjectCfg] = {}
        cloth_z = TABLE_TOP_Z + CLOTH_PARTICLE_RADIUS + 0.002

        for lane, lane_y in enumerate(lane_centers):
            lane_path = f"{{ENV_REGEX_NS}}/Lane{lane}"
            setattr(
                self,
                f"table_{lane}",
                AssetBaseCfg(
                    prim_path=f"{lane_path}/Table",
                    spawn=VisualTableUsdFileCfg(
                        usd_path=KITCHEN_ISLAND_USD,
                        scale=KITCHEN_ISLAND_SCALE,
                        variants={"Physics": "none"},
                        make_uninstanceable=True,
                        rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=False)],
                        collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)],
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, lane_y, 0.0), rot=KITCHEN_ISLAND_ROTATION),
                ),
            )
            setattr(
                self,
                f"tabletop_collider_{lane}",
                AssetBaseCfg(
                    prim_path=f"{lane_path}/TabletopCollider",
                    spawn=tabletop_collider_cfg(
                        (2.0 * TABLE_HALF_WIDTH, 2.0 * TABLE_HALF_DEPTH, 2.0 * TABLETOP_HALF_HEIGHT),
                        friction=1.00,
                        contact_stiffness=1.0e5,
                        contact_damping=1.0e2,
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, lane_y, TABLE_TOP_Z - TABLETOP_HALF_HEIGHT)),
                ),
            )

            setattr(
                self,
                f"cloth_{lane}",
                DeformableObjectCfg(
                    prim_path=f"{lane_path}/Cloth",
                    spawn=sim_utils.MeshRectangleCfg(
                        size=(CLOTH_WIDTH, CLOTH_DEPTH),
                        edge_refinement=24,
                        deformable_props=NewtonDeformableBodyPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.76, 0.05, 0.05)),
                        physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                            density=0.24,
                            particle_radius=CLOTH_PARTICLE_RADIUS,
                            tri_ke=5.0e4,
                            tri_ka=5.0e4,
                            tri_kd=5.0e1,
                            edge_ke=0.10,
                            edge_kd=1.0e-3,
                        ),
                    ),
                    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, lane_y, cloth_z)),
                ),
            )

            tableware[f"bowl_{lane}"] = rigid_object_cfg(
                usd_path=BOWL_USD,
                position=(-0.22, lane_y - 0.08, cloth_z + TABLEWARE_CLEARANCE - BOWL_SCALE * BOWL_LOCAL_Z_MIN),
                mass=0.45,
                collision_proxy_cfg=sim_utils.CylinderCfg(
                    radius=BOWL_LOCAL_RADIUS,
                    height=BOWL_LOCAL_HEIGHT,
                    visible=False,
                    collision_props=collision_properties(),
                    physics_material=rigid_material(
                        density=None,
                        friction=0.70,
                        contact_stiffness=1.0e3,
                        contact_damping=1.0e1,
                    ),
                ),
                collision_proxy_position=(0.0, 0.0, BOWL_LOCAL_Z_MIN + 0.5 * BOWL_LOCAL_HEIGHT),
                scale=(BOWL_SCALE,) * 3,
            ).replace(prim_path=f"{lane_path}/Bowl")
            tableware[f"glass_{lane}"] = rigid_object_cfg(
                usd_path=WINE_GLASS_USD,
                position=(0.02, lane_y + 0.12, cloth_z + TABLEWARE_CLEARANCE - WINE_GLASS_LOCAL_Z_MIN),
                mass=0.35,
                collision_proxy_cfg=sim_utils.CylinderCfg(
                    radius=0.035,
                    height=0.006,
                    visible=False,
                    collision_props=collision_properties(),
                    physics_material=rigid_material(
                        density=None,
                        friction=0.70,
                        contact_stiffness=1.0e3,
                        contact_damping=1.0e1,
                    ),
                ),
                collision_proxy_position=(0.0, 0.0, 0.003),
                center_of_mass=WINE_GLASS_CENTER_OF_MASS,
                diagonal_inertia=WINE_GLASS_DIAGONAL_INERTIA,
                # Approximate the 22.7 cm visual with cheap base, stem, and cup shapes.
                additional_collision_proxy_cfgs=(
                    sim_utils.CylinderCfg(
                        radius=0.004,
                        height=0.096,
                        visible=False,
                        collision_props=collision_properties(),
                        physics_material=rigid_material(
                            density=None,
                            friction=0.70,
                            contact_stiffness=1.0e3,
                            contact_damping=1.0e1,
                        ),
                    ),
                    sim_utils.CapsuleCfg(
                        radius=0.041,
                        height=0.045,
                        visible=False,
                        collision_props=collision_properties(),
                        physics_material=rigid_material(
                            density=None,
                            friction=0.70,
                            contact_stiffness=1.0e3,
                            contact_damping=1.0e1,
                        ),
                    ),
                ),
                additional_collision_proxy_positions=((0.0, 0.0, 0.054), (0.0, 0.0, 0.165)),
                visual_color=(0.55, 0.75, 0.90),
                visual_opacity=0.90,
                visual_roughness=0.12,
            ).replace(prim_path=f"{lane_path}/Glass")
            tableware[f"fork_{lane}"] = rigid_object_cfg(
                usd_path=FORK_USD,
                position=(0.10, lane_y - 0.11, cloth_z + TABLEWARE_CLEARANCE - FORK_SCALE * FORK_LOCAL_Z_MIN),
                mass=0.07,
                collision_proxy_cfg=sim_utils.CuboidCfg(
                    size=(0.014, 0.160, 0.003),
                    visible=False,
                    collision_props=collision_properties(),
                    physics_material=rigid_material(
                        density=None,
                        friction=0.70,
                        contact_stiffness=1.0e3,
                        contact_damping=1.0e1,
                    ),
                ),
                collision_proxy_position=(0.0, 0.0092, 0.0016),
                orientation=FORK_ROTATION,
                scale=(FORK_SCALE,) * 3,
                center_of_mass=FORK_CENTER_OF_MASS,
                diagonal_inertia=FORK_DIAGONAL_INERTIA,
            ).replace(prim_path=f"{lane_path}/Fork")

        self.tableware = RigidObjectCollectionCfg(rigid_objects=tableware)
        self.light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
        )


@wp.kernel
def _advance_pull(
    speeds: wp.array(dtype=float),
    elapsed: wp.array(dtype=float),
    distances: wp.array(dtype=float),
    dt: float,
):
    env_id = wp.tid()
    if elapsed[0] >= SETTLE_TIME:
        ramp_u = wp.clamp((elapsed[0] - SETTLE_TIME) / PULL_RAMP_TIME, 0.0, 1.0)
        ramp = ramp_u * ramp_u * (3.0 - 2.0 * ramp_u)
        distances[env_id] = wp.min(distances[env_id] + ramp * speeds[env_id] * dt, PULL_DISTANCE)


@wp.kernel
def _advance_time(elapsed: wp.array(dtype=float), dt: float):
    elapsed[0] += dt


@wp.kernel
def _update_pulled_edge(
    rest_positions: wp.array2d(dtype=wp.vec3f),
    distances: wp.array(dtype=float),
    lane: int,
    targets: wp.array2d(dtype=wp.vec4f),
):
    env_id, node_id = wp.tid()
    rest = rest_positions[env_id, node_id]
    if rest[0] >= 0.5 * CLOTH_WIDTH - 1.0e-4:
        distance = distances[lane]
        descent = 0.0
        if distance > PULL_DESCENT_START:
            descent = 0.30 * wp.clamp((distance - PULL_DESCENT_START) / (PULL_DISTANCE - PULL_DESCENT_START), 0.0, 1.0)
        targets[env_id, node_id] = wp.vec4f(rest[0] + distance, rest[1], rest[2] - descent, 0.0)


class _TableclothPullController:
    """Advance the five cloth edges at the VBD solver-substep rate."""

    def __init__(self, cloths: list[DeformableObject], device: str, solver_dt: float):
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

        self._cloths = cloths
        self._solver_dt = solver_dt
        self._speeds = wp.array(PULL_SPEEDS, dtype=float, device=device)
        self._distances = wp.zeros(len(PULL_SPEEDS), dtype=float, device=device)
        self._elapsed = wp.zeros(1, dtype=float, device=device)
        self._rest_positions = []
        self._targets = []
        self._physics_ready_handle = NewtonManager.register_callback(
            self._on_physics_ready,
            PhysicsEvent.PHYSICS_READY,
            order=100,
            name="tablecloth_pull_controller",
        )

    def _on_physics_ready(self, _event) -> None:
        """Bind asset buffers before the solver CUDA graph is captured."""
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

        self._elapsed.zero_()
        self._distances.zero_()
        self._rest_positions = [wp.clone(cloth.data.nodal_pos_w.warp) for cloth in self._cloths]
        self._targets = [cloth.data.nodal_kinematic_target.warp for cloth in self._cloths]
        NewtonManager.register_state_force_callback(self._update)

    def _update(self, _state) -> None:
        """Move and enforce each kinematic edge immediately before a solver substep."""
        wp.launch(_advance_time, dim=1, inputs=[self._elapsed, self._solver_dt])
        wp.launch(
            _advance_pull,
            dim=len(PULL_SPEEDS),
            inputs=[self._speeds, self._elapsed, self._distances, self._solver_dt],
        )
        for lane, cloth in enumerate(self._cloths):
            wp.launch(
                _update_pulled_edge,
                dim=(cloth.num_instances, cloth.max_sim_vertices_per_body),
                inputs=[self._rest_positions[lane], self._distances, lane, self._targets[lane]],
            )
            cloth.write_data_to_sim()


def main() -> None:
    """Launch the five-speed tablecloth demo."""
    if args_cli.video and "none" in (args_cli.visualizer or []):
        raise ValueError("--video requires a capture-capable visualizer; omit --visualizer none")
    if args_cli.video and "kit" in (args_cli.visualizer or []):
        args_cli.enable_cameras = True
    max_steps = VIDEO_STEPS if args_cli.video and args_cli.max_steps < 0 else args_cli.max_steps

    physics_cfg = NewtonCfg(
        num_substeps=SUBSTEPS,
        # Fast cloth can cross the contact band within one frame, so collide on every substep.
        collision_decimation=1,
        default_shape_cfg=NewtonShapeCfg(gap=0.001, ke=1.0e5, kd=1.0e2, mu=0.70),
        # Use a polished cloth/tableware interface so realistic lightweight props are not kicked by a fast pull.
        soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=2.0e4, soft_contact_kd=2.0e1, soft_contact_mu=0.03),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="nxn",
            soft_contact_margin=0.005,
            enable_rigid_soft_full_surface_contact=True,
        ),
        solver_cfg=VBDSolverCfg(
            iterations=15,
            rigid_compliant_alm=True,
            rigid_body_particle_contact_buffer_size=8192,
        ),
    )
    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli) as resolved_physics_cfg:
        from isaaclab.scene import InteractiveScene  # noqa: PLC0415

        default_visualizer_cfg = None
        if args_cli.video and "kit" in (args_cli.visualizer or []):
            from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: PLC0415

            default_visualizer_cfg = KitVisualizerCfg(
                window_width=VIDEO_RESOLUTION[0], window_height=VIDEO_RESOLUTION[1]
            )

        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1.0 / FPS,
                device=args_cli.device,
                physics=resolved_physics_cfg,
                default_visualizer_cfg=default_visualizer_cfg,
            )
        )
        sim.set_camera_view(eye=(3.61, -5.225, 2.8825), target=(0.0, 0.0, 0.40))
        scene = InteractiveScene(TableclothSceneCfg(num_envs=1, env_spacing=1.0))
        cloths = [scene[f"cloth_{lane}"] for lane in range(len(PULL_SPEEDS))]
        _pull_controller = _TableclothPullController(cloths, scene.device, sim.get_physics_dt() / SUBSTEPS)
        sim.reset()

        sim_dt = sim.get_physics_dt()
        video_recorder = create_video_recorder(
            sim,
            enabled=args_cli.video,
            output_dir=VIDEO_OUTPUT_DIR,
            filename_prefix="newton_tablecloth",
            video_length=max_steps,
            fps=FPS,
        )

        print("[INFO]: Setup complete. Five side-by-side Isaac Lab tablecloth trials are ready.", flush=True)
        step = 0
        try:
            while sim.is_headless_or_exist_active_visualizer() and (max_steps < 0 or step < max_steps):
                sim.step()
                scene.update(sim_dt)
                if video_recorder is not None:
                    video_recorder.step()
                step += 1
        finally:
            if video_recorder is not None:
                video_recorder.close()


if __name__ == "__main__":
    main()
