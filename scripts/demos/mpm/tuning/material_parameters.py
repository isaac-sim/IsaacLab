# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare Newton MPM material parameters in controlled scenes.

The comparisons borrow the visual grammar and parameter contexts of Daviet's
mixed-MPM Figure 23: equal-volume specimens fall onto identical cylinders while
one material field changes. Elasticity comparisons use spherical impactors on
plates because compression and rebound read more clearly there. Every preset
uses a 25 mm MPM grid with two particles per voxel axis; spatial resolution is
therefore not another experimental variable.

.. code-block:: bash

    uv run python scripts/demos/mpm/tuning/material_parameters.py \
      --preset young_modulus --visualizer kit
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation


class TuningPreset(NamedTuple):
    """Definition of one controlled MPM material comparison."""

    title: str
    subtitle: str
    field: str
    values: tuple[float, ...]
    labels: tuple[str, ...]
    colors: tuple[tuple[float, float, float], ...]
    geometry: str
    fixed: dict[str, float]


PRESETS = {
    "young_modulus": TuningPreset(
        title="Elastic stiffness",
        subtitle="Young's modulus E · equal spheres · yielding disabled",
        field="young_modulus",
        values=(1.0e4, 1.0e5, 1.0e6),
        labels=("SOFT · 10 kPa", "COMPLIANT · 100 kPa", "STIFF ELASTIC · 1 MPa"),
        colors=((0.10, 0.42, 0.92), (0.52, 0.24, 0.86), (0.92, 0.16, 0.12)),
        geometry="sphere_plate",
        fixed={
            "friction": 0.0,
            "damping": 0.02,
            "yield_pressure": 1.0e20,
            "tensile_yield_ratio": 1.0,
            "yield_stress": 1.0e20,
        },
    ),
    "poisson_ratio": TuningPreset(
        title="Compressibility",
        subtitle="Poisson ratio ν · equal elastic spheres · E = 20 kPa",
        field="poisson_ratio",
        values=(0.05, 0.30, 0.499),
        labels=("COMPRESSIBLE · ν = 0.05", "BALANCED · ν = 0.30", "NEAR-INCOMPRESSIBLE · ν = 0.499"),
        colors=((0.12, 0.60, 0.88), (0.22, 0.72, 0.48), (0.94, 0.48, 0.10)),
        geometry="sphere_plate",
        fixed={
            "young_modulus": 2.0e4,
            "friction": 0.0,
            "damping": 0.02,
            "yield_pressure": 1.0e20,
            "tensile_yield_ratio": 1.0,
            "yield_stress": 1.0e20,
        },
    ),
    "friction": TuningPreset(
        title="Granular friction",
        subtitle="Internal friction μ only · cohesion unchanged · paper-inspired cylinder drop",
        field="friction",
        values=(0.0, 0.68, 2.0),
        labels=("NO FRICTION · μ = 0", "DRY SAND · μ = 0.68", "HIGH-FRICTION WET SAND · μ = 2"),
        colors=((0.18, 0.62, 0.92), (0.78, 0.47, 0.16), (0.68, 0.20, 0.12)),
        geometry="cuboid_cylinder",
        fixed={"young_modulus": 1.0e15, "yield_pressure": 1.0e12, "tensile_yield_ratio": 0.0},
    ),
    "tensile_yield_ratio": TuningPreset(
        title="Tensile strength",
        subtitle="Tensile yield ratio β · p_c = 4 MPa · tensile limits 0 / 40 / 200 kPa",
        field="tensile_yield_ratio",
        values=(0.0, 0.01, 0.05),
        labels=("NO TENSION · β = 0", "FRIABLE · β = 0.01", "COHESIVE · β = 0.05"),
        colors=((0.88, 0.50, 0.12), (0.34, 0.64, 0.88), (0.72, 0.22, 0.62)),
        geometry="cuboid_cylinder",
        fixed={"young_modulus": 1.0e7, "friction": 0.68, "yield_pressure": 4.0e6},
    ),
    "yield_pressure": TuningPreset(
        title="Pressure yielding",
        subtitle="Yield pressure p_c · β = 0.5 · ξ = 5 · θ = 1",
        field="yield_pressure",
        values=(1.0e5, 1.0e6, 4.0e6),
        labels=("YIELDS EARLY · 100 kPa", "INTERMEDIATE · 1 MPa", "STRONG · 4 MPa"),
        colors=((0.12, 0.60, 0.88), (0.95, 0.50, 0.08), (0.42, 0.24, 0.72)),
        geometry="cuboid_cylinder",
        fixed={"young_modulus": 1.0e7, "tensile_yield_ratio": 0.5, "hardening": 5.0, "dilatancy": 1.0},
    ),
    "hardening": TuningPreset(
        title="Plastic hardening",
        subtitle="ξ = 0 keeps full p_c · ξ > 0 builds strength after compaction",
        field="hardening",
        values=(0.0, 0.05, 5.0),
        labels=("FIXED STRENGTH · ξ = 0", "SLOW BUILD-UP · ξ = 0.05", "RAPID BUILD-UP · ξ = 5"),
        colors=((0.12, 0.62, 0.88), (0.94, 0.66, 0.10), (0.48, 0.20, 0.78)),
        geometry="cuboid_cylinder",
        fixed={
            "young_modulus": 1.0e7,
            "yield_pressure": 4.0e6,
            "tensile_yield_ratio": 0.5,
            "dilatancy": 1.0,
        },
    ),
    "dilatancy": TuningPreset(
        title="Shear dilatancy",
        subtitle="Dilatancy θ · p_c = 4 MPa · β = 0.05 · ξ = 5",
        field="dilatancy",
        values=(0.0, 0.1, 1.0),
        labels=("COMPACTS · θ = 0", "LOW DILATANCY · θ = 0.1", "EXPANDS · θ = 1"),
        colors=((0.16, 0.58, 0.92), (0.18, 0.72, 0.48), (0.96, 0.52, 0.10)),
        geometry="cuboid_cylinder",
        fixed={
            "young_modulus": 1.0e7,
            "yield_pressure": 4.0e6,
            "tensile_yield_ratio": 0.05,
            "hardening": 5.0,
        },
    ),
    "yield_stress": TuningPreset(
        title="Cohesive yield stress",
        subtitle="Von Mises yield stress τ_c · frictionless clay-like impact · β = 1",
        field="yield_stress",
        values=(0.0, 1.0e4, 2.0e4),
        labels=("NONE · 0 Pa", "COHESIVE · 10 kPa", "STRONG · 20 kPa"),
        colors=((0.14, 0.62, 0.90), (0.76, 0.34, 0.18), (0.48, 0.18, 0.12)),
        geometry="cuboid_cylinder",
        fixed={
            "young_modulus": 1.0e7,
            "friction": 0.0,
            "yield_pressure": 1.0e12,
            "tensile_yield_ratio": 1.0,
        },
    ),
    "viscosity": TuningPreset(
        title="Plastic viscosity",
        subtitle="Viscosity η · paper-inspired granular cylinder drop",
        field="viscosity",
        values=(0.0, 10.0, 500.0),
        labels=("FREE-FLOWING · 0 Pa·s", "PASTE · 10 Pa·s", "HIGH VISCOSITY · 500 Pa·s"),
        colors=((0.08, 0.62, 0.95), (0.24, 0.40, 0.84), (0.42, 0.16, 0.68)),
        geometry="cuboid_cylinder",
        fixed={"young_modulus": 1.0e7, "friction": 0.68, "yield_pressure": 1.0e12},
    ),
    "particle_jitter": TuningPreset(
        title="Particle packing",
        subtitle="Same dry sand · only initial lattice jitter changes",
        field="particle_jitter_fraction",
        values=(0.0, 0.30),
        labels=("ALIGNED GRID · NO JITTER", "IRREGULAR PACKING · 30% JITTER"),
        colors=((0.84, 0.48, 0.14), (0.90, 0.66, 0.18)),
        geometry="cuboid_cylinder",
        fixed={"young_modulus": 1.0e15, "friction": 0.68, "yield_pressure": 1.0e12},
    ),
}

SIMULATION_HZ = 120
DEFAULT_VOXEL_SIZE = 0.025
PARTICLES_PER_VOXEL_AXIS = 2.0
DEFAULT_PARTICLE_JITTER_FRACTION = 0.30
DEFAULT_PARTICLE_SEED = 42
DENSITY = 1000.0
DEFAULT_CAMERA_EYE = (0.0, -7.4, 3.0)
DEFAULT_CAMERA_TARGET = (0.0, 0.0, 0.70)

parser = argparse.ArgumentParser(description="Controlled Newton MPM material tuning comparisons.")
parser.add_argument("--preset", choices=tuple(PRESETS), default="young_modulus", help="Material comparison to run.")
parser.add_argument(
    "--variant_index",
    type=int,
    default=None,
    help="Render one zero-based material variant centered in frame; omit for the side-by-side scene.",
)
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
parser.add_argument(
    "--voxel_size",
    type=float,
    default=DEFAULT_VOXEL_SIZE,
    help=f"MPM grid voxel size [m]. Defaults to {DEFAULT_VOXEL_SIZE:g}.",
)
parser.add_argument(
    "--particle_jitter_fraction",
    type=float,
    default=DEFAULT_PARTICLE_JITTER_FRACTION,
    help="Maximum per-axis particle jitter as a fraction of particle spacing in [0, 0.5].",
)
parser.add_argument(
    "--particle_seed",
    type=int,
    default=DEFAULT_PARTICLE_SEED,
    help=f"Seed for deterministic particle jitter. Defaults to {DEFAULT_PARTICLE_SEED}.",
)
parser.add_argument(
    "--simulation_hz",
    type=int,
    default=SIMULATION_HZ,
    help=f"Physics frequency [Hz]. Defaults to {SIMULATION_HZ}.",
)
parser.add_argument(
    "--solver_iterations",
    type=int,
    default=60,
    help="Maximum implicit MPM solver iterations per physics step. Defaults to 60.",
)
parser.add_argument("--disable_cuda_graph", action="store_true", help="Disable CUDA graph capture for debugging.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

if not 0.0 <= args_cli.particle_jitter_fraction <= 0.5:
    parser.error("--particle_jitter_fraction must be in [0, 0.5].")
if not np.isfinite(args_cli.voxel_size) or args_cli.voxel_size <= 0:
    parser.error("--voxel_size must be finite and positive.")
if not np.isfinite(args_cli.simulation_hz) or args_cli.simulation_hz <= 0:
    parser.error("--simulation_hz must be finite and positive.")
if args_cli.solver_iterations <= 0:
    parser.error("--solver_iterations must be positive.")

PRESET = PRESETS[args_cli.preset]
if args_cli.preset in {"friction", "yield_pressure", "yield_stress"}:
    CAMERA_EYE = (0.0, -10.5, 4.3)
    CAMERA_TARGET = (0.0, 0.0, 0.65)
else:
    CAMERA_EYE = DEFAULT_CAMERA_EYE
    CAMERA_TARGET = DEFAULT_CAMERA_TARGET
if args_cli.preset == "yield_pressure":
    SPECIMEN_INITIAL_Z = 1.90
    SPECIMEN_INITIAL_VELOCITY_Z = -8.0
elif PRESET.geometry == "sphere_plate":
    SPECIMEN_INITIAL_Z = 1.25
    SPECIMEN_INITIAL_VELOCITY_Z = -0.20
else:
    SPECIMEN_INITIAL_Z = 1.55
    SPECIMEN_INITIAL_VELOCITY_Z = 0.0
if args_cli.variant_index is not None and not 0 <= args_cli.variant_index < len(PRESET.values):
    parser.error(f"--variant_index must be between 0 and {len(PRESET.values) - 1} for {args_cli.preset!r}")
ACTIVE_VARIANT_INDICES = (
    tuple(range(len(PRESET.values))) if args_cli.variant_index is None else (args_cli.variant_index,)
)
PARTICLE_SPACING = args_cli.voxel_size / PARTICLES_PER_VOXEL_AXIS
PARTICLE_RADIUS = 0.5 * PARTICLE_SPACING
PARTICLE_MASS = PARTICLE_SPACING**3 * DENSITY
SPECIMEN_NAMES = (
    tuple(f"specimen_{index}" for index in ACTIVE_VARIANT_INDICES) if args_cli.variant_index is None else ("specimen",)
)


def particle_jitter_fraction_for_variant(index: int) -> float:
    """Return the resolved initial-packing jitter fraction for one variant."""
    if PRESET.field == "particle_jitter_fraction":
        return PRESET.values[index]
    return args_cli.particle_jitter_fraction


def material_values_for_variant(index: int) -> dict[str, float]:
    """Return material values authored explicitly for one specimen variant."""
    material_values = {
        "density": DENSITY,
        "young_modulus": 1.0e7,
        "poisson_ratio": 0.3,
        "friction": 0.68,
        "yield_pressure": 1.0e12,
        **PRESET.fixed,
    }
    if PRESET.field != "particle_jitter_fraction":
        material_values[PRESET.field] = PRESET.values[index]
    return material_values


def create_visualizer_cfgs():
    """Create the configured Kit or Newton visualizer."""
    requested = args_cli.visualizer or []
    cfgs = []
    if "kit" in requested:
        from isaaclab_visualizers.kit import KitVisualizerCfg

        cfgs.append(KitVisualizerCfg(eye=CAMERA_EYE, lookat=CAMERA_TARGET, focal_length=32.0))
    if not {"newton", "newton_gl", "newton_rtx"}.intersection(requested):
        return cfgs

    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

    for name in requested:
        if name not in {"newton", "newton_gl", "newton_rtx"}:
            continue
        cfg_type = NewtonRTXVisualizerCfg if name == "newton_rtx" else NewtonGLVisualizerCfg
        visualizer_kwargs = {"rtx_environment": "studio"} if name == "newton_rtx" else {}
        cfgs.append(
            cfg_type(
                eye=CAMERA_EYE,
                lookat=CAMERA_TARGET,
                streaming_view=False,
                show_particles=True,
                **visualizer_kwargs,
            )
        )
    return cfgs


def create_sim_cfg():
    """Create the shared implicit-MPM solver configuration."""
    from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg

    import isaaclab.sim as sim_utils

    return sim_utils.SimulationCfg(
        dt=1.0 / args_cli.simulation_hz,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        visualizer_cfgs=create_visualizer_cfgs(),
        physics=NewtonCfg(
            solver_cfg=MPMSolverCfg(
                voxel_size=args_cli.voxel_size,
                grid_type="sparse",
                grid_padding=0,
                max_active_cell_count=1 << 18,
                max_leaf_node_count=1 << 15,
                max_lower_node_count=1 << 11,
                max_upper_node_count=1 << 9,
                max_iterations=args_cli.solver_iterations,
                tolerance=1.0e-4,
                solver="auto",
                warmstart_mode="auto",
                transfer_scheme="apic",
                integration_scheme="pic",
                strain_basis="P0",
                velocity_basis="Q1",
                collider_basis="S2",
                air_drag=1.0e-3,
            ),
            use_cuda_graph=not args_cli.disable_cuda_graph,
        ),
    )


def create_scene_cfg():
    """Create the side-by-side tuning specimens and their impact plinths."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass

    centers_x = np.linspace(-1.45, 1.45, len(PRESET.values)) if len(PRESET.values) > 1 else np.zeros(1)

    def specimen_cfg(index: int, center_x: float) -> MPMObjectCfg:
        points = create_specimen_points(PRESET.geometry, index)
        velocities = np.zeros_like(points)
        velocities[:, 2] = SPECIMEN_INITIAL_VELOCITY_Z
        return MPMObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Specimen_{index}",
            spawn=MPMPointsCfg(
                positions=points.tolist(),
                velocities=velocities.tolist(),
                mass=PARTICLE_MASS,
                radius=PARTICLE_RADIUS,
                material=MPMParticleMaterialCfg(**material_values_for_variant(index)),
                visual_color=PRESET.colors[index],
            ),
            init_state=MPMObjectCfg.InitialStateCfg(pos=(center_x, 0.0, SPECIMEN_INITIAL_Z)),
        )

    def collider_cfg(index: int, center_x: float) -> AssetBaseCfg:
        if PRESET.geometry == "sphere_plate":
            spawn = sim_utils.CuboidCfg(
                size=(1.05, 0.90, 0.10),
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_margin=PARTICLE_SPACING,
                ),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=0.6,
                    dynamic_friction=0.6,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.20, 0.22, 0.26),
                    roughness=0.3,
                ),
            )
            position = (center_x, 0.0, 0.05)
            orientation = (0.0, 0.0, 0.0, 1.0)
        else:
            # The cylinder axes point along the camera direction so the complete
            # split, drape, and final angle of repose remain visible in profile.
            spawn = sim_utils.CylinderCfg(
                radius=0.20,
                height=1.15,
                axis="Y",
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_margin=PARTICLE_SPACING,
                ),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=0.55,
                    dynamic_friction=0.55,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.24, 0.25, 0.27),
                    roughness=0.18,
                    metallic=0.15,
                ),
            )
            position = (center_x, 0.12, 0.42)
            orientation = (0.0, 0.0, 0.0, 1.0)
        return AssetBaseCfg(
            prim_path=f"/World/Collider_{index}",
            spawn=spawn,
            init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=orientation),
        )

    ground_cfg = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg(size=(20.0, 16.0)))
    light_cfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.82, 0.84, 0.88)),
    )

    if args_cli.variant_index is not None:
        variant_index = args_cli.variant_index

        @configclass
        class SingleMaterialTuningSceneCfg(InteractiveSceneCfg):
            """Scene containing one centered MPM material variant."""

            ground = ground_cfg
            dome_light = light_cfg
            collider = collider_cfg(variant_index, 0.0)
            specimen = specimen_cfg(variant_index, 0.0)

        return SingleMaterialTuningSceneCfg(num_envs=1, env_spacing=0.0)

    @configclass
    class MaterialTuningSceneCfg(InteractiveSceneCfg):
        """Scene containing up to three independently authored MPM materials."""

        ground = ground_cfg
        dome_light = light_cfg

        collider_0 = collider_cfg(0, float(centers_x[0]))
        collider_1: AssetBaseCfg | None = collider_cfg(1, float(centers_x[1])) if len(PRESET.values) > 1 else None
        collider_2: AssetBaseCfg | None = collider_cfg(2, float(centers_x[2])) if len(PRESET.values) > 2 else None

        specimen_0 = specimen_cfg(0, float(centers_x[0]))
        specimen_1: MPMObjectCfg | None = specimen_cfg(1, float(centers_x[1])) if len(PRESET.values) > 1 else None
        specimen_2: MPMObjectCfg | None = specimen_cfg(2, float(centers_x[2])) if len(PRESET.values) > 2 else None

    return MaterialTuningSceneCfg(num_envs=1, env_spacing=0.0)


def create_specimen_points(geometry: str, variant_index: int) -> np.ndarray:
    """Create deterministic cell-centered particles for a cuboid or sphere."""
    spacing = PARTICLE_SPACING
    radius = 0.38
    if geometry == "cuboid_cylinder":
        x_axis = np.arange(-0.34 + 0.5 * spacing, 0.34, spacing)
        y_axis = np.arange(-0.25 + 0.5 * spacing, 0.25, spacing)
        z_axis = np.arange(-0.36 + 0.5 * spacing, 0.36, spacing)
        points = np.stack(np.meshgrid(x_axis, y_axis, z_axis, indexing="ij"), axis=-1).reshape(-1, 3)
    elif geometry == "sphere_plate":
        axis = np.arange(-radius + 0.5 * spacing, radius, spacing)
        points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        points = points[np.linalg.norm(points, axis=1) <= radius - 0.25 * spacing]
    else:
        raise ValueError(f"Unsupported specimen geometry: {geometry!r}.")
    if points.shape[0] == 0:
        raise RuntimeError("Specimen generation produced no particles; reduce --voxel_size.")
    # Reuse one seeded offset field so every material variant starts from the
    # same irregular packing and the comparison continues to isolate one field.
    jitter_fraction = particle_jitter_fraction_for_variant(variant_index)
    jitter = jitter_fraction * spacing
    # Every material variant receives the same deterministic normalized offsets;
    # this keeps packing noise from masquerading as a constitutive difference.
    points += np.random.default_rng(args_cli.particle_seed).uniform(-jitter, jitter, points.shape)
    return points.astype(np.float32)


def run_simulator(sim, scene) -> None:
    """Run the comparison until the viewer closes or the step limit is reached."""
    sim_dt = sim.get_physics_dt()
    count = 0
    while sim.is_headless_or_exist_active_visualizer() and (args_cli.max_steps < 0 or count < args_cli.max_steps):
        sim.step(render=False)
        scene.update(sim_dt)
        if sim.is_rendering:
            sim.render()
        count += 1


def main() -> None:
    """Launch the selected MPM material comparison."""
    sim_cfg = create_sim_cfg()
    with launch_simulation(sim_cfg, args_cli):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(create_scene_cfg())
        sim.reset()
        sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
        particle_count = sum(scene[name].particles_per_object for name in SPECIMEN_NAMES)
        print(
            f"[INFO]: {PRESET.title}: {PRESET.subtitle}. "
            f"Spawned {len(SPECIMEN_NAMES)} specimens with {particle_count} particles.",
            flush=True,
        )
        for index in ACTIVE_VARIANT_INDICES:
            print(f"[INFO]: {PRESET.labels[index]}", flush=True)
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
